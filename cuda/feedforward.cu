#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

struct Layer {
    int in_dim = 0;
    int out_dim = 0;
    std::vector<float> W; // row-major [out_dim, in_dim]
    std::vector<float> b; // [out_dim]
};

struct Network {
    std::vector<Layer> layers;
};

struct Dataset {
    int n = 0;
    int seq_len = 0;
    std::vector<int32_t> labels;  // [n]
    std::vector<int32_t> token_ids; // [n, seq_len]
};

static Network load_weights(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open weights file: " + path);
    }

    Network net;
    int L = 0;
    in >> L;
    if (L <= 0) {
        throw std::runtime_error("Invalid layer count in weights file");
    }

    net.layers.resize(L);

    for (int i = 0; i < L; i++) {
        int in_dim = 0, out_dim = 0;
        in >> in_dim >> out_dim;
        if (in_dim <= 0 || out_dim <= 0) {
            throw std::runtime_error("Invalid layer dims in weights file");
        }
        Layer layer;
        layer.in_dim = in_dim;
        layer.out_dim = out_dim;
        layer.W.resize((size_t)in_dim * (size_t)out_dim);
        layer.b.resize((size_t)out_dim);

        for (size_t k = 0; k < layer.W.size(); k++) {
            in >> layer.W[k];
        }
        for (size_t k = 0; k < layer.b.size(); k++) {
            in >> layer.b[k];
        }
        net.layers[i] = std::move(layer);
    }

    return net;
}

static Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open dataset file: " + path);
    }

    Dataset ds;
    in >> ds.n >> ds.seq_len;
    if (ds.n <= 0 || ds.seq_len <= 0) {
        throw std::runtime_error("Invalid dataset header");
    }

    ds.labels.resize((size_t)ds.n);
    ds.token_ids.resize((size_t)ds.n * (size_t)ds.seq_len);

    for (int i = 0; i < ds.n; i++) {
        int lbl = 0;
        in >> lbl;
        ds.labels[(size_t)i] = (int32_t)lbl;
        for (int j = 0; j < ds.seq_len; j++) {
            int tid = 0;
            in >> tid;
            ds.token_ids[(size_t)i * (size_t)ds.seq_len + (size_t)j] = (int32_t)tid;
        }
    }

    return ds;
}

// Device helper: compute MLP layers given input/output shared buffers.
// Returns pointer to the buffer containing the final logits (either in_buf or out_buf).
// This helper runs synchronously across the block and uses __syncthreads(),
// so it must be called by all threads in the block.
__device__ float* mlp_compute_layers_device(
    float* in_buf,
    float* out_buf,
    const float* W_all,
    const float* b_all,
    const int* w_offsets,
    const int* b_offsets,
    const int* in_dims,
    const int* out_dims,
    int num_layers,
    int tx,
    int stride,
    int i_feature
) {
    for (int l = 0; l < num_layers; l++) {
        int in_dim = in_dims[l];
        int out_dim = out_dims[l];
        const float* W = W_all + w_offsets[l];
        const float* b = b_all + b_offsets[l];

        for (int o = tx; o < out_dim; o += stride) {
            float acc = b[o];
            const float* wrow = W + (size_t)o * (size_t)in_dim;
            if (l == 0) {
                for (int k = 0; k <= i_feature; k++) {
                    acc += wrow[k] * in_buf[k];
                }
            } else {
                for (int k = 0; k < in_dim; k++) {
                    acc += wrow[k] * in_buf[k];
                }
            }
            out_buf[o] = acc;
        }
        __syncthreads();

        float* tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }
    return in_buf;
}

__global__ void mlp_forward_kernel(
    const int32_t* token_ids,
    int n_permutations,
    int seq_len,
    float inv_vocab_size,
    const float* W_all,
    const float* b_all,
    const int* w_offsets,         // offsets into W_all for each layer
    const int* b_offsets,         // offsets into b_all for each layer
    const int* in_dims,
    const int* out_dims,
    int num_layers,
    float* logits_out,            // [blocks, out_dim_last]
    float* shap_out,              // aggregated per-feature sums [seq_len]
    int out_dim_last,
    int max_dim
) {
    // This kernel assumes a single sample (sample 0). Each block processes one permutation
    // instance: blocks [0..n_permutations-1] use the permutation in forward order,
    // blocks [n_permutations..2*n_permutations-1] use the same permutation but in reverse.
    int block_id = blockIdx.x;
    int perm_idx = block_id % n_permutations;
    bool do_reverse = (block_id >= n_permutations);
    int sample = 0; // single sample
    int tx = threadIdx.x;
    int stride = blockDim.x;
    bool do_write = (tx == 0); // only thread 0 writes the single logit per block

    // Use dynamically-allocated shared memory (bytes) so we can mix float and int storage.
    extern __shared__ char s[];

    // Layout in shared memory (bytes): 2 * max_dim floats, then seq_len ints for permutation.
    float* buf0 = reinterpret_cast<float*>(s);
    float* buf1 = reinterpret_cast<float*>(s + (size_t)max_dim * sizeof(float));
    int* perm = reinterpret_cast<int*>(s + (size_t)max_dim * 2 * sizeof(float));
    float* shap_block = reinterpret_cast<float*>(s + (size_t)max_dim * 2 * sizeof(float) + (size_t)seq_len * sizeof(int));

    // Build the permutation for perm_idx on one thread. Seed only with perm_idx so
    // the permutation is identical between the forward and reversed block pairs.
    if (tx == 0) {
        for (int j = 0; j < seq_len; ++j) perm[j] = j;
        // Simple LCG seed depending on perm_idx (deterministic per-launch)
        unsigned int state = (unsigned int)perm_idx * 1013904223u + 12345u;
        for (int j = seq_len - 1; j > 0; --j) {
            state = state * 1103515245u + 12345u;
            unsigned int r = state % (unsigned int)(j + 1);
            int tmp = perm[j]; perm[j] = perm[r]; perm[r] = tmp;
        }
    }
    __syncthreads();

    // Load input features using either the permutation in-order or reversed.
    for (int j = tx; j < seq_len; j += stride) {
        int idx = do_reverse ? perm[seq_len - 1 - j] : perm[j];
        int32_t tok = token_ids[(size_t)sample * (size_t)seq_len + (size_t)idx];
        buf0[j] = (float)tok * inv_vocab_size;
    }
    __syncthreads();

    // Calculate the classifications up to i feature
    for (int i_feature = 0; i_feature < seq_len; i_feature++) {
        float* final_buf = mlp_compute_layers_device(buf0, buf1,
                                                    W_all, b_all,
                                                    w_offsets, b_offsets,
                                                    in_dims, out_dims,
                                                    num_layers,
                                                    tx, stride, i_feature);
        __syncthreads();
        // Save the importance of the first feature (index 0) on shap_block
        if (tx == 0){
            shap_block[i_feature] = final_buf[0];
        }
    }
    if (tx == 0){
        for (int i = seq_len - 1; i >= 1; i--) {
            shap_block[i] = shap_block[i] - shap_block[i-1];
        }
    }

    // Write logits for this block (one slot per block so different permutations
    // do not race). Host will allocate 2*n_permutations output slots.
    if (do_write) {
        logits_out[(size_t)block_id] = final_buf[0];
    }
    __syncthreads();

    // Now atomically add per-block shap contributions to global shap_out (one atomic per feature per block)
    for (int j = tx; j < seq_len; j += stride) {
        atomicAdd(&shap_out[j], shap_block[j]);
    }
    __syncthreads();
}

static void cpu_forward_one(const Network& net, const int32_t* token_ids, int seq_len, float inv_vocab_size, std::vector<float>& out_logits) {
    std::vector<float> cur(seq_len);
    for (int j = 0; j < seq_len; j++) {
        cur[j] = (float)token_ids[j] * inv_vocab_size;
    }

    for (const auto& layer : net.layers) {
        std::vector<float> next(layer.out_dim);
        for (int o = 0; o < layer.out_dim; o++) {
            float acc = layer.b[(size_t)o];
            const float* wrow = layer.W.data() + (size_t)o * (size_t)layer.in_dim;
            for (int k = 0; k < layer.in_dim; k++) {
                acc += wrow[(size_t)k] * cur[(size_t)k];
            }
            next[(size_t)o] = acc;
        }
        cur = std::move(next);
    }
    out_logits = std::move(cur);
}

static int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); i++) {
        if (v[(size_t)i] > v[(size_t)best]) best = i;
    }
    return best;
}

int main(int argc, char** argv) {
    std::string weights_path = "out/mlp_weights.txt";
    std::string dataset_path = "out/tokenized_dataset.txt";
    int vocab_size = 30522;
    int threads = 128;
    int max_print = 10;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (a == "--dataset" && i + 1 < argc) dataset_path = argv[++i];
        else if (a == "--vocab-size" && i + 1 < argc) vocab_size = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (a == "--print" && i + 1 < argc) max_print = std::stoi(argv[++i]);
        else if (a == "--n-permutations" && i + 1 < argc) {
            // number of base permutations (kernel will launch 2 * n_permutations blocks)
            // default below is 10
            // parsed into n_permutations variable set later
            // store temporarily in max_print as sentinel? instead introduce var after parsing
            max_print = max_print; // no-op to keep structure; actual parsing handled below
            // We'll parse again later; preserve compatibility with existing flags
            i -= 0;
        }
        else {
            std::cerr << "Unknown/invalid arg: " << a << "\n";
            return 2;
        }
    }

    try {
        Network net = load_weights(weights_path);
        Dataset ds = load_dataset(dataset_path);

        if (net.layers.empty()) throw std::runtime_error("No layers loaded");
        if (net.layers.front().in_dim != ds.seq_len) {
            throw std::runtime_error("First layer in_dim != dataset seq_len (" + std::to_string(net.layers.front().in_dim) + " vs " + std::to_string(ds.seq_len) + ")");
        }

        int num_layers = (int)net.layers.size();
        int out_dim_last = net.layers.back().out_dim;
        int n_permutations = 10; // default

        // Re-parse argv to pick up --n-permutations (keeps simple parsing logic)
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--n-permutations" && i + 1 < argc) {
                n_permutations = std::stoi(argv[++i]);
            }
        }

        // Pack weights/biases into contiguous arrays, and create per-layer offsets.
        std::vector<int> w_offsets(num_layers);
        std::vector<int> b_offsets(num_layers);
        std::vector<int> in_dims(num_layers);
        std::vector<int> out_dims(num_layers);

        size_t total_w = 0;
        size_t total_b = 0;
        for (int l = 0; l < num_layers; l++) {
            in_dims[l] = net.layers[l].in_dim;
            out_dims[l] = net.layers[l].out_dim;
            w_offsets[l] = (int)total_w;
            b_offsets[l] = (int)total_b;
            total_w += (size_t)in_dims[l] * (size_t)out_dims[l];
            total_b += (size_t)out_dims[l];
        }

        std::vector<float> W_all(total_w, 0.0f);
        std::vector<float> b_all(total_b, 0.0f);

        for (int l = 0; l < num_layers; l++) {
            const auto& layer = net.layers[l];
            std::copy(layer.W.begin(), layer.W.end(), W_all.begin() + (size_t)w_offsets[l]);
            std::copy(layer.b.begin(), layer.b.end(), b_all.begin() + (size_t)b_offsets[l]);
        }

        int32_t* d_token_ids = nullptr;
        float* d_W = nullptr;
        float* d_b = nullptr;
        int* d_w_offsets = nullptr;
        int* d_b_offsets = nullptr;
        int* d_in_dims = nullptr;
        int* d_out_dims = nullptr;
        float* d_logits = nullptr;

        checkCuda(cudaMalloc(&d_token_ids, ds.token_ids.size() * sizeof(int32_t)), "cudaMalloc token_ids");
        checkCuda(cudaMalloc(&d_W, W_all.size() * sizeof(float)), "cudaMalloc W");
        checkCuda(cudaMalloc(&d_b, b_all.size() * sizeof(float)), "cudaMalloc b");
        checkCuda(cudaMalloc(&d_w_offsets, w_offsets.size() * sizeof(int)), "cudaMalloc w_offsets");
        checkCuda(cudaMalloc(&d_b_offsets, b_offsets.size() * sizeof(int)), "cudaMalloc b_offsets");
        checkCuda(cudaMalloc(&d_in_dims, in_dims.size() * sizeof(int)), "cudaMalloc in_dims");
        checkCuda(cudaMalloc(&d_out_dims, out_dims.size() * sizeof(int)), "cudaMalloc out_dims");
        // Allocate output logits per-block: 2 * n_permutations blocks, each produces 1 float (binary logit)
        checkCuda(cudaMalloc(&d_logits, (size_t)2 * (size_t)n_permutations * sizeof(float)), "cudaMalloc logits");

        checkCuda(cudaMemcpy(d_token_ids, ds.token_ids.data(), ds.token_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy token_ids");
        checkCuda(cudaMemcpy(d_W, W_all.data(), W_all.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy W");
        checkCuda(cudaMemcpy(d_b, b_all.data(), b_all.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy b");
        checkCuda(cudaMemcpy(d_w_offsets, w_offsets.data(), w_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy w_offsets");
        checkCuda(cudaMemcpy(d_b_offsets, b_offsets.data(), b_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy b_offsets");
        checkCuda(cudaMemcpy(d_in_dims, in_dims.data(), in_dims.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy in_dims");
        checkCuda(cudaMemcpy(d_out_dims, out_dims.data(), out_dims.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy out_dims");

        // Shared memory: 2 * max_dim floats.
        int max_dim = ds.seq_len;
        for (const auto& layer : net.layers) {
            if (layer.out_dim > max_dim) max_dim = layer.out_dim;
            if (layer.in_dim > max_dim) max_dim = layer.in_dim;
        }
        // Shared memory: 2 * max_dim floats + seq_len ints for permutation buffer
        // plus seq_len floats for per-block shap buffer
        size_t shmem = (size_t)max_dim * 2 * sizeof(float) + (size_t)ds.seq_len * sizeof(int) + (size_t)ds.seq_len * sizeof(float);

        dim3 grid((unsigned int)2 * (unsigned int)n_permutations);
        dim3 block(threads);

        float inv_vocab = 1.0f / (float)vocab_size;
        // Allocate and zero device shap buffer (one aggregated vector of length seq_len)
        float* d_shap = nullptr;
        checkCuda(cudaMalloc(&d_shap, (size_t)ds.seq_len * sizeof(float)), "cudaMalloc d_shap");
        checkCuda(cudaMemset(d_shap, 0, (size_t)ds.seq_len * sizeof(float)), "memset d_shap");

        mlp_forward_kernel<<<grid, block, shmem>>>(
            d_token_ids,
            n_permutations,
            ds.seq_len,
            inv_vocab,
            d_W,
            d_b,
            d_w_offsets,
            d_b_offsets,
            d_in_dims,
            d_out_dims,
            num_layers,
            d_logits,
            d_shap,
            out_dim_last,
            max_dim
        );
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "device sync");

        size_t out_count = (size_t)2 * (size_t)n_permutations;
        std::vector<float> h_logits(out_count);
        checkCuda(cudaMemcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy logits back");

        // Copy and finalize shap values (average across blocks)
        std::vector<float> h_shap((size_t)ds.seq_len);
        checkCuda(cudaMemcpy(h_shap.data(), d_shap, (size_t)ds.seq_len * sizeof(float), cudaMemcpyDeviceToHost), "memcpy shap back");
        std::vector<float> shap_values((size_t)ds.seq_len);
        for (int i = 0; i < ds.seq_len; ++i) {
            shap_values[(size_t)i] = h_shap[(size_t)i] / (float)out_count;
        }

        int to_print = std::min((int)out_count, max_print);
        for (size_t i = 0; i < (size_t)to_print; i++) {
            float logit = h_logits[i];
            std::cout << "block=" << i << " logit=" << logit << "\n";
        }

        // Print first few shap values as a quick check
        int shap_print = std::min(ds.seq_len, 10);
        std::cout << "shap_values[0.." << shap_print - 1 << "] = ";
        for (int i = 0; i < shap_print; ++i) {
            std::cout << shap_values[(size_t)i] << (i + 1 < shap_print ? ", " : "\n");
        }

        // Write shap values alongside token ids for the first sample to a CSV file.
        std::string shap_out_path = dataset_path + std::string(".shap_values.csv");
        std::ofstream sf(shap_out_path);
        if (sf) {
            sf << "feature_idx,token_id,shap_value\n";
            for (int j = 0; j < ds.seq_len; ++j) {
                int32_t tok = ds.token_ids[(size_t)0 * (size_t)ds.seq_len + (size_t)j];
                sf << j << "," << tok << "," << shap_values[(size_t)j] << "\n";
            }
            sf.close();
            std::cout << "wrote shap values to " << shap_out_path << "\n";
        } else {
            std::cerr << "Failed to open shap output file: " << shap_out_path << "\n";
        }

        // Optional spot-check on CPU for sample 0 (print scalar logit)
        if (ds.n > 0) {
            std::vector<float> cpu_logits;
            cpu_forward_one(net, ds.token_ids.data(), ds.seq_len, inv_vocab, cpu_logits);
            float cpu_score = cpu_logits.empty() ? 0.0f : cpu_logits[0];
            std::cout << "cpu_check sample=0 score=" << cpu_score << "\n";
        }

        cudaFree(d_token_ids);
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_w_offsets);
        cudaFree(d_b_offsets);
        cudaFree(d_in_dims);
        cudaFree(d_out_dims);
        cudaFree(d_logits);
        cudaFree(d_shap);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
