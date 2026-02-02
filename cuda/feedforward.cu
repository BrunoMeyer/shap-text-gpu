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

__global__ void mlp_forward_kernel(
    const int32_t* token_ids,
    int n,
    int seq_len,
    float inv_vocab_size,
    const float* W_all,
    const float* b_all,
    const int* w_offsets,         // offsets into W_all for each layer
    const int* b_offsets,         // offsets into b_all for each layer
    const int* in_dims,
    const int* out_dims,
    int num_layers,
    float* logits_out,            // [n, out_dim_last]
    int out_dim_last,
    int max_dim
) {
    int sample = blockIdx.x;
    int tid = threadIdx.x;
    if (sample >= n) return;
    bool do_write = (tid < out_dim_last);

    // Naive forward for one output neuron of final layer.
    // We compute full hidden vectors in local memory (very slow but simple).
    extern __shared__ float scratch[];

    // scratch layout: max_dim * 2 (ping/pong)
    float* buf0 = scratch;
    float* buf1 = scratch + max_dim;

    // Load input features (token_id / vocab_size)
    for (int j = tid; j < seq_len; j += blockDim.x) {
        int32_t tid = token_ids[sample * seq_len + j];
        buf0[j] = (float)tid * inv_vocab_size;
    }
    __syncthreads();

    // Forward through layers; for simplicity we only support first layer in_dim == seq_len.
    // Each layer output is computed by all threads cooperatively.
    float* in_buf = buf0;
    float* out_buf = buf1;

    for (int l = 0; l < num_layers; l++) {
        int in_dim = in_dims[l];
        int out_dim = out_dims[l];
        const float* W = W_all + w_offsets[l];
        const float* b = b_all + b_offsets[l];

        // Each thread computes multiple output neurons.
        for (int o = tid; o < out_dim; o += blockDim.x) {
            float acc = b[o];
            const float* wrow = W + (size_t)o * (size_t)in_dim;
            for (int k = 0; k < in_dim; k++) {
                acc += wrow[k] * in_buf[k];
            }
            out_buf[o] = acc; // identity activation
        }
        __syncthreads();

        // swap buffers
        in_buf = out_buf;
        out_buf = (out_buf == buf0) ? buf1 : buf0;
    }

    // After final layer, in_buf holds logits
    if (do_write) {
        logits_out[sample * out_dim_last + tid] = in_buf[tid];
    }
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
        checkCuda(cudaMalloc(&d_logits, (size_t)ds.n * (size_t)out_dim_last * sizeof(float)), "cudaMalloc logits");

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
        size_t shmem = (size_t)max_dim * 2 * sizeof(float);

        dim3 grid(ds.n);
        dim3 block(threads);

        float inv_vocab = 1.0f / (float)vocab_size;
        mlp_forward_kernel<<<grid, block, shmem>>>(
            d_token_ids,
            ds.n,
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
            out_dim_last,
            max_dim
        );
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "device sync");

        std::vector<float> h_logits((size_t)ds.n * (size_t)out_dim_last);
        checkCuda(cudaMemcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "memcpy logits back");

        int correct = 0;
        int to_print = std::min(ds.n, max_print);
        for (int i = 0; i < ds.n; i++) {
            std::vector<float> logits(out_dim_last);
            for (int k = 0; k < out_dim_last; k++) {
                logits[(size_t)k] = h_logits[(size_t)i * (size_t)out_dim_last + (size_t)k];
            }
            int pred = argmax(logits);
            if (pred == (int)ds.labels[(size_t)i]) correct++;

            if (i < to_print) {
                std::cout << "sample=" << i << " label=" << ds.labels[(size_t)i] << " pred=" << pred << " logits[0]=" << logits[0] << "\n";
            }
        }
        std::cout << "cuda_forward accuracy=" << (double)correct / (double)ds.n << " (on exported split)\n";

        // Optional spot-check on CPU for sample 0
        if (ds.n > 0) {
            std::vector<float> cpu_logits;
            cpu_forward_one(net, ds.token_ids.data(), ds.seq_len, inv_vocab, cpu_logits);
            int cpu_pred = argmax(cpu_logits);
            std::cout << "cpu_check sample=0 pred=" << cpu_pred << " logits[0]=" << cpu_logits[0] << "\n";
        }

        cudaFree(d_token_ids);
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_w_offsets);
        cudaFree(d_b_offsets);
        cudaFree(d_in_dims);
        cudaFree(d_out_dims);
        cudaFree(d_logits);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
