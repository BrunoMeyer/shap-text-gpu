# linear_mlp_bert_cuda

End-to-end demo:

1) Python: downloads a public text dataset, tokenizes with BERT tokenizer, trains an MLP (linear layers with identity activations), and exports:
   - tokenized dataset to a plain-text file
   - network weights to a plain-text file

2) CUDA: loads the exported dataset + weights and runs a GPU feedforward kernel (linear activation).

## Quickstart

```bash
cd linear_mlp_bert_cuda
./scripts/run_all.sh \
  --dataset ag_news \
  --max-len 64 \
  --hidden-dim 128 \
  --num-layers 3 \
  --epochs 1 \
  --train-samples 2000 \
  --test-samples 256
```

Artifacts are written under `out/`.

## Notes

- The feature vector is a simple numeric transform of BERT token IDs: `x = token_id / vocab_size` (float). This is intentionally simple so the exported dataset is easy to consume from CUDA.
- The MLP uses identity activations between linear layers (so it is functionally equivalent to a single linear map, but kept as multiple layers for demonstration).
