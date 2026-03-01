#!/usr/bin/env python3
"""Compute SHAP values for a sample using the exported model and dataset.

This script loads the plain-text weights exported by `train_export.py`, builds
an equivalent PyTorch linear MLP, loads a sample from the exported dataset
file, and computes per-feature SHAP values using one of the supported
explainers: `linear`, `kernel`, or `permutation`.

Notes for fair comparison with the CUDA implementation:
- Inputs are normalized as `token_id / vocab_size` (same as CUDA and exporter).
- Missing features are masked by replacing with zeros (CUDA treats omitted
  features as absent/zero when computing marginal contributions).
- The script can truncate the sample to `--sample-size` to match the CUDA
  single-sample dataset produced by `scripts/run_all.sh`.

Example:
  python python/compute_shap.py \
    --weights out/mlp_weights.txt \
    --dataset out/tokenized_dataset.txt \
    --meta out/export_meta.json \
    --sample 0 --sample-size 64 --explainer permutation --nsamples 100
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import time
import csv


def parse_args():
    p = argparse.ArgumentParser(description="Compute SHAP values for an exported MLP and a dataset sample")
    p.add_argument("--weights", type=str, required=True, help="Path to exported weights (text)")
    p.add_argument("--dataset", type=str, required=True, help="Exported dataset file (token ids)")
    p.add_argument("--meta", type=str, required=True, help="Exported meta JSON (contains tokenizer, vocab_size)")
    p.add_argument("--sample-size", type=int, default=None, help="Truncate the input to this many features (must be <= exported seq_len)")
    p.add_argument("--sample", type=int, default=None, help="Sample index to compute (default: all samples)")
    p.add_argument("--explainer", type=str, default="permutation", choices=["linear", "kernel", "permutation"], help="Explainer algorithm to use")
    p.add_argument("--nsamples", type=int, default=1000, help="Number of samples (for Kernel/Permutation) or budget for explainer")
    p.add_argument("--target-index", type=int, default=0, help="Target output index/class to explain (default 0)")
    p.add_argument("--out", type=str, default=None, help="Output text path (defaults to <dataset>.sampleX.shap.txt)")
    p.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer name in meta")
    p.add_argument("--npermutations", type=int, default=1000, help="Number of permutations for permutation explainer (if supported by shap version)")
    return p.parse_args()


def read_dataset(dataset_path: str) -> Tuple[int, int, List[List[int]]]:
    # Returns (n, seq_len, token_rows)
    with open(dataset_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) < 2:
            raise RuntimeError("Invalid dataset header")
        n = int(header[0]); seq_len = int(header[1])
        rows = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # first entry is label
            ids = [int(x) for x in parts[1:1+seq_len]]
            rows.append(ids)
    return n, seq_len, rows


def load_weights_text(path: str):
    # Reads weights file and returns a list of (W,b) numpy arrays where W.shape=(out_dim,in_dim)
    layers = []
    with open(path, "r", encoding="utf-8") as f:
        L = int(f.readline().strip())
        for _ in range(L):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in weights file")
            in_dim, out_dim = [int(x) for x in line.strip().split()]
            w_count = out_dim * in_dim
            w_vals = []
            while len(w_vals) < w_count:
                parts = f.readline().strip().split()
                w_vals.extend([float(x) for x in parts])
            W = np.array(w_vals, dtype=np.float32).reshape((out_dim, in_dim))
            # read biases (may be on one line)
            b_vals = []
            while len(b_vals) < out_dim:
                parts = f.readline().strip().split()
                b_vals.extend([float(x) for x in parts])
            b = np.array(b_vals, dtype=np.float32)
            layers.append((W, b))
    return layers


def build_torch_model_from_weights(layers: List[Tuple[np.ndarray, np.ndarray]], input_cut: Optional[int] = None) -> torch.nn.Module:
    # If input_cut is provided and smaller than first layer in_dim, slice the first W accordingly.
    modules = []
    for i, (W, b) in enumerate(layers):
        out_dim, in_dim = W.shape
        if i == 0 and input_cut is not None and input_cut < in_dim:
            in_dim_use = input_cut
            W_use = W[:, :in_dim_use]
        else:
            in_dim_use = in_dim
            W_use = W

        lin = torch.nn.Linear(in_dim_use, out_dim, bias=True)
        lin.weight.data = torch.from_numpy(W_use).to(torch.float32)
        lin.bias.data = torch.from_numpy(b).to(torch.float32)
        modules.append(lin)
    # Build a sequential model applying linears in order with no activations
    model = torch.nn.Sequential(*modules)
    model.eval()
    return model


def compute_effective_input_weights(layers: List[Tuple[np.ndarray, np.ndarray]], input_cut: Optional[int] = None) -> np.ndarray:
    # Compute W_eff = W_L @ W_{L-1} @ ... @ W_1  (shape: out_dim_last, in_dim_first)
    mats = [W for (W, b) in layers]
    # Possibly slice first matrix columns
    if input_cut is not None and input_cut < mats[0].shape[1]:
        mats[0] = mats[0][:, :input_cut]
    W_eff = mats[0]
    for M in mats[1:]:
        W_eff = M @ W_eff
    return W_eff


def main():
    args = parse_args()

    # Load meta
    with open(args.meta, "r", encoding="utf-8") as mf:
        meta = json.load(mf)
    tokenizer_name = args.tokenizer if args.tokenizer else meta.get("tokenizer")
    if tokenizer_name is None:
        raise RuntimeError("Tokenizer not found in meta; provide --tokenizer")
    vocab_size = int(meta.get("vocab_size", 30522))

    # Read dataset
    n, seq_len, rows = read_dataset(args.dataset)

    # Determine sample_size (truncate to requested)
    if args.sample_size is None:
        sample_size = seq_len
    else:
        if args.sample_size > seq_len:
            raise ValueError("--sample-size cannot be larger than dataset seq_len")
        sample_size = args.sample_size

    # Which samples to process: single index or all
    if args.sample is None:
        sample_indices = list(range(n))
    else:
        if args.sample < 0 or args.sample >= n:
            raise IndexError("sample index out of range")
        sample_indices = [args.sample]

    # Load weights once
    layers = load_weights_text(args.weights)

    # Build torch model; slice first layer inputs to sample_size if needed
    model = build_torch_model_from_weights(layers, input_cut=sample_size)

    # For fair comparison, background is zero baseline of sample_size
    background = np.zeros((1, sample_size), dtype=np.float32)

    # Import shap once
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap library is required for explainers. Install via 'pip install shap'") from e

    # Load tokenizer once for detokenization
    try:
        from transformers import AutoTokenizer
    except Exception:
        raise RuntimeError("transformers is required to detokenize tokens. Install via 'pip install transformers'")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    target = args.target_index

    # Prepare combined CSV output (one row per token across samples)
    csv_out = args.out if args.out else os.path.splitext(args.dataset)[0] + ".all_shap_values.csv"
    csv_f = open(csv_out, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_f)
    # header: sample_idx,feature_idx,token_id,token_str,shap_value,detokenized_full_text
    csv_writer.writerow(["sample_idx", "feature_idx", "token_id", "token_str", "shap_value", "detokenized_full_text"])

    # Process each requested sample
    for sample_idx in sample_indices:
        token_ids = rows[sample_idx][:sample_size]
        x = np.array(token_ids, dtype=np.float32) / float(vocab_size)
        x = x.reshape(1, -1)


        # Compute SHAP depending on explainer
        if args.explainer == "linear":
            W_eff = compute_effective_input_weights(layers, input_cut=sample_size)
            if target >= W_eff.shape[0]:
                raise IndexError("target-index out of range for model outputs")

            zero_in = torch.zeros((1, x.shape[1]), dtype=torch.float32)
            with torch.no_grad():
                b_eff = model(zero_in).cpu().numpy().reshape(-1)

            class _SimpleLinearModel:
                def __init__(self, W, b):
                    self.coef_ = W
                    self.intercept_ = b
                def predict(self, X: np.ndarray) -> np.ndarray:
                    return X.dot(self.coef_.T) + self.intercept_

            lm = _SimpleLinearModel(W_eff.astype(np.float32), b_eff.astype(np.float32))
            expl = shap.LinearExplainer(lm, background)
            t0 = time.perf_counter()
            shap_out = expl.shap_values(x)
            t1 = time.perf_counter()
            print(f"linear_explainer_eval_time={t1-t0:.3f}s sample={sample_idx}")
            if isinstance(shap_out, list):
                shap_vals = np.array(shap_out[target]).reshape(-1)[:x.shape[1]]
            else:
                shap_vals = np.array(shap_out).reshape(-1)[:x.shape[1]]

        else:
            def f(X: np.ndarray) -> np.ndarray:
                xt = torch.from_numpy(X.astype(np.float32))
                with torch.no_grad():
                    out = model(xt).cpu().numpy()
                if out.ndim == 1:
                    return out.reshape(-1)
                return out[:, target]

            if args.explainer == "kernel":
                expl = shap.KernelExplainer(f, background)
                shap_out = expl.shap_values(x, nsamples=args.nsamples)
                shap_vals = np.array(shap_out).reshape(-1)[:x.shape[1]]

            elif args.explainer == "permutation":
                try:
                    masker = shap.maskers.Independent(background)
                    expl = shap.Explainer(f, masker, algorithm="permutation")
                    t0 = time.perf_counter()
                    exp = expl(x, max_evals=args.npermutations)
                    t1 = time.perf_counter()
                    print(f"permutation_explainer_eval_time={t1-t0:.3f}s sample={sample_idx}")
                    shap_vals = np.array(exp.values).reshape(-1, x.shape[1])[0]
                except Exception:
                    expl = shap.Explainer(f, algorithm="permutation")
                    t0 = time.perf_counter()
                    exp = expl(x, max_evals=args.npermutations)
                    t1 = time.perf_counter()
                    print(f"permutation_explainer_eval_time={t1-t0:.3f}s sample={sample_idx}")
                    shap_vals = np.array(exp.values).reshape(-1, x.shape[1])[0]

            else:
                raise ValueError("Unsupported explainer")

        # Detokenize and append rows to combined CSV
        tokens = tok.convert_ids_to_tokens(token_ids)
        text = tok.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for i, tid in enumerate(token_ids):
            tok_str = tok.convert_tokens_to_string([tokens[i]])
            sv = float(shap_vals[i]) if i < len(shap_vals) else 0.0
            csv_writer.writerow([sample_idx, i, tid, tok_str, sv, text])
        print(f"Appended sample {sample_idx} to CSV: {csv_out}")

    csv_f.close()
    print(f"Wrote combined SHAP CSV to: {csv_out}")


if __name__ == "__main__":
    main()
