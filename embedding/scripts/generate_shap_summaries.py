#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from typing import List, Tuple


def read_vocab(vocab_path: str) -> List[str]:
    toks = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            toks.append(line.rstrip('\n'))
    return toks


def read_cuda_shap(cuda_csv_path: str, vocab: List[str]) -> List[Tuple[str, float]]:
    vals = []
    if not os.path.exists(cuda_csv_path):
        return vals
    with open(cuda_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(row.get('token_id', row.get('tokenId', '')))
            except Exception:
                tid = None
            try:
                sv = float(row.get('shap_value', row.get('shapValue', '0')))
            except Exception:
                sv = 0.0
            tok = vocab[tid] if (tid is not None and 0 <= tid < len(vocab)) else '<unk>'
            vals.append((tok, sv))
    return vals


def read_permutation_detok(detok_path: str) -> Tuple[int, List[Tuple[str, float]]]:
    # returns (target, list of (token, shap))
    vals = []
    target = 0
    if not os.path.exists(detok_path):
        return target, vals
    with open(detok_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # look for header line containing target
    for ln in lines[:5]:
        m = re.search(r'target=(\d+)', ln)
        if m:
            target = int(m.group(1))
            break
    # find the table header line starting with idx\t
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith('idx'):
            start = i + 1
            break
    for ln in lines[start:]:
        if not ln.strip():
            continue
        parts = ln.rstrip('\n').split('\t')
        if len(parts) < 4:
            continue
        tok = parts[2]
        try:
            sv = float(parts[3])
        except Exception:
            sv = 0.0
        vals.append((tok, sv))
    return target, vals


def top_bottom(vals: List[Tuple[str, float]], top_k=5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    if not vals:
        return [], []
    sorted_vals = sorted(vals, key=lambda x: x[1])
    bottom = sorted_vals[:top_k]
    top = sorted_vals[-top_k:][::-1]
    # ensure lengths
    def pad(a):
        while len(a) < top_k:
            a.append(("", 0.0))
        return a
    return pad(top), pad(bottom)


def parse_time_from_file(file_path: str, pattern: str, to_seconds: bool = False) -> float:
    if not os.path.exists(file_path):
        return 0.0
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    m = re.search(pattern, txt)
    if not m:
        return 0.0
    val = m.group(1)
    try:
        v = float(val)
    except Exception:
        v = 0.0
    if to_seconds:
        return v / 1000.0
    return v


def write_summary_csv(path: str, sample: int, cls: int, logit: float, time_s: float, top: List[Tuple[str, float]], bottom: List[Tuple[str, float]]):
    header = ['sample', 'class', 'logit', 'time']
    for i in range(1, 6):
        header += [f'top{i}_token', f'top{i}_shap']
    for i in range(1, 6):
        header += [f'bottom{i}_token', f'bottom{i}_shap']

    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        row = [sample, cls, f"{logit:.9g}", f"{time_s:.6f}"]
        for tok, sv in top:
            row += [tok, f"{sv:.6f}"]
        for tok, sv in bottom:
            row += [tok, f"{sv:.6f}"]
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True)
    p.add_argument('--sample', type=int, required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--cuda-shap-csv', required=True)
    p.add_argument('--cuda-log', required=True)
    p.add_argument('--perm-detok', required=True)
    p.add_argument('--perm-log', required=True)
    p.add_argument('--logits-file', required=False, help='Path to test_logits_sorted.csv (optional)')
    args = p.parse_args()

    vocab = read_vocab(args.vocab)

    # read logits mapping if available
    logits_map = {}
    if args.logits_file and os.path.exists(args.logits_file):
        try:
            with open(args.logits_file, 'r', encoding='utf-8') as lf:
                rdr = csv.DictReader(lf)
                for row in rdr:
                    try:
                        idx = int(row.get('sample_idx', row.get('index', '')))
                        val = float(row.get('logit', row.get('score', '0')))
                        logits_map[idx] = val
                    except Exception:
                        continue
        except Exception:
            logits_map = {}

    # CUDA
    cuda_vals = read_cuda_shap(args.cuda_shap_csv, vocab)
    top_cuda, bottom_cuda = top_bottom(cuda_vals, top_k=5)
    cuda_time = parse_time_from_file(args.cuda_log, r'cuda_kernel_time_ms=([0-9.]+)', to_seconds=True)
    # class: default 0
    cuda_logit = logits_map.get(args.sample, 0.0)
    write_summary_csv(os.path.join(args.out_dir, 'cuda_shap_summary.csv'), args.sample, 0, cuda_logit, cuda_time, top_cuda, bottom_cuda)

    # Permutation
    perm_target, perm_vals = read_permutation_detok(args.perm_detok)
    top_perm, bottom_perm = top_bottom(perm_vals, top_k=5)
    perm_time = parse_time_from_file(args.perm_log, r'permutation_explainer_eval_time=([0-9.]+)s')
    perm_logit = logits_map.get(args.sample, 0.0)
    write_summary_csv(os.path.join(args.out_dir, 'permutation_shap_summary.csv'), args.sample, perm_target, perm_logit, perm_time, top_perm, bottom_perm)


if __name__ == '__main__':
    main()
