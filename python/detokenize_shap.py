#!/usr/bin/env python3
"""Detokenize tokens for a sample and save token strings with their SHAP values.

Reads the exported dataset file (text format) and the SHAP CSV produced by
`feedforward.cu` (dataset_file.shap_values.csv). Uses the tokenizer from the
export meta to convert token ids -> token strings.

Example:
  python python/detokenize_shap.py --dataset out/tokenized_dataset.txt --meta out/export_meta.json --sample 0 --out out/sample0_shap.txt
"""

import argparse
import csv
import json
import os
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Detokenize a sample and save per-token SHAP values")
    p.add_argument("--dataset", type=str, default="out/tokenized_dataset.txt")
    p.add_argument("--meta", type=str, default="out/export_meta.json")
    p.add_argument("--shap-csv", type=str, default=None, help="Path to shap csv (defaults to dataset + '.shap_values.csv')")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="Output txt file path")
    p.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer name to override meta")
    return p.parse_args()


def read_dataset_tokens(dataset_path: str, sample_idx: int) -> List[int]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) < 2:
            raise ValueError("Invalid dataset header")
        n = int(header[0])
        seq_len = int(header[1])

        if sample_idx < 0 or sample_idx >= n:
            raise IndexError(f"sample index {sample_idx} out of range (N={n})")

        for i, line in enumerate(f):
            if i == sample_idx:
                parts = line.strip().split()
                # first entry is label
                ids = [int(x) for x in parts[1:1+seq_len]]
                if len(ids) != seq_len:
                    raise ValueError("Unexpected token count in dataset line")
                return ids

    raise RuntimeError("Reached EOF without finding sample")


def read_shap_values(shap_csv_path: str, seq_len: int) -> List[float]:
    vals = [0.0] * seq_len
    with open(shap_csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            idx = int(row.get("feature_idx") or row.get("feature") or row.get("idx"))
            val = float(row.get("shap_value") or row.get("shap") or row.get("value"))
            if 0 <= idx < seq_len:
                vals[idx] = val
    return vals


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    # Read header to get seq_len
    with open(args.dataset, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        seq_len = int(header[1])

    shap_csv = args.shap_csv if args.shap_csv else args.dataset + ".shap_values.csv"
    if not os.path.exists(shap_csv):
        raise FileNotFoundError(f"SHAP CSV not found: {shap_csv}")

    token_ids = read_dataset_tokens(args.dataset, args.sample)
    shap_vals = read_shap_values(shap_csv, seq_len)

    # Lazy import transformers to avoid hard dependency if user only runs training
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required to detokenize. Install via 'pip install transformers'") from e

    tokenizer_name = args.tokenizer
    if tokenizer_name is None:
        if not os.path.exists(args.meta):
            raise FileNotFoundError(f"Meta file not found and no tokenizer override provided: {args.meta}")
        with open(args.meta, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
            tokenizer_name = meta.get("tokenizer")
    if tokenizer_name is None:
        raise RuntimeError("Tokenizer name not found in meta and not provided via --tokenizer")

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Convert ids -> token strings
    tokens = tok.convert_ids_to_tokens(token_ids)

    # Full detokenized text (for context)
    full_text = tok.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    out_path = args.out if args.out else os.path.splitext(shap_csv)[0] + f".sample{args.sample}.txt"
    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write(f"sample={args.sample} detokenized_full_text: {full_text}\n")
        outf.write("idx\ttoken_id\ttoken_str\tshap_value\n")
        for i, tid in enumerate(token_ids):
            token_str = tok.convert_tokens_to_string([tokens[i]])
            shap = shap_vals[i] if i < len(shap_vals) else 0.0
            outf.write(f"{i}\t{tid}\t{token_str}\t{shap}\n")

    print(f"Wrote detokenized SHAP output to: {out_path}")


if __name__ == "__main__":
    main()
