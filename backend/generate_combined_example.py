#!/usr/bin/env python3
import json
import os
import sys
import random
from datetime import datetime


POS_WORDS = {
    "good",
    "great",
    "excellent",
    "positive",
    "love",
    "nice",
    "best",
    "happy",
    "awesome",
}

NEG_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "worst",
    "poor",
    "sad",
    "angry",
}


def make_tokens_with_shap(text):
    parts = text.strip().split()
    tokens = []
    for i, p in enumerate(parts):
        w = p.strip().lower().strip(".,!?;:\"")
        if w in POS_WORDS:
            base = 1.0
        elif w in NEG_WORDS:
            base = -1.0
        else:
            base = 0.0

        # add small deterministic noise based on hash to vary values
        noise = (hash(p) % 100) / 1000.0
        shap = base + (noise if base >= 0 else -noise)

        tokens.append({
            "idx": i,
            "token_id": i + 1,
            "token_str": p,
            "shap_value": float(shap),
        })

    return tokens


def main():
    if len(sys.argv) < 3:
        print("Usage: generate_combined_example.py <out_dir> <textfile>")
        sys.exit(2)

    out_dir = sys.argv[1]
    textfile = sys.argv[2]

    with open(textfile, "r", encoding="utf-8") as f:
        text = f.read().strip()

    tokens = make_tokens_with_shap(text)

    # create a GPU variant with small per-token noise
    shap_gpu_tokens = []
    for t in tokens:
        jitter = (hash(t['token_str'] + 'gpu') % 50) / 1000.0
        shap_gpu_tokens.append({
            "idx": t["idx"],
            "token_id": t["token_id"],
            "token_str": t["token_str"],
            "shap_value": float(t["shap_value"] + (jitter if t["shap_value"] >= 0 else -jitter)),
        })

    obj = {
        "file_name": f"adhoc_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.txt",
        "metadata": {"sample": 0},
        "detokenized_full_text": text,
        "tokens": tokens,
        "shap_gpu_tokens": shap_gpu_tokens,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "combined_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([obj], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
