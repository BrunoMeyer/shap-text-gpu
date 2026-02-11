#!/usr/bin/env python3
"""Compare two detokenized SHAP output files and compute per-token differences.

Writes a tab-separated comparison with columns:
  idx\ttoken_id\ttoken_str\tshap_a\tshap_b\tdiff\tabs_diff

Also prints and writes mean difference and mean absolute difference.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np


def read_shap(path: str) -> List[Tuple[int, int, str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("idx\t"):
                break
        for line in f:
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) < 4:
                continue
            idx_s, tid_s, tok_str, shap_s = parts
            try:
                idx = int(idx_s)
                tid = int(tid_s)
                sv = float(shap_s)
            except Exception:
                continue
            rows.append((idx, tid, tok_str, sv))
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    p = argparse.ArgumentParser(description="Compare two detokenized SHAP outputs")
    p.add_argument("a", help="First SHAP file (baseline)")
    p.add_argument("b", help="Second SHAP file to compare")
    p.add_argument("--out", help="Output compare file (default: <a>.compare.txt)")
    args = p.parse_args()

    a_rows = read_shap(args.a)
    b_rows = read_shap(args.b)

    if len(a_rows) != len(b_rows):
        print(f"Warning: row count differs: {len(a_rows)} vs {len(b_rows)}")

    n = min(len(a_rows), len(b_rows))

    diffs = []
    abs_diffs = []
    out_lines = []
    for i in range(n):
        ia, ta, sa, va = a_rows[i]
        ib, tb, sb, vb = b_rows[i]
        if ia != ib:
            # Align by index where possible
            pass
        if ta != tb or sa != sb:
            # token mismatch; keep tokens from file a and note mismatch
            tok = f"{sa} ||| {sb}"
        else:
            tok = sa
        diff = vb - va
        absd = abs(diff)
        diffs.append(diff)
        abs_diffs.append(absd)
        out_lines.append((ia, ta, tok, va, vb, diff, absd))

    diffs = np.array(diffs, dtype=np.float64)
    abs_diffs = np.array(abs_diffs, dtype=np.float64)
    mean_diff = float(diffs.mean()) if diffs.size else 0.0
    mean_abs_diff = float(abs_diffs.mean()) if abs_diffs.size else 0.0

    out_path = args.out if args.out else os.path.splitext(args.a)[0] + ".compare.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"file_a: {args.a}\n")
        f.write(f"file_b: {args.b}\n")
        f.write(f"n_compared: {n}\n")
        f.write(f"mean_diff: {mean_diff}\n")
        f.write(f"mean_abs_diff: {mean_abs_diff}\n")
        f.write("idx\ttoken_id\ttoken_str\tshap_a\tshap_b\tdiff\tabs_diff\n")
        for row in out_lines:
            idx, tid, tok, va, vb, diff, absd = row
            f.write(f"{idx}\t{tid}\t{tok}\t{va}\t{vb}\t{diff}\t{absd}\n")

    print(f"Wrote comparison to: {out_path}")
    print(f"mean_diff={mean_diff} mean_abs_diff={mean_abs_diff}")


if __name__ == "__main__":
    main()
