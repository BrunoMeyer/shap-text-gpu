#!/usr/bin/env python3
"""Parse a SHAP comparison file and produce two files sorted by each SHAP column.

Usage: python sort_shap_comparison.py comparison_file.txt [--out-dir DIR]

Outputs: two TSV files with columns: token_id, token_str, shap
"""
import argparse
import os
import re
import sys


def parse_comparison(path):
    file_a = None
    file_b = None
    rows = []
    table_started = False
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not table_started:
                m = re.match(r'file_a:\s*(.+)', line)
                if m:
                    file_a = m.group(1).strip()
                    continue
                m = re.match(r'file_b:\s*(.+)', line)
                if m:
                    file_b = m.group(1).strip()
                    continue
                if line.startswith('idx\t'):
                    table_started = True
                    continue
            else:
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) < 7:
                    # try splitting on whitespace as fallback
                    parts = line.split()
                if len(parts) < 7:
                    # cannot parse this row, skip
                    continue
                # expected columns: idx, token_id, token_str, shap_a, shap_b, diff, abs_diff
                try:
                    token_id = parts[1]
                    token_str = parts[2]
                    shap_a = float(parts[3])
                    shap_b = float(parts[4])
                except Exception:
                    continue
                rows.append((token_id, token_str, shap_a, shap_b))
    return file_a, file_b, rows


def write_sorted(rows, key_index, out_path):
    # key_index: 2 for shap_a, 3 for shap_b (0-based on tuple)
    sorted_rows = sorted(rows, key=lambda r: r[key_index], reverse=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('token_id\ttoken_str\tshap\n')
        for token_id, token_str, shap_a, shap_b in sorted_rows:
            shap = shap_a if key_index == 2 else shap_b
            f.write(f"{token_id}\t{token_str}\t{shap}\n")


def main():
    p = argparse.ArgumentParser(description='Split and sort SHAP comparison file')
    p.add_argument('comparison_file')
    p.add_argument('--out-dir', '-o', default=None, help='Output directory')
    args = p.parse_args()

    file_a, file_b, rows = parse_comparison(args.comparison_file)
    if not rows:
        print('No table rows found in comparison file.', file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.comparison_file))
    os.makedirs(out_dir, exist_ok=True)

    # derive base names for outputs
    if file_a:
        base_a = os.path.basename(file_a)
    else:
        base_a = 'file_a'
    if file_b:
        base_b = os.path.basename(file_b)
    else:
        base_b = 'file_b'

    out_a = os.path.join(out_dir, f"{base_a}_sorted.txt")
    out_b = os.path.join(out_dir, f"{base_b}_sorted.txt")

    # In rows tuple: (token_id, token_str, shap_a, shap_b)
    # write sorted by shap_a and shap_b respectively
    write_sorted(rows, 2, out_a)
    write_sorted(rows, 3, out_b)

    print('Wrote:', out_a)
    print('Wrote:', out_b)


if __name__ == '__main__':
    main()
