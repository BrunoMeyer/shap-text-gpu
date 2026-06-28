#!/usr/bin/env bash
set -euo pipefail

# Usage: example_pipeline.sh <out_dir> <textfile>
OUT_DIR="$1"
TEXTFILE="$2"

mkdir -p "$OUT_DIR"

python3 "$(dirname "$0")/generate_combined_example.py" "$OUT_DIR" "$TEXTFILE"

echo "Wrote example combined_output.json to: $OUT_DIR/combined_output.json"
