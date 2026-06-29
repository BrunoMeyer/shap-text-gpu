#!/usr/bin/env bash
set -euo pipefail

# Usage: real_embedding_pipeline.sh <out_dir> <textfile>

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

OUT_DIR="$1"
TEXTFILE="$2"

python3 "$SCRIPT_DIR/real_embedding_pipeline.py" "$OUT_DIR" "$TEXTFILE"
