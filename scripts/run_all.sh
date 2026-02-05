#!/usr/bin/env bash
set -euo pipefail

# Orchestrates:
# 1) Python train + export
# 2) Build CUDA binary
# 3) Run CUDA feedforward

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR_DEFAULT="$ROOT_DIR/out"

DATASET="imdb"
TOKENIZER="bert-base-uncased"
MAX_LEN=64
HIDDEN_DIM=128
NUM_LAYERS=3
BATCH_SIZE=64
EPOCHS=1
LR=1e-3
TRAIN_SAMPLES=2000
TEST_SAMPLES=256
DEVICE="auto"

THREADS=128
PRINT=10

OUT_DIR="$OUT_DIR_DEFAULT"
WEIGHTS_FILE="mlp_weights.txt"
DATASET_FILE="tokenized_dataset.txt"
META_FILE="export_meta.json"

usage() {
  cat <<EOF
Usage: $0 [options]

Python/training options:
  --dataset NAME            (default: $DATASET)
  --tokenizer NAME          (default: $TOKENIZER)
  --max-len N               (default: $MAX_LEN)
  --hidden-dim N            (default: $HIDDEN_DIM)
  --num-layers N            (default: $NUM_LAYERS)
  --batch-size N            (default: $BATCH_SIZE)
  --epochs N                (default: $EPOCHS)
  --lr FLOAT                (default: $LR)
  --train-samples N         (default: $TRAIN_SAMPLES)
  --test-samples N          (default: $TEST_SAMPLES)
  --device auto|cpu|cuda    (default: $DEVICE)

Export/output options:
  --out-dir PATH            (default: $OUT_DIR)
  --weights-file NAME       (default: $WEIGHTS_FILE)
  --dataset-file NAME       (default: $DATASET_FILE)
  --meta-file NAME          (default: $META_FILE)

CUDA options:
  --threads N               (default: $THREADS)
  --print N                 (default: $PRINT)

Other:
  --skip-build              skip nvcc build
  -h|--help

Example:
  $0 --dataset imdb --max-len 64 --hidden-dim 128 --num-layers 3 --epochs 1
EOF
}

SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;
    --max-len) MAX_LEN="$2"; shift 2;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2;;
    --num-layers) NUM_LAYERS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --train-samples) TRAIN_SAMPLES="$2"; shift 2;;
    --test-samples) TEST_SAMPLES="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;

    --out-dir) OUT_DIR="$2"; shift 2;;
    --weights-file) WEIGHTS_FILE="$2"; shift 2;;
    --dataset-file) DATASET_FILE="$2"; shift 2;;
    --meta-file) META_FILE="$2"; shift 2;;

    --threads) THREADS="$2"; shift 2;;
    --print) PRINT="$2"; shift 2;;

    --skip-build) SKIP_BUILD=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

mkdir -p "$OUT_DIR"

echo "[1/3] Python train+export -> $OUT_DIR"
python3 "$ROOT_DIR/python/train_export.py" \
  --dataset "$DATASET" \
  --tokenizer "$TOKENIZER" \
  --max-len "$MAX_LEN" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --train-samples "$TRAIN_SAMPLES" \
  --test-samples "$TEST_SAMPLES" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --weights-file "$WEIGHTS_FILE" \
  --dataset-file "$DATASET_FILE" \
  --meta-file "$META_FILE"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[2/3] Build CUDA binary"
  make -C "$ROOT_DIR/cuda" all
else
  echo "[2/3] Skipping CUDA build"
fi

# Read vocab_size from meta json (no jq requirement): small python one-liner
VOCAB_SIZE="$(python3 - <<PY
import json
p = "${OUT_DIR}/${META_FILE}"
with open(p, 'r', encoding='utf-8') as f:
    print(json.load(f)['vocab_size'])
PY
)"

echo "[3/3] CUDA feedforward"
# Create a single-sample dataset file (first sample) and pass that to the CUDA binary
SINGLE_DATASET="$OUT_DIR/${DATASET_FILE}.single"
seq_len=$(awk 'NR==1{print $2; exit}' "$OUT_DIR/$DATASET_FILE")
sample_line=$(sed -n '2p' "$OUT_DIR/$DATASET_FILE")
printf "1 %s\n%s\n" "$seq_len" "$sample_line" > "$SINGLE_DATASET"

"$ROOT_DIR/out/feedforward" \
  --weights "$OUT_DIR/$WEIGHTS_FILE" \
  --dataset "$SINGLE_DATASET" \
  --vocab-size "$VOCAB_SIZE" \
  --threads "$THREADS" \
  --print "$PRINT"
