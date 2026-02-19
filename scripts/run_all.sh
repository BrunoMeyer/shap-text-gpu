#!/usr/bin/env bash
set -euo pipefail

# Orchestrates:
# 1) Python train + export
# 2) Build CUDA binary
# 3) Run CUDA feedforward
# 4) Compute SHAP (linear & permutation) with Python
# 5) Detokenize SHAP for sample 0
# 6, 7 and 8) Compare SHAP outputs


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

N_SAMPLES=1
#SAMPLE_ID=0
N_PERMUTATIONS=129

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

SHAP options:
  --nsamples N              (default: $N_SAMPLES)
  --npermutations N         (default: $N_PERMUTATIONS)
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

    #--sample) SAMPLE_ID="$2"; shift 2;;
    --nsamples) N_SAMPLES="$2"; shift 2;;
    --npermutations) N_PERMUTATIONS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --print) PRINT="$2"; shift 2;;

    --skip-build) SKIP_BUILD=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

mkdir -p "$OUT_DIR"

echo "[1/5] Python train+export -> $OUT_DIR"
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
  echo "[2/5] Build CUDA binary"
  make -C "$ROOT_DIR/cuda" all
else
  echo "[2/5] Skipping CUDA build"
fi

# Read vocab_size from meta json (no jq requirement): small python one-liner
VOCAB_SIZE="$(python3 - <<PY
import json
p = "${OUT_DIR}/${META_FILE}"
with open(p, 'r', encoding='utf-8') as f:
    print(json.load(f)['vocab_size'])
PY
)"

echo "[3/5] CUDA feedforward (running per-sample)"
# Read seq_len and total samples from dataset header
seq_len=$(awk 'NR==1{print $2; exit}' "$OUT_DIR/$DATASET_FILE")
total_samples=$(awk 'NR==1{print $1; exit}' "$OUT_DIR/$DATASET_FILE")

# Run feedforward for each sample (dataset lines start at line 2).
# Prepare time accumulation
sum_times=0.0
count_times=0
TIMES_FILE="$OUT_DIR/shap_times_ms.csv"
echo "sample_idx,cuda_kernel_time_ms" > "$TIMES_FILE"
for ((i=0;i<total_samples;i++)); do
  SAMPLE_DATASET="$OUT_DIR/${DATASET_FILE}.single.$i"
  sample_line=$(sed -n "$((2 + i))p" "$OUT_DIR/$DATASET_FILE")
  printf "1 %s\n%s\n" "$seq_len" "$sample_line" > "$SAMPLE_DATASET"

  echo "  running feedforward for sample $i"
  # Capture output so we can extract the cuda kernel timing
  ff_out=$("$ROOT_DIR/out/feedforward" \
    --weights "$OUT_DIR/$WEIGHTS_FILE" \
    --dataset "$SAMPLE_DATASET" \
    --vocab-size "$VOCAB_SIZE" \
    --threads "$THREADS" \
    --print "$PRINT" \
    2>&1)
  # Print the captured output to console
  printf "%s\n" "$ff_out"

  # Extract cuda kernel time in milliseconds (line like: cuda_kernel_time_ms=123.45)
  t_ms=$(printf "%s\n" "$ff_out" | awk -F"=" '/^cuda_kernel_time_ms=/{print $2; exit}')
  if [[ -n "$t_ms" ]]; then
    # append per-sample time
    echo "$i,$t_ms" >> "$TIMES_FILE"
    # accumulate using python for floating-point arithmetic
    sum_times=$(python3 - <<PY
  print(float("$sum_times") + float("$t_ms"))
PY
    )
    count_times=$((count_times+1))
  else
    echo "warning: cuda_kernel_time_ms not found for sample $i" >&2
  fi
done

# Compute average time per sample (ms) and write summary
if [[ $count_times -gt 0 ]]; then
  avg_time_ms=$(python3 - <<PY
print(float("$sum_times") / float($count_times))
PY
)
else
  avg_time_ms=0
fi
TIME_SUMMARY="$OUT_DIR/shap_times_summary.txt"
cat > "$TIME_SUMMARY" <<EOF
n_samples_timed: $count_times
total_time_ms: $sum_times
avg_time_per_sample_ms: $avg_time_ms
EOF
echo "Wrote SHAP timing summary to: $TIME_SUMMARY"

# Combine per-sample shap CSVs into a single CSV for all samples
ALL_SHAP="$OUT_DIR/${DATASET_FILE}.all_shap_values.csv"
echo "sample_idx,feature_idx,token_id,shap_value" > "$ALL_SHAP"
for ((i=0;i<total_samples;i++)); do
  src="$OUT_DIR/${DATASET_FILE}.single.$i.shap_values.csv"
  if [[ -f "$src" ]]; then
    # Skip header and prefix each row with the sample index
    tail -n +2 "$src" | while IFS= read -r line; do
      printf "%d,%s\n" "$i" "$line" >> "$ALL_SHAP"
    done
  else
    echo "warning: missing per-sample shap file: $src" >&2
  fi
done


echo "[4/5] Compute SHAP (linear & permutation) with Python"
python3 "$ROOT_DIR/python/compute_shap.py" \
  --weights "$OUT_DIR/$WEIGHTS_FILE" \
  --dataset "$DATASET_FILE" \
  --meta "$OUT_DIR/$META_FILE" \
  --sample-size "$seq_len" \
  --explainer linear \
  --nsamples "$N_SAMPLES" \
  --out "$OUT_DIR/sample${SAMPLE_ID}_shap_linear.txt" \
  --tokenizer "$TOKENIZER"

echo "[5/5] Compare Python permutation vs CUDA across dataset"
# For each sample: detokenize CUDA shap CSV, compare against Python permutation SHAP,
# collect per-sample mean_diff and mean_abs_diff, then compute dataset averages.
sum_diff=0.0
sum_abs_diff=0.0
count_cmp=0
for ((i=0;i<total_samples;i++)); do
  PY_FILE="$OUT_DIR/sample${i}_shap_permutation.txt"
  CUDA_CSV="$OUT_DIR/${DATASET_FILE}.single.$i.shap_values.csv"
  CUDA_DETOK="$OUT_DIR/sample${i}_shap_cuda.txt"
  COMP_OUT="$OUT_DIR/sample${i}_shap_permutation_vs_cuda.compare.txt"

  if [[ ! -f "$PY_FILE" ]]; then
    echo "warning: missing python shap file: $PY_FILE" >&2
    continue
  fi
  if [[ ! -f "$CUDA_CSV" ]]; then
    echo "warning: missing cuda shap csv: $CUDA_CSV" >&2
    continue
  fi

  # Detokenize CUDA shap CSV for sample i
  python3 "$ROOT_DIR/python/detokenize_shap.py" \
    --dataset "$OUT_DIR/${DATASET_FILE}.single.$i" \
    --meta "$OUT_DIR/$META_FILE" \
    --shap-csv "$CUDA_CSV" \
    --sample 0 \
    --out "$CUDA_DETOK"

  # Compare python vs cuda for this sample
  python3 "$ROOT_DIR/python/compare_shap.py" \
    "$PY_FILE" \
    "$CUDA_DETOK" \
    --out "$COMP_OUT"

  # Extract mean_diff and mean_abs_diff from compare output
  md=$(awk -F": " '/^mean_diff:/{print $2; exit}' "$COMP_OUT" || echo "0")
  mad=$(awk -F": " '/^mean_abs_diff:/{print $2; exit}' "$COMP_OUT" || echo "0")
  # Ensure numeric (fallback to 0)
  md=$(printf "%f" "$md" 2>/dev/null || echo "0")
  mad=$(printf "%f" "$mad" 2>/dev/null || echo "0")

  sum_diff=$(python3 - <<PY
print(float("$sum_diff") + float("$md"))
PY
)
  sum_abs_diff=$(python3 - <<PY
print(float("$sum_abs_diff") + float("$mad"))
PY
)
  count_cmp=$((count_cmp+1))
done

if [[ $count_cmp -gt 0 ]]; then
  avg_diff=$(python3 - <<PY
print(float("$sum_diff") / float($count_cmp))
PY
)
  avg_abs_diff=$(python3 - <<PY
print(float("$sum_abs_diff") / float($count_cmp))
PY
)
else
  avg_diff=0
  avg_abs_diff=0
fi

SUMMARY_OUT="$OUT_DIR/shap_dataset_compare_summary.txt"
cat > "$SUMMARY_OUT" <<EOF
n_compared: $count_cmp
mean_diff_across_dataset: $avg_diff
mean_abs_diff_across_dataset: $avg_abs_diff
EOF

echo "Wrote dataset comparison summary to: $SUMMARY_OUT"
