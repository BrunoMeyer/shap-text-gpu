#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST="127.0.0.1"
PORT="8000"
KEEP_TMP="0"
BACKEND_CORS_ORIGINS="*"
SHAP_PIPELINE_CMD_DEFAULT="bash $ROOT_DIR/backend/real_embedding_pipeline.sh {dir} {textfile}"
SHAP_PIPELINE_CMD="${SHAP_PIPELINE_CMD:-$SHAP_PIPELINE_CMD_DEFAULT}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --host HOST              (default: $HOST)
  --port PORT              (default: $PORT)
  --pipeline CMD           pipeline command template; use {dir} and {textfile}
                           (default: $SHAP_PIPELINE_CMD_DEFAULT)
  --cors-origins ORIGINS   comma-separated origins, or * (default: $BACKEND_CORS_ORIGINS)
  --keep-tmp               keep request temp directories for debugging
  -h|--help

Example:
  $0 --pipeline "bash $ROOT_DIR/backend/example_pipeline.sh {dir} {textfile}" --port 8000

The backend will expose POST /api/analyze and expects the pipeline command to write
combined_output.json into the request directory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --pipeline) SHAP_PIPELINE_CMD="$2"; shift 2;;
    --cors-origins) BACKEND_CORS_ORIGINS="$2"; shift 2;;
    --keep-tmp) KEEP_TMP="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

export SHAP_PIPELINE_CMD
export BACKEND_CORS_ORIGINS
export KEEP_TMP

cd "$ROOT_DIR"
exec uvicorn backend.main:app --reload --host "$HOST" --port "$PORT"
