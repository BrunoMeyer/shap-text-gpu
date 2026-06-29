SHAP Viewer Backend
===================

This minimal backend exposes a single endpoint to analyze ad-hoc text and return processed SHAP sample JSON compatible with the viewer.

How it works
------------
- The backend creates a unique temporary directory per request.
- It writes the provided text to a file in that directory.
- It runs a shell command configured via the `SHAP_PIPELINE_CMD` environment variable. That command is expected to produce `combined_output.json` (or tokenized/SHAP files which the repository `utils/parse_txt_output.py` can turn into `combined_output.json`).

Configuration
-------------
Set the environment variable `SHAP_PIPELINE_CMD` to a shell command template. The template can use `{dir}` and `{textfile}` which will be replaced at runtime. Example:

```
export SHAP_PIPELINE_CMD="bash /path/to/my_pipeline.sh {dir} {textfile}"
```

`my_pipeline.sh` should use `{textfile}` as the user-provided text input and write outputs into `{dir}`. The backend will look for `{dir}/combined_output.json` after the command finishes. If missing, it will try to run the repository `utils/parse_txt_output.py` on the directory.

Real embedding pipeline
-----------------------
The default launcher now uses `backend/real_embedding_pipeline.sh`, which:
- tokenizes the input text using the exported `vocab.txt`
- builds a one-sample temporary dataset
- runs `embedding/python/compute_shap_emb.py`
- runs `embedding/python/detokenize_shap_emb.py`
- writes viewer-ready `combined_output.json`

By default it reuses artifacts from `out/`:
- `out/mlp_weights.txt`
- `out/embedding_matrix.txt`
- `out/export_meta.json`
- `out/vocab.txt`
- `out/tokenized_dataset.txt`

You can override those via environment variables:

```
EMBEDDING_EXPORT_DIR=./out \
EMBEDDING_WEIGHTS_FILE=./out/mlp_weights.txt \
EMBEDDING_MATRIX_FILE=./out/embedding_matrix.txt \
EMBEDDING_META_FILE=./out/export_meta.json \
EMBEDDING_VOCAB_FILE=./out/vocab.txt \
EMBEDDING_TEMPLATE_DATASET=./out/tokenized_dataset.txt \
EMBEDDING_EXPLAINER=permutation \
EMBEDDING_NPERMUTATIONS=129 \
./backend/run_backend.sh
```

Notes
-----
- This keeps the backend minimal and flexible: the heavy model / SHAP computation can stay in your existing scripts. The backend simply isolates each run in a temporary directory and returns the parsed JSON to the frontend.
- The service cleans up temp directories by default. For debugging, you can set `KEEP_TMP=1` to preserve the temporary output directory.
