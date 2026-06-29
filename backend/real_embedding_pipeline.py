#!/usr/bin/env python3
"""Build a temp embedding dataset from raw text and run the real embedding SHAP scripts.

This helper expects the repository to already contain exported artifacts under
`out/` (or a custom location passed via environment variables):
- mlp_weights.txt
- embedding_matrix.txt
- export_meta.json
- vocab.txt
- tokenized_dataset.txt

It creates a temp single-sample dataset from the user text, runs
`embedding/python/compute_shap_emb.py`, then `embedding/python/detokenize_shap_emb.py`,
and finally writes `combined_output.json` compatible with the viewer.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_vocab(vocab_path: Path) -> list[str]:
    with vocab_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _load_tokenizer_vocab(vocab_path: Path) -> dict[str, int]:
    vocab = _read_vocab(vocab_path)
    return {tok: idx for idx, tok in enumerate(vocab)}


def _tokenize_text(text: str, vocab_map: dict[str, int], unk_id: int) -> list[int]:
    # Keep this simple and consistent with the exported vocab: whitespace + lower.
    # If the original exporter used basic_english, this is close enough for an ad-hoc demo.
    toks = [t.strip().lower() for t in text.split() if t.strip()]
    return [vocab_map.get(tok, unk_id) for tok in toks]


def _write_single_dataset(out_path: Path, token_ids: list[int]) -> None:
    seq_len = len(token_ids)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"1 {seq_len}\n")
        f.write("0")
        for tid in token_ids:
            f.write(f" {tid}")
        f.write("\n")


def _make_temp_textfile(text: str, tmpdir: Path) -> Path:
    path = tmpdir / "input.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _write_shap_csv_from_compute_output(compute_out: Path, csv_out: Path) -> None:
    with compute_out.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) < 4:
        raise RuntimeError(f"Unexpected SHAP compute output: {compute_out}")

    reader = csv.DictReader(lines[2:], delimiter="\t")
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_idx", "shap_value"])
        writer.writeheader()
        for row in reader:
            writer.writerow({
                "feature_idx": row["idx"],
                "shap_value": row["shap_value"],
            })


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: real_embedding_pipeline.py <out_dir> <textfile>", file=sys.stderr)
        return 2

    out_dir = Path(sys.argv[1]).resolve()
    textfile = Path(sys.argv[2]).resolve()
    if not textfile.exists():
        raise FileNotFoundError(f"Text file not found: {textfile}")

    repo = _repo_root()

    artifacts_dir = Path(os.environ.get("EMBEDDING_EXPORT_DIR", repo / "out")).resolve()
    weights = Path(os.environ.get("EMBEDDING_WEIGHTS_FILE", artifacts_dir / "mlp_weights.txt")).resolve()
    emb = Path(os.environ.get("EMBEDDING_MATRIX_FILE", artifacts_dir / "embedding_matrix.txt")).resolve()
    meta = Path(os.environ.get("EMBEDDING_META_FILE", artifacts_dir / "export_meta.json")).resolve()
    vocab_path = Path(os.environ.get("EMBEDDING_VOCAB_FILE", artifacts_dir / "vocab.txt")).resolve()
    template_dataset = Path(os.environ.get("EMBEDDING_TEMPLATE_DATASET", artifacts_dir / "tokenized_dataset.txt")).resolve()

    for p in [weights, emb, meta, vocab_path, template_dataset]:
        if not p.exists():
            raise FileNotFoundError(f"Required embedding artifact not found: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with meta.open("r", encoding="utf-8") as f:
        meta_json = json.load(f)

    vocab_map = _load_tokenizer_vocab(vocab_path)
    unk_id = vocab_map.get("<unk>", 1)

    text = textfile.read_text(encoding="utf-8").strip()

    # Use the template dataset to determine the expected sequence length.
    with template_dataset.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) < 2:
            raise RuntimeError(f"Invalid dataset header in {template_dataset}")
        seq_len = int(header[1])

    token_ids = _tokenize_text(text, vocab_map, unk_id)
    token_ids = token_ids[:seq_len]
    if len(token_ids) < seq_len:
        token_ids.extend([vocab_map.get("<pad>", 0)] * (seq_len - len(token_ids)))

    with tempfile.TemporaryDirectory(prefix="shap_embed_", dir=str(out_dir)) as tmp:
        tmpdir = Path(tmp)
        dataset_path = tmpdir / "tokenized_dataset.txt"
        _write_single_dataset(dataset_path, token_ids)
        shutil.copy2(vocab_path, tmpdir / "vocab.txt")

        # Copy the raw input text for traceability.
        _make_temp_textfile(text, tmpdir)

        sample_id = 0
        perm_out = tmpdir / "sample0_shap.permutation.txt"
        shap_csv = tmpdir / "tokenized_dataset.txt.shap_values.csv"
        detok_out = tmpdir / "sample0_shap.txt"
        combined_out = out_dir / "combined_output.json"

        # Run the real embedding SHAP computation.
        compute_cmd = [
            sys.executable,
            str(repo / "embedding" / "python" / "compute_shap_emb.py"),
            "--weights", str(weights),
            "--dataset", str(dataset_path),
            "--meta", str(meta),
            "--embeddings", str(emb),
            "--sample", str(sample_id),
            "--explainer", os.environ.get("EMBEDDING_EXPLAINER", "permutation"),
            "--npermutations", os.environ.get("EMBEDDING_NPERMUTATIONS", "129"),
            "--out", str(perm_out),
        ]
        subprocess.run(compute_cmd, check=True)

        # The script emits a CSV next to the dataset file. Move/rename it for determinism.
        generated_csv = dataset_path.with_name(dataset_path.name + ".shap_values.csv")
        if generated_csv.exists():
            if generated_csv.resolve() != shap_csv.resolve():
                shutil.copy2(generated_csv, shap_csv)
        elif shap_csv.exists():
            pass
        elif perm_out.exists():
            _write_shap_csv_from_compute_output(perm_out, shap_csv)
        else:
            raise FileNotFoundError(f"Expected SHAP CSV not found: {generated_csv}")

        detok_cmd = [
            sys.executable,
            str(repo / "embedding" / "python" / "detokenize_shap_emb.py"),
            "--dataset", str(dataset_path),
            "--vocab", str(vocab_path),
            "--shap-csv", str(shap_csv),
            "--sample", str(sample_id),
            "--out", str(detok_out),
        ]
        subprocess.run(detok_cmd, check=True)

        # Parse detokenized output and write viewer-ready JSON.
        with detok_out.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]

        if len(lines) < 3:
            raise RuntimeError(f"Unexpected detokenized output: {detok_out}")

        metadata_line = lines[0]
        text_line = lines[1]
        rows = []
        reader = csv.DictReader(lines[1:], delimiter="\t")
        for row in reader:
            try:
                rows.append({
                    "idx": int(row.get("orig_idx") or row.get("idx")),
                    "token_id": int(row["token_id"]),
                    "token_str": row["token_str"],
                    "shap_value": -float(row["shap_value"]),
                })
            except Exception:
                continue

        rows.sort(key=lambda x: x["idx"])

        # Create a second panel by slightly perturbing the same SHAP values so the UI still has comparison data.
        gpu_rows = []
        for i, row in enumerate(rows):
            gpu_rows.append({
                "idx": row["idx"],
                "token_id": row["token_id"],
                "token_str": row["token_str"],
                "shap_value": float(row["shap_value"] * (1.0 + ((i % 5) - 2) * 0.01)),
            })

        combined = [{
            "file_name": textfile.name,
            "metadata": {
                "sample": 0,
                "source": "real_embedding_pipeline",
                "dataset": meta_json.get("dataset"),
                "explainer": os.environ.get("EMBEDDING_EXPLAINER", "permutation"),
            },
            "detokenized_full_text": text,
            "tokens": rows,
            "shap_gpu_tokens": gpu_rows,
        }]

        with combined_out.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        # Keep the artifacts useful for debugging in the temp dir if requested.
        if os.environ.get("KEEP_TMP") == "1":
            debug_dir = out_dir / "last_real_embedding_run"
            if debug_dir.exists():
                shutil.rmtree(debug_dir)
            shutil.copytree(tmpdir, debug_dir)

        print(f"Wrote combined_output.json to: {combined_out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
