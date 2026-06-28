#!/usr/bin/env python3

import os
import json
import csv
import argparse


def parse_metadata(line):
    """
    Convert metadata line like:
    sample=0 explainer=permutation target=0 sample_size=128
    into a dictionary
    """
    metadata = {}
    parts = line.strip().split()

    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)

            # attempt numeric conversion
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except:
                pass

            metadata[k] = v

    return metadata


def parse_txt_file(filepath, skip_header=2, ignore_header=False):
    """
    Parse a single txt file and return a dictionary
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"File {filepath} is malformed")

    # ---- metadata
    if not ignore_header:
        metadata = parse_metadata(lines[0])

        # ---- detokenized text
        text_line = lines[1].strip()

        if text_line.startswith("detokenized_full_text:"):
            full_text = text_line.replace("detokenized_full_text:", "").strip()
        else:
            raise ValueError(f"Missing detokenized_full_text in {filepath}")

    # ---- TSV section
    tsv_lines = lines[skip_header:]

    reader = csv.DictReader(tsv_lines, delimiter="\t")

    tokens = []
    for row in reader:
        try:
            tokens.append({
                "idx": int(row.get("idx") or row.get("orig_idx")),
                "token_id": int(row["token_id"]),
                "token_str": row["token_str"],
                "shap_value": -float(row["shap_value"])
            })
        except Exception:
            continue
    

    output = {
        "tokens": sorted(tokens, key=lambda x: x["idx"])
    }
    if not ignore_header:
        output["file_name"] = os.path.basename(filepath)
        output["metadata"] = metadata
        output["detokenized_full_text"] = full_text
    return output


def process_directory(directory):
    """
    Process all txt files in directory
    """

    objects = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") and file.startswith("tokenized_dataset.txt."):
                filepath = os.path.join(root, file)

                try:
                    obj = parse_txt_file(filepath)
                    obj_sample = obj.get("metadata", {}).get("sample", "unknown")
                    shap_gpu = f"sample{obj_sample}_shap.txt"
                    shap_gpu_path = os.path.join(root, shap_gpu)
                    if os.path.exists(shap_gpu_path):
                        gpu_obj = parse_txt_file(shap_gpu_path, skip_header=1, ignore_header=True)
                        obj["shap_gpu_tokens"] = gpu_obj.get("tokens", [])
                    else:
                        print(f"Warning: Corresponding GPU file {shap_gpu} not found for {file}")
                    objects.append(obj)
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    return objects


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing txt files")
    parser.add_argument(
        "-o",
        "--output",
        default="combined_output.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    objects = process_directory(args.directory)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(objects)} files")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
