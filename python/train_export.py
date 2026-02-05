#!/usr/bin/env python3

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a linear-activation MLP on a public text dataset tokenized with BERT; export weights + tokenized dataset.")

    p.add_argument("--dataset", type=str, default="ag_news", help="HuggingFace datasets name (e.g. ag_news, imdb)")
    p.add_argument("--text-field", type=str, default=None, help="Optional override for the text column")
    p.add_argument("--label-field", type=str, default=None, help="Optional override for the label column")

    p.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="Tokenizer name from transformers")
    p.add_argument("--max-len", type=int, default=64)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3, help="Total number of Linear layers including output")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--test-samples", type=int, default=256)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--out-dir", type=str, default="out")
    p.add_argument("--weights-file", type=str, default="mlp_weights.txt")
    p.add_argument("--dataset-file", type=str, default="tokenized_dataset.txt")
    p.add_argument("--meta-file", type=str, default="export_meta.json")

    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DatasetSpec:
    text_field: str
    label_field: str


def infer_dataset_spec(dataset_name: str, text_field: Optional[str], label_field: Optional[str], column_names: List[str]) -> DatasetSpec:
    if text_field and label_field:
        return DatasetSpec(text_field=text_field, label_field=label_field)

    # Common defaults
    if dataset_name == "ag_news":
        return DatasetSpec(text_field=text_field or "text", label_field=label_field or "label")
    if dataset_name == "imdb":
        return DatasetSpec(text_field=text_field or "text", label_field=label_field or "label")

    # Heuristic: prefer these names if present
    preferred_text = ["text", "sentence", "review", "content"]
    preferred_label = ["label", "labels", "category"]

    chosen_text = text_field
    chosen_label = label_field

    if chosen_text is None:
        for cand in preferred_text:
            if cand in column_names:
                chosen_text = cand
                break
    if chosen_label is None:
        for cand in preferred_label:
            if cand in column_names:
                chosen_label = cand
                break

    if chosen_text is None or chosen_label is None:
        raise ValueError(f"Could not infer text/label fields from columns={column_names}. Provide --text-field/--label-field.")

    return DatasetSpec(text_field=chosen_text, label_field=chosen_label)


class LinearMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: List[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity activation between layers ("linear activations")
        for layer in self.layers:
            x = layer(x)
        return x


def export_weights_text(model: LinearMLP, path: str) -> None:
    # Plain-text, easy to parse from C++:
    # L
    # in0 out0
    # W (out0 * in0 floats row-major)
    # b (out0 floats)
    # ...
    layers = list(model.layers)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(layers)}\n")
        for layer in layers:
            assert isinstance(layer, nn.Linear)
            w = layer.weight.detach().cpu().contiguous().float()
            b = layer.bias.detach().cpu().contiguous().float()
            out_dim, in_dim = w.shape
            f.write(f"{in_dim} {out_dim}\n")
            # weights row-major: row = out, col = in
            flat_w = w.reshape(-1).tolist()
            f.write(" ".join(f"{v:.9g}" for v in flat_w) + "\n")
            f.write(" ".join(f"{v:.9g}" for v in b.tolist()) + "\n")


def export_dataset_text(token_ids: torch.Tensor, labels: torch.Tensor, path: str) -> None:
    # Format:
    # N seq_len
    # label id0 id1 ... id(seq_len-1)
    token_ids = token_ids.detach().cpu().to(torch.int32)
    labels = labels.detach().cpu().to(torch.int32)
    n, seq_len = token_ids.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n} {seq_len}\n")
        for i in range(n):
            ids = token_ids[i].tolist()
            f.write(str(int(labels[i].item())))
            for t in ids:
                f.write(" ")
                f.write(str(int(t)))
            f.write("\n")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        total_loss += float(loss_fn(logits, yb).item())
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    weights_path = os.path.join(args.out_dir, args.weights_file)
    dataset_path = os.path.join(args.out_dir, args.dataset_file)
    meta_path = os.path.join(args.out_dir, args.meta_file)

    device = pick_device(args.device)

    # Lazy imports so error messages are clearer if deps missing.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(args.dataset)
    if "train" not in ds:
        raise ValueError(f"Dataset '{args.dataset}' has no 'train' split")

    train_split = ds["train"]
    test_split = ds["test"] if "test" in ds else ds["train"]

    spec = infer_dataset_spec(args.dataset, args.text_field, args.label_field, train_split.column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab_size = int(getattr(tokenizer, "vocab_size", 30522))

    # Subsample for speed
    train_n = min(args.train_samples, len(train_split))
    test_n = min(args.test_samples, len(test_split))

    train_split = train_split.select(range(train_n))
    test_split = test_split.select(range(test_n))

    def tok_batch(batch):
        enc = tokenizer(
            batch[spec.text_field],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
            return_attention_mask=False,
        )
        return {"input_ids": enc["input_ids"]}

    train_tok = train_split.map(tok_batch, batched=True, remove_columns=[c for c in train_split.column_names if c != spec.label_field])
    test_tok = test_split.map(tok_batch, batched=True, remove_columns=[c for c in test_split.column_names if c != spec.label_field])

    # Convert to tensors
    train_ids = torch.tensor(train_tok["input_ids"], dtype=torch.int64)
    train_y = torch.tensor(train_tok[spec.label_field], dtype=torch.int64)
    test_ids = torch.tensor(test_tok["input_ids"], dtype=torch.int64)
    test_y = torch.tensor(test_tok[spec.label_field], dtype=torch.int64)

    # Features: token_id / vocab_size (float)
    train_x = train_ids.to(torch.float32) / float(vocab_size)
    test_x = test_ids.to(torch.float32) / float(vocab_size)

    num_classes = int(train_y.max().item() + 1)
    input_dim = int(train_x.shape[1])

    model = LinearMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=num_classes, num_layers=args.num_layers).to(device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch} train_loss={running/max(seen,1):.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    # Export (CUDA will use token IDs and do the same normalization)
    export_weights_text(model, weights_path)
    export_dataset_text(test_ids, test_y, dataset_path)

    meta = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "max_len": args.max_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_classes": num_classes,
        "export": {
            "weights_file": os.path.basename(weights_path),
            "dataset_file": os.path.basename(dataset_path),
        },
        "feature_transform": "x = token_id / vocab_size",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote: {weights_path}")
    print(f"wrote: {dataset_path}")
    print(f"wrote: {meta_path}")


if __name__ == "__main__":
    main()
