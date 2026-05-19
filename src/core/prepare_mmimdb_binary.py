from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import torch
from datasets import load_dataset as hf_load_dataset
from .utils import data_root


def _extract_message_content(messages: object, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != role:
            continue
        return str(message.get("content", "")).strip()
    return ""


def _extract_mmimdb_labels(messages: object) -> list[str]:
    assistant_message = _extract_message_content(messages, "assistant")
    parts = [segment.strip().lower() for segment in assistant_message.split(",")]
    return sorted({part for part in parts if part})


def _stratified_train_val_test_split(
    labels: list[int],
    seed: int,
    val_split_ratio: float,
    test_split_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
    if not labels:
        return [], [], []
    if val_split_ratio < 0.0 or test_split_ratio < 0.0 or val_split_ratio + test_split_ratio >= 1.0:
        raise ValueError("val_split_ratio and test_split_ratio must be non-negative and sum to less than 1")

    label_tensor = torch.tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    train_indices: list[torch.Tensor] = []
    val_indices: list[torch.Tensor] = []
    test_indices: list[torch.Tensor] = []

    for label in label_tensor.unique(sorted=True).tolist():
        class_indices = torch.where(label_tensor == label)[0]
        permuted = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        n_total = int(permuted.numel())

        if n_total <= 1:
            n_val = 0
            n_test = 0
        elif n_total == 2:
            n_val = 0
            n_test = 1
        else:
            n_val = int(round(n_total * val_split_ratio))
            n_test = int(round(n_total * test_split_ratio))
            n_val = max(1, n_val) if val_split_ratio > 0.0 else 0
            n_test = max(1, n_test) if test_split_ratio > 0.0 else 0
            while n_val + n_test > n_total - 1:
                if n_test >= n_val and n_test > 0:
                    n_test -= 1
                elif n_val > 0:
                    n_val -= 1
                else:
                    break

        val_indices.append(permuted[:n_val])
        test_indices.append(permuted[n_val : n_val + n_test])
        train_indices.append(permuted[n_val + n_test :])

    def _concat(parts: list[torch.Tensor]) -> list[int]:
        non_empty = [part for part in parts if part.numel() > 0]
        if not non_empty:
            return []
        return torch.cat(non_empty).tolist()

    return _concat(train_indices), _concat(val_indices), _concat(test_indices)


def prepare_mmimdb_binary(
    data_dir: Path,
    output_path: Path,
    seed: int,
    val_split_ratio: float,
    test_split_ratio: float,
) -> dict[str, object]:
    parquet_dir = data_dir / "data"
    parquet_files = sorted(str(path) for path in parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Missing MMIMDb parquet shards under {parquet_dir}")

    dataset = hf_load_dataset("parquet", data_files={"train": parquet_files}, split="train")

    atomic_label_counts: Counter[str] = Counter()
    row_atomic_labels: list[list[str]] = []
    for row in dataset:
        labels = _extract_mmimdb_labels(row.get("messages"))
        row_atomic_labels.append(labels)
        atomic_label_counts.update(labels)

    if len(atomic_label_counts) < 2:
        raise ValueError("Could not derive at least two MMIMDb atomic labels")

    selected_labels = [
        label
        for label, _ in sorted(atomic_label_counts.items(), key=lambda item: (-item[1], item[0]))[:2]
    ]
    label_to_id = {label: idx for idx, label in enumerate(selected_labels)}

    filtered_dataset_indices: list[int] = []
    filtered_labels: list[int] = []
    for dataset_index, atomic_labels in enumerate(row_atomic_labels):
        matched = [label for label in atomic_labels if label in label_to_id]
        if len(matched) != 1:
            continue
        filtered_dataset_indices.append(dataset_index)
        filtered_labels.append(label_to_id[matched[0]])

    train_positions, val_positions, test_positions = _stratified_train_val_test_split(
        labels=filtered_labels,
        seed=seed,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
    )

    def _records(positions: list[int]) -> list[dict[str, int]]:
        return [
            {
                "dataset_index": int(filtered_dataset_indices[position]),
                "label": int(filtered_labels[position]),
            }
            for position in positions
        ]

    payload = {
        "task": "mmimdb_top2_binary",
        "selected_labels": selected_labels,
        "label_counts": {label: int(atomic_label_counts[label]) for label in selected_labels},
        "seed": int(seed),
        "val_split_ratio": float(val_split_ratio),
        "test_split_ratio": float(test_split_ratio),
        "num_source_examples": int(len(dataset)),
        "num_filtered_examples": int(len(filtered_dataset_indices)),
        "train": _records(train_positions),
        "val": _records(val_positions),
        "test": _records(test_positions),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MM-IMDb as a fixed top-2 binary classification task.")
    parser.add_argument("--data-dir", type=Path, default=data_root() / "mmimdb")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--test-split-ratio", type=float, default=0.1)
    args = parser.parse_args()

    output_path = args.output or (args.data_dir / "prepared_top2_binary.json")
    payload = prepare_mmimdb_binary(
        data_dir=args.data_dir,
        output_path=output_path,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        test_split_ratio=args.test_split_ratio,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
