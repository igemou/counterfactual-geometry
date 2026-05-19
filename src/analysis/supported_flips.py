from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .common import DATASET_ORDER, dataset_label, load_cached_split, load_json, main_experiment_paths, metric_value, model_label, split_cache_path, spearman, write_text


def _within_class_knn_medians(embeddings: torch.Tensor, labels: torch.Tensor, k: int) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    for class_id in sorted(int(value) for value in labels.unique().tolist()):
        class_embeddings = embeddings[labels == class_id]
        if class_embeddings.size(0) < 2:
            thresholds[class_id] = float("inf")
            continue
        distances = torch.cdist(class_embeddings, class_embeddings)
        distances.fill_diagonal_(float("inf"))
        effective_k = min(k, class_embeddings.size(0) - 1)
        knn_values, _ = torch.topk(distances, k=effective_k, largest=False, dim=1)
        radii = knn_values[:, -1]
        thresholds[class_id] = float(torch.median(radii).item())
    return thresholds


def run_supported_flips_analysis(
    compare_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    k: int,
) -> dict[str, object]:
    model_rows: list[dict[str, object]] = []
    for path in main_experiment_paths(compare_dir):
        payload = load_json(path)
        reference_split = str(payload.get("reference_split", "val"))
        reference_embeddings, reference_labels = load_cached_split(split_cache_path(payload, cache_dir, reference_split))
        thresholds = _within_class_knn_medians(reference_embeddings, reference_labels, k=k)
        raw_results = [row for row in payload.get("raw_results", []) if isinstance(row, dict)]
        supported = unsupported = no_flip = 0
        for row in raw_results:
            if not bool(row.get("counterfactual_success", False)):
                no_flip += 1
                continue
            target_label = row.get("target_label")
            target_support_radius = row.get("target_support_radius", row.get("target_density"))
            if target_label is None or target_support_radius is None:
                no_flip += 1
                continue
            threshold = thresholds.get(int(target_label), float("inf"))
            if float(target_support_radius) <= threshold:
                supported += 1
            else:
                unsupported += 1
        total = max(len(raw_results), 1)
        model_rows.append({
            "dataset": str(payload.get("dataset", "")).lower(),
            "model": model_label(payload),
            "supported_flip_rate": supported / total,
            "unsupported_flip_rate": unsupported / total,
            "no_flip_rate": no_flip / total,
            "cf_suc": metric_value(payload, "counterfactual_success_mean"),
        })
    dataset_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        rows = [row for row in model_rows if row["dataset"] == dataset]
        if not rows:
            continue
        dataset_rows.append({
            "dataset": dataset,
            "mean_supported_flip_rate": float(np.mean([float(row["supported_flip_rate"]) for row in rows])),
            "mean_unsupported_flip_rate": float(np.mean([float(row["unsupported_flip_rate"]) for row in rows])),
            "mean_no_flip_rate": float(np.mean([float(row["no_flip_rate"]) for row in rows])),
            "corr_supported_flip_rate_cf_suc": spearman(
                [float(row["supported_flip_rate"]) for row in rows],
                [float(row["cf_suc"]) for row in rows],
            ),
        })
    payload = {
        "config": {
            "k": k,
            "support_threshold_statistic": "class_median_knn_radius",
            "support_definition": "A successful flip is marked supported when its target-class kNN radius at the search endpoint is less than or equal to the median within-class kNN radius of that target class on the reference split.",
        },
        "model_rows": model_rows,
        "dataset_rows": dataset_rows,
    }
    write_text(output_dir / "supported_flips.json", json.dumps(payload, indent=2) + "\n")
    return payload

__all__ = ["run_supported_flips_analysis"]
