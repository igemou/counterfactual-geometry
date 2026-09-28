from __future__ import annotations

from ..core.utils import write_json

from pathlib import Path

import numpy as np
import torch

from ..core.geometry import class_support_thresholds, endpoint_supported
from .common import DATASET_ORDER, load_cached_split, load_json, main_experiment_paths, metric_value, model_label, split_cache_path, spearman


def _within_class_knn_medians(embeddings: torch.Tensor, labels: torch.Tensor, k: int) -> dict[int, float]:
    return class_support_thresholds(embeddings, labels, k=k, quantile=0.5)


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
            target_support_radius = row["target_support_radius"]
            if target_label is None or target_support_radius is None:
                unsupported += 1
                continue
            threshold = thresholds.get(int(target_label), float("inf"))
            if endpoint_supported(float(target_support_radius), threshold):
                supported += 1
            else:
                unsupported += 1
        total = max(len(raw_results), 1)
        model_rows.append({
            "dataset": str(payload.get("dataset", "")).lower(),
            "model": model_label(payload),
            "seed": payload.get("seed"),
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
    write_json(output_dir / "supported_flips.json", payload)
    return payload

__all__ = ["run_supported_flips_analysis"]
