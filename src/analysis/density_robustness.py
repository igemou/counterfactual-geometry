from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.geometry import class_knn_radius
from .common import DATASET_ORDER, dataset_label, load_cached_split, load_json, main_experiment_paths, metric_value, split_cache_path, spearman, write_text
from .geometry_prediction import _fit_and_score_ols, _held_out_split


def _global_feature_std(embeddings: torch.Tensor) -> float:
    if embeddings.numel() == 0:
        return 1.0
    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0)
    scale = float(torch.sqrt(torch.clamp(variance.mean(), min=1e-12)).item())
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _median_pairwise_distance(embeddings: torch.Tensor, max_points: int = 512) -> float:
    if embeddings.size(0) < 2:
        return 1.0
    if embeddings.size(0) > max_points:
        indices = torch.linspace(0, embeddings.size(0) - 1, steps=max_points, dtype=torch.long)
        sample = embeddings[indices]
    else:
        sample = embeddings
    distances = torch.cdist(sample.float(), sample.float())
    keep = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
    values = distances[keep]
    if values.numel() == 0:
        return 1.0
    scale = float(torch.median(values).item())
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _rows_for_k(compare_dir: Path, cache_dir: Path, k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in main_experiment_paths(compare_dir):
        payload = load_json(path)
        eval_split = str(payload.get("eval_split", "test"))
        reference_split = str(payload.get("reference_split", "val"))
        eval_embeddings, _ = load_cached_split(split_cache_path(payload, cache_dir, eval_split))
        reference_embeddings, reference_labels = load_cached_split(split_cache_path(payload, cache_dir, reference_split))
        feature_std_scale = _global_feature_std(reference_embeddings)
        pairwise_distance_scale = _median_pairwise_distance(reference_embeddings)
        raw_results = [row for row in payload.get("raw_results", []) if isinstance(row, dict)]
        for index, raw_result in enumerate(raw_results):
            if index >= eval_embeddings.size(0):
                break
            target_label = raw_result.get("target_label")
            if target_label is None:
                continue
            target_refs = reference_embeddings[reference_labels == int(target_label)]
            support_radius = class_knn_radius(eval_embeddings[index], target_refs, k=k)
            rows.append({
                "dataset": str(payload.get("dataset", "")).lower(),
                "boundary_distance": metric_value(raw_result, "boundary_distance"),
                "local_support_radius": support_radius,
                "local_support_radius_feature_std": support_radius / feature_std_scale,
                "local_support_radius_pairwise_median": support_radius / pairwise_distance_scale,
                "counterfactual_success": float(bool(raw_result.get("counterfactual_success", False))),
                "example_index": index,
                "model": str(payload.get("encoder", payload.get("multimodal_encoder", ""))),
            })
    return rows


def _normalization_summary(
    rows: list[dict[str, Any]],
    *,
    predictor_key: str,
    test_fraction: float,
    seed: int,
) -> dict[str, float]:
    normalized_rows = [
        {
            **row,
            "local_support_radius": float(row[predictor_key]),
        }
        for row in rows
    ]
    train_rows, test_rows = _held_out_split(normalized_rows, test_fraction=test_fraction, seed=seed)
    return {
        "spearman_support_cf_suc": spearman(
            [float(row[predictor_key]) for row in rows],
            [float(row["counterfactual_success"]) for row in rows],
        ),
        "held_out_r2_cf_suc": _fit_and_score_ols(
            train_rows,
            test_rows,
            predictors=("boundary_distance", "local_support_radius"),
            outcome="counterfactual_success",
        ),
    }


def run_density_robustness(
    compare_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    ks: list[int],
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        by_k: list[dict[str, Any]] = []
        for k in ks:
            rows = [row for row in _rows_for_k(compare_dir, cache_dir, k=k) if row["dataset"] == dataset]
            if not rows:
                continue
            train_rows, test_rows = _held_out_split(rows, test_fraction=test_fraction, seed=seed)
            by_k.append({
                "k": k,
                "spearman_support_cf_suc": spearman(
                    [float(row["local_support_radius"]) for row in rows],
                    [float(row["counterfactual_success"]) for row in rows],
                ),
                "held_out_r2_cf_suc": _fit_and_score_ols(
                    train_rows,
                    test_rows,
                    predictors=("boundary_distance", "local_support_radius"),
                    outcome="counterfactual_success",
                ),
                "scale_normalizations": {
                    "feature_std": _normalization_summary(
                        rows,
                        predictor_key="local_support_radius_feature_std",
                        test_fraction=test_fraction,
                        seed=seed,
                    ),
                    "median_pairwise_distance": _normalization_summary(
                        rows,
                        predictor_key="local_support_radius_pairwise_median",
                        test_fraction=test_fraction,
                        seed=seed,
                    ),
                },
            })
        if by_k:
            dataset_rows.append({"dataset": dataset, "results": by_k})

    payload = {
        "config": {
            "ks": ks,
            "test_fraction": test_fraction,
            "seed": seed,
            "scale_normalizations": {
                "feature_std": "local_support_radius divided by sqrt(mean per-feature variance) on the reference split",
                "median_pairwise_distance": "local_support_radius divided by the median pairwise embedding distance on the reference split",
            },
        },
        "datasets": dataset_rows,
    }
    write_text(output_dir / "density_robustness.json", json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = ["run_density_robustness"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run k-sensitivity robustness analysis for local support.")
    parser.add_argument("--compare-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_density_robustness(args.compare_dir, args.cache_dir, args.output_dir, ks=args.ks, test_fraction=args.test_fraction, seed=args.seed)


if __name__ == "__main__":
    main()
