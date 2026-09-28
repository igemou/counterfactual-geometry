from __future__ import annotations

from ..core.utils import write_json

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..analysis.common import DATASET_ORDER, model_label, spearman
from ..core.geometry import endpoint_supported
from ..analysis.supported_flips import _within_class_knn_medians
from ..counterfactuals.search import build_baseline_config, generate_counterfactual
from ..analysis.common import DEFAULT_CACHE_DIR, DEFAULT_COMPARE_DIR, DEFAULT_OUTPUT_DIR, load_main_payloads, load_payload_probe, load_payload_splits, evaluation_indices


def _mahalanobis_score(z: torch.Tensor, mean: torch.Tensor, precision: torch.Tensor) -> float:
    delta = (z.float() - mean.float()).unsqueeze(0)
    value = delta @ precision.float() @ delta.transpose(0, 1)
    score = float(torch.sqrt(torch.clamp(value.squeeze(), min=0.0)).item())
    return score


def _class_precision(class_embeddings: torch.Tensor, regularization: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean = class_embeddings.float().mean(dim=0)
    centered = class_embeddings.float() - mean
    feature_dim = class_embeddings.size(1)
    if class_embeddings.size(0) < 2:
        ridge = torch.eye(feature_dim, dtype=torch.float32) / max(regularization, 1e-8)
        return mean, ridge
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(class_embeddings.size(0) - 1, 1)
    average_variance = float(torch.trace(covariance).item()) / max(feature_dim, 1)
    ridge_scale = regularization * max(average_variance, 1e-6)
    covariance = covariance + ridge_scale * torch.eye(feature_dim, dtype=torch.float32)
    precision = torch.linalg.pinv(covariance)
    return mean, precision


def _within_class_mahalanobis_medians(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    regularization: float,
) -> tuple[dict[int, float], dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    thresholds: dict[int, float] = {}
    stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for class_id in sorted(int(value) for value in labels.unique().tolist()):
        class_embeddings = embeddings[labels == class_id]
        if class_embeddings.numel() == 0:
            thresholds[class_id] = float("inf")
            continue
        mean, precision = _class_precision(class_embeddings, regularization=regularization)
        stats[class_id] = (mean, precision)
        if class_embeddings.size(0) < 2:
            thresholds[class_id] = float("inf")
            continue
        scores = [_mahalanobis_score(class_embeddings[index], mean, precision) for index in range(class_embeddings.size(0))]
        thresholds[class_id] = float(torch.median(torch.tensor(scores, dtype=torch.float32)).item())
    return thresholds, stats


def _evaluate_payload(
    payload: dict[str, Any],
    cache_dir: Path,
    *,
    max_examples: int | None,
    regularization: float,
) -> dict[str, Any]:
    probe, _ = load_payload_probe(payload)
    splits = load_payload_splits(payload, cache_dir)
    eval_split = str(payload.get("eval_split", "test"))
    reference_split = str(payload.get("reference_split", "val"))
    eval_embeddings, _ = splits[eval_split]
    reference_embeddings, reference_labels = splits[reference_split]
    k = int(payload.get("k", 20))
    indices = evaluation_indices(payload, len(eval_embeddings), max_examples)

    config = build_baseline_config(
        step_size=float(payload.get("step_size", 1e-2)),
        trust_radius=float(payload.get("trust_radius", 1.0)),
        max_steps=int(payload.get("max_steps", 300)),
        shift_weight=float(payload.get("shift_weight", 0.0)),
        tangent_dim=int(payload.get("tangent_dim", 2)),
    )
    knn_thresholds = _within_class_knn_medians(reference_embeddings, reference_labels, k=k)
    mahalanobis_thresholds, mahalanobis_stats = _within_class_mahalanobis_medians(
        reference_embeddings,
        reference_labels,
        regularization=regularization,
    )

    rows: list[dict[str, Any]] = []
    for raw_index in indices:
        index = int(raw_index)
        result = generate_counterfactual(
            z0=eval_embeddings[index],
            classifier_head=probe,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            config=config,
            k=k,
        )
        target_label = int(result.target_label) if result.target_label is not None else None
        target_knn_threshold = knn_thresholds.get(target_label, float("inf")) if target_label is not None else float("inf")
        if target_label is not None and target_label in mahalanobis_stats:
            mean, precision = mahalanobis_stats[target_label]
            mahalanobis_score = _mahalanobis_score(result.final_embedding.cpu(), mean, precision)
        else:
            mahalanobis_score = float("inf")
        target_mahalanobis_threshold = (
            mahalanobis_thresholds.get(target_label, float("inf")) if target_label is not None else float("inf")
        )
        rows.append(
            {
                "example_index": index,
                "counterfactual_success": bool(result.success),
                "target_label": target_label,
                "target_knn_radius": float(result.density),
                "target_knn_threshold": float(target_knn_threshold),
                "target_mahalanobis": float(mahalanobis_score),
                "target_mahalanobis_threshold": float(target_mahalanobis_threshold),
                "supported_flip_knn": bool(result.success and endpoint_supported(float(result.density), float(target_knn_threshold))),
                "supported_flip_mahalanobis": bool(
                    result.success and endpoint_supported(float(mahalanobis_score), float(target_mahalanobis_threshold))
                ),
                "optimization_effort": int(result.optimization_effort),
                "counterfactual_distance": float(result.distance),
            }
        )

    total_count = max(len(rows), 1)
    supported_knn = sum(float(row["supported_flip_knn"]) for row in rows)
    supported_mahalanobis = sum(float(row["supported_flip_mahalanobis"]) for row in rows)
    success_rate = sum(float(row["counterfactual_success"]) for row in rows) / total_count
    agreement = (
        sum(float(bool(row["supported_flip_knn"]) == bool(row["supported_flip_mahalanobis"])) for row in rows) / total_count
        if rows
        else 0.0
    )
    return {
        "dataset": str(payload.get("dataset", "")).lower(),
        "model": model_label(payload),
        "num_examples": len(rows),
        "counterfactual_success_rate": success_rate,
        "supported_flip_rate_knn": supported_knn / total_count,
        "supported_flip_rate_mahalanobis": supported_mahalanobis / total_count,
        "supported_flip_rate_gap": (supported_mahalanobis - supported_knn) / total_count,
        "supported_flip_label_agreement": agreement,
        "mean_target_knn_radius_success_only": float(
            np.mean([float(row["target_knn_radius"]) for row in rows if bool(row["counterfactual_success"])])
        )
        if any(bool(row["counterfactual_success"]) for row in rows)
        else 0.0,
        "mean_target_mahalanobis_success_only": float(
            np.mean([float(row["target_mahalanobis"]) for row in rows if bool(row["counterfactual_success"])])
        )
        if any(bool(row["counterfactual_success"]) for row in rows)
        else 0.0,
    }


def run_support_estimator_mahalanobis(
    compare_dir: Path = DEFAULT_COMPARE_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    max_examples: int | None = None,
    regularization: float = 1e-3,
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    payloads = load_main_payloads(compare_dir)
    if datasets is not None:
        selected = {dataset.lower() for dataset in datasets}
        payloads = [payload for payload in payloads if str(payload.get("dataset", "")).lower() in selected]
    model_rows = [
        _evaluate_payload(
            payload,
            cache_dir,
            max_examples=max_examples,
            regularization=regularization,
        )
        for payload in payloads
    ]

    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        rows = [row for row in model_rows if row["dataset"] == dataset]
        if not rows:
            continue
        knn_rates = [float(row["supported_flip_rate_knn"]) for row in rows]
        mahalanobis_rates = [float(row["supported_flip_rate_mahalanobis"]) for row in rows]
        dataset_rows.append(
            {
                "dataset": dataset,
                "num_models": len(rows),
                "mean_counterfactual_success_rate": float(np.mean([float(row["counterfactual_success_rate"]) for row in rows])),
                "mean_supported_flip_rate_knn": float(np.mean(knn_rates)),
                "mean_supported_flip_rate_mahalanobis": float(np.mean(mahalanobis_rates)),
                "mean_supported_flip_rate_gap": float(np.mean(np.asarray(mahalanobis_rates) - np.asarray(knn_rates))),
                "mean_supported_flip_agreement": float(
                    np.mean([float(row["supported_flip_label_agreement"]) for row in rows])
                ),
                "spearman_model_ranking_knn_vs_mahalanobis": spearman(knn_rates, mahalanobis_rates),
            }
        )

    payload = {
        "config": {
            "max_examples": max_examples,
            "regularization": regularization,
            "datasets": sorted({str(payload.get("dataset", "")).lower() for payload in payloads}),
            "support_estimators": {
                "knn": "Supported if the target-class kNN radius at the replayed counterfactual endpoint is below the class-median within-class kNN radius on the reference split.",
                "mahalanobis": "Supported if the target-class Mahalanobis distance at the replayed counterfactual endpoint is below the class-median within-class Mahalanobis distance on the reference split.",
            },
        },
        "model_rows": model_rows,
        "dataset_rows": dataset_rows,
    }
    write_json(output_dir / "support_estimator_mahalanobis.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix experiment comparing Euclidean kNN and Mahalanobis support estimators.")
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()
    run_support_estimator_mahalanobis(
        compare_dir=args.compare_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        regularization=args.regularization,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
