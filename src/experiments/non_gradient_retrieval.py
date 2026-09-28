from __future__ import annotations

from ..core.utils import write_json

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..analysis.common import DATASET_ORDER, model_label, spearman
from ..analysis.supported_flips import _within_class_knn_medians
from ..core.geometry import choose_target_label, class_knn_radius, endpoint_supported
from ..counterfactuals.evaluation import evaluate_embeddings
from ..core.utils import ensure_2d
from ..analysis.common import DEFAULT_CACHE_DIR, DEFAULT_COMPARE_DIR, DEFAULT_OUTPUT_DIR, load_main_payloads, load_payload_probe, load_payload_splits, evaluation_indices


def _retrieve_target_example(
    z: torch.Tensor,
    *,
    classifier_head,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    trust_radius: float,
    k: int,
) -> dict[str, float | int | bool]:
    with torch.no_grad():
        logits = classifier_head(ensure_2d(z)).squeeze(0)
        start_label = int(torch.argmax(logits).item())
        target_label = choose_target_label(logits)

    target_refs = reference_embeddings[reference_labels == target_label]
    if target_refs.numel() == 0:
        return {
            "counterfactual_success": False,
            "start_label": start_label,
            "target_label": target_label,
            "final_label": start_label,
            "counterfactual_distance": float("inf"),
            "target_support_radius": float("inf"),
            "within_trust_radius": False,
        }

    distances = torch.cdist(ensure_2d(z), target_refs).squeeze(0)
    nearest_distance, nearest_index = torch.min(distances, dim=0)
    within_trust_radius = float(nearest_distance.item()) <= trust_radius
    if within_trust_radius:
        final_point = target_refs[int(nearest_index.item())]
        with torch.no_grad():
            final_logits = classifier_head(ensure_2d(final_point)).squeeze(0)
            final_label = int(torch.argmax(final_logits).item())
        target_support = class_knn_radius(final_point, target_refs, k=k)
        success = final_label == target_label
    else:
        final_label = start_label
        target_support = float("inf")
        success = False

    return {
        "counterfactual_success": success,
        "start_label": start_label,
        "target_label": target_label,
        "final_label": final_label,
        "counterfactual_distance": float(nearest_distance.item()),
        "target_support_radius": target_support,
        "within_trust_radius": within_trust_radius,
    }


def _evaluate_payload(payload: dict[str, Any], cache_dir: Path, *, max_examples: int | None) -> dict[str, Any]:
    probe, _ = load_payload_probe(payload)
    splits = load_payload_splits(payload, cache_dir)
    eval_split = str(payload.get("eval_split", "test"))
    reference_split = str(payload.get("reference_split", "val"))
    eval_embeddings, _ = splits[eval_split]
    reference_embeddings, reference_labels = splits[reference_split]
    indices = evaluation_indices(payload, len(eval_embeddings), max_examples)
    _, gradient_summary = evaluate_embeddings(eval_embeddings[indices], probe, splits[eval_split][1][indices],
                                               reference_embeddings, reference_labels, example_indices=indices,
                                               k=int(payload.get("k",20)), step_size=float(payload["step_size"]),
                                               trust_radius=float(payload["trust_radius"]), max_steps=int(payload.get("max_steps",300)),
                                               shift_weight=float(payload.get("shift_weight",0.)), tangent_dim=int(payload.get("tangent_dim",2)))
    trust_radius = float(payload.get("trust_radius", 1.0))
    k = int(payload.get("k", 20))
    thresholds = _within_class_knn_medians(reference_embeddings, reference_labels, k=k)

    raw_rows = []
    for raw_index in indices:
        index = int(raw_index)
        result = _retrieve_target_example(
            eval_embeddings[index],
            classifier_head=probe,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            trust_radius=trust_radius,
            k=k,
        )
        target_label = int(result["target_label"])
        target_support = float(result["target_support_radius"])
        threshold = thresholds.get(target_label, float("inf"))
        raw_rows.append(
            {
                "example_index": index,
                **result,
                "supported_flip": bool(result["counterfactual_success"] and endpoint_supported(target_support, threshold)),
                "support_threshold": float(threshold),
            }
        )

    total_count = max(len(raw_rows), 1)
    success_rate = float(np.mean([float(row["counterfactual_success"]) for row in raw_rows])) if raw_rows else 0.0
    supported_flip_rate = float(np.mean([float(row["supported_flip"]) for row in raw_rows])) if raw_rows else 0.0
    mean_distance = float(np.mean([float(row["counterfactual_distance"]) for row in raw_rows if row["counterfactual_success"]])) if raw_rows else 0.0
    within_radius_rate = float(np.mean([float(row["within_trust_radius"]) for row in raw_rows])) if raw_rows else 0.0

    return {
        "dataset": str(payload.get("dataset", "")).lower(),
        "model": model_label(payload),
        "num_examples": len(raw_rows),
        "trust_radius": trust_radius,
        "counterfactual_success_rate": success_rate,
        "supported_flip_rate": supported_flip_rate,
        "mean_counterfactual_distance": mean_distance,
        "within_trust_radius_rate": within_radius_rate,
        "gradient_based_counterfactual_success_mean": float(gradient_summary["counterfactual_success_mean"]),
        "gradient_based_counterfactual_distance_mean": float(gradient_summary["counterfactual_distance_mean"]),
        "raw_results": raw_rows,
    }


def run_non_gradient_retrieval(
    compare_dir: Path = DEFAULT_COMPARE_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    datasets: list[str] | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    payloads = load_main_payloads(compare_dir)
    if datasets is not None:
        keep = {dataset.lower() for dataset in datasets}
        payloads = [payload for payload in payloads if str(payload.get("dataset", "")).lower() in keep]

    model_rows = [_evaluate_payload(payload, cache_dir, max_examples=max_examples) for payload in payloads]
    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        rows = [row for row in model_rows if row["dataset"] == dataset]
        if not rows:
            continue
        retrieval_success = [float(row["counterfactual_success_rate"]) for row in rows]
        gradient_success = [float(row["gradient_based_counterfactual_success_mean"]) for row in rows]
        retrieval_distance = [float(row["mean_counterfactual_distance"]) for row in rows]
        gradient_distance = [float(row["gradient_based_counterfactual_distance_mean"]) for row in rows]
        dataset_rows.append(
            {
                "dataset": dataset,
                "num_models": len(rows),
                "mean_retrieval_success_rate": float(np.mean(retrieval_success)),
                "mean_gradient_success_rate": float(np.mean(gradient_success)),
                "mean_retrieval_supported_flip_rate": float(np.mean([float(row["supported_flip_rate"]) for row in rows])),
                "mean_within_trust_radius_rate": float(np.mean([float(row["within_trust_radius_rate"]) for row in rows])),
                "spearman_model_ranking_retrieval_vs_gradient_success": spearman(retrieval_success, gradient_success),
                "mean_retrieval_distance": float(np.mean(retrieval_distance)),
                "mean_gradient_distance": float(np.mean(gradient_distance)),
            }
        )

    payload = {
        "config": {
            "datasets": datasets,
            "max_examples": max_examples,
            "baseline": "nearest_target_class_within_trust_radius",
            "protocol": "For each example, pick the second-best target label under the probe, then retrieve the nearest reference embedding from that target class that lies within the same trust-radius locality constraint.",
        },
        "model_rows": model_rows,
        "dataset_rows": dataset_rows,
    }
    write_json(output_dir / "non_gradient_retrieval.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-gradient retrieval baseline for appendix.")
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    run_non_gradient_retrieval(
        compare_dir=args.compare_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
