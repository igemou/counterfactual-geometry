from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..core.utils import load_probe
from ..counterfactuals.evaluation import evaluate_embeddings
from .common import DATASET_ORDER, dataset_label, load_cached_split, load_json, main_experiment_paths, model_label, split_cache_path, spearman, write_text


def _evaluate_payload(
    payload: dict[str, Any],
    cache_dir: Path,
    *,
    optimizer_name: str,
    step_size: float,
    max_steps: int,
    trust_radius: float,
) -> float:
    classifier_head, _ = load_probe(payload["probe_checkpoint"], map_location="cpu")
    classifier_head = classifier_head.eval()
    eval_split = str(payload.get("eval_split", "test"))
    reference_split = str(payload.get("reference_split", "val"))
    eval_embeddings, eval_labels = load_cached_split(split_cache_path(payload, cache_dir, eval_split))
    reference_embeddings, reference_labels = load_cached_split(split_cache_path(payload, cache_dir, reference_split))
    _, summary = evaluate_embeddings(
        embeddings=eval_embeddings,
        classifier_head=classifier_head,
        labels=eval_labels,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        same_reference_pool=eval_split == reference_split,
        counterfactual_mode=str(payload.get("counterfactual_mode", "targeted")),
        k=int(payload.get("k", 20)),
        step_size=step_size,
        max_steps=max_steps,
        trust_radius=trust_radius,
        optimizer_name=optimizer_name,
    )
    return float(summary.get("counterfactual_success_mean", 0.0))


def run_optimization_robustness(
    compare_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    optimizer_names: list[str],
    step_size_multipliers: list[float],
    trust_radius_multipliers: list[float],
    max_step_values: list[int],
) -> dict[str, Any]:
    experiment_payloads = [load_json(path) for path in main_experiment_paths(compare_dir)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for payload in experiment_payloads:
        grouped.setdefault(str(payload.get("dataset", "")).lower(), []).append(payload)

    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        payloads = grouped.get(dataset, [])
        if not payloads:
            continue
        default_scores: dict[str, float] = {}
        settings: list[dict[str, Any]] = []
        for payload in payloads:
            label = model_label(payload)
            default_scores[label] = _evaluate_payload(
                payload,
                cache_dir,
                optimizer_name="sgd",
                step_size=float(payload.get("step_size", 1e-2)),
                max_steps=int(payload.get("max_steps", 500)),
                trust_radius=float(payload.get("trust_radius", 1.0)),
            )

        def _append_setting(kind: str, name: str, score_map: dict[str, float]) -> None:
            shared_models = sorted(set(default_scores) & set(score_map))
            settings.append({
                "kind": kind,
                "name": name,
                "spearman_rank_correlation": spearman(
                    [default_scores[model] for model in shared_models],
                    [score_map[model] for model in shared_models],
                ),
            })

        for optimizer_name in optimizer_names:
            if optimizer_name == "sgd":
                continue
            score_map = {}
            for payload in payloads:
                label = model_label(payload)
                score_map[label] = _evaluate_payload(
                    payload,
                    cache_dir,
                    optimizer_name=optimizer_name,
                    step_size=float(payload.get("step_size", 1e-2)),
                    max_steps=int(payload.get("max_steps", 500)),
                    trust_radius=float(payload.get("trust_radius", 1.0)),
                )
            _append_setting("optimizer", optimizer_name, score_map)

        for multiplier in step_size_multipliers:
            score_map = {}
            for payload in payloads:
                label = model_label(payload)
                score_map[label] = _evaluate_payload(
                    payload,
                    cache_dir,
                    optimizer_name="sgd",
                    step_size=float(payload.get("step_size", 1e-2)) * multiplier,
                    max_steps=int(payload.get("max_steps", 500)),
                    trust_radius=float(payload.get("trust_radius", 1.0)),
                )
            _append_setting("step_size", f"{multiplier:g}x", score_map)

        for multiplier in trust_radius_multipliers:
            score_map = {}
            for payload in payloads:
                label = model_label(payload)
                score_map[label] = _evaluate_payload(
                    payload,
                    cache_dir,
                    optimizer_name="sgd",
                    step_size=float(payload.get("step_size", 1e-2)),
                    max_steps=int(payload.get("max_steps", 500)),
                    trust_radius=float(payload.get("trust_radius", 1.0)) * multiplier,
                )
            _append_setting("trust_radius", f"{multiplier:g}x", score_map)

        for max_steps in max_step_values:
            score_map = {}
            for payload in payloads:
                label = model_label(payload)
                score_map[label] = _evaluate_payload(
                    payload,
                    cache_dir,
                    optimizer_name="sgd",
                    step_size=float(payload.get("step_size", 1e-2)),
                    max_steps=max_steps,
                    trust_radius=float(payload.get("trust_radius", 1.0)),
                )
            _append_setting("max_steps", str(max_steps), score_map)

        dataset_rows.append({"dataset": dataset, "settings": settings})

    payload = {
        "config": {
            "optimizer_names": optimizer_names,
            "step_size_multipliers": step_size_multipliers,
            "trust_radius_multipliers": trust_radius_multipliers,
            "max_step_values": max_step_values,
        },
        "datasets": dataset_rows,
    }
    write_text(output_dir / "optimization_robustness.json", json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = ["run_optimization_robustness"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimization-setting robustness analysis for counterfactual evaluation.")
    parser.add_argument("--compare-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--optimizer-names", nargs="+", default=["sgd", "adam", "adamw"])
    parser.add_argument("--step-size-multipliers", nargs="+", type=float, default=[0.5, 2.0])
    parser.add_argument("--trust-radius-multipliers", nargs="+", type=float, default=[0.5, 2.0])
    parser.add_argument("--max-step-values", nargs="+", type=int, default=[250, 1000])
    args = parser.parse_args()
    run_optimization_robustness(
        args.compare_dir,
        args.cache_dir,
        args.output_dir,
        optimizer_names=args.optimizer_names,
        step_size_multipliers=args.step_size_multipliers,
        trust_radius_multipliers=args.trust_radius_multipliers,
        max_step_values=args.max_step_values,
    )


if __name__ == "__main__":
    main()
