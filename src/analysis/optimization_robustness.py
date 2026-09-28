from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..core.utils import write_json
from ..counterfactuals.evaluation import evaluate_embeddings
from .common import (
    DEFAULT_CACHE_DIR, DEFAULT_COMPARE_DIR, DEFAULT_OUTPUT_DIR,
    evaluation_indices, load_main_payloads, load_payload_probe, load_payload_splits,
    model_label, spearman,
)


def run_optimization_robustness(
    compare_dir, cache_dir, output_dir, optimizer_names, step_size_multipliers,
    trust_radius_multipliers, max_step_values,
):
    settings = [("baseline", "sgd", {})]
    settings += [("optimizer", name, {"optimizer_name": name})
                 for name in optimizer_names if name != "sgd"]
    settings += [("step_size", f"{value:g}x", {"step_multiplier": value})
                 for value in step_size_multipliers]
    settings += [("trust_radius", f"{value:g}x", {"radius_multiplier": value})
                 for value in trust_radius_multipliers]
    settings += [("max_steps", str(value), {"max_steps": value}) for value in max_step_values]
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for payload in load_main_payloads(compare_dir):
        probe, _ = load_payload_probe(payload)
        splits = load_payload_splits(payload, cache_dir)
        eval_split = payload.get("eval_split", "test")
        reference_split = payload.get("reference_split", "val")
        embeddings, labels = splits[eval_split]
        refs, ref_labels = splits[reference_split]
        indices = evaluation_indices(payload, len(embeddings))
        for kind, name, overrides in settings:
            _, summary = evaluate_embeddings(
                embeddings[indices], probe, labels[indices], refs, ref_labels,
                example_indices=indices, same_reference_pool=eval_split == reference_split,
                k=int(payload.get("k", 20)),
                step_size=float(payload["step_size"]) * overrides.get("step_multiplier", 1),
                trust_radius=float(payload["trust_radius"]) * overrides.get("radius_multiplier", 1),
                max_steps=overrides.get("max_steps", int(payload.get("max_steps", 300))),
                optimizer_name=overrides.get("optimizer_name", "sgd"),
                shift_weight=float(payload.get("shift_weight", 0)),
                tangent_dim=int(payload.get("tangent_dim", 2)),
            )
            scores[payload["dataset"]][(kind, name)][model_label(payload)].append(
                summary["counterfactual_success_mean"])

    dataset_rows = []
    for dataset, setting_scores in sorted(scores.items()):
        # Rank model means across seeds, rather than silently retaining the last seed.
        means = {setting: {model: float(np.mean(values)) for model, values in models.items()}
                 for setting, models in setting_scores.items()}
        baseline = means[("baseline", "sgd")]
        models = sorted(baseline)
        rows = [{"kind": kind, "name": name,
                 "spearman_rank_correlation": spearman(
                     [baseline[model] for model in models], [values[model] for model in models])}
                for (kind, name), values in means.items() if kind != "baseline"]
        dataset_rows.append({"dataset": dataset, "settings": rows})
    result = {
        "config": {"optimizer_names": optimizer_names,
                   "step_size_multipliers": step_size_multipliers,
                   "trust_radius_multipliers": trust_radius_multipliers,
                   "max_step_values": max_step_values},
        "datasets": dataset_rows,
    }
    write_json(output_dir / "optimization_robustness.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimization-setting robustness analysis for counterfactual evaluation.")
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
