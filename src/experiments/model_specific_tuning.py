from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
import numpy as np

from ..analysis.common import (
    DEFAULT_CACHE_DIR, DEFAULT_COMPARE_DIR, DEFAULT_OUTPUT_DIR,
    evaluation_indices, load_main_payloads, load_payload_probe, load_payload_splits, model_label,
)
from ..core.utils import write_json
from ..counterfactuals.evaluation import evaluate_embeddings


def validation_key(summary):
    """Minimize lexicographically: highest success, then smallest successful distance."""
    distance = summary["counterfactual_distance_mean"]
    distance = float(distance) if distance is not None else float("inf")
    return (-summary["counterfactual_success_mean"], distance if math.isfinite(distance) else float("inf"))


def tune_model(payload, cache_dir, step_multipliers, radius_multipliers, max_examples=None):
    probe, _ = load_payload_probe(payload)
    splits = load_payload_splits(payload, cache_dir)
    refs, ref_labels = splits[payload.get("reference_split", "val")]

    def evaluate(split, config):
        embeddings, labels = splits[split]
        indices = (evaluation_indices(payload, len(embeddings), max_examples) if split == "test"
                   else np.arange(len(embeddings))[:max_examples])
        _, summary = evaluate_embeddings(
            embeddings[indices], probe, labels[indices], refs, ref_labels,
            same_reference_pool=split == payload.get("reference_split", "val"),
            max_steps=300, k=int(payload.get("k", 20)),
            shift_weight=float(payload.get("shift_weight", 0.0)),
            tangent_dim=int(payload.get("tangent_dim", 2)), **config)
        return summary

    baseline = {key: float(payload[key]) for key in ("step_size", "trust_radius")}
    # Include the shared baseline so validation tuning cannot omit the original choice.
    candidates = [baseline]
    for step, radius in product(step_multipliers, radius_multipliers):
        if step <= 0 or radius <= 0:
            raise ValueError("Search multipliers must be positive")
        config = {"step_size": baseline["step_size"] * step, "trust_radius": baseline["trust_radius"] * radius}
        if config not in candidates:
            candidates.append(config)
    trials = [{"config": config, "validation": evaluate("val", config)} for config in candidates]
    best = min(trials, key=lambda trial: validation_key(trial["validation"]))
    return {"dataset": payload["dataset"], "model": model_label(payload), "seed": payload.get("seed"),
            "baseline_config": baseline, "tuned_config": best["config"], "trials": trials,
            "baseline_test": evaluate("test", baseline), "tuned_test": evaluate("test", best["config"])}


def run_model_specific_tuning(compare_dir=DEFAULT_COMPARE_DIR, cache_dir=DEFAULT_CACHE_DIR,
                              output_dir=DEFAULT_OUTPUT_DIR, *, step_multipliers=(0.5, 1.0, 2.0),
                              radius_multipliers=(0.5, 1.0, 2.0), max_examples=None):
    rows = [tune_model(payload, cache_dir, step_multipliers, radius_multipliers, max_examples)
            for payload in load_main_payloads(compare_dir)]
    datasets = []
    for dataset in sorted({row["dataset"] for row in rows}):
        models = [row for row in rows if row["dataset"] == dataset]
        datasets.append({"dataset": dataset, "num_runs": len(models), **{
            variant: {metric: float(np.mean([r[variant][metric] for r in models]))
                      for metric in ("counterfactual_success_mean", "counterfactual_distance_mean")}
            for variant in ("baseline_test", "tuned_test")}})
    result = {"config": {"max_steps": 300, "step_multipliers": step_multipliers,
                         "radius_multipliers": radius_multipliers, "max_examples": max_examples,
                         "selection": "highest validation CF-Suc, then lowest validation CF-Dist"},
              "model_rows": rows, "dataset_rows": datasets}
    write_json(output_dir / "model_specific_tuning.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-multipliers", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--radius-multipliers", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--max-examples", type=int)
    run_model_specific_tuning(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
