from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from .common import DEFAULT_CACHE_DIR, DEFAULT_COMPARE_DIR, DEFAULT_OUTPUT_DIR, load_main_payloads, load_cached_split, split_cache_path, model_label, spearman
from ..core.geometry import class_support_thresholds, endpoint_supported
from ..core.utils import write_json


def supported_rate(rows, thresholds):
    if not rows:
        raise ValueError("Support evaluation requires search results")
    supported = sum(bool(row["counterfactual_success"]) and row.get("target_support_radius") is not None
                    and endpoint_supported(row["target_support_radius"], thresholds.get(row["target_label"], float("nan")))
                    for row in rows)
    return supported / len(rows)


def run_support_threshold_sensitivity(compare_dir=DEFAULT_COMPARE_DIR, cache_dir=DEFAULT_CACHE_DIR,
                                      output_dir=DEFAULT_OUTPUT_DIR, *, quantiles=(0.5, 0.75, 0.9)):
    model_rows = []
    for payload in load_main_payloads(compare_dir):
        refs, labels = load_cached_split(split_cache_path(payload, cache_dir, payload.get("reference_split", "val")))
        k = int(payload.get("k", 20))
        baseline = supported_rate(payload["raw_results"], class_support_thresholds(refs, labels, k, 0.5))
        rates = [{"quantile": p, "supported_success": supported_rate(
            payload["raw_results"], class_support_thresholds(refs, labels, k, p))}
                 for p in quantiles]
        model_rows.append({"dataset": payload["dataset"], "model": model_label(payload),
                           "seed": payload.get("seed"), "k": k, "baseline_supported_success": baseline, "rates": rates})
    dataset_rows = [{"dataset": dataset, "rates": [
        {"quantile": p, "supported_success": float(np.mean([
            row["rates"][i]["supported_success"] for row in model_rows if row["dataset"] == dataset]))}
        for i, p in enumerate(quantiles)]} for dataset in sorted({r["dataset"] for r in model_rows})]
    for dataset_row in dataset_rows:
        runs = [row for row in model_rows if row['dataset'] == dataset_row['dataset']]
        models = sorted({row['model'] for row in runs})
        baseline = [float(np.mean([row['baseline_supported_success'] for row in runs if row['model'] == model]))
                    for model in models]
        for i, rate in enumerate(dataset_row['rates']):
            scores = [float(np.mean([row['rates'][i]['supported_success'] for row in runs if row['model'] == model]))
                      for model in models]
            rate['spearman_vs_p05'] = (spearman(baseline, scores)
                if len(models) > 1 and np.std(baseline) > 0 and np.std(scores) > 0 else None)
    result = {"config": {"quantiles": list(quantiles), "baseline_quantile": 0.5,
                          "k": "use each input run's recorded k; main experiments use 20"},
              "model_rows": model_rows, "dataset_rows": dataset_rows}
    write_json(output_dir / "support_threshold_sensitivity.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    run_support_threshold_sensitivity(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
