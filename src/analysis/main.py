from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry_prediction import run_geometry_prediction
from .predictive_metric_comparison import run_predictive_metric_comparison
from .supported_flips import run_supported_flips_analysis
from .svm_probe_comparison import run_svm_probe_comparison
from .common import write_text


def run_all(
    compare_dir: Path,
    cache_dir: Path,
    interventions_dir: Path,
    output_dir: Path,
    test_fraction: float,
    split_seed: int,
    k: int,
    eval_split: str,
    svm_c: float,
) -> dict[str, object]:
    predictive_metrics = run_predictive_metric_comparison(compare_dir, cache_dir, interventions_dir, output_dir)
    geometry = run_geometry_prediction(
        compare_dir,
        cache_dir,
        output_dir,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    supported = run_supported_flips_analysis(
        compare_dir,
        cache_dir,
        output_dir,
        k=k,
    )
    svm = run_svm_probe_comparison(compare_dir, cache_dir, output_dir, eval_split=eval_split, svm_c=svm_c)
    payload = {"predictive_metric_comparison": predictive_metrics, "geometry_prediction": geometry, "supported_flips": supported, "svm_probe_comparison": svm}
    write_text(output_dir / "report_index.json", json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run analyses over encoder-comparison and classifier-head-variation outputs."
    )
    parser.add_argument("--compare-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/embeddings"))
    parser.add_argument("--interventions-dir", type=Path, default=Path("outputs/interventions"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hypotheses"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--task", choices=["all", "predictive_metrics", "geometry", "supported_flips", "svm_probe"], default="all")
    args = parser.parse_args()
    if args.task == "predictive_metrics":
        run_predictive_metric_comparison(args.compare_dir, args.cache_dir, args.interventions_dir, args.output_dir)
        return
    if args.task == "geometry":
        run_geometry_prediction(args.compare_dir, args.cache_dir, args.output_dir, test_fraction=args.test_fraction, seed=args.split_seed)
        return
    if args.task == "supported_flips":
        run_supported_flips_analysis(args.compare_dir, args.cache_dir, args.output_dir, k=args.k)
        return
    if args.task == "svm_probe":
        run_svm_probe_comparison(args.compare_dir, args.cache_dir, args.output_dir, eval_split=args.eval_split, svm_c=args.svm_c)
        return
    run_all(args.compare_dir, args.cache_dir, args.interventions_dir, args.output_dir, args.test_fraction, args.split_seed, args.k, args.eval_split, args.svm_c)


if __name__ == "__main__":
    main()
