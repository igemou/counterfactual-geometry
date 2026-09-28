from __future__ import annotations

from ..core.utils import write_json

from pathlib import Path
from typing import Any

import numpy as np

from .common import DATASET_ORDER, load_json, main_experiment_paths, metric_value, model_label, r2_score, validation_cross_entropy


GEOMETRY_MODELS = {
    "accuracy_only": ("accuracy",),
    "ce_only": ("cross_entropy",),
    "support_only": ("local_support_radius",),
    "curvature_only": ("local_curvature",),
    "boundary_curvature": ("boundary_distance", "local_curvature"),
    "support_curvature": ("local_support_radius", "local_curvature"),
    "boundary_support_curvature": ("boundary_distance", "local_support_radius", "local_curvature"),
    "boundary_only": ("boundary_distance",),
    "boundary_support": ("boundary_distance", "local_support_radius"),

}
GEOMETRY_OUTCOMES = ("counterfactual_success", "counterfactual_distance")


def _fit_and_score_ols(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    predictors: tuple[str, ...],
    outcome: str,
) -> float:
    def usable(row):
        return ((outcome != "counterfactual_distance" or row["counterfactual_success"])
                and all(np.isfinite(float(row[key])) for key in (*predictors, outcome)))

    train_rows = [row for row in train_rows if usable(row)]
    test_rows = [row for row in test_rows if usable(row)]
    if not train_rows or not test_rows:
        return float("nan")
    train = np.asarray([[row[key] for key in predictors] for row in train_rows], dtype=float)
    test = np.asarray([[row[key] for key in predictors] for row in test_rows], dtype=float)
    mean, std = train.mean(0), train.std(0)
    std[std == 0] = 1
    train_columns = [np.ones((len(train_rows), 1)), (train - mean) / std]
    test_columns = [np.ones((len(test_rows), 1)), (test - mean) / std]
    # Same model/dataset controls in every comparison; categories come only from training.
    for key in ("model", "dataset"):
        categories = sorted({str(row.get(key, "")) for row in train_rows})
        for category in categories[1:]:
            train_columns.append(np.array([str(row.get(key, "")) == category for row in train_rows])[:, None])
            test_columns.append(np.array([str(row.get(key, "")) == category for row in test_rows])[:, None])
    y_train = np.array([row[outcome] for row in train_rows], dtype=float)
    y_test = np.array([row[outcome] for row in test_rows], dtype=float)
    coefficients = np.linalg.lstsq(np.column_stack(train_columns), y_train, rcond=None)[0]
    return r2_score(y_test, np.column_stack(test_columns) @ coefficients)


def _geometry_rows(compare_dir: Path, cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in main_experiment_paths(compare_dir):
        payload = load_json(path)
        current_model = model_label(payload)
        accuracy = float(payload["test_accuracy"])
        cross_entropy = validation_cross_entropy(payload, cache_dir)
        for raw_result in payload.get("raw_results", []):
            if not isinstance(raw_result, dict):
                continue
            rows.append({
                "dataset": str(payload.get("dataset", "")).lower(),
                "model": current_model,
                "accuracy": accuracy,
                "cross_entropy": cross_entropy,
                "local_curvature": metric_value(raw_result, "local_curvature"),
                "boundary_distance": metric_value(raw_result, "boundary_distance"),
                "local_support_radius": metric_value(raw_result, "local_support_radius", "local_density_radius"),
                "counterfactual_success": float(bool(raw_result.get("counterfactual_success", False))),
                "counterfactual_distance": metric_value(raw_result, "counterfactual_distance"),
                "example_index": int(raw_result.get("example_index", len(rows))),
            })
    return rows


def _held_out_split(rows: list[dict[str, Any]], test_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    # All repetitions of an example stay on one side, even across encoders/seeds.
    ids = sorted({(str(r.get("dataset", "")), int(r["example_index"])) for r in rows})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    count = min(len(ids) - 1, max(1, round(len(ids) * test_fraction))) if len(ids) > 1 else 0
    test_ids = {ids[i] for i in order[:count]}
    is_test = lambda r: (str(r.get("dataset", "")), int(r["example_index"])) in test_ids
    return [r for r in rows if not is_test(r)], [r for r in rows if is_test(r)]


def run_geometry_prediction(
    compare_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    rows = _geometry_rows(compare_dir, cache_dir)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    dataset_results: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        dataset_rows = by_dataset.get(dataset, [])
        if not dataset_rows:
            continue
        train_rows, test_rows = _held_out_split(dataset_rows, test_fraction=test_fraction, seed=seed)
        outcomes: dict[str, dict[str, float]] = {}
        for outcome in GEOMETRY_OUTCOMES:
            outcomes[outcome] = {}
            for model_name, predictors in GEOMETRY_MODELS.items():
                outcomes[outcome][model_name] = _fit_and_score_ols(
                    train_rows,
                    test_rows,
                    predictors=predictors,
                    outcome=outcome,
                )
        dataset_results.append({
            "dataset": dataset,
            "num_examples": len(dataset_rows),
            "num_train_examples": len(train_rows),
            "num_test_examples": len(test_rows),
            "outcomes": outcomes,
        })
    payload = {"config": {"test_fraction": test_fraction, "seed": seed, "controls": ["model", "dataset"]}, "datasets": dataset_results}
    write_json(output_dir / "geometry_predicts_behavior.json", payload)
    return payload

__all__ = ["run_geometry_prediction"]
