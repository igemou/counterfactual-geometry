from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import DATASET_ORDER, dataset_label, load_json, main_experiment_paths, metric_value, model_label, r2_score, validation_cross_entropy, write_text


GEOMETRY_MODELS = {
    "accuracy_only": ("accuracy",),
    "ce_only": ("cross_entropy",),
    "boundary_only": ("boundary_distance",),
    "boundary_support": ("boundary_distance", "local_support_radius"),
    "boundary_support_interaction": (
        "boundary_distance",
        "local_support_radius",
        "boundary_distance*local_support_radius",
    ),
}
GEOMETRY_OUTCOMES = ("counterfactual_success", "counterfactual_distance")


def _fit_and_score_ols(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    predictors: tuple[str, ...],
    outcome: str,
) -> float:
    if not train_rows or not test_rows:
        return 0.0
    base_predictors = [predictor for predictor in predictors if "*" not in predictor]

    def _row_is_finite(row: dict[str, Any]) -> bool:
        values = [float(row[outcome])]
        values.extend(float(row[predictor]) for predictor in base_predictors)
        return all(np.isfinite(values))

    train_rows = [row for row in train_rows if _row_is_finite(row)]
    test_rows = [row for row in test_rows if _row_is_finite(row)]
    if not train_rows or not test_rows:
        return 0.0

    standardized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for predictor in predictors:
        if "*" in predictor:
            continue
        train_values = np.asarray([float(row[predictor]) for row in train_rows], dtype=np.float64)
        test_values = np.asarray([float(row[predictor]) for row in test_rows], dtype=np.float64)
        mean = float(train_values.mean())
        std = float(train_values.std())
        if std == 0.0:
            standardized[predictor] = (np.zeros_like(train_values), np.zeros_like(test_values))
        else:
            standardized[predictor] = ((train_values - mean) / std, (test_values - mean) / std)

    train_columns = [np.ones(len(train_rows), dtype=np.float64)]
    test_columns = [np.ones(len(test_rows), dtype=np.float64)]
    for predictor in predictors:
        if "*" in predictor:
            left_name, right_name = predictor.split("*", maxsplit=1)
            left_train, left_test = standardized[left_name]
            right_train, right_test = standardized[right_name]
            train_columns.append(left_train * right_train)
            test_columns.append(left_test * right_test)
            continue
        train_column, test_column = standardized[predictor]
        train_columns.append(train_column)
        test_columns.append(test_column)
    train_design = np.column_stack(train_columns)
    test_design = np.column_stack(test_columns)
    y_train = np.asarray([float(row[outcome]) for row in train_rows], dtype=np.float64)
    y_test = np.asarray([float(row[outcome]) for row in test_rows], dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(train_design, y_train, rcond=None)
    predictions = test_design @ coefficients
    return r2_score(y_test, predictions)


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
                "boundary_distance": metric_value(raw_result, "boundary_distance"),
                "local_support_radius": metric_value(raw_result, "local_support_radius", "local_density_radius"),
                "counterfactual_success": float(bool(raw_result.get("counterfactual_success", False))),
                "counterfactual_distance": metric_value(raw_result, "counterfactual_distance"),
                "example_index": int(raw_result.get("example_index", len(rows))),
            })
    return rows


def _held_out_split(rows: list[dict[str, Any]], test_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    rng = np.random.default_rng(seed)
    for model_rows in grouped.values():
        ordered = sorted(model_rows, key=lambda row: int(row["example_index"]))
        if len(ordered) < 2:
            train_rows.extend(ordered)
            continue
        indices = np.arange(len(ordered))
        rng.shuffle(indices)
        test_count = max(1, int(round(len(ordered) * test_fraction)))
        test_index_set = set(indices[:test_count].tolist())
        for index, row in enumerate(ordered):
            if index in test_index_set:
                test_rows.append(row)
            else:
                train_rows.append(row)
    return train_rows, test_rows


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
    payload = {"config": {"test_fraction": test_fraction, "seed": seed}, "datasets": dataset_results}
    write_text(output_dir / "geometry_predicts_behavior.json", json.dumps(payload, indent=2) + "\n")
    return payload

__all__ = ["run_geometry_prediction"]
