from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

from .common import (
    DATASET_ORDER,
    MULTIMODAL_FUSION_REPRESENTATION,
    all_intervention_paths,
    attach_cross_entropy,
    dataset_label,
    encoder_label,
    experiment_summary_row,
    load_json,
    main_experiment_paths,
    mean_std,
    metric_value,
    model_label,
    write_text,
)


def _run_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    seed = row.get("seed")
    if seed is None:
        return (-1, str(row["path"]))
    return (int(seed), str(row["path"]))


def _paired_runs(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_by_seed = {row["seed"]: row for row in left_rows if row.get("seed") is not None}
    right_by_seed = {row["seed"]: row for row in right_rows if row.get("seed") is not None}
    common_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if common_seeds:
        return [(left_by_seed[seed], right_by_seed[seed]) for seed in common_seeds]
    ordered_left = sorted(left_rows, key=_run_sort_key)
    ordered_right = sorted(right_rows, key=_run_sort_key)
    count = min(len(ordered_left), len(ordered_right))
    return list(zip(ordered_left[:count], ordered_right[:count]))


def _aggregate_pairwise_gaps(rows: list[dict[str, Any]], gap_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), {}).setdefault(str(row["model"]), []).append(row)
    output: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        dataset_rows = grouped.get(dataset, {})
        for left_model, right_model in combinations(sorted(dataset_rows), 2):
            pairs = _paired_runs(dataset_rows[left_model], dataset_rows[right_model])
            if not pairs:
                continue
            gap_values = [abs(float(left[gap_key]) - float(right[gap_key])) for left, right in pairs]
            cf_suc_gaps = [abs(float(left["cf_suc"]) - float(right["cf_suc"])) for left, right in pairs]
            cf_dist_gaps = [abs(float(left["cf_dist"]) - float(right["cf_dist"])) for left, right in pairs]
            opt_eff_gaps = [abs(float(left["opt_eff"]) - float(right["opt_eff"])) for left, right in pairs]
            gap_mean, gap_std = mean_std(gap_values)
            cf_suc_mean, cf_suc_std = mean_std(cf_suc_gaps)
            cf_dist_mean, cf_dist_std = mean_std(cf_dist_gaps)
            opt_eff_mean, opt_eff_std = mean_std(opt_eff_gaps)
            output.append({
                "dataset": dataset,
                "model_a": left_model,
                "model_b": right_model,
                gap_key: gap_mean,
                f"{gap_key}_std": gap_std,
                "cf_suc_gap": cf_suc_mean,
                "cf_suc_gap_std": cf_suc_std,
                "cf_dist_gap": cf_dist_mean,
                "cf_dist_gap_std": cf_dist_std,
                "opt_eff_gap": opt_eff_mean,
                "opt_eff_gap_std": opt_eff_std,
                "num_runs": len(pairs),
            })
    output.sort(
        key=lambda row: (
            DATASET_ORDER.index(str(row["dataset"])),
            float(row[gap_key]),
            str(row["model_a"]),
            str(row["model_b"]),
        )
    )
    return output


def _select_best_pair_per_dataset(rows: list[dict[str, Any]], gap_key: str) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        matches = [row for row in rows if row["dataset"] == dataset]
        if matches:
            best_rows.append(matches[0])
    return best_rows


def _format_mean_pm_std(mean: float, std: float, decimals: int) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _format_model_gap_table_md(best_rows: list[dict[str, Any]], gap_key: str, gap_label: str) -> str:
    lines = [
        f"| Dataset | Model A | Model B | {gap_label} | CF-Suc Gap | CF-Dist Gap | OptEff Gap |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in best_rows:
        lines.append(
            f"| {dataset_label(str(row['dataset']))} | {row['model_a']} | {row['model_b']} | "
            f"{float(row[gap_key]):.4f} +/- {float(row[f'{gap_key}_std']):.4f} | "
            f"{float(row['cf_suc_gap']):.4f} +/- {float(row['cf_suc_gap_std']):.4f} | "
            f"{float(row['cf_dist_gap']):.4f} +/- {float(row['cf_dist_gap_std']):.4f} | "
            f"{float(row['opt_eff_gap']):.4f} +/- {float(row['opt_eff_gap_std']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def _pair_count_caption_suffix(rows: list[dict[str, Any]]) -> str:
    pair_counts = {int(row["num_runs"]) for row in rows if "num_runs" in row}
    if pair_counts == {5}:
        return "Reported values show mean differences with standard deviation across five seeds."
    return "Reported values show mean differences with standard deviation across paired runs."


def _fixed_representation_range_rows(interventions_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in all_intervention_paths(interventions_dir):
        payload = load_json(path)
        dataset = str(payload.get("dataset", "")).lower()
        if dataset not in DATASET_ORDER:
            continue
        representation = str(payload.get("representation", "")).lower()
        if dataset == "mmimdb":
            multimodal_rows = {None, "", "multimodal"}
            if representation not in multimodal_rows and representation != MULTIMODAL_FUSION_REPRESENTATION:
                continue
            current_model_label = model_label(payload)
            if "+" not in current_model_label and current_model_label not in {"CLIP", "SigLIP2"}:
                continue
        else:
            current_model_label = encoder_label(str(payload.get("encoder", "")).lower())
        variants: list[dict[str, Any]] = []
        for variant in payload.get("variants", []):
            if not isinstance(variant, dict):
                continue
            training = variant.get("training", {})
            variation_type = str(training.get("variation_type", "baseline"))
            if variant.get("name") == "baseline_checkpoint" or variation_type == "seed_only":
                variants.append(variant)
        if not variants:
            continue
        accuracies = [float(variant["test_accuracy"]) for variant in variants]
        cf_sucs = [metric_value(variant, "counterfactual_success_mean") for variant in variants]
        rows.append({
            "dataset": dataset,
            "encoder": current_model_label,
            "acc_range": max(accuracies) - min(accuracies),
            "cf_suc_range": max(cf_sucs) - min(cf_sucs),
        })
    rows.sort(key=lambda row: (DATASET_ORDER.index(str(row["dataset"])), str(row["encoder"])))
    return rows


def _format_fixed_representation_range_table_md(rows: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Encoder | Delta ACC Range | Delta CF-Suc Range |", "| --- | --- | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {dataset_label(str(row['dataset']))} | {row['encoder']} | {float(row['acc_range']):.4f} | {float(row['cf_suc_range']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_predictive_metric_comparison(
    compare_dir: Path,
    cache_dir: Path,
    interventions_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    summary_rows = [experiment_summary_row(path) for path in main_experiment_paths(compare_dir)]
    attach_cross_entropy(summary_rows, cache_dir)
    accuracy_pairs = _aggregate_pairwise_gaps(summary_rows, gap_key="test_accuracy")
    for row in accuracy_pairs:
        row["acc_gap"] = row.pop("test_accuracy")
        row["acc_gap_std"] = row.pop("test_accuracy_std")
    loss_pairs = _aggregate_pairwise_gaps(summary_rows, gap_key="val_ce")
    fixed_representation_rows = _fixed_representation_range_rows(interventions_dir)
    best_accuracy_pairs = _select_best_pair_per_dataset(accuracy_pairs, gap_key="acc_gap")
    best_loss_pairs = _select_best_pair_per_dataset(loss_pairs, gap_key="val_ce")
    payload = {
        "accuracy_matched_best_pairs": best_accuracy_pairs,
        "accuracy_matched_all_pairs": accuracy_pairs,
        "loss_matched_best_pairs": best_loss_pairs,
        "loss_matched_all_pairs": loss_pairs,
        "fixed_representation_head_ranges": fixed_representation_rows,
    }
    write_text(output_dir / "paper_tables.json", json.dumps(payload, indent=2) + "\n")
    return payload

__all__ = ["run_predictive_metric_comparison"]
