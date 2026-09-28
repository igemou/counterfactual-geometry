from __future__ import annotations

from ..core.utils import write_json

from pathlib import Path

import torch
from sklearn.svm import LinearSVC

from ..core.utils import load_probe
from .common import DATASET_ORDER, load_cached_split, load_json, main_experiment_paths, all_intervention_paths, metric_value, model_label, spearman, split_cache_path


def _linear_params_from_probe(probe) -> tuple[torch.Tensor, torch.Tensor]:
    layer = getattr(probe, "head", None)
    if layer is None:
        layer = getattr(probe, "linear", None)
    return layer.weight.detach().cpu().float(), layer.bias.detach().cpu().float()


def _encode_if_needed(probe, embeddings: torch.Tensor) -> torch.Tensor:
    projection = getattr(probe, "projection", None)
    if projection is None:
        return embeddings
    with torch.no_grad():
        return probe.encode(embeddings).detach().cpu().float()


def _probe_logits(probe, embeddings: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        projection = getattr(probe, "projection", None)
        if projection is None:
            return probe(embeddings)
        encoded = probe.encode(embeddings)
        return probe.classify_encoded(encoded)


def _linear_params_from_svm(model: LinearSVC, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.tensor(model.coef_, dtype=torch.float32)
    bias = torch.tensor(model.intercept_, dtype=torch.float32)
    if num_classes == 2 and weight.ndim == 2 and weight.size(0) == 1:
        weight = torch.cat([-weight, weight], dim=0)
        bias = torch.cat([-bias, bias], dim=0)
    return weight, bias


def _center_linear_params(weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered_weight = weight - weight.mean(dim=0, keepdim=True)
    centered_bias = bias - bias.mean()
    return centered_weight, centered_bias


def _angle_degrees(
    probe_weight: torch.Tensor,
    probe_bias: torch.Tensor,
    svm_weight: torch.Tensor,
    svm_bias: torch.Tensor,
) -> float:
    probe_weight, probe_bias = _center_linear_params(probe_weight, probe_bias)
    svm_weight, svm_bias = _center_linear_params(svm_weight, svm_bias)
    left = probe_weight.reshape(-1)
    right = svm_weight.reshape(-1)
    denom = torch.norm(left) * torch.norm(right)
    if float(denom.item()) == 0.0:
        return 0.0
    cosine = torch.clamp(torch.dot(left, right) / denom, min=-1.0, max=1.0)
    return float(torch.rad2deg(torch.acos(cosine)).item())


def _linear_logits(embeddings: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return embeddings @ weight.T + bias


def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=min(2, logits.size(1)), dim=1).values
    if top2.size(1) == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def run_svm_probe_comparison(
    compare_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    eval_split: str,
    svm_c: float,
    interventions_dir: Path | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    payloads = [load_json(path) for path in main_experiment_paths(compare_dir)]
    if interventions_dir is not None:
        for path in all_intervention_paths(interventions_dir):
            family = load_json(path)
            for variant in family.get("variants", []):
                if not variant.get("probe_checkpoint"):
                    raise ValueError("Head variants require saved checkpoints; rerun the head experiment")
                payloads.append({**family, **variant, "variant_name": variant["name"]})
    for payload in payloads:
        probe_checkpoint = payload.get("probe_checkpoint")
        if not probe_checkpoint:
            continue
        raw_train_embeddings, train_labels = load_cached_split(split_cache_path(payload, cache_dir, "train"))
        raw_eval_embeddings, eval_labels = load_cached_split(split_cache_path(payload, cache_dir, eval_split))
        probe, _ = load_probe(probe_checkpoint, map_location="cpu")
        probe.eval()
        probe_weight, probe_bias = _linear_params_from_probe(probe)
        train_embeddings = _encode_if_needed(probe, raw_train_embeddings)
        eval_embeddings = _encode_if_needed(probe, raw_eval_embeddings)
        svm = LinearSVC(C=svm_c, dual="auto", max_iter=20000, multi_class="crammer_singer", random_state=0)
        svm.fit(train_embeddings.numpy(), train_labels.numpy())
        svm_weight, svm_bias = _linear_params_from_svm(svm, num_classes=int(train_labels.max().item()) + 1)
        probe_logits = _probe_logits(probe, raw_eval_embeddings)
        svm_logits = _linear_logits(eval_embeddings, svm_weight, svm_bias)
        rows.append({
            "dataset": str(payload.get("dataset", "")).lower(),
            "model": model_label(payload),
            "variant": payload.get("variant_name", "encoder_baseline"),
            "weight_angle_deg": _angle_degrees(probe_weight, probe_bias, svm_weight, svm_bias),
            "margin_gap": float((signed_geometric_margins(eval_embeddings, probe_weight, probe_bias, eval_labels) - signed_geometric_margins(eval_embeddings, svm_weight, svm_bias, eval_labels)).mean()),
            "cf_suc": metric_value(payload, "counterfactual_success_mean"),
            "cf_dist": metric_value(payload, "counterfactual_distance_mean"),
        })
    correlations: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        correlations.append({
            "dataset": dataset,
            "rho_weight_angle_cf_suc": spearman(
                [float(row["weight_angle_deg"]) for row in dataset_rows],
                [float(row["cf_suc"]) for row in dataset_rows],
            ),
            "rho_weight_angle_cf_dist": spearman(
                [float(row["weight_angle_deg"]) for row in dataset_rows],
                [float(row["cf_dist"]) for row in dataset_rows],
            ),
            "rho_margin_gap_cf_suc": spearman(
                [float(row["margin_gap"]) for row in dataset_rows],
                [float(row["cf_suc"]) for row in dataset_rows],
            ),
            "rho_margin_gap_cf_dist": spearman(
                [float(row["margin_gap"]) for row in dataset_rows],
                [float(row["cf_dist"]) for row in dataset_rows],
            ),
        })
    payload = {"rows": rows, "correlations": correlations, "config": {"eval_split": eval_split, "svm_c": svm_c}}
    write_json(output_dir / "svm_probe_baseline.json", payload)
    return payload

__all__ = ["run_svm_probe_comparison"]


def signed_geometric_margins(embeddings, weight, bias, labels):
    logits = embeddings @ weight.T + bias
    competitor = logits.clone()
    competitor[torch.arange(len(labels)), labels] = -torch.inf
    other = competitor.argmax(1)
    gap = logits[torch.arange(len(labels)), labels] - logits[torch.arange(len(labels)), other]
    normal = (weight[labels] - weight[other]).norm(dim=1)
    return gap / normal.clamp_min(1e-8)
