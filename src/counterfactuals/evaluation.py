from __future__ import annotations

import torch
from ..core.geometry import choose_target_label, decision_margin, estimate_local_geometry, class_support_thresholds, endpoint_supported
from ..core.utils import ensure_2d, mean_std
from .search import build_baseline_config, generate_counterfactual


SUMMARY_SKIP_KEYS = {"start_label", "final_label", "target_label", "example_index"}
SUCCESS_ONLY_KEYS = {"counterfactual_distance", "optimization_effort"}


def _module_device(module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_single_example(
    z: torch.Tensor,
    classifier_head,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    k: int = 20,
    step_size: float = 1e-2,
    max_steps: int = 300,
    trust_radius: float = 1.0,
    optimizer_name: str = "sgd",
    exclude_self: bool = False,
    record_trajectory: bool = False,
    max_trajectory_points: int = 10,
    shift_weight: float = 0.0,
    tangent_dim: int = 2,
    config=None,
    target_label: int | None = None,
    support_thresholds: dict[int, float] | None = None,
    support_quantile: float = 0.5,
    record_endpoint: bool = False,
) -> dict[str, float | int | bool]:
    device = _module_device(classifier_head)
    z = z.to(device)
    reference_embeddings = reference_embeddings.to(device)
    reference_labels = reference_labels.to(device)

    with torch.no_grad():
        logits = classifier_head(ensure_2d(z)).squeeze(0)
        predicted_label = int(torch.argmax(logits).item())
        if target_label is None:
            target_label = choose_target_label(logits)
        initial_margin = decision_margin(logits, predicted_label, target_label)

    geometry_stats = estimate_local_geometry(
        z=z,
        predicted_label=predicted_label,
        classifier_head=classifier_head,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        neighborhood_label=target_label,
        k=k,
        exclude_self=exclude_self,
        tangent_dim=config.tangent_dim if config is not None else tangent_dim,
        curvature_eps=config.curvature_eps if config is not None else 1e-8,
    )

    config = config or build_baseline_config(
        step_size=step_size,
        trust_radius=trust_radius,
        max_steps=max_steps,
        optimizer_name=optimizer_name,
        shift_weight=shift_weight,
        tangent_dim=tangent_dim,
    )
    search_result = generate_counterfactual(
        z0=z,
        classifier_head=classifier_head,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        config=config,
        k=k,
        target_label=target_label,
        record_trajectory=record_trajectory,
        max_trajectory_points=max_trajectory_points,
    )

    result = {
        **geometry_stats,
        "counterfactual_distance": search_result.distance,
        "counterfactual_margin": search_result.margin,
        "decision_margin": initial_margin,
        "optimization_effort": search_result.optimization_effort,
        "counterfactual_success": search_result.success,
        "start_label": search_result.start_label,
        "final_label": search_result.final_label,
    }
    result["target_support_radius"] = search_result.density
    result["target_label"] = search_result.target_label
    thresholds = support_thresholds
    if thresholds is None:
        thresholds = class_support_thresholds(reference_embeddings, reference_labels, k, support_quantile)
    threshold = thresholds.get(search_result.target_label, float("nan"))
    result["support_threshold"] = threshold
    result["supported_counterfactual_success"] = search_result.success and endpoint_supported(search_result.density, threshold)
    if record_endpoint:
        result["final_embedding"] = search_result.final_embedding.cpu().tolist()
    if search_result.trajectory is not None:
        result["counterfactual_trajectory"] = [point.tolist() for point in search_result.trajectory]
    return result


def summarize_metrics(results: list[dict[str, float | int | bool]]) -> dict[str, float]:
    if not results:
        return {}

    summary: dict[str, float] = {}
    keys = [
        key
        for key, value in results[0].items()
        if isinstance(value, (int, float, bool)) and key not in SUMMARY_SKIP_KEYS
    ]
    for key in keys:
        selected = [row for row in results if row["counterfactual_success"]] if key in SUCCESS_ONLY_KEYS else results
        values = [float(result[key]) for result in selected]
        mean, std = mean_std(values) if values else (float("nan"), float("nan"))
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    summary["num_evaluated"] = len(results)
    summary["num_successful"] = sum(bool(row["counterfactual_success"]) for row in results)
    return summary


def evaluate_embeddings(
    embeddings: torch.Tensor,
    classifier_head,
    labels: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    max_examples: int | None = None,
    same_reference_pool: bool = False,
    example_indices: torch.Tensor | None = None,
    record_trajectory: bool = False,
    max_trajectory_points: int = 10,
    **kwargs,
) -> tuple[list[dict[str, float | int | bool]], dict[str, float]]:
    del labels
    if len(embeddings) == 0:
        raise ValueError("Evaluation requires at least one example")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive")
    if "support_thresholds" not in kwargs:
        kwargs["support_thresholds"] = class_support_thresholds(reference_embeddings, reference_labels,
                                                               kwargs.get("k", 20), kwargs.get("support_quantile", 0.5))
    results = []
    total = embeddings.size(0) if max_examples is None else min(max_examples, embeddings.size(0))
    for index in range(total):
        result = evaluate_single_example(
            z=embeddings[index],
            classifier_head=classifier_head,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            exclude_self=same_reference_pool,
            record_trajectory=record_trajectory,
            max_trajectory_points=max_trajectory_points,
            **kwargs,
        )
        result["example_index"] = int(example_indices[index].item()) if example_indices is not None else index
        results.append(result)
    return results, summarize_metrics(results)
