from __future__ import annotations

import torch
from ..core.geometry import choose_target_label, estimate_local_geometry
from ..core.utils import ensure_2d, mean_std
from .search import build_baseline_config, generate_counterfactual, target_density


SUMMARY_SKIP_KEYS = {"start_label", "final_label", "target_label"}


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
    counterfactual_mode: str = "targeted",
    k: int = 20,
    step_size: float = 1e-2,
    max_steps: int = 500,
    trust_radius: float = 1.0,
    optimizer_name: str = "sgd",
    exclude_self: bool = False,
    target_strategy: str = "second_best",
    record_trajectory: bool = False,
    max_trajectory_points: int = 10,
) -> dict[str, float | int | bool]:
    device = _module_device(classifier_head)
    z = z.to(device)
    reference_embeddings = reference_embeddings.to(device)
    reference_labels = reference_labels.to(device)

    with torch.no_grad():
        logits = classifier_head(ensure_2d(z)).squeeze(0)
        predicted_label = int(torch.argmax(logits).item())
        target_label = None
        if logits.numel() > 1 and counterfactual_mode == "targeted":
            target_label = choose_target_label(logits, strategy=target_strategy)
        elif logits.numel() > 1:
            competitor_logits = logits.clone()
            competitor_logits[predicted_label] = -torch.inf
            target_label = int(torch.argmax(competitor_logits).item())

    geometry_stats = estimate_local_geometry(
        z=z,
        predicted_label=predicted_label,
        classifier_head=classifier_head,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        neighborhood_label=target_label,
        k=k,
        exclude_self=exclude_self,
    )

    config = build_baseline_config(
        step_size=step_size,
        trust_radius=trust_radius,
        max_steps=max_steps,
        optimizer_name=optimizer_name,
    )
    search_result = generate_counterfactual(
        z0=z,
        classifier_head=classifier_head,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        config=config,
        k=k,
        mode=counterfactual_mode,
        target_strategy=target_strategy,
        record_trajectory=record_trajectory,
        max_trajectory_points=max_trajectory_points,
    )

    result = {
        **geometry_stats,
        "counterfactual_distance": search_result.distance,
        "counterfactual_margin": search_result.margin,
        "decision_margin": search_result.margin,
        "optimization_effort": search_result.optimization_effort,
        "counterfactual_success": search_result.success,
        "start_label": search_result.start_label,
        "final_label": search_result.final_label,
    }
    if search_result.target_label is not None:
        target_refs = reference_embeddings[reference_labels == search_result.target_label]
        result["target_support_radius"] = target_density(search_result.final_embedding, target_refs, k=k)
        result["target_label"] = search_result.target_label
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
        values = [float(result[key]) for result in results]
        mean, std = mean_std(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
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
        if example_indices is not None:
            result["example_index"] = int(example_indices[index].item())
        results.append(result)
    return results, summarize_metrics(results)
