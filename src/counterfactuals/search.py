from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import torch
from ..core.geometry import (
    class_knn_radius,
    choose_target_label,
    decision_margin,
    estimate_local_geometry,
    logit_gap,
    project_to_l2_ball,
    untargeted_decision_margin,
)
from ..core.utils import ensure_2d
from .tuning import GeometryScales, TunedCFConfig


@dataclass
class CounterfactualResult:
    success: bool
    start_label: int
    target_label: int | None
    final_label: int
    margin: float
    distance: float
    density: float
    optimization_effort: int
    final_embedding: torch.Tensor
    trajectory: list[torch.Tensor] | None = None


def downsample_trajectory(trajectory: list[torch.Tensor], max_points: int) -> list[torch.Tensor]:
    if max_points <= 0 or len(trajectory) <= max_points:
        return trajectory
    positions = torch.linspace(0, len(trajectory) - 1, steps=max_points)
    indices = torch.round(positions).to(torch.long).tolist()
    deduped: list[int] = []
    for index in indices:
        if not deduped or deduped[-1] != index:
            deduped.append(index)
    if deduped[0] != 0:
        deduped.insert(0, 0)
    if deduped[-1] != len(trajectory) - 1:
        deduped.append(len(trajectory) - 1)
    return [trajectory[index] for index in deduped]


def target_density(z: torch.Tensor, target_reference_embeddings: torch.Tensor, k: int = 20) -> float:
    if target_reference_embeddings.numel() == 0:
        return float("inf")
    distances = torch.cdist(ensure_2d(z), target_reference_embeddings).squeeze(0)
    k = min(k, distances.numel())
    values, _ = torch.topk(distances, k=k, largest=False)
    return float(values[-1].item())


def build_baseline_config(
    step_size: float,
    trust_radius: float,
    max_steps: int,
    optimizer_name: str = "sgd",
) -> TunedCFConfig:
    return TunedCFConfig(
        step_size=step_size,
        trust_radius=trust_radius,
        shift_weight=0.0,
        knn_weight=0.0,
        max_steps=max_steps,
        density_scale=1.0,
        boundary_scale=1.0,
        optimizer_name=optimizer_name,
        prior_scaling="baseline",
    )


def _knn_plausibility_loss(z: torch.Tensor, target_refs: torch.Tensor, k: int) -> torch.Tensor:
    if target_refs.numel() == 0:
        return z.new_tensor(0.0)
    distances = torch.cdist(ensure_2d(z), target_refs).squeeze(0)
    k = min(k, distances.numel())
    values, _ = torch.topk(distances, k=k, largest=False)
    return values.pow(2).mean()

def _reference_boundary_scale(
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    classifier_head,
    k: int,
) -> float:
    values = []
    for index in range(reference_embeddings.size(0)):
        stats = estimate_local_geometry(
            z=reference_embeddings[index],
            predicted_label=int(reference_labels[index].item()),
            classifier_head=classifier_head,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            k=k,
            exclude_self=True,
        )
        values.append(float(stats["boundary_distance"]))
    if not values:
        return 1.0
    return float(torch.tensor(values, dtype=torch.float32).median().item())


def _reference_target_support_scale(
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    classifier_head,
    k: int,
) -> float:
    values = []
    with torch.no_grad():
        for index in range(reference_embeddings.size(0)):
            logits = classifier_head(ensure_2d(reference_embeddings[index])).squeeze(0)
            if logits.numel() < 2:
                continue
            _, runner_up, _ = logit_gap(logits)
            target_refs = reference_embeddings[reference_labels == runner_up]
            radius = class_knn_radius(reference_embeddings[index], target_refs, k=k, exclude_self=False)
            if torch.isfinite(torch.tensor(radius)):
                values.append(float(radius))
    if not values:
        return 1.0
    return float(torch.tensor(values, dtype=torch.float32).median().item())


def build_geometry_scales(
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    classifier_head,
    k: int,
) -> GeometryScales:
    eps = 1e-8
    density_scale = max(
        _reference_target_support_scale(
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            classifier_head=classifier_head,
            k=k,
        ),
        eps,
    )
    boundary_scale = max(
        _reference_boundary_scale(
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            classifier_head=classifier_head,
            k=k,
        ),
        eps,
    )
    return GeometryScales(density_scale=density_scale, boundary_scale=boundary_scale)


def build_geometry_aware_config(
    scales: GeometryScales,
    step_scale: float,
    trust_scale: float,
    shift_scale: float,
    knn_scale: float,
    max_steps: int,
    optimizer_name: str = "sgd",
) -> TunedCFConfig:
    density_scale = max(scales.density_scale, 1e-8)
    boundary_scale = max(scales.boundary_scale, 1e-8)
    return TunedCFConfig(
        step_size=step_scale * density_scale,
        trust_radius=trust_scale * boundary_scale,
        shift_weight=shift_scale / density_scale,
        knn_weight=knn_scale / (density_scale**2),
        max_steps=max_steps,
        density_scale=scales.density_scale,
        boundary_scale=scales.boundary_scale,
        optimizer_name=optimizer_name,
        prior_scaling="geometry",
    )


def _build_optimizer(name: str, parameter: torch.nn.Parameter, lr: float):
    lowered = name.lower()
    if lowered == "sgd":
        return torch.optim.SGD([parameter], lr=lr)
    if lowered == "adam":
        return torch.optim.Adam([parameter], lr=lr)
    if lowered == "adamw":
        return torch.optim.AdamW([parameter], lr=lr)
    raise ValueError(f"Unsupported optimizer_name: {name}")


def generate_counterfactual(
    z0: torch.Tensor,
    classifier_head,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    config: TunedCFConfig,
    k: int,
    mode: Literal["untargeted", "targeted"] = "targeted",
    target_label: int | None = None,
    target_strategy: str = "second_best",
    record_trajectory: bool = False,
    max_trajectory_points: int = 10,
) -> CounterfactualResult:
    center = z0.detach().clone()

    with torch.no_grad():
        logits0 = classifier_head(ensure_2d(center)).squeeze(0)
        start_label = int(torch.argmax(logits0).item())
        if mode == "untargeted":
            _, inferred_target_label = untargeted_decision_margin(logits0, start_label)
            target_label = inferred_target_label
        elif target_label is None:
            target_label = choose_target_label(logits0, strategy=target_strategy)

    target_refs = reference_embeddings[reference_labels == target_label]
    z = torch.nn.Parameter(center.clone())
    optimizer = _build_optimizer(config.optimizer_name, z, lr=config.step_size)
    trajectory = [center.detach().cpu()] if record_trajectory else None
    final_label = start_label
    final_margin = float("-inf")
    steps_taken = 0
    success = False

    if mode == "untargeted":
        final_margin, target_label = untargeted_decision_margin(logits0, start_label)
    elif target_label is not None:
        final_margin = decision_margin(logits0, start_label, target_label)
    else:
        raise ValueError("target_label must be set when mode='targeted'")

    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad()
        logits = classifier_head(ensure_2d(z)).squeeze(0)
        shift_loss = torch.norm(z - center, p=2)
        knn_loss = _knn_plausibility_loss(z, target_refs, k=k)
        if mode == "untargeted":
            competitor_logits = logits.clone()
            competitor_logits[start_label] = -torch.inf
            objective = logits[start_label] - torch.max(competitor_logits)
        else:
            if target_label is None:
                raise ValueError("target_label must be set when mode='targeted'")
            objective = -logits[target_label]
        loss = objective + config.shift_weight * shift_loss + config.knn_weight * knn_loss
        loss.backward()

        with torch.no_grad():
            optimizer.step()
            z.copy_(project_to_l2_ball(z, center, config.trust_radius))
            if trajectory is not None:
                trajectory.append(z.detach().cpu())
            logits = classifier_head(ensure_2d(z)).squeeze(0)
            final_label = int(torch.argmax(logits).item())
            if mode == "untargeted":
                final_margin, current_target_label = untargeted_decision_margin(logits, start_label)
                target_label = current_target_label
            else:
                if target_label is None:
                    raise ValueError("target_label must be set when mode='targeted'")
                final_margin = decision_margin(logits, start_label, target_label)
            steps_taken = step
            if final_label != start_label:
                success = True
                break

    if target_label is not None:
        final_target_refs = reference_embeddings[reference_labels == target_label]
        density_value = target_density(z, final_target_refs, k=k)
    else:
        density_value = float("inf")
    return CounterfactualResult(
        success=success,
        start_label=start_label,
        target_label=target_label,
        final_label=final_label,
        margin=final_margin,
        distance=float(torch.norm(z - center, p=2).item()),
        density=density_value,
        optimization_effort=steps_taken,
        final_embedding=z.detach(),
        trajectory=None if trajectory is None else downsample_trajectory(trajectory, max_points=max_trajectory_points),
    )


def evaluate_counterfactual_example(
    z: torch.Tensor,
    classifier_head,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    config: TunedCFConfig,
    k: int = 20,
    exclude_self: bool = False,
    record_trajectory: bool = False,
    max_trajectory_points: int = 10,
) -> dict[str, float | int | bool]:
    with torch.no_grad():
        logits = classifier_head(ensure_2d(z)).squeeze(0)
        start_label = int(torch.argmax(logits).item())
        target_label = choose_target_label(logits, strategy="second_best")

    geometry_stats = estimate_local_geometry(
        z=z,
        predicted_label=start_label,
        classifier_head=classifier_head,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        neighborhood_label=target_label,
        k=k,
        exclude_self=exclude_self,
    )
    cf = generate_counterfactual(
        z0=z,
        classifier_head=classifier_head,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        config=config,
        k=k,
        mode="targeted",
        target_label=target_label,
        record_trajectory=record_trajectory,
        max_trajectory_points=max_trajectory_points,
    )

    result: dict[str, float | int | bool] = {
        **geometry_stats,
        "counterfactual_success": bool(cf.success),
        "counterfactual_margin": float(cf.margin),
        "counterfactual_distance": float(cf.distance),
        "counterfactual_density": float(cf.density),
        "optimization_effort": int(cf.optimization_effort),
        "start_label": int(cf.start_label),
        "target_label": int(cf.target_label) if cf.target_label is not None else -1,
        "final_label": int(cf.final_label),
    }
    if cf.trajectory is not None:
        result["counterfactual_trajectory"] = [point.tolist() for point in cf.trajectory]
    return result


def evaluate_counterfactual_embeddings(
    embeddings: torch.Tensor,
    example_indices: torch.Tensor,
    classifier_head,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    config: TunedCFConfig,
    k: int,
    exclude_self: bool,
    record_trajectory: bool,
    max_trajectory_points: int,
) -> list[dict[str, float | int | bool]]:
    results = []
    for index in range(embeddings.size(0)):
        result = evaluate_counterfactual_example(
            z=embeddings[index],
            classifier_head=classifier_head,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            config=config,
            k=k,
            exclude_self=exclude_self,
            record_trajectory=record_trajectory,
            max_trajectory_points=max_trajectory_points,
        )
        result["example_index"] = int(example_indices[index].item())
        results.append(result)
    return results
