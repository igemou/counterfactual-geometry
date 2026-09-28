from __future__ import annotations

from dataclasses import dataclass
import torch
from ..core.geometry import (
    class_knn_radius,
    local_curvature,
    choose_target_label,
    decision_margin,
    project_to_l2_ball,
)
from ..core.utils import ensure_2d


@dataclass
class SearchConfig:
    step_size: float
    trust_radius: float
    shift_weight: float
    knn_weight: float
    max_steps: int
    optimizer_name: str = "sgd"
    curvature_alpha: float = 0.0
    curvature_scale: float = 1.0
    tangent_dim: int = 2
    curvature_eps: float = 1e-8


@dataclass
class CounterfactualResult:
    success: bool
    start_label: int
    target_label: int
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


def build_baseline_config(
    step_size: float,
    trust_radius: float,
    max_steps: int,
    optimizer_name: str = "sgd",
    shift_weight: float = 0.0,
    tangent_dim: int = 2,
) -> SearchConfig:
    return SearchConfig(
        step_size=step_size,
        trust_radius=trust_radius,
        shift_weight=shift_weight,
        tangent_dim=tangent_dim,
        knn_weight=0.0,
        max_steps=max_steps,
        optimizer_name=optimizer_name,
    )


def _knn_plausibility_loss(z: torch.Tensor, target_refs: torch.Tensor, k: int) -> torch.Tensor:
    if target_refs.numel() == 0:
        return z.new_tensor(0.0)
    distances = torch.cdist(ensure_2d(z), target_refs).squeeze(0)
    k = min(k, distances.numel())
    values, _ = torch.topk(distances, k=k, largest=False)
    return values.pow(2).mean()

def _build_optimizer(name: str, parameter: torch.nn.Parameter, lr: float):
    lowered = name.lower()
    if lowered == "sgd":
        return torch.optim.SGD([parameter], lr=lr)
    if lowered == "adam":
        return torch.optim.Adam([parameter], lr=lr)
    if lowered == "adamw":
        return torch.optim.AdamW([parameter], lr=lr, weight_decay=0.0)
    raise ValueError(f"Unsupported optimizer_name: {name}")


def generate_counterfactual(
    z0: torch.Tensor,
    classifier_head,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    config: SearchConfig,
    k: int,
    target_label: int | None = None,
    record_trajectory: bool = False,
    max_trajectory_points: int = 10,
) -> CounterfactualResult:
    if config.step_size <= 0 or config.trust_radius < 0 or config.max_steps < 0:
        raise ValueError("Invalid search budget")
    if min(config.shift_weight, config.knn_weight, config.curvature_alpha, config.curvature_scale) < 0:
        raise ValueError("Search weights and curvature scale must be nonnegative")
    center = z0.detach().clone()

    with torch.no_grad():
        logits0 = classifier_head(ensure_2d(center)).squeeze(0)
        start_label = int(torch.argmax(logits0).item())
        if target_label is None:
            target_label = choose_target_label(logits0)

    if target_label == start_label or not 0 <= target_label < logits0.numel():
        raise ValueError("Target must be a valid alternative to the initial prediction")
    target_refs = reference_embeddings[reference_labels == target_label]
    if config.knn_weight > 0 and len(target_refs) < k:
        raise ValueError("Support objective requires at least k target references")
    z = torch.nn.Parameter(center.clone())
    optimizer = _build_optimizer(config.optimizer_name, z, lr=config.step_size)
    trajectory = [center.detach().cpu().clone()] if record_trajectory else None
    final_label = start_label
    final_margin = float("-inf")
    steps_taken = 0
    success = False

    final_margin = decision_margin(logits0, start_label, target_label)

    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad()
        logits = classifier_head(ensure_2d(z)).squeeze(0)
        shift_loss = torch.norm(z - center, p=2)
        knn_loss = _knn_plausibility_loss(z, target_refs, k=k) if config.knn_weight else z.new_tensor(0.0)
        objective = -logits[target_label]
        loss = objective + config.shift_weight * shift_loss + config.knn_weight * knn_loss
        z.grad = torch.autograd.grad(loss, z)[0]
        if config.curvature_alpha:
            curvature = local_curvature(z.detach(), reference_embeddings, k, config.tangent_dim, config.curvature_eps)
            lr = config.step_size / (1 + config.curvature_alpha * curvature / (config.curvature_scale + config.curvature_eps))
            for group in optimizer.param_groups:
                group["lr"] = lr

        with torch.no_grad():
            optimizer.step()
            z.copy_(project_to_l2_ball(z, center, config.trust_radius))
            if trajectory is not None:
                trajectory.append(z.detach().cpu().clone())
            logits = classifier_head(ensure_2d(z)).squeeze(0)
            final_label = int(torch.argmax(logits).item())
            final_margin = decision_margin(logits, start_label, target_label)
            steps_taken = step
            if final_label == target_label:
                success = True
                break

    return CounterfactualResult(
        success=success,
        start_label=start_label,
        target_label=target_label,
        final_label=final_label,
        margin=final_margin,
        distance=float(torch.norm(z - center, p=2).item()),
        density=class_knn_radius(z.detach(), target_refs, k=k),
        optimization_effort=steps_taken,
        final_embedding=z.detach(),
        trajectory=None if trajectory is None else downsample_trajectory(trajectory, max_points=max_trajectory_points),
    )
