from __future__ import annotations

import math
import torch
from .utils import ensure_2d


def _maybe_exclude_self(
    z: torch.Tensor,
    reference_embeddings: torch.Tensor,
    distances: torch.Tensor,
    exclude_self: bool,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not exclude_self or reference_embeddings.numel() == 0 or distances.numel() == 0:
        return reference_embeddings, distances

    min_distance, min_index = torch.min(distances, dim=0)
    if float(min_distance.item()) > tol:
        return reference_embeddings, distances

    keep = torch.ones(reference_embeddings.size(0), dtype=torch.bool, device=reference_embeddings.device)
    keep[int(min_index.item())] = False
    return reference_embeddings[keep], distances[keep]


def class_knn_radius(
    z: torch.Tensor,
    reference_embeddings: torch.Tensor,
    k: int = 20,
    exclude_self: bool = False,
) -> float:
    if reference_embeddings.numel() == 0:
        return float("inf")
    z = ensure_2d(z)
    distances = torch.cdist(z, reference_embeddings).squeeze(0)
    reference_embeddings, distances = _maybe_exclude_self(z, reference_embeddings, distances, exclude_self=exclude_self)
    if distances.numel() == 0:
        return float("inf")
    k = min(k, distances.numel())
    values, _ = torch.topk(distances, k=k, largest=False)
    return float(values[-1].item())


def dataset_density_scale(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 20) -> float:
    radii = []
    for index in range(embeddings.size(0)):
        same_class = labels == labels[index]
        same_class[index] = False
        refs = embeddings[same_class]
        if refs.numel() == 0:
            continue
        radius = class_knn_radius(embeddings[index], refs, k=k)
        if math.isfinite(radius):
            radii.append(radius)
    if not radii:
        raise ValueError("Unable to compute dataset_density_scale: no finite class-conditional kNN radii were available.")
    return float(torch.tensor(radii).median().item())


def logit_gap(logits: torch.Tensor) -> tuple[int, int, torch.Tensor]:
    top2 = torch.topk(logits, k=2)
    predicted = int(top2.indices[0].item())
    runner_up = int(top2.indices[1].item())
    gap = logits[predicted] - logits[runner_up]
    return predicted, runner_up, gap


def approx_boundary_distance(z: torch.Tensor, classifier_head, eps: float = 1e-8) -> float:
    z = z.detach().clone().requires_grad_(True)
    logits = classifier_head(ensure_2d(z)).squeeze(0)
    _, _, gap = logit_gap(logits)
    gradient = torch.autograd.grad(gap, z, retain_graph=False, create_graph=False)[0]
    return float(gap.abs().item() / (gradient.norm(p=2).item() + eps))


def choose_target_label(logits: torch.Tensor, strategy: str = "second_best") -> int:
    if strategy == "second_best":
        return int(torch.topk(logits, k=2).indices[1].item())
    if strategy == "least_likely":
        return int(torch.argmin(logits).item())
    raise ValueError(f"Unknown target strategy: {strategy}")


def decision_margin(logits: torch.Tensor, original_label: int, target_label: int) -> float:
    return float((logits[target_label] - logits[original_label]).item())


def untargeted_decision_margin(logits: torch.Tensor, original_label: int) -> tuple[float, int]:
    competitor_logits = logits.clone()
    competitor_logits[original_label] = -torch.inf
    target_label = int(torch.argmax(competitor_logits).item())
    margin = float((logits[target_label] - logits[original_label]).item())
    return margin, target_label


def project_to_l2_ball(z: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    delta = z - center
    norm = delta.norm(p=2)
    if norm <= radius:
        return z
    return center + delta * (radius / norm)


def estimate_local_geometry(
    z: torch.Tensor,
    predicted_label: int,
    classifier_head,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    neighborhood_label: int | None = None,
    k: int = 20,
    exclude_self: bool = False,
) -> dict[str, float]:
    label_for_local_geometry = predicted_label if neighborhood_label is None else neighborhood_label
    same_class = reference_labels == label_for_local_geometry
    class_references = reference_embeddings[same_class]
    local_support = class_knn_radius(z, class_references, k=k, exclude_self=exclude_self)
    boundary_distance = approx_boundary_distance(z, classifier_head)
    return {
        "local_support_radius": local_support,
        "boundary_distance": boundary_distance,
    }
