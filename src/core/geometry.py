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
    if k < 1:
        raise ValueError("k must be positive")
    if reference_embeddings.size(0) < k:
        return float("inf")
    z = ensure_2d(z)
    distances = torch.linalg.vector_norm(reference_embeddings - z.reshape(-1), dim=1)
    reference_embeddings, distances = _maybe_exclude_self(z, reference_embeddings, distances, exclude_self=exclude_self)
    if distances.numel() < k:
        return float("inf")
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


def choose_target_label(logits: torch.Tensor) -> int:
    """Highest-scoring alternative to the current prediction."""
    return int(torch.topk(logits, k=2).indices[1].item())


def decision_margin(logits: torch.Tensor, original_label: int, target_label: int) -> float:
    return float((logits[target_label] - logits[original_label]).item())


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
    tangent_dim: int = 2,
    curvature_eps: float = 1e-8,
) -> dict[str, float]:
    label_for_local_geometry = predicted_label if neighborhood_label is None else neighborhood_label
    same_class = reference_labels == label_for_local_geometry
    class_references = reference_embeddings[same_class]
    local_support = class_knn_radius(z, class_references, k=k, exclude_self=exclude_self)
    boundary_distance = approx_boundary_distance(z, classifier_head)
    return {
        "local_support_radius": local_support,
        "boundary_distance": boundary_distance,
        "local_curvature": local_curvature(z, reference_embeddings, k, tangent_dim, curvature_eps, exclude_self),
    }


def tangent_geometry(offsets: torch.Tensor, tangent_dim: int, eps: float = 1e-8):
    """Query-centered SVD; no centering at the neighborhood mean."""
    if offsets.ndim != 2 or not 1 <= tangent_dim < min(offsets.shape):
        raise ValueError("tangent_dim must be positive and smaller than both k and embedding dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")
    _, _, vh = torch.linalg.svd(offsets, full_matrices=False)
    basis = vh[:tangent_dim].T
    tangent = offsets @ basis
    normal = offsets - tangent @ basis.T
    curvature = normal.norm(dim=1).mean() / (tangent.square().sum(dim=1).mean() + eps)
    return curvature, basis


@torch.no_grad()
def local_curvature(z, reference_embeddings, k=20, tangent_dim=2, eps=1e-8, exclude_self=False):
    if reference_embeddings.size(0) < k + int(exclude_self):
        raise ValueError("Not enough reference points for the requested curvature neighborhood")
    z = z.reshape(-1)
    distances = torch.linalg.vector_norm(reference_embeddings - z, dim=1)
    refs, distances = _maybe_exclude_self(z, reference_embeddings, distances, exclude_self)
    indices = torch.topk(distances, k=k, largest=False).indices
    value, _ = tangent_geometry(refs[indices] - z, tangent_dim, eps)
    return float(value.item())


@torch.no_grad()
def calibrate_curvature(validation_embeddings, reference_embeddings, k=20, tangent_dim=2, eps=1e-8,
                        same_reference_pool=False):
    if len(validation_embeddings) == 0:
        raise ValueError("Curvature calibration requires validation examples")
    values = [local_curvature(z, reference_embeddings, k, tangent_dim, eps, same_reference_pool)
              for z in validation_embeddings]
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), 0.5).item())


@torch.no_grad()
def class_support_thresholds(embeddings, labels, k=20, quantile=0.5):
    """Linear-interpolated quantiles of exact leave-one-out kth radii.

    A class with fewer than k+1 points has no estimable threshold (NaN),
    and must never count as supported.
    """
    if k < 1 or not 0 <= quantile <= 1:
        raise ValueError("Require k >= 1 and quantile in [0, 1]")
    thresholds = {}
    for c in labels.unique().tolist():
        refs = embeddings[labels == c]
        if len(refs) <= k:
            thresholds[int(c)] = float("nan")
            continue
        radii = []
        for i, z in enumerate(refs):
            distances = torch.linalg.vector_norm(refs - z, dim=1)
            distances[i] = torch.inf
            radii.append(distances.kthvalue(k).values)
        thresholds[int(c)] = float(torch.quantile(torch.stack(radii), quantile).item())
    return thresholds


def endpoint_supported(radius, threshold):
    return math.isfinite(radius) and math.isfinite(threshold) and radius <= threshold
