from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from .classifier import build_classifier


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    configured = os.environ.get("GEOMETRY_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    scratch_data = Path.home() / "scratch" / "data"
    if scratch_data.exists():
        return scratch_data.resolve()

    return project_root() / "data"


def hf_cache_root() -> Path | None:
    configured = os.environ.get("GEOMETRY_HF_CACHE") or os.environ.get("HF_HOME")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.exists():
            return path

    scratch_cache = Path.home() / "scratch" / "hf_cache"
    if scratch_cache.exists():
        return scratch_cache.resolve()

    return None


def _safe_cache_component(value: str | None, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    normalized = normalized.strip("-._")
    return normalized or fallback


def embedding_cache_dir(root: str | Path | None = None) -> Path:
    if root is not None:
        base = Path(root)
    else:
        configured = os.environ.get("GEOMETRY_EMBEDDING_CACHE_DIR")
        base = Path(configured) if configured else (project_root() / "outputs" / "cache" / "embeddings")
    return base.expanduser().resolve()


def embedding_cache_path(
    dataset_name: str,
    encoder_name: str,
    encoder_model_name: str | None,
    split: str,
    root: str | Path | None = None,
    version: str = "v1",
) -> Path:
    dataset_key = _safe_cache_component(dataset_name.lower(), "dataset")
    encoder_key = _safe_cache_component(encoder_name.lower(), "encoder")
    model_key = _safe_cache_component(encoder_model_name, "default-model")
    split_key = _safe_cache_component(split.lower(), "split")
    filename = f"{dataset_key}_{encoder_key}_{model_key}_{split_key}_{version}.pt"
    return embedding_cache_dir(root) / filename


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def to_device(batch, device: torch.device | str):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        values = [to_device(value, device) for value in batch]
        return type(batch)(values)
    return batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.norm(x - y, p=2).item())


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if tensor.numel() == 0:
        return 0.0, 0.0
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def load_probe_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    if "classifier_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a classifier_state_dict")
    return checkpoint


def load_probe(checkpoint_path: str | Path, map_location: str | torch.device = "cpu"):
    checkpoint = load_probe_checkpoint(checkpoint_path, map_location=map_location)
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint["num_classes"])
    metadata = checkpoint.get("metadata", {})
    projection_dim = metadata.get("projection_dim")
    projection_dim = int(projection_dim) if projection_dim is not None else None
    classifier = build_classifier(input_dim=input_dim, num_classes=num_classes, projection_dim=projection_dim)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()
    return classifier, checkpoint
