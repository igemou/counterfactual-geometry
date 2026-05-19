from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.utils import embedding_cache_path, load_probe, load_probe_checkpoint


DATASET_ORDER = ["shapes", "imdb", "mnist", "chestxray", "mmimdb"]
MAIN_STANDARD_ENCODERS = {
    "shapes": ("resnet50", "vit", "dinov2"),
    "imdb": ("bert", "distilbert", "roberta"),
    "mnist": ("resnet50", "vit", "dinov2"),
    "chestxray": ("resnet50", "vit", "dinov2"),
}
MAIN_MMIMDB_MULTIMODAL_ENCODERS = ("clip", "siglip2")
MULTIMODAL_FUSION_REPRESENTATION = "fused"


def dataset_label(name: str) -> str:
    return {
        "shapes": "Shapes",
        "imdb": "IMDB",
        "mnist": "MNIST",
        "chestxray": "ChestXray",
        "mmimdb": "MM-IMDb",
    }.get(name, name)


def encoder_label(name: str) -> str:
    return {
        "resnet50": "ResNet50",
        "vit": "ViT",
        "dinov2": "DINOv2",
        "bert": "BERT",
        "distilbert": "DistilBERT",
        "roberta": "RoBERTa",
        "clip": "CLIP",
        "siglip2": "SigLIP2",
    }.get(name, name)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or ys.size < 2:
        return 0.0
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = np.linalg.norm(xs) * np.linalg.norm(ys)
    if denom == 0.0:
        return 0.0
    return float(np.dot(xs, ys) / denom)


def spearman(xs: list[float], ys: list[float]) -> float:
    x_array = np.asarray(xs, dtype=np.float64)
    y_array = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[mask]
    y_array = y_array[mask]
    if x_array.size < 2 or y_array.size < 2:
        return 0.0
    return pearson(rankdata(x_array), rankdata(y_array))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - y_true.mean()) ** 2)
    if total == 0.0:
        return 0.0
    return float(1.0 - residual / total)


def main_experiment_paths(compare_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for dataset, encoders in MAIN_STANDARD_ENCODERS.items():
        for encoder in encoders:
            candidates = [
                compare_dir / f"{dataset}_{encoder}_encoder_comparison.json",
                compare_dir / f"{dataset}_{encoder}_experiment.json",
            ]
            for path in candidates:
                if path.exists():
                    paths.append(path)
                    break
    for encoder in MAIN_MMIMDB_MULTIMODAL_ENCODERS:
        candidates = [
            compare_dir / "mmimdb_suite" / f"mmimdb_multimodal_{encoder}_encoder_comparison.json",
            compare_dir / f"mmimdb_multimodal_{encoder}_encoder_comparison.json",
            compare_dir / f"multimodal_multimodal_{encoder}_encoder_comparison.json",
            compare_dir / "mmimdb_suite" / f"mmimdb_multimodal_{encoder}_experiment.json",
            compare_dir / f"mmimdb_multimodal_{encoder}_experiment.json",
            compare_dir / f"multimodal_multimodal_{encoder}_experiment.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                paths.append(candidate)
                break
    return paths


def all_intervention_paths(interventions_dir: Path) -> list[Path]:
    candidates = list(interventions_dir.glob("*_classifier_head_variation.json"))
    if not candidates:
        candidates = list(interventions_dir.glob("*_boundary_intervention.json"))
    return sorted(path for path in candidates if path.is_file())


def payload_seed(payload: dict[str, Any]) -> int | None:
    raw_seed = payload.get("seed")
    if raw_seed is not None:
        return int(raw_seed)
    checkpoint_path = payload.get("probe_checkpoint")
    if not checkpoint_path:
        return None
    checkpoint = load_probe_checkpoint(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    seed = metadata.get("seed")
    return int(seed) if seed is not None else None


def model_label(payload: dict[str, Any]) -> str:
    dataset = str(payload.get("dataset", "")).lower()
    if dataset != "mmimdb":
        return encoder_label(str(payload.get("encoder", "")).lower())
    representation = str(payload.get("representation", "")).lower()
    image_encoder = str(payload.get("image_encoder", "")).lower()
    text_encoder = str(payload.get("text_encoder", "")).lower()
    multimodal_encoder = str(payload.get("multimodal_encoder", "")).lower()
    encoder = str(payload.get("encoder", "")).lower()
    if representation == "multimodal" and multimodal_encoder:
        return encoder_label(multimodal_encoder)
    if representation == "fused" and image_encoder and text_encoder:
        return f"{encoder_label(image_encoder)}+{encoder_label(text_encoder)}"
    if representation == "image" and image_encoder:
        return encoder_label(image_encoder)
    if representation == "text" and text_encoder:
        return encoder_label(text_encoder)
    if encoder.endswith("_fused"):
        tokens = encoder.split("_")
        if len(tokens) >= 2:
            return f"{encoder_label(tokens[0])}+{encoder_label(tokens[1])}"
    return encoder_label(encoder)


def metric_value(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return float(value)
    raise KeyError(f"Missing metric keys {keys}")


def experiment_summary_row(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "dataset": str(payload.get("dataset", "")).lower(),
        "model": model_label(payload),
        "seed": payload_seed(payload),
        "path": str(path),
        "payload": payload,
        "val_accuracy": float(payload["val_accuracy"]),
        "test_accuracy": float(payload["test_accuracy"]),
        "cf_suc": metric_value(payload, "counterfactual_success_mean"),
        "cf_dist": metric_value(payload, "counterfactual_distance_mean"),
        "opt_eff": metric_value(payload, "optimization_effort_mean"),
    }


def split_cache_path(payload: dict[str, Any], cache_dir: Path, split: str) -> Path:
    dataset = str(payload.get("dataset", "")).lower()
    if dataset != "mmimdb":
        return embedding_cache_path(
            dataset_name=dataset,
            encoder_name=str(payload.get("encoder", "")),
            encoder_model_name=str(payload.get("encoder_model_name", "")),
            split=split,
            root=cache_dir,
        )
    representation = str(payload.get("representation", "")).lower()
    if representation == "image":
        return embedding_cache_path(
            "mmimdb",
            str(payload.get("image_encoder", "")),
            str(payload.get("image_encoder_model_name", "")),
            split,
            root=cache_dir,
        )
    if representation == "text":
        return embedding_cache_path(
            "mmimdb",
            str(payload.get("text_encoder", "")),
            str(payload.get("text_encoder_model_name", "")),
            split,
            root=cache_dir,
        )
    if representation == "fused":
        fusion_key = f"{payload.get('image_encoder', '')}-{payload.get('text_encoder', '')}"
        return embedding_cache_path("mmimdb_fused", fusion_key, None, split, root=cache_dir)
    if representation == "multimodal":
        return embedding_cache_path(
            "mmimdb_multimodal",
            str(payload.get("multimodal_encoder", "")),
            str(payload.get("multimodal_encoder_model_name", "")),
            split,
            root=cache_dir,
        )
    raise ValueError(f"Unsupported multimodal representation: {representation}")


def load_cached_split(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_path, map_location="cpu")
    embeddings = payload.get("embeddings")
    labels = payload.get("labels")
    if not isinstance(embeddings, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError(f"Invalid cached split: {cache_path}")
    return embeddings.float(), labels.long()


def validation_cross_entropy(payload: dict[str, Any], cache_dir: Path) -> float:
    checkpoint_path = payload.get("probe_checkpoint")
    if not checkpoint_path:
        raise ValueError("Missing probe checkpoint.")
    classifier, _ = load_probe(checkpoint_path, map_location="cpu")
    classifier.eval()
    embeddings, labels = load_cached_split(split_cache_path(payload, cache_dir=cache_dir, split="val"))
    with torch.no_grad():
        logits = classifier(embeddings)
        return float(F.cross_entropy(logits, labels).item())


def attach_cross_entropy(rows: list[dict[str, Any]], cache_dir: Path) -> None:
    for row in rows:
        row["val_ce"] = validation_cross_entropy(row["payload"], cache_dir)
