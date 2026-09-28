from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.utils import embedding_cache_path, load_probe


DATASET_ORDER = ["shapes", "imdb", "mnist", "chestxray", "mmimdb", "papalexi"]
MAIN_STANDARD_ENCODERS = {
    "papalexi": ("scgpt", "geneformer"),
    "shapes": ("resnet50", "vit", "dinov2"),
    "imdb": ("bert", "distilbert", "roberta"),
    "mnist": ("resnet50", "vit", "dinov2"),
    "chestxray": ("resnet50", "vit", "dinov2"),
}
MAIN_MMIMDB_MULTIMODAL_ENCODERS = ("clip", "siglip2")
MULTIMODAL_FUSION_REPRESENTATION = "fused"


def dataset_label(name: str) -> str:
    return {
        "papalexi": "Papalexi",
        "shapes": "Shapes",
        "imdb": "IMDB",
        "mnist": "MNIST",
        "chestxray": "ChestXray",
        "mmimdb": "MM-IMDb",
    }.get(name, name)


def encoder_label(name: str) -> str:
    return {
        "scgpt": "scGPT",
        "geneformer": "Geneformer",
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
    """Read result metadata in the input directory and its seed subdirectories."""
    directories = [compare_dir] + sorted(p for p in compare_dir.glob("seed*") if p.is_dir())
    paths = []
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            payload = load_json(path)
            if not isinstance(payload, dict) or "raw_results" not in payload:
                continue
            if payload.get("variant") or payload.get("search_variant"):
                continue
            dataset = payload.get("dataset")
            encoder = payload.get("encoder") or payload.get("multimodal_encoder")
            allowed = (MAIN_MMIMDB_MULTIMODAL_ENCODERS if dataset == "mmimdb"
                       else MAIN_STANDARD_ENCODERS.get(dataset, ()))
            if encoder in allowed:
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No main experiment results found in {compare_dir} or its seed subdirectories")
    return paths


def all_intervention_paths(interventions_dir: Path) -> list[Path]:
    candidates = list(interventions_dir.glob("*_classifier_head_variation.json"))
    for directory in sorted(interventions_dir.glob("seed*")):
        candidates.extend(directory.glob("*_classifier_head_variation.json"))
    return sorted(path for path in candidates if path.is_file())


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
        if key in mapping:
            return float(value) if value is not None else float("nan")
    raise KeyError(f"Missing metric keys {keys}")


def experiment_summary_row(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "dataset": str(payload.get("dataset", "")).lower(),
        "model": model_label(payload),
        "seed": int(payload["seed"]),
        "path": str(path),
        "payload": payload,
        "val_accuracy": float(payload["val_accuracy"]),
        "test_accuracy": float(payload["test_accuracy"]),
        "cf_suc": metric_value(payload, "counterfactual_success_mean"),
        "cf_dist": metric_value(payload, "counterfactual_distance_mean"),
        "opt_eff": metric_value(payload, "optimization_effort_mean"),
    }


def split_cache_path(payload: dict[str, Any], cache_dir: Path, split: str) -> Path:
    if payload.get("split_paths"):
        return Path(payload["split_paths"][split])
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
    if "val_ce" in payload:
        return float(payload["val_ce"])
    checkpoint_path = payload.get("probe_checkpoint")
    classifier, _ = load_probe(checkpoint_path, map_location="cpu")
    classifier.eval()
    embeddings, labels = load_cached_split(split_cache_path(payload, cache_dir=cache_dir, split="val"))
    with torch.no_grad():
        logits = classifier(embeddings)
        return float(F.cross_entropy(logits, labels).item())


def attach_cross_entropy(rows: list[dict[str, Any]], cache_dir: Path) -> None:
    for row in rows:
        row["val_ce"] = validation_cross_entropy(row["payload"], cache_dir)


DEFAULT_COMPARE_DIR = Path("outputs")
DEFAULT_CACHE_DIR = Path("outputs/cache/embeddings")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis")


def load_main_payloads(compare_dir):
    payloads = [load_json(path) for path in main_experiment_paths(compare_dir)]
    return payloads


def load_payload_probe(payload):
    probe, checkpoint = load_probe(payload["probe_checkpoint"], map_location="cpu")
    return probe.eval(), checkpoint


def load_payload_splits(payload, cache_dir):
    return {split: load_cached_split(split_cache_path(payload, cache_dir, split))
            for split in ("train", "val", "test")}


def evaluation_indices(payload, count, max_examples=None):
    """Replay stored row IDs so comparisons use exactly the same examples."""
    rows = payload.get("raw_results", [])
    indices = [int(row["example_index"]) for row in rows] if rows else list(range(count))
    if max_examples is not None:
        indices = indices[:max_examples]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError("Invalid or duplicate evaluation example indices")
    return torch.tensor(indices, dtype=torch.long)
