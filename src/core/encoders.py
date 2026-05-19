from __future__ import annotations

from typing import Any

from torch import nn
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from .multimodal_encoders import CLIPMultimodalEncoder, SigLIP2MultimodalEncoder
from .text_encoders import HuggingFaceTextEncoder
from .vision_encoders import (
    CLIPVisionEncoder,
    DinoV2Encoder,
    ResNet50Encoder,
    SigLIP2VisionEncoder,
    TorchvisionViTEncoder,
)


VISION_ENCODERS = {"resnet50", "vit", "dinov2", "clip", "siglip2"}
TEXT_ENCODERS = {"distilbert", "bert", "roberta"}
MULTIMODAL_ENCODERS = {"clip", "siglip2"}


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    return encoder


def build_encoder(name: str, **kwargs: Any) -> nn.Module:
    lowered = name.lower()
    model_name = kwargs.get("model_name")
    multimodal = bool(kwargs.get("multimodal", False))
    if lowered == "resnet50":
        return ResNet50Encoder(pretrained=kwargs.get("pretrained", True))
    if lowered == "vit":
        return TorchvisionViTEncoder(pretrained=kwargs.get("pretrained", True))
    if lowered == "distilbert":
        return HuggingFaceTextEncoder(model_name=model_name or "distilbert-base-uncased")
    if lowered == "bert":
        return HuggingFaceTextEncoder(model_name=model_name or "bert-base-uncased")
    if lowered == "roberta":
        return HuggingFaceTextEncoder(model_name=model_name or "roberta-base")
    if lowered == "dinov2":
        return DinoV2Encoder(model_name=model_name or "facebook/dinov2-base")
    if lowered == "siglip2":
        if multimodal:
            return SigLIP2MultimodalEncoder(model_name=model_name or "google/siglip2-base-patch16-224")
        return SigLIP2VisionEncoder(model_name=model_name or "google/siglip2-base-patch16-224")
    if lowered == "clip":
        if multimodal:
            return CLIPMultimodalEncoder(model_name=model_name or "openai/clip-vit-base-patch32")
        return CLIPVisionEncoder(model_name=model_name or "openai/clip-vit-base-patch32")
    raise ValueError(f"Unsupported encoder: {name}")


def build_processor(name: str, model_name: str | None = None):
    lowered = name.lower()
    if lowered == "distilbert":
        return AutoTokenizer.from_pretrained(model_name or "distilbert-base-uncased")
    if lowered == "bert":
        return AutoTokenizer.from_pretrained(model_name or "bert-base-uncased")
    if lowered == "roberta":
        return AutoTokenizer.from_pretrained(model_name or "roberta-base")
    if lowered == "dinov2":
        return AutoImageProcessor.from_pretrained(model_name or "facebook/dinov2-base")
    if lowered in {"siglip2", "clip"}:
        default_model = "google/siglip2-base-patch16-224" if lowered == "siglip2" else "openai/clip-vit-base-patch32"
        return AutoProcessor.from_pretrained(model_name or default_model)
    return None


def unpack_batch(batch):
    if isinstance(batch, dict):
        if "labels" in batch:
            labels = batch["labels"]
        elif "label" in batch:
            labels = batch["label"]
        else:
            labels = None
        features = {key: value for key, value in batch.items() if key not in {"label", "labels"}}
        return features, labels

    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]

    return batch, None


def encode_batch(encoder: nn.Module, batch):
    features, labels = unpack_batch(batch)
    if isinstance(features, dict):
        if "text" in features and "image" not in features and "pixel_values" not in features:
            raise ValueError("Text batches must be tokenized before calling encode_batch.")
        embeddings = encoder(**features)
    else:
        embeddings = encoder(features)
    return embeddings, labels
