from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn.functional as F
from torchvision import models

from torch import nn
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer, AutoModel, Dinov2Model, CLIPModel, SiglipModel

class HuggingFaceTextEncoder(nn.Module):
    uses_processor = False

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size for {model_name}")
        self.output_dim = hidden_size
        signature = inspect.signature(self.model.forward)
        self._accepted_kwargs = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in self._accepted_kwargs}
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **filtered_kwargs)
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

class ResNet50Encoder(nn.Module):
    uses_processor = False

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        self.output_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TorchvisionViTEncoder(nn.Module):
    uses_processor = False

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        self.output_dim = model.heads.head.in_features
        model.heads = nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HuggingFaceVisionEncoder(nn.Module):
    uses_processor = True

    def __init__(self, model_name: str, model_cls):
        super().__init__()
        self.model_name = model_name
        self.model = model_cls.from_pretrained(model_name)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            vision_config = getattr(self.model.config, "vision_config", None)
            hidden_size = getattr(vision_config, "hidden_size", None)
        if hidden_size is None:
            projection_dim = getattr(self.model.config, "projection_dim", None)
            hidden_size = projection_dim
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size for {model_name}")
        self.output_dim = hidden_size

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values, **kwargs)
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        raise ValueError("Unsupported output structure for Hugging Face vision encoder")


class DinoV2Encoder(HuggingFaceVisionEncoder):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__(model_name=model_name, model_cls=Dinov2Model)

def _infer_multimodal_projection_dim(config) -> int | None:
    candidates = [
        getattr(config, "projection_dim", None),
        getattr(config, "projection_size", None),
    ]
    for nested_name in ("text_config", "vision_config"):
        nested = getattr(config, nested_name, None)
        if nested is None:
            continue
        candidates.extend(
            [
                getattr(nested, "projection_dim", None),
                getattr(nested, "projection_size", None),
                getattr(nested, "hidden_size", None),
            ]
        )
    for value in candidates:
        if value is not None:
            return int(value)
    return None


class HuggingFaceMultimodalEncoder(nn.Module):
    uses_processor = True

    def __init__(self, model_name: str, model_cls):
        super().__init__()
        self.model_name = model_name
        self.model = model_cls.from_pretrained(model_name)
        projection_dim = _infer_multimodal_projection_dim(self.model.config)
        if projection_dim is None:
            raise ValueError(f"Could not infer projection dimension for {model_name}")
        self.output_dim = int(projection_dim) * 2

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)
        return torch.cat([image_embeds, text_embeds], dim=-1)


class SigLIP2MultimodalEncoder(HuggingFaceMultimodalEncoder):
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__(model_name=model_name, model_cls=SiglipModel)


class CLIPMultimodalEncoder(HuggingFaceMultimodalEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__(model_name=model_name, model_cls=CLIPModel)


VISION_ENCODERS = {"resnet50", "vit", "dinov2"}
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
        raise ValueError("SigLIP2 is used as a multimodal encoder; set multimodal=True")
    if lowered == "clip":
        if multimodal:
            return CLIPMultimodalEncoder(model_name=model_name or "openai/clip-vit-base-patch32")
        raise ValueError("CLIP is used as a multimodal encoder; set multimodal=True")
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
