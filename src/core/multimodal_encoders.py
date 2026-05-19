from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel, SiglipModel


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
        if not hasattr(outputs, "image_embeds") or outputs.image_embeds is None:
            raise ValueError("Multimodal encoder did not return image_embeds")
        if not hasattr(outputs, "text_embeds") or outputs.text_embeds is None:
            raise ValueError("Multimodal encoder did not return text_embeds")
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)
        return torch.cat([image_embeds, text_embeds], dim=-1)


class SigLIP2MultimodalEncoder(HuggingFaceMultimodalEncoder):
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__(model_name=model_name, model_cls=SiglipModel)


class CLIPMultimodalEncoder(HuggingFaceMultimodalEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__(model_name=model_name, model_cls=CLIPModel)
