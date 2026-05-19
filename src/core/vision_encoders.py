from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from transformers import CLIPVisionModel, Dinov2Model, SiglipVisionModel


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


class SigLIP2VisionEncoder(HuggingFaceVisionEncoder):
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__(model_name=model_name, model_cls=SiglipVisionModel)


class CLIPVisionEncoder(HuggingFaceVisionEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__(model_name=model_name, model_cls=CLIPVisionModel)
