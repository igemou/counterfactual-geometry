from __future__ import annotations

import inspect

import torch
from torch import nn
from transformers import AutoModel


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
