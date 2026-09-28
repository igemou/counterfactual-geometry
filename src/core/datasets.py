from __future__ import annotations

from .datasets_image import ChestXrayDataModule, MNISTDataModule, ShapesDataModule
from .datasets_multimodal import MMIMDbDataModule
from .datasets_text import IMDBDataModule


def build_datamodule(name: str, **kwargs):
    lowered = name.lower()
    if lowered == "mnist":
        return MNISTDataModule(**kwargs)
    if lowered in {"chestxray", "chest_xray"}:
        return ChestXrayDataModule(**kwargs)
    if lowered == "shapes":
        return ShapesDataModule(**kwargs)
    if lowered == "imdb":
        return IMDBDataModule(**kwargs)
    if lowered == "mmimdb":
        return MMIMDbDataModule(**kwargs)
    raise ValueError(f"Unsupported dataset: {name}")
