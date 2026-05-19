from __future__ import annotations

from .dataset_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageFolderDataModule,
    _build_image_transform,
    _dl_kwargs,
    make_weighted_sampler,
)
from .datasets_image import ChestXrayDataModule, MNISTDataModule, ShapesDataModule
from .datasets_multimodal import (
    MMIMDbDataModule,
    MMIMDbImageDataset,
    MMIMDbMultimodalDataset,
    MMIMDbTextDataset,
)
from .datasets_text import IMDBDataModule, IMDBTextDataset, TokenizedTextDataset


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
