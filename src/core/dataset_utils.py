from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def make_weighted_sampler(
    targets: list[int],
    num_classes: Optional[int] = None,
    eps: float = 1e-6,
    seed: Optional[int] = None,
) -> WeightedRandomSampler:
    if not targets:
        raise ValueError("targets must be non-empty")

    if num_classes is None:
        num_classes = int(max(targets)) + 1

    counts = torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes).float()
    weights_per_class = 1.0 / (counts + eps)
    sample_weights = weights_per_class[torch.tensor(targets, dtype=torch.long)]

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(targets),
        replacement=True,
        generator=generator,
    )


def _dl_kwargs(num_workers: int) -> dict[str, object]:
    return {
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }


def _build_image_transform(
    image_size: int = 224,
    to_rgb: bool = True,
    normalize: bool = True,
):
    steps: list[Callable] = [transforms.Resize((image_size, image_size))]
    if to_rgb:
        steps.append(transforms.Lambda(lambda image: image.convert("RGB")))
    else:
        steps.append(transforms.Grayscale(num_output_channels=1))
    steps.append(transforms.ToTensor())
    if normalize:
        if to_rgb:
            steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        else:
            steps.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    return transforms.Compose(steps)


class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        to_rgb: bool = True,
        normalize: bool = True,
        val_split: str = "val",
        test_split: str = "test",
        train_split: str = "train",
        fallback_val_to_test: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.to_rgb = to_rgb
        self.normalize = normalize
        self.val_split = val_split
        self.test_split = test_split
        self.train_split = train_split
        self.fallback_val_to_test = fallback_val_to_test
        self.seed = seed

        self.transform = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _split_path(self, split_name: str) -> Path:
        return self.data_dir / split_name

    def _load_imagefolder(self, split_name: str):
        if self.transform is None:
            self.transform = _build_image_transform(
                image_size=self.image_size,
                to_rgb=self.to_rgb,
                normalize=self.normalize,
            )
        split_path = self._split_path(split_name)
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split '{split_name}' at {split_path}")
        return datasets.ImageFolder(str(split_path), transform=self.transform)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            if self.train_ds is None:
                self.train_ds = self._load_imagefolder(self.train_split)
            if self.val_ds is None:
                val_path = self._split_path(self.val_split)
                if val_path.exists():
                    self.val_ds = self._load_imagefolder(self.val_split)
                elif self.fallback_val_to_test:
                    self.val_ds = self._load_imagefolder(self.test_split)
                else:
                    raise FileNotFoundError(f"Missing validation split at {val_path}")

        if stage == "validate":
            val_path = self._split_path(self.val_split)
            if val_path.exists():
                self.val_ds = self._load_imagefolder(self.val_split)
            elif self.fallback_val_to_test:
                self.val_ds = self._load_imagefolder(self.test_split)
            else:
                raise FileNotFoundError(f"Missing validation split at {val_path}")

        if stage in (None, "test"):
            if self.test_ds is None:
                self.test_ds = self._load_imagefolder(self.test_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            **_dl_kwargs(self.num_workers),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            **_dl_kwargs(self.num_workers),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            **_dl_kwargs(self.num_workers),
        )

    @property
    def num_classes(self) -> int:
        dataset = self.train_ds or self.val_ds or self.test_ds
        if dataset is None:
            raise ValueError("Call setup() first.")
        return len(dataset.classes)

    @property
    def class_names(self) -> list[str]:
        dataset = self.train_ds or self.val_ds or self.test_ds
        if dataset is None:
            raise ValueError("Call setup() first.")
        return list(dataset.classes)
