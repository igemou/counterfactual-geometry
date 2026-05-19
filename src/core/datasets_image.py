from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .dataset_utils import ImageFolderDataModule, _build_image_transform, _dl_kwargs
from .utils import data_root


class MNISTDataModule(ImageFolderDataModule):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        to_rgb: bool = True,
        normalize: bool = True,
        val_split_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir or (data_root() / "mnist"),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            to_rgb=to_rgb,
            normalize=normalize,
            val_split="val",
            test_split="test",
            train_split="train",
            fallback_val_to_test=False,
            seed=seed,
        )
        self.val_split_ratio = val_split_ratio

    def _stratified_train_val_split(self, dataset) -> tuple[Subset, Subset]:
        if not 0.0 < self.val_split_ratio < 1.0:
            raise ValueError("val_split_ratio must be in (0, 1) for MNISTDataModule")

        targets = torch.tensor(dataset.targets, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed)
        train_indices = []
        val_indices = []
        for label in targets.unique(sorted=True).tolist():
            class_idx = torch.where(targets == label)[0]
            permuted = class_idx[torch.randperm(class_idx.numel(), generator=generator)]
            n_val = max(1, int(round(class_idx.numel() * self.val_split_ratio)))
            n_val = min(n_val, max(class_idx.numel() - 1, 0))
            val_indices.append(permuted[:n_val])
            train_indices.append(permuted[n_val:])

        train_idx = torch.cat(train_indices).tolist()
        val_idx = torch.cat(val_indices).tolist()
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def _labels_for_subset(self, subset: Subset) -> list[int]:
        base_targets = torch.tensor(subset.dataset.targets, dtype=torch.long)
        indices = torch.tensor(subset.indices, dtype=torch.long)
        return base_targets[indices].tolist()

    def setup(self, stage: str | None = None):
        if self.transform is None:
            self.transform = _build_image_transform(
                image_size=self.image_size,
                to_rgb=self.to_rgb,
                normalize=self.normalize,
            )

        if stage in (None, "fit"):
            if self.train_ds is None or self.val_ds is None:
                base_train = self._load_imagefolder(self.train_split)
                self.train_ds, self.val_ds = self._stratified_train_val_split(base_train)

        if stage == "validate":
            if self.val_ds is None:
                base_train = self._load_imagefolder(self.train_split)
                _, self.val_ds = self._stratified_train_val_split(base_train)

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

    @property
    def num_classes(self) -> int:
        dataset = self.train_ds or self.val_ds or self.test_ds
        if dataset is None:
            raise ValueError("Call setup() first.")
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        return len(dataset.classes)

    @property
    def class_names(self) -> list[str]:
        dataset = self.train_ds or self.val_ds or self.test_ds
        if dataset is None:
            raise ValueError("Call setup() first.")
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        return list(dataset.classes)


class ChestXrayDataModule(ImageFolderDataModule):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        to_rgb: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir or (data_root() / "chestxray"),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            to_rgb=to_rgb,
            normalize=normalize,
            val_split="val",
            test_split="test",
            train_split="train",
            fallback_val_to_test=False,
            seed=seed,
        )


class ShapesDataModule(ImageFolderDataModule):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        to_rgb: bool = True,
        normalize: bool = True,
        val_split_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir or (data_root() / "shapes"),
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            to_rgb=to_rgb,
            normalize=normalize,
            val_split="val",
            test_split="test",
            train_split="train",
            fallback_val_to_test=False,
            seed=seed,
        )
        self.val_split_ratio = val_split_ratio

    def _stratified_train_val_split(self, dataset) -> tuple[Subset, Subset]:
        if not 0.0 < self.val_split_ratio < 1.0:
            raise ValueError("val_split_ratio must be in (0, 1) for ShapesDataModule")

        targets = torch.tensor(dataset.targets, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed)
        train_indices = []
        val_indices = []
        for label in targets.unique(sorted=True).tolist():
            class_idx = torch.where(targets == label)[0]
            permuted = class_idx[torch.randperm(class_idx.numel(), generator=generator)]
            n_val = max(1, int(round(class_idx.numel() * self.val_split_ratio)))
            n_val = min(n_val, max(class_idx.numel() - 1, 0))
            val_indices.append(permuted[:n_val])
            train_indices.append(permuted[n_val:])

        train_idx = torch.cat(train_indices).tolist()
        val_idx = torch.cat(val_indices).tolist()
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def _labels_for_subset(self, subset: Subset) -> list[int]:
        base_targets = torch.tensor(subset.dataset.targets, dtype=torch.long)
        indices = torch.tensor(subset.indices, dtype=torch.long)
        return base_targets[indices].tolist()

    def setup(self, stage: str | None = None):
        if self.transform is None:
            self.transform = _build_image_transform(
                image_size=self.image_size,
                to_rgb=self.to_rgb,
                normalize=self.normalize,
            )

        if stage in (None, "fit"):
            if self.train_ds is None or self.val_ds is None:
                base_train = self._load_imagefolder(self.train_split)
                self.train_ds, self.val_ds = self._stratified_train_val_split(base_train)

        if stage == "validate":
            if self.val_ds is None:
                base_train = self._load_imagefolder(self.train_split)
                _, self.val_ds = self._stratified_train_val_split(base_train)

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

    @property
    def num_classes(self) -> int:
        dataset = self.train_ds or self.val_ds or self.test_ds
        if dataset is None:
            raise ValueError("Call setup() first.")
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        return len(dataset.classes)

    @property
    def class_names(self) -> list[str]:
        dataset = self.train_ds or self.val_ds or self.test_ds
        if dataset is None:
            raise ValueError("Call setup() first.")
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        return list(dataset.classes)
