from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_utils import _dl_kwargs
from .utils import data_root


class IMDBTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        expected = {"text", "label"}
        missing = expected.difference(frame.columns)
        if missing:
            raise ValueError(f"IMDB dataframe missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        return {
            "text": str(row["text"]),
            "label": int(row["label"]),
        }


class TokenizedTextDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Callable, max_length: int = 256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


class IMDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        tokenizer: Callable | None = None,
        max_length: int = 256,
        multimodal: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else (data_root() / "imdb")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multimodal = multimodal
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio
        self.seed = seed

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._class_names = ["negative", "positive"]

    def _read_csv(self, name: str) -> pd.DataFrame:
        csv_path = self.data_dir / name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing IMDB split at {csv_path}")
        return pd.read_csv(csv_path)

    def _maybe_tokenize(self, dataset: Dataset) -> Dataset:
        if self.tokenizer is None:
            return dataset
        return TokenizedTextDataset(dataset, tokenizer=self.tokenizer, max_length=self.max_length)

    def _split_train_val(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not 0.0 < self.val_split_ratio < 1.0:
            return frame, frame.iloc[0:0].copy()

        generator = torch.Generator().manual_seed(self.seed)
        labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.long)
        train_indices = []
        val_indices = []
        for label in labels.unique(sorted=True).tolist():
            class_idx = torch.where(labels == label)[0]
            permuted = class_idx[torch.randperm(class_idx.numel(), generator=generator)]
            n_val = max(1, int(round(class_idx.numel() * self.val_split_ratio)))
            n_val = min(n_val, max(class_idx.numel() - 1, 0))
            val_indices.append(permuted[:n_val])
            train_indices.append(permuted[n_val:])

        train_idx = torch.cat(train_indices).tolist()
        val_idx = torch.cat(val_indices).tolist()
        return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[val_idx].reset_index(drop=True)

    def setup(self, stage: str | None = None):
        train_frame = self._read_csv("imdb_train.csv")
        test_frame = self._read_csv("imdb_test.csv")
        train_frame, val_frame = self._split_train_val(train_frame)
        if len(val_frame) == 0:
            val_frame = test_frame.copy()

        if stage in (None, "fit"):
            if self.train_ds is None:
                self.train_ds = self._maybe_tokenize(IMDBTextDataset(train_frame))
            if self.val_ds is None:
                self.val_ds = self._maybe_tokenize(IMDBTextDataset(val_frame))

        if stage == "validate":
            self.val_ds = self._maybe_tokenize(IMDBTextDataset(val_frame))

        if stage in (None, "test"):
            if self.test_ds is None:
                self.test_ds = self._maybe_tokenize(IMDBTextDataset(test_frame))

    def _labels_for_dataset(self, dataset: Dataset) -> list[int]:
        if isinstance(dataset, TokenizedTextDataset):
            dataset = dataset.dataset
        return [int(dataset[index]["label"]) for index in range(len(dataset))]

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
        return 2

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)
