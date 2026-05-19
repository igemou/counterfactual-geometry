from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Callable
import pytorch_lightning as pl
import torch
from PIL import Image
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset
from .dataset_utils import _build_image_transform, _dl_kwargs
from .datasets_text import TokenizedTextDataset
from .utils import data_root


def _extract_message_content(messages: object, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != role:
            continue
        return str(message.get("content", "")).strip()
    return ""


def _extract_mmimdb_text(messages: object) -> str:
    user_message = _extract_message_content(messages, "user")
    if not user_message:
        return ""
    for pattern in (
        r"plot\s*:\s*(.+)",
        r"plot of the movie\s*:\s*(.+)",
        r"corresponding plot of the movie\s*:\s*(.+)",
    ):
        match = re.search(pattern, user_message, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            return match.group(1).strip()
    return user_message


def _decode_hf_image(image_obj: object):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict):
        image_bytes = image_obj.get("bytes")
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = image_obj.get("path")
        if image_path:
            return Image.open(image_path).convert("RGB")
    raise TypeError(f"Unsupported MMIMDb image payload: {type(image_obj)!r}")


class MMIMDbImageDataset(Dataset):
    def __init__(self, hf_dataset, indices: list[int], labels: list[int], transform: Callable | None = None):
        self.hf_dataset = hf_dataset
        self.indices = list(indices)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.hf_dataset[int(self.indices[index])]
        images = row.get("images") or []
        if not images:
            raise ValueError("MMIMDb example is missing an image.")
        image = _decode_hf_image(images[0])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class MMIMDbTextDataset(Dataset):
    def __init__(self, hf_dataset, indices: list[int], labels: list[int]):
        self.hf_dataset = hf_dataset
        self.indices = list(indices)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.hf_dataset[int(self.indices[index])]
        return {
            "text": _extract_mmimdb_text(row.get("messages")),
            "label": int(self.labels[index]),
        }


class MMIMDbMultimodalDataset(Dataset):
    def __init__(self, hf_dataset, indices: list[int], labels: list[int], transform: Callable | None = None):
        self.hf_dataset = hf_dataset
        self.indices = list(indices)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.hf_dataset[int(self.indices[index])]
        images = row.get("images") or []
        if not images:
            raise ValueError("MMIMDb example is missing an image.")
        image = _decode_hf_image(images[0])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "text": _extract_mmimdb_text(row.get("messages")),
            "label": int(self.labels[index]),
        }


class MMIMDbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        tokenizer: Callable | None = None,
        max_length: int = 256,
        multimodal: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        to_rgb: bool = True,
        normalize: bool = True,
        val_split_ratio: float = 0.1,
        test_split_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else (data_root() / "mmimdb")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multimodal = multimodal
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.to_rgb = to_rgb
        self.normalize = normalize
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.seed = seed

        self.transform = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._hf_dataset = None
        self._class_names: list[str] = []
        self._label_by_dataset_index: dict[int, int] = {}
        self._train_indices: list[int] | None = None
        self._val_indices: list[int] | None = None
        self._test_indices: list[int] | None = None

    def _prepared_manifest_path(self) -> Path:
        return self.data_dir / "prepared_top2_binary.json"

    def _load_prepared_manifest(self) -> bool:
        manifest_path = self._prepared_manifest_path()
        if not manifest_path.exists():
            return False

        payload = json.loads(manifest_path.read_text())
        selected_labels = list(payload.get("selected_labels", []))
        train_records = list(payload.get("train", []))
        val_records = list(payload.get("val", []))
        test_records = list(payload.get("test", []))

        if len(selected_labels) != 2:
            raise ValueError(f"Prepared MMIMDb manifest at {manifest_path} must contain exactly two selected_labels")

        def _decode_records(records: list[dict]) -> tuple[list[int], list[int]]:
            dataset_indices: list[int] = []
            labels: list[int] = []
            for record in records:
                dataset_indices.append(int(record["dataset_index"]))
                labels.append(int(record["label"]))
            return dataset_indices, labels

        train_indices, train_labels = _decode_records(train_records)
        val_indices, val_labels = _decode_records(val_records)
        test_indices, test_labels = _decode_records(test_records)

        self._class_names = selected_labels
        self._train_indices = train_indices
        self._val_indices = val_indices
        self._test_indices = test_indices
        self._label_by_dataset_index = {}
        for dataset_index, label in zip(train_indices, train_labels):
            self._label_by_dataset_index[int(dataset_index)] = int(label)
        for dataset_index, label in zip(val_indices, val_labels):
            self._label_by_dataset_index[int(dataset_index)] = int(label)
        for dataset_index, label in zip(test_indices, test_labels):
            self._label_by_dataset_index[int(dataset_index)] = int(label)
        return True

    def _dataset_files(self) -> list[str]:
        data_dir = self.data_dir / "data"
        files = sorted(str(path) for path in data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Missing MMIMDb parquet shards under {data_dir}")
        return files

    def _ensure_loaded(self) -> None:
        if self._hf_dataset is not None:
            return
        dataset = hf_load_dataset("parquet", data_files={"train": self._dataset_files()}, split="train")
        self._hf_dataset = dataset

        if not self._load_prepared_manifest():
            raise FileNotFoundError(
                f"Missing prepared MMIMDb manifest at {self._prepared_manifest_path()}. "
                "Run src.core.prepare_mmimdb_binary first."
            )

    def _make_base_dataset(self, indices: list[int]) -> Dataset:
        self._ensure_loaded()
        if self.multimodal:
            if self.transform is None:
                self.transform = _build_image_transform(
                    image_size=self.image_size,
                    to_rgb=self.to_rgb,
                    normalize=self.normalize,
                )
            return MMIMDbMultimodalDataset(
                self._hf_dataset,
                indices=indices,
                labels=[self._label_by_dataset_index[int(index)] for index in indices],
                transform=self.transform,
            )

        if self.tokenizer is not None:
            return MMIMDbTextDataset(
                self._hf_dataset,
                indices=indices,
                labels=[self._label_by_dataset_index[int(index)] for index in indices],
            )

        if self.transform is None:
            self.transform = _build_image_transform(
                image_size=self.image_size,
                to_rgb=self.to_rgb,
                normalize=self.normalize,
            )
        return MMIMDbImageDataset(
            self._hf_dataset,
            indices=indices,
            labels=[self._label_by_dataset_index[int(index)] for index in indices],
            transform=self.transform,
        )

    def _maybe_tokenize(self, dataset: Dataset) -> Dataset:
        if self.tokenizer is None:
            return dataset
        return TokenizedTextDataset(dataset, tokenizer=self.tokenizer, max_length=self.max_length)

    def setup(self, stage: str | None = None):
        self._ensure_loaded()

        if stage in (None, "fit"):
            if self.train_ds is None:
                self.train_ds = self._maybe_tokenize(self._make_base_dataset(self._train_indices or []))
            if self.val_ds is None:
                self.val_ds = self._maybe_tokenize(self._make_base_dataset(self._val_indices or []))

        if stage == "validate":
            self.val_ds = self._maybe_tokenize(self._make_base_dataset(self._val_indices or []))

        if stage in (None, "test"):
            if self.test_ds is None:
                self.test_ds = self._maybe_tokenize(self._make_base_dataset(self._test_indices or []))

    def _labels_for_dataset(self, dataset: Dataset) -> list[int]:
        if isinstance(dataset, TokenizedTextDataset):
            dataset = dataset.dataset
        if isinstance(dataset, MMIMDbImageDataset):
            return list(dataset.labels)
        if isinstance(dataset, MMIMDbTextDataset):
            return list(dataset.labels)
        if isinstance(dataset, MMIMDbMultimodalDataset):
            return list(dataset.labels)
        raise TypeError(f"Unsupported dataset type for MMIMDb labels: {type(dataset)!r}")

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
        return len(self._class_names)

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)
