from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..core.datasets import build_datamodule
from ..core.encoders import build_encoder, build_processor, freeze_encoder, unpack_batch
from ..core.utils import embedding_cache_path, to_device

PROCESSOR_VISION_ENCODERS = {"dinov2", "siglip2", "clip"}
TORCHVISION_VISION_ENCODERS = {"resnet50", "vit"}

def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def copy_split_to_device(
    split_to_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    split: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, labels = split_to_embeddings[split]
    return embeddings.to(device), labels.to(device)


VISION_BACKBONES = ("resnet50", "vit", "dinov2")
TEXT_BACKBONES = ("distilbert", "bert", "roberta")
MULTIMODAL_BACKBONES = ("clip", "siglip2")
REPRESENTATIONS = ("image", "text", "multimodal", "fused")
SUITE_MULTIMODAL_ENCODERS = MULTIMODAL_BACKBONES


def suite_output_path(
    output_dir: Path,
    representation: str,
    image_encoder: str | None,
    text_encoder: str | None,
    multimodal_encoder: str | None = None,
) -> Path:
    parts = ["multimodal", representation]
    if image_encoder:
        parts.append(image_encoder)
    if text_encoder:
        parts.append(text_encoder)
    if multimodal_encoder:
        parts.append(multimodal_encoder)
    return output_dir / ("_".join(parts) + "_encoder_comparison.json")


class EncodedLinearHead(nn.Module):
    def __init__(self, probe) -> None:
        super().__init__()
        if getattr(probe, "head", None) is not None:
            reference = probe.head
        else:
            reference = probe.linear
        if reference is None:
            raise ValueError("Probe does not expose a final linear layer")
        self.linear = nn.Linear(reference.in_features, reference.out_features)
        self.linear.load_state_dict(reference.state_dict())

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


def build_multimodal_datamodule(
    representation: str,
    batch_size: int,
    num_workers: int,
    image_encoder_name: str | None = None,
    text_processor=None,
):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if representation == "image":
        if image_encoder_name is not None and image_encoder_name.lower() in PROCESSOR_VISION_ENCODERS:
            kwargs["normalize"] = False
        return build_datamodule("mmimdb", **kwargs)
    if representation == "text":
        if text_processor is None:
            raise ValueError("A tokenizer is required for multimodal text experiments.")
        kwargs["tokenizer"] = text_processor
        return build_datamodule("mmimdb", **kwargs)
    if representation in {"multimodal", "fused"}:
        kwargs["multimodal"] = True
        kwargs["normalize"] = False
        return build_datamodule("mmimdb", **kwargs)
    raise ValueError(f"Unsupported multimodal representation: {representation}")


def _prepare_image_inputs(features, processor, device: torch.device):
    if processor is None:
        return features.to(device)
    processed = processor(images=[image.cpu() for image in features], return_tensors="pt")
    return to_device(dict(processed), device)


def _prepare_text_inputs(features, tokenizer, device: torch.device):
    if tokenizer is None:
        return to_device(features, device)
    processed = tokenizer(
        list(features["text"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return to_device(dict(processed), device)


def extract_fused_embeddings(
    dataloader,
    image_encoder,
    text_encoder,
    device: torch.device,
    image_processor=None,
    text_processor=None,
    cache_path: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path is not None:
        resolved_cache_path = Path(cache_path)
        if resolved_cache_path.exists():
            payload = torch.load(resolved_cache_path, map_location="cpu")
            embeddings = payload.get("embeddings")
            labels = payload.get("labels")
            if not isinstance(embeddings, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise ValueError(f"Invalid fused embedding cache at {resolved_cache_path}")
            return embeddings, labels

    image_encoder.eval()
    text_encoder.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features, labels = unpack_batch(batch)
            labels = to_device(labels, device)

            image_inputs = _prepare_image_inputs(features["image"], image_processor, device=device)
            if isinstance(image_inputs, dict):
                image_embeddings = image_encoder(**image_inputs)
            else:
                image_embeddings = image_encoder(image_inputs)

            text_inputs = _prepare_text_inputs({"text": features["text"]}, text_processor, device=device)
            text_embeddings = text_encoder(**text_inputs)

            fused_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
            all_embeddings.append(fused_embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if cache_path is not None:
        resolved_cache_path = Path(cache_path)
        resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings, "labels": labels}, resolved_cache_path)
    return embeddings, labels


def project_if_needed(classifier_head, embeddings: torch.Tensor) -> torch.Tensor:
    if getattr(classifier_head, "projection", None) is None:
        return embeddings
    with torch.no_grad():
        return classifier_head.encode(embeddings.to(next(classifier_head.parameters()).device)).cpu()


def load_multimodal_split_embeddings(
    representation: str,
    split_loaders: dict[str, object],
    resolved_device: torch.device,
    image_encoder_name: str | None,
    text_encoder_name: str | None,
    multimodal_encoder_name: str | None,
    image_encoder_model_name: str | None,
    text_encoder_model_name: str | None,
    multimodal_encoder_model_name: str | None,
    text_processor=None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    split_to_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    if representation == "image":
        image_encoder = freeze_encoder(build_encoder(image_encoder_name, model_name=image_encoder_model_name)).to(resolved_device)
        image_processor = None
        if image_encoder_name is not None and image_encoder_name.lower() in PROCESSOR_VISION_ENCODERS:
            image_processor = build_processor(image_encoder_name, model_name=image_encoder_model_name)

        from .unimodal_encoder_comparison import extract_embeddings

        for split, loader in split_loaders.items():
            cache_path = embedding_cache_path("mmimdb", image_encoder_name, image_encoder_model_name, split)
            split_to_embeddings[split] = extract_embeddings(
                loader,
                encoder=image_encoder,
                device=resolved_device,
                processor=image_processor,
                cache_path=cache_path,
            )
        return split_to_embeddings

    if representation == "text":
        text_encoder = freeze_encoder(build_encoder(text_encoder_name, model_name=text_encoder_model_name)).to(resolved_device)
        from .unimodal_encoder_comparison import extract_embeddings

        for split, loader in split_loaders.items():
            cache_path = embedding_cache_path("mmimdb", text_encoder_name, text_encoder_model_name, split)
            split_to_embeddings[split] = extract_embeddings(
                loader,
                encoder=text_encoder,
                device=resolved_device,
                processor=None,
                cache_path=cache_path,
            )
        return split_to_embeddings

    if representation == "fused":
        image_encoder = freeze_encoder(build_encoder(image_encoder_name, model_name=image_encoder_model_name)).to(resolved_device)
        text_encoder = freeze_encoder(build_encoder(text_encoder_name, model_name=text_encoder_model_name)).to(resolved_device)
        image_processor = None
        if image_encoder_name is not None and image_encoder_name.lower() in PROCESSOR_VISION_ENCODERS:
            image_processor = build_processor(image_encoder_name, model_name=image_encoder_model_name)

        for split, loader in split_loaders.items():
            model_key = f"{image_encoder_name}-{text_encoder_name}"
            cache_path = embedding_cache_path("mmimdb_fused", model_key, None, split)
            split_to_embeddings[split] = extract_fused_embeddings(
                loader,
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                device=resolved_device,
                image_processor=image_processor,
                text_processor=text_processor,
                cache_path=cache_path,
            )
        return split_to_embeddings

    multimodal_encoder = freeze_encoder(
        build_encoder(
            multimodal_encoder_name,
            model_name=multimodal_encoder_model_name,
            multimodal=True,
        )
    ).to(resolved_device)
    multimodal_processor = build_processor(multimodal_encoder_name, model_name=multimodal_encoder_model_name)
    from .unimodal_encoder_comparison import extract_embeddings

    for split, loader in split_loaders.items():
        cache_path = embedding_cache_path("mmimdb_multimodal", multimodal_encoder_name, multimodal_encoder_model_name, split)
        split_to_embeddings[split] = extract_embeddings(
            loader,
            encoder=multimodal_encoder,
            device=resolved_device,
            processor=multimodal_processor,
            cache_path=cache_path,
        )
    return split_to_embeddings
