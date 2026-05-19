from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..core.classifier import build_classifier, train_linear_probe
from ..core.datasets import build_datamodule
from ..core.encoders import build_encoder, build_processor, freeze_encoder, unpack_batch
from ..counterfactuals.evaluation import evaluate_embeddings
from ..core.geometry import dataset_density_scale
from ..core.utils import embedding_cache_path, set_seed, to_device
from .common import (
    PROCESSOR_VISION_ENCODERS,
    TORCHVISION_VISION_ENCODERS,
    copy_split_to_device,
    resolve_device,
)


MULTIMODAL_ENCODERS = {"clip", "siglip2"}
IMAGE_ENCODERS = PROCESSOR_VISION_ENCODERS | TORCHVISION_VISION_ENCODERS
TEXT_ENCODERS = {"distilbert", "bert", "roberta"}
IMAGE_DATASETS = {"mnist", "chestxray", "shapes"}
TEXT_DATASETS = {"imdb"}
MULTIMODAL_DATASETS = {"mmimdb"}


def _validate_dataset_encoder_pair(dataset_name: str, encoder_name: str) -> None:
    dataset_name = dataset_name.lower()
    encoder_name = encoder_name.lower()

    if dataset_name in MULTIMODAL_DATASETS and encoder_name in MULTIMODAL_ENCODERS:
        return
    if dataset_name in IMAGE_DATASETS and encoder_name in IMAGE_ENCODERS:
        return
    if dataset_name in TEXT_DATASETS and encoder_name in TEXT_ENCODERS:
        return

    if dataset_name in IMAGE_DATASETS and encoder_name in TEXT_ENCODERS and dataset_name not in TEXT_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' is image-based and requires an image encoder. "
            f"Supported image encoders: {sorted(IMAGE_ENCODERS)}"
        )

    if dataset_name in TEXT_DATASETS and encoder_name in IMAGE_ENCODERS and dataset_name not in IMAGE_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' is text-based and requires a text encoder. "
            f"Supported text encoders: {sorted(TEXT_ENCODERS)}"
        )

    raise ValueError(f"Unsupported dataset/encoder combination: dataset={dataset_name}, encoder={encoder_name}")


def _split_accuracy(
    classifier_head,
    split_to_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    split: str,
    device: torch.device,
) -> float:
    embeddings, labels = copy_split_to_device(split_to_embeddings, split, device)
    return _classification_accuracy(classifier_head, embeddings, labels)


def _classification_accuracy(classifier_head, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        logits = classifier_head(embeddings)
        predictions = logits.argmax(dim=1)
        return float((predictions == labels).float().mean().item())


def _use_multimodal_inputs(dataset_name: str, encoder_name: str) -> bool:
    return dataset_name.lower() in MULTIMODAL_DATASETS and encoder_name.lower() in MULTIMODAL_ENCODERS


def _prepare_encoder_inputs(features, processor, device: torch.device):
    if processor is None:
        if isinstance(features, dict):
            return to_device(features, device)
        return features.to(device)

    if isinstance(features, torch.Tensor):
        processed = processor(images=[image.cpu() for image in features], return_tensors="pt")
        return to_device(dict(processed), device)

    if isinstance(features, dict):
        if "image" in features and "text" in features:
            processed = processor(
                images=[image.cpu() for image in features["image"]],
                text=list(features["text"]),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            return to_device(dict(processed), device)
        return to_device(features, device)

    raise TypeError(f"Unsupported feature type for encoder inputs: {type(features)!r}")


def extract_embeddings(
    dataloader,
    encoder,
    device: torch.device,
    processor=None,
    cache_path: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path is not None:
        resolved_cache_path = Path(cache_path)
        if resolved_cache_path.exists():
            print(f"Loading cached embeddings from {resolved_cache_path}")
            payload = torch.load(resolved_cache_path, map_location="cpu")
            embeddings = payload.get("embeddings")
            labels = payload.get("labels")
            if not isinstance(embeddings, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise ValueError(f"Invalid embedding cache at {resolved_cache_path}")
            return embeddings, labels

    encoder.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features, labels = unpack_batch(batch)
            labels = to_device(labels, device)
            prepared_features = _prepare_encoder_inputs(features, processor=processor, device=device)
            if isinstance(prepared_features, dict):
                embeddings = encoder(**prepared_features)
            else:
                embeddings = encoder(prepared_features)
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if cache_path is not None:
        resolved_cache_path = Path(cache_path)
        resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings, "labels": labels}, resolved_cache_path)
        print(f"Saved embeddings cache to {resolved_cache_path}")
    return embeddings, labels


def _build_datamodule_for_experiment(
    dataset_name: str,
    encoder_name: str,
    batch_size: int,
    num_workers: int,
    text_processor=None,
):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    lowered_dataset = dataset_name.lower()
    lowered_encoder = encoder_name.lower()
    if lowered_dataset == "imdb":
        if text_processor is None:
            raise ValueError("A tokenizer is required for IMDB experiments.")
        kwargs["tokenizer"] = text_processor
    elif lowered_dataset in MULTIMODAL_DATASETS and lowered_encoder in MULTIMODAL_ENCODERS:
        kwargs["multimodal"] = True
        kwargs["normalize"] = False
    elif lowered_dataset in MULTIMODAL_DATASETS and lowered_encoder in TEXT_ENCODERS:
        if text_processor is None:
            raise ValueError("A tokenizer is required for text-only multimodal experiments.")
        kwargs["tokenizer"] = text_processor
    elif lowered_encoder in PROCESSOR_VISION_ENCODERS:
        kwargs["normalize"] = False

    return build_datamodule(dataset_name, **kwargs)


def _checkpoint_stem(dataset_name: str, encoder_name: str, encoder_model_name: str | None = None) -> str:
    stem = f"{dataset_name}_{encoder_name}"
    if encoder_model_name:
        safe_model_name = encoder_model_name.replace('/', '_')
        stem = f"{stem}_{safe_model_name}"
    return stem


def save_probe_checkpoint(
    classifier_head,
    checkpoint_dir: str | Path,
    dataset_name: str,
    encoder_name: str,
    encoder_model_name: str | None,
    seed: int,
    input_dim: int,
    num_classes: int,
    metadata: dict[str, object],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{_checkpoint_stem(dataset_name, encoder_name, encoder_model_name)}_probe.pt"
    payload = {
        "classifier_state_dict": classifier_head.state_dict(),
        "dataset": dataset_name,
        "encoder": encoder_name,
        "encoder_model_name": encoder_model_name or "",
        "seed": seed,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def run_unimodal_encoder_comparison(
    dataset_name: str,
    encoder_name: str,
    encoder_model_name: str | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
    seed: int = 42,
    probe_epochs: int = 100,
    probe_lr: float = 1e-3,
    probe_weight_decay: float = 1e-4,
    eval_split: str = "test",
    reference_split: str = "val",
    max_examples: int | None = None,
    counterfactual_mode: str = "targeted",
    k: int = 20,
    step_size: float = 1e-2,
    max_steps: int = 500,
    trust_radius: float = 1.0,
    save_probe_dir: str | Path | None = None,
) -> dict[str, float | str | int]:
    _validate_dataset_encoder_pair(dataset_name, encoder_name)
    set_seed(seed)
    resolved_device = resolve_device(device)

    text_processor = None
    if encoder_name.lower() in TEXT_ENCODERS:
        text_processor = build_processor(encoder_name, model_name=encoder_model_name)

    datamodule = _build_datamodule_for_experiment(
        dataset_name=dataset_name,
        encoder_name=encoder_name,
        batch_size=batch_size,
        num_workers=num_workers,
        text_processor=text_processor,
    )
    datamodule.setup(None)

    encoder = build_encoder(
        encoder_name,
        model_name=encoder_model_name,
        multimodal=_use_multimodal_inputs(dataset_name, encoder_name),
    )
    encoder = freeze_encoder(encoder).to(resolved_device)

    image_processor = None
    if getattr(encoder, "uses_processor", False):
        image_processor = build_processor(encoder_name, model_name=encoder_model_name)

    train_cache_path = embedding_cache_path(dataset_name, encoder_name, encoder_model_name, "train")
    train_embeddings, train_labels = extract_embeddings(
        datamodule.train_dataloader(),
        encoder=encoder,
        device=resolved_device,
        processor=image_processor,
        cache_path=train_cache_path,
    )
    val_cache_path = embedding_cache_path(dataset_name, encoder_name, encoder_model_name, "val")
    val_embeddings, val_labels = extract_embeddings(
        datamodule.val_dataloader(),
        encoder=encoder,
        device=resolved_device,
        processor=image_processor,
        cache_path=val_cache_path,
    )
    test_cache_path = embedding_cache_path(dataset_name, encoder_name, encoder_model_name, "test")
    test_embeddings, test_labels = extract_embeddings(
        datamodule.test_dataloader(),
        encoder=encoder,
        device=resolved_device,
        processor=image_processor,
        cache_path=test_cache_path,
    )

    classifier_head = build_classifier(input_dim=train_embeddings.size(1), num_classes=datamodule.num_classes)
    classifier_head = classifier_head.to(resolved_device)
    classifier_head, probe_training_stats = train_linear_probe(
        classifier=classifier_head,
        embeddings=train_embeddings.to(resolved_device),
        labels=train_labels.to(resolved_device),
        val_embeddings=val_embeddings.to(resolved_device),
        val_labels=val_labels.to(resolved_device),
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
    )
    classifier_head = classifier_head.eval()

    split_to_embeddings = {
        "train": (train_embeddings, train_labels),
        "val": (val_embeddings, val_labels),
        "test": (test_embeddings, test_labels),
    }
    if eval_split not in split_to_embeddings:
        raise ValueError(f"Unsupported eval split: {eval_split}")
    if reference_split not in split_to_embeddings:
        raise ValueError(f"Unsupported reference split: {reference_split}")

    eval_embeddings, eval_labels = copy_split_to_device(split_to_embeddings, eval_split, resolved_device)
    reference_embeddings, reference_labels = copy_split_to_device(split_to_embeddings, reference_split, resolved_device)
    same_reference_pool = eval_split == reference_split

    results, summary = evaluate_embeddings(
        embeddings=eval_embeddings,
        classifier_head=classifier_head,
        labels=eval_labels,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        max_examples=max_examples,
        same_reference_pool=same_reference_pool,
        counterfactual_mode=counterfactual_mode,
        k=k,
        step_size=step_size,
        max_steps=max_steps,
        trust_radius=trust_radius,
    )

    support_scale = dataset_density_scale(reference_embeddings, reference_labels, k=k)
    output = {
        **summary,
        "dataset": dataset_name,
        "encoder": encoder_name,
        "encoder_model_name": encoder_model_name or "",
        "seed": seed,
        "eval_split": eval_split,
        "reference_split": reference_split,
        "counterfactual_mode": counterfactual_mode,
        "target_strategy": "second_best",
        "k": k,
        "step_size": step_size,
        "max_steps": max_steps,
        "trust_radius": trust_radius,
        "num_train": int(train_embeddings.size(0)),
        "num_val": int(val_embeddings.size(0)),
        "num_test": int(test_embeddings.size(0)),
        "train_accuracy": _split_accuracy(classifier_head, split_to_embeddings, "train", resolved_device),
        "val_accuracy": _split_accuracy(classifier_head, split_to_embeddings, "val", resolved_device),
        "test_accuracy": _split_accuracy(classifier_head, split_to_embeddings, "test", resolved_device),
        "reference_support_scale": support_scale,
        "probe_lr": probe_lr,
        "probe_weight_decay": probe_weight_decay,
        "probe_epochs": probe_epochs,
        "probe_selection_metric": str(probe_training_stats["selection_metric"]),
        "probe_best_epoch": int(probe_training_stats["best_epoch"]),
        "probe_best_score": float(probe_training_stats["best_score"]),
        "num_evaluated": len(results),
        "raw_results": results,
    }
    if save_probe_dir is not None:
        checkpoint_path = save_probe_checkpoint(
            classifier_head=classifier_head,
            checkpoint_dir=save_probe_dir,
            dataset_name=dataset_name,
            encoder_name=encoder_name,
            encoder_model_name=encoder_model_name,
            seed=seed,
            input_dim=int(train_embeddings.size(1)),
            num_classes=datamodule.num_classes,
            metadata={
                "batch_size": batch_size,
                "num_workers": num_workers,
                "seed": seed,
                "probe_epochs": probe_epochs,
                "probe_lr": probe_lr,
                "probe_weight_decay": probe_weight_decay,
                "eval_split": eval_split,
                "reference_split": reference_split,
                "counterfactual_mode": counterfactual_mode,
                "target_strategy": "second_best",
                "k": k,
                "step_size": step_size,
                "max_steps": max_steps,
                "trust_radius": trust_radius,
                "probe_selection_metric": output["probe_selection_metric"],
                "probe_best_epoch": output["probe_best_epoch"],
                "probe_best_score": output["probe_best_score"],
            },
        )
        output["probe_checkpoint"] = str(checkpoint_path)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run representation geometry and counterfactual probe experiments.")
    parser.add_argument("--dataset", required=True, choices=sorted(IMAGE_DATASETS | TEXT_DATASETS))
    parser.add_argument(
        "--encoder",
        required=True,
        choices=sorted(IMAGE_ENCODERS | TEXT_ENCODERS),
    )
    parser.add_argument("--encoder-model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--counterfactual-mode", choices=["untargeted", "targeted"], default="targeted")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--trust-radius", type=float, default=1.0)
    parser.add_argument("--save-probe-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = run_unimodal_encoder_comparison(
        dataset_name=args.dataset,
        encoder_name=args.encoder,
        encoder_model_name=args.encoder_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_weight_decay=args.probe_weight_decay,
        eval_split=args.eval_split,
        reference_split=args.reference_split,
        max_examples=args.max_examples,
        counterfactual_mode=args.counterfactual_mode,
        k=args.k,
        step_size=args.step_size,
        max_steps=args.max_steps,
        trust_radius=args.trust_radius,
        save_probe_dir=args.save_probe_dir,
    )

    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
