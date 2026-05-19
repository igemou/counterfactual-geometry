from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from ..core.classifier import build_classifier, train_linear_probe
from ..core.encoders import build_processor
from ..counterfactuals.evaluation import evaluate_embeddings
from .unimodal_encoder_comparison import (
    _classification_accuracy,
    save_probe_checkpoint,
)
from ..core.geometry import dataset_density_scale
from ..core.utils import load_probe, set_seed
from .common import (
    EncodedLinearHead,
    MULTIMODAL_BACKBONES,
    REPRESENTATIONS,
    SUITE_MULTIMODAL_ENCODERS,
    TEXT_BACKBONES,
    VISION_BACKBONES,
    build_multimodal_datamodule,
    load_multimodal_split_embeddings,
    project_if_needed,
    resolve_device,
    suite_output_path,
)

# Backward-compatible re-exports for callers that previously imported helper
# functions from this runner module directly.
_build_multimodal_datamodule = build_multimodal_datamodule
_load_multimodal_split_embeddings = load_multimodal_split_embeddings


def run_multimodal_suite(
    output_dir: Path,
    device: str | None = None,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    probe_epochs: int = 100,
    probe_lr: float = 1e-3,
    probe_weight_decay: float = 1e-4,
    fusion_projection_dim: int | None = None,
    save_probe_dir: Path | None = None,
    max_examples: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []

    for image_encoder in VISION_BACKBONES:
        output = run_multimodal_encoder_comparison(
            representation="image",
            image_encoder_name=image_encoder,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
            probe_epochs=probe_epochs,
            probe_lr=probe_lr,
            probe_weight_decay=probe_weight_decay,
            max_examples=max_examples,
            save_probe_dir=save_probe_dir,
        )
        output_path = suite_output_path(output_dir, "image", image_encoder, None)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        records.append({"representation": "image", "image_encoder": image_encoder, "output": str(output_path)})

    for text_encoder in TEXT_BACKBONES:
        output = run_multimodal_encoder_comparison(
            representation="text",
            text_encoder_name=text_encoder,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
            probe_epochs=probe_epochs,
            probe_lr=probe_lr,
            probe_weight_decay=probe_weight_decay,
            max_examples=max_examples,
            save_probe_dir=save_probe_dir,
        )
        output_path = suite_output_path(output_dir, "text", None, text_encoder)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        records.append({"representation": "text", "text_encoder": text_encoder, "output": str(output_path)})

    for multimodal_encoder in SUITE_MULTIMODAL_ENCODERS:
        output = run_multimodal_encoder_comparison(
            representation="multimodal",
            multimodal_encoder_name=multimodal_encoder,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
            probe_epochs=probe_epochs,
            probe_lr=probe_lr,
            probe_weight_decay=probe_weight_decay,
            max_examples=max_examples,
            save_probe_dir=save_probe_dir,
        )
        output_path = suite_output_path(output_dir, "multimodal", None, None, multimodal_encoder)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        records.append({"representation": "multimodal", "multimodal_encoder": multimodal_encoder, "output": str(output_path)})

    for image_encoder in VISION_BACKBONES:
        for text_encoder in TEXT_BACKBONES:
            output = run_multimodal_encoder_comparison(
                representation="fused",
                image_encoder_name=image_encoder,
                text_encoder_name=text_encoder,
                fusion_projection_dim=fusion_projection_dim,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                seed=seed,
                probe_epochs=probe_epochs,
                probe_lr=probe_lr,
                probe_weight_decay=probe_weight_decay,
                max_examples=max_examples,
                save_probe_dir=save_probe_dir,
            )
            output_path = suite_output_path(output_dir, "fused", image_encoder, text_encoder)
            output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            records.append({
                "representation": "fused",
                "image_encoder": image_encoder,
                "text_encoder": text_encoder,
                "output": str(output_path),
            })

    summary_path = output_dir / "multimodal_representation_suite_index.json"
    summary_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return {"num_runs": len(records), "index": str(summary_path)}


def run_multimodal_encoder_comparison(
    representation: str,
    image_encoder_name: str | None = None,
    text_encoder_name: str | None = None,
    multimodal_encoder_name: str | None = None,
    image_encoder_model_name: str | None = None,
    text_encoder_model_name: str | None = None,
    multimodal_encoder_model_name: str | None = None,
    fusion_projection_dim: int | None = None,
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
    record_trajectories: bool = False,
    max_trajectory_points: int = 10,
    save_probe_dir: str | Path | None = None,
    probe_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    requested_representation = representation.lower()
    if requested_representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {requested_representation}")
    canonical_representation = requested_representation

    loaded_checkpoint: dict[str, object] | None = None
    if probe_checkpoint is not None:
        loaded_classifier, loaded_checkpoint = load_probe(probe_checkpoint, map_location="cpu")
        metadata = loaded_checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        checkpoint_representation = str(metadata.get("representation", requested_representation or "")).lower()
        normalized_checkpoint_representation = checkpoint_representation
        if checkpoint_representation:
            requested_representation = checkpoint_representation
            canonical_representation = normalized_checkpoint_representation
        image_encoder_name = image_encoder_name or str(metadata.get("image_encoder", "") or "") or None
        text_encoder_name = text_encoder_name or str(metadata.get("text_encoder", "") or "") or None
        multimodal_encoder_name = multimodal_encoder_name or str(metadata.get("multimodal_encoder", "") or "") or None
        checkpoint_projection_dim = metadata.get("projection_dim")
        if checkpoint_projection_dim is not None:
            fusion_projection_dim = int(checkpoint_projection_dim)

    if canonical_representation in {"image", "fused"} and image_encoder_name not in VISION_BACKBONES:
        raise ValueError(
            f"image_encoder_name must be one of {VISION_BACKBONES} for representation={requested_representation}"
        )
    if canonical_representation in {"text", "fused"} and text_encoder_name not in TEXT_BACKBONES:
        raise ValueError(
            f"text_encoder_name must be one of {TEXT_BACKBONES} for representation={requested_representation}"
        )
    if canonical_representation == "multimodal" and multimodal_encoder_name not in MULTIMODAL_BACKBONES:
        raise ValueError(f"multimodal_encoder_name must be one of {MULTIMODAL_BACKBONES} for representation=multimodal")

    set_seed(seed)
    resolved_device = resolve_device(device)

    text_processor = None
    if canonical_representation in {"text", "fused"}:
        text_processor = build_processor(text_encoder_name, model_name=text_encoder_model_name)

    datamodule = build_multimodal_datamodule(
        representation=canonical_representation,
        batch_size=batch_size,
        num_workers=num_workers,
        image_encoder_name=image_encoder_name,
        text_processor=text_processor if canonical_representation == "text" else None,
    )
    datamodule.setup(None)

    split_loaders = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    split_to_embeddings = load_multimodal_split_embeddings(
        representation=canonical_representation,
        split_loaders=split_loaders,
        resolved_device=resolved_device,
        image_encoder_name=image_encoder_name,
        text_encoder_name=text_encoder_name,
        multimodal_encoder_name=multimodal_encoder_name,
        image_encoder_model_name=image_encoder_model_name,
        text_encoder_model_name=text_encoder_model_name,
        multimodal_encoder_model_name=multimodal_encoder_model_name,
        text_processor=text_processor,
    )

    train_embeddings, train_labels = split_to_embeddings["train"]
    val_embeddings, val_labels = split_to_embeddings["val"]
    test_embeddings, test_labels = split_to_embeddings["test"]

    projection_dim = None
    if canonical_representation == "fused" and fusion_projection_dim is not None and int(fusion_projection_dim) > 0:
        projection_dim = int(fusion_projection_dim)
    if loaded_checkpoint is None:
        classifier_head = build_classifier(
            input_dim=train_embeddings.size(1),
            num_classes=datamodule.num_classes,
            projection_dim=projection_dim,
        ).to(resolved_device)
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
    else:
        classifier_head = loaded_classifier.to(resolved_device).eval()
        metadata = loaded_checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        probe_training_stats = {
            "best_epoch": int(metadata.get("probe_best_epoch", 0)),
            "best_score": float(metadata.get("probe_best_score", 0.0)),
            "selection_metric": str(metadata.get("probe_selection_metric", "loaded_checkpoint")),
        }

    eval_classifier = classifier_head
    if projection_dim is not None:
        split_to_embeddings = {
            split: (project_if_needed(classifier_head, embeddings), labels)
            for split, (embeddings, labels) in split_to_embeddings.items()
        }
        eval_classifier = EncodedLinearHead(classifier_head).to(resolved_device).eval()

    if eval_split not in split_to_embeddings or reference_split not in split_to_embeddings:
        raise ValueError("Unsupported eval/reference split")

    eval_embeddings, eval_labels = split_to_embeddings[eval_split]
    eval_indices = torch.arange(eval_embeddings.size(0), dtype=torch.long)
    reference_embeddings, reference_labels = split_to_embeddings[reference_split]
    if max_examples is not None:
        eval_embeddings = eval_embeddings[:max_examples]
        eval_labels = eval_labels[:max_examples]
        eval_indices = eval_indices[:max_examples]

    results, summary = evaluate_embeddings(
        embeddings=eval_embeddings.to(resolved_device),
        classifier_head=eval_classifier,
        labels=eval_labels.to(resolved_device),
        reference_embeddings=reference_embeddings.to(resolved_device),
        reference_labels=reference_labels.to(resolved_device),
        max_examples=max_examples,
        same_reference_pool=eval_split == reference_split,
        example_indices=eval_indices,
        record_trajectory=record_trajectories,
        max_trajectory_points=max_trajectory_points,
        counterfactual_mode=counterfactual_mode,
        k=k,
        step_size=step_size,
        max_steps=max_steps,
        trust_radius=trust_radius,
    )

    output: dict[str, object] = {
        "dataset": "mmimdb",
        "seed": seed,
        "representation": requested_representation,
        "image_encoder": image_encoder_name or "",
        "text_encoder": text_encoder_name or "",
        "multimodal_encoder": multimodal_encoder_name or "",
        "image_encoder_model_name": image_encoder_model_name or "",
        "text_encoder_model_name": text_encoder_model_name or "",
        "multimodal_encoder_model_name": multimodal_encoder_model_name or "",
        "fusion_projection_dim": int(projection_dim) if projection_dim is not None else 0,
        "eval_split": eval_split,
        "reference_split": reference_split,
        "counterfactual_mode": counterfactual_mode,
        "target_strategy": "second_best",
        "k": k,
        "step_size": step_size,
        "max_steps": max_steps,
        "trust_radius": trust_radius,
        "test_accuracy": _classification_accuracy(
            eval_classifier,
            split_to_embeddings["test"][0].to(resolved_device),
            split_to_embeddings["test"][1].to(resolved_device),
        ),
        "val_accuracy": _classification_accuracy(
            eval_classifier,
            split_to_embeddings["val"][0].to(resolved_device),
            split_to_embeddings["val"][1].to(resolved_device),
        ),
        "train_accuracy": _classification_accuracy(
            eval_classifier,
            split_to_embeddings["train"][0].to(resolved_device),
            split_to_embeddings["train"][1].to(resolved_device),
        ),
        "num_train": int(split_to_embeddings["train"][0].size(0)),
        "num_val": int(split_to_embeddings["val"][0].size(0)),
        "num_test": int(split_to_embeddings["test"][0].size(0)),
        "num_evaluated": len(results),
        "class_names": datamodule.class_names,
        "support_scale": dataset_density_scale(split_to_embeddings["train"][0], split_to_embeddings["train"][1], k=k),
        "probe_lr": probe_lr,
        "probe_weight_decay": probe_weight_decay,
        "probe_epochs": probe_epochs,
        "probe_best_epoch": int(probe_training_stats["best_epoch"]),
        "probe_best_score": float(probe_training_stats["best_score"]),
        "probe_selection_metric": str(probe_training_stats["selection_metric"]),
        **summary,
        "raw_results": results,
    }

    if loaded_checkpoint is not None:
        output["probe_checkpoint"] = str(probe_checkpoint)
    elif save_probe_dir is not None:
        encoder_key = image_encoder_name if canonical_representation == "image" else text_encoder_name
        if canonical_representation == "fused":
            encoder_key = f"{image_encoder_name}_{text_encoder_name}_fused"
        elif canonical_representation == "multimodal":
            encoder_key = multimodal_encoder_name
        checkpoint_path = save_probe_checkpoint(
            classifier_head=classifier_head,
            checkpoint_dir=save_probe_dir,
            dataset_name="mmimdb",
            encoder_name=str(encoder_key),
            encoder_model_name=None,
            seed=seed,
            input_dim=train_embeddings.size(1),
            num_classes=datamodule.num_classes,
            metadata={
                "representation": requested_representation,
                "image_encoder": image_encoder_name or "",
                "text_encoder": text_encoder_name or "",
                "multimodal_encoder": multimodal_encoder_name or "",
                "projection_dim": projection_dim,
                "probe_lr": probe_lr,
                "probe_weight_decay": probe_weight_decay,
                "probe_epochs": probe_epochs,
                "eval_split": eval_split,
                "reference_split": reference_split,
                "counterfactual_mode": counterfactual_mode,
                "target_strategy": "second_best",
                "k": k,
                "step_size": step_size,
                "max_steps": max_steps,
                "trust_radius": trust_radius,
                "probe_best_epoch": int(probe_training_stats["best_epoch"]),
                "probe_best_score": float(probe_training_stats["best_score"]),
                "probe_selection_metric": str(probe_training_stats["selection_metric"]),
            },
        )
        output["probe_checkpoint"] = str(checkpoint_path)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal experiments for image, text, multimodal, or fused representations.")
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--image-encoder", choices=VISION_BACKBONES, default=None)
    parser.add_argument("--text-encoder", choices=TEXT_BACKBONES, default=None)
    parser.add_argument("--multimodal-encoder", choices=MULTIMODAL_BACKBONES, default=None)
    parser.add_argument("--image-encoder-model-name", default=None)
    parser.add_argument("--text-encoder-model-name", default=None)
    parser.add_argument("--multimodal-encoder-model-name", default=None)
    parser.add_argument("--fusion-projection-dim", type=int, default=0)
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
    parser.add_argument("--record-trajectories", action="store_true")
    parser.add_argument("--max-trajectory-points", type=int, default=10)
    parser.add_argument("--save-probe-dir", type=Path, default=None)
    parser.add_argument("--probe-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = run_multimodal_encoder_comparison(
        representation=args.representation,
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        multimodal_encoder_name=args.multimodal_encoder,
        image_encoder_model_name=args.image_encoder_model_name,
        text_encoder_model_name=args.text_encoder_model_name,
        multimodal_encoder_model_name=args.multimodal_encoder_model_name,
        fusion_projection_dim=args.fusion_projection_dim,
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
        record_trajectories=args.record_trajectories,
        max_trajectory_points=args.max_trajectory_points,
        save_probe_dir=args.save_probe_dir,
        probe_checkpoint=args.probe_checkpoint,
    )

    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
