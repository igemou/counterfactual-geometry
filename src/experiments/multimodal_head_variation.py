from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import torch

from ..core.classifier import build_classifier, train_linear_probe
from ..counterfactuals.evaluation import evaluate_embeddings
from ..core.utils import load_probe, set_seed
from ..core.encoders import build_processor
from .common import MULTIMODAL_BACKBONES, TEXT_BACKBONES, VISION_BACKBONES, build_multimodal_datamodule, load_multimodal_split_embeddings


def _classification_accuracy(classifier_head, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        logits = classifier_head(embeddings)
        predictions = logits.argmax(dim=1)
        return float((predictions == labels).float().mean().item())


def _parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return list(default)
    return [float(value) for value in values]


def _parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return list(default)
    return [int(value) for value in values]


def _build_variant_name(seed: int, probe_lr: float, probe_weight_decay: float, probe_epochs: int) -> str:
    return (
        "retrained"
        f"_seed{seed}"
        f"_lr{probe_lr:g}"
        f"_wd{probe_weight_decay:g}"
        f"_ep{probe_epochs}"
    )


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_probe_on_fixed_embeddings(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    seed: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    projection_dim: int | None = None,
):
    set_seed(seed)
    classifier_head = build_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        projection_dim=projection_dim,
    ).to(device)
    classifier_head, training_stats = train_linear_probe(
        classifier=classifier_head,
        embeddings=train_embeddings.to(device),
        labels=train_labels.to(device),
        val_embeddings=val_embeddings.to(device),
        val_labels=val_labels.to(device),
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
    )
    return classifier_head.eval(), training_stats


def _baseline_probe_hparams(metadata: dict[str, Any]) -> tuple[float, float]:
    baseline_lr = metadata.get("probe_lr", 1e-3)
    baseline_weight_decay = metadata.get("probe_weight_decay", 1e-4)
    return float(baseline_lr), float(baseline_weight_decay)


def run_multimodal_classifier_head_variation(
    representation: str,
    image_encoder_name: str | None,
    text_encoder_name: str | None,
    probe_checkpoint: str | Path,
    multimodal_encoder_name: str | None = None,
    image_encoder_model_name: str | None = None,
    text_encoder_model_name: str | None = None,
    multimodal_encoder_model_name: str | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
    seed: int = 42,
    eval_split: str = "test",
    reference_split: str = "val",
    max_examples: int | None = None,
    counterfactual_mode: str = "targeted",
    k: int = 20,
    step_size: float = 1e-2,
    max_steps: int = 500,
    trust_radius: float = 1.0,
    intervention_seeds: list[int] | None = None,
    intervention_probe_lrs: list[float] | None = None,
    intervention_probe_weight_decays: list[float] | None = None,
    intervention_probe_epochs: int = 100,
) -> dict[str, Any]:
    representation = representation.lower()
    if representation == "fused":
        if image_encoder_name not in VISION_BACKBONES or text_encoder_name not in TEXT_BACKBONES:
            raise ValueError("fused representation requires valid image_encoder_name and text_encoder_name")
    elif representation == "multimodal":
        if multimodal_encoder_name not in MULTIMODAL_BACKBONES:
            raise ValueError("multimodal representation requires multimodal_encoder_name to be clip or siglip2")
    else:
        raise ValueError("representation must be 'fused' or 'multimodal'")

    set_seed(seed)
    resolved_device = _resolve_device(device)

    baseline_head, checkpoint = load_probe(probe_checkpoint, map_location="cpu")
    baseline_head = baseline_head.to(resolved_device).eval()

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    projection_dim = metadata.get("projection_dim")
    if projection_dim is not None:
        projection_dim = int(projection_dim)
        if projection_dim <= 0:
            projection_dim = None
    if representation == "multimodal":
        projection_dim = None

    text_processor = None
    if representation == "fused":
        text_processor = build_processor(text_encoder_name, model_name=text_encoder_model_name)
    datamodule = build_multimodal_datamodule(
        representation=representation,
        batch_size=batch_size,
        num_workers=num_workers,
        image_encoder_name=image_encoder_name,
        text_processor=text_processor,
    )
    datamodule.setup(None)

    split_loaders = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    split_to_embeddings = load_multimodal_split_embeddings(
        representation=representation,
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
    eval_embeddings, eval_labels = split_to_embeddings[eval_split]
    reference_embeddings, reference_labels = split_to_embeddings[reference_split]
    same_reference_pool = eval_split == reference_split

    if max_examples is not None:
        eval_embeddings = eval_embeddings[:max_examples]
        eval_labels = eval_labels[:max_examples]

    eval_embeddings = eval_embeddings.to(resolved_device)
    eval_labels = eval_labels.to(resolved_device)
    reference_embeddings = reference_embeddings.to(resolved_device)
    reference_labels = reference_labels.to(resolved_device)

    num_classes = int(checkpoint["num_classes"])
    input_dim = int(checkpoint["input_dim"])

    baseline_probe_lr, baseline_probe_weight_decay = _baseline_probe_hparams(metadata)

    variants: list[dict[str, Any]] = [
        {
            "name": "baseline_checkpoint",
            "classifier_head": baseline_head,
            "training": {
                "source": "checkpoint",
                "variation_type": "baseline",
                "probe_lr": baseline_probe_lr,
                "probe_weight_decay": baseline_probe_weight_decay,
                "best_epoch": int(metadata.get("probe_best_epoch", 0)),
                "best_score": float(metadata.get("probe_best_score", 0.0)),
                "selection_metric": str(metadata.get("probe_selection_metric", "unknown")),
            },
        }
    ]

    seeds = intervention_seeds or [seed + 1, seed + 2, seed + 3, seed + 4]
    probe_lrs = intervention_probe_lrs or [baseline_probe_lr]
    probe_weight_decays = intervention_probe_weight_decays or [baseline_probe_weight_decay]

    for variant_seed, probe_lr, probe_weight_decay in product(seeds, probe_lrs, probe_weight_decays):
        varies_lr = abs(probe_lr - baseline_probe_lr) > 1e-12
        varies_weight_decay = abs(probe_weight_decay - baseline_probe_weight_decay) > 1e-12
        if varies_lr and varies_weight_decay:
            variation_type = "seed_lr_weight_decay"
        elif varies_lr:
            variation_type = "seed_lr"
        elif varies_weight_decay:
            variation_type = "seed_weight_decay"
        else:
            variation_type = "seed_only"
        classifier_head, training_stats = _train_probe_on_fixed_embeddings(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            input_dim=input_dim,
            num_classes=num_classes,
            device=resolved_device,
            seed=variant_seed,
            probe_epochs=intervention_probe_epochs,
            probe_lr=probe_lr,
            probe_weight_decay=probe_weight_decay,
            projection_dim=projection_dim,
        )
        variants.append(
            {
                "name": _build_variant_name(
                    seed=variant_seed,
                    probe_lr=probe_lr,
                    probe_weight_decay=probe_weight_decay,
                    probe_epochs=intervention_probe_epochs,
                ),
                "classifier_head": classifier_head,
                "training": {
                    "source": "retrained_on_fixed_embeddings",
                    "variation_type": variation_type,
                    "seed": variant_seed,
                    "probe_lr": probe_lr,
                    "probe_weight_decay": probe_weight_decay,
                    "probe_epochs": intervention_probe_epochs,
                    **training_stats,
                },
            }
        )

    variant_outputs: list[dict[str, Any]] = []
    baseline_summary: dict[str, float] | None = None
    baseline_accuracy: float | None = None

    for variant in variants:
        classifier_head = variant.pop("classifier_head")
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

        output = {
            **variant,
            **summary,
            "train_accuracy": _classification_accuracy(
                classifier_head,
                train_embeddings.to(resolved_device),
                train_labels.to(resolved_device),
            ),
            "val_accuracy": _classification_accuracy(
                classifier_head,
                val_embeddings.to(resolved_device),
                val_labels.to(resolved_device),
            ),
            "test_accuracy": _classification_accuracy(
                classifier_head,
                split_to_embeddings["test"][0].to(resolved_device),
                split_to_embeddings["test"][1].to(resolved_device),
            ),
            "num_evaluated": len(results),
            "raw_results": results,
        }

        if output["name"] == "baseline_checkpoint":
            baseline_summary = summary
            baseline_accuracy = output["test_accuracy"]
        else:
            if baseline_summary is not None:
                output["delta_counterfactual_success_mean"] = (
                    output.get("counterfactual_success_mean", 0.0)
                    - baseline_summary.get("counterfactual_success_mean", 0.0)
                )
                output["delta_counterfactual_distance_mean"] = (
                    output.get("counterfactual_distance_mean", 0.0) - baseline_summary.get("counterfactual_distance_mean", 0.0)
                )
                output["delta_boundary_distance_mean"] = (
                    output.get("boundary_distance_mean", 0.0) - baseline_summary.get("boundary_distance_mean", 0.0)
                )
            if baseline_accuracy is not None:
                output["delta_test_accuracy"] = output["test_accuracy"] - baseline_accuracy

        variant_outputs.append(output)

    return {
        "dataset": "mmimdb",
        "representation": representation,
        "encoder": (
            f"{image_encoder_name}_{text_encoder_name}_fused"
            if representation == "fused"
            else str(multimodal_encoder_name)
        ),
        "image_encoder": image_encoder_name or "",
        "text_encoder": text_encoder_name or "",
        "multimodal_encoder": multimodal_encoder_name or "",
        "image_encoder_model_name": image_encoder_model_name or "",
        "text_encoder_model_name": text_encoder_model_name or "",
        "multimodal_encoder_model_name": multimodal_encoder_model_name or "",
        "probe_checkpoint": str(probe_checkpoint),
        "fixed_embeddings": True,
        "intervention_family": "retrained_linear_probe",
        "baseline_probe_lr": baseline_probe_lr,
        "baseline_probe_weight_decay": baseline_probe_weight_decay,
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
        "num_test": int(split_to_embeddings["test"][0].size(0)),
        "variants": variant_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal fixed-representation classifier-head variation on MM-IMDb.")
    parser.add_argument("--representation", choices=["fused", "multimodal"], default="fused")
    parser.add_argument("--image-encoder", default=None)
    parser.add_argument("--text-encoder", default=None)
    parser.add_argument("--multimodal-encoder", choices=MULTIMODAL_BACKBONES, default=None)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--image-encoder-model-name", default=None)
    parser.add_argument("--text-encoder-model-name", default=None)
    parser.add_argument("--multimodal-encoder-model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--reference-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--counterfactual-mode", choices=["untargeted", "targeted"], default="targeted")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--trust-radius", type=float, default=1.0)
    parser.add_argument("--intervention-seed", action="append", default=None)
    parser.add_argument("--intervention-probe-lr", action="append", default=None)
    parser.add_argument("--intervention-probe-weight-decay", action="append", default=None)
    parser.add_argument("--intervention-probe-epochs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = run_multimodal_classifier_head_variation(
        representation=args.representation,
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        probe_checkpoint=args.probe_checkpoint,
        multimodal_encoder_name=args.multimodal_encoder,
        image_encoder_model_name=args.image_encoder_model_name,
        text_encoder_model_name=args.text_encoder_model_name,
        multimodal_encoder_model_name=args.multimodal_encoder_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        eval_split=args.eval_split,
        reference_split=args.reference_split,
        max_examples=args.max_examples,
        counterfactual_mode=args.counterfactual_mode,
        k=args.k,
        step_size=args.step_size,
        max_steps=args.max_steps,
        trust_radius=args.trust_radius,
        intervention_seeds=_parse_int_list(args.intervention_seed, default=[]),
        intervention_probe_lrs=_parse_float_list(args.intervention_probe_lr, default=[]),
        intervention_probe_weight_decays=_parse_float_list(args.intervention_probe_weight_decay, default=[]),
        intervention_probe_epochs=args.intervention_probe_epochs,
    )

    payload = json.dumps(output, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
