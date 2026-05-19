from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import torch

from ..core.classifier import build_classifier, train_linear_probe
from ..counterfactuals.evaluation import evaluate_embeddings
from ..core.utils import embedding_cache_path, load_probe, set_seed
from .common import copy_split_to_device, resolve_device


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
    classifier_head = build_classifier(input_dim=input_dim, num_classes=num_classes, projection_dim=projection_dim).to(device)
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


def _load_split_embeddings(
    dataset_name: str,
    encoder_name: str,
    encoder_model_name: str | None,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = embedding_cache_path(dataset_name, encoder_name, encoder_model_name, split)
    payload = torch.load(cache_path, map_location="cpu")
    embeddings = payload.get("embeddings")
    labels = payload.get("labels")
    if not isinstance(embeddings, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError(f"Invalid embedding cache at {cache_path}")
    return embeddings.float(), labels.long()


def run_unimodal_classifier_head_variation(
    dataset_name: str,
    encoder_name: str,
    probe_checkpoint: str | Path,
    encoder_model_name: str | None = None,
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
    set_seed(seed)
    resolved_device = resolve_device(device)

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

    inferred_model_name = encoder_model_name or checkpoint.get("encoder_model_name") or None
    if inferred_model_name == "":
        inferred_model_name = None

    split_to_embeddings = {}
    for split in {"train", "val", "test", eval_split, reference_split}:
        split_to_embeddings[split] = _load_split_embeddings(
            dataset_name=dataset_name,
            encoder_name=encoder_name,
            encoder_model_name=inferred_model_name,
            split=split,
        )

    train_embeddings, train_labels = split_to_embeddings["train"]
    val_embeddings, val_labels = split_to_embeddings["val"]
    eval_embeddings, eval_labels = copy_split_to_device(split_to_embeddings, eval_split, resolved_device)
    reference_embeddings, reference_labels = copy_split_to_device(split_to_embeddings, reference_split, resolved_device)
    same_reference_pool = eval_split == reference_split

    if max_examples is not None:
        eval_embeddings = eval_embeddings[:max_examples]
        eval_labels = eval_labels[:max_examples]

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
                "best_epoch": int(checkpoint.get("metadata", {}).get("probe_best_epoch", 0)),
                "best_score": float(checkpoint.get("metadata", {}).get("probe_best_score", 0.0)),
                "selection_metric": str(checkpoint.get("metadata", {}).get("probe_selection_metric", "unknown")),
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
            "train_accuracy": _classification_accuracy(classifier_head, train_embeddings.to(resolved_device), train_labels.to(resolved_device)),
            "val_accuracy": _classification_accuracy(classifier_head, val_embeddings.to(resolved_device), val_labels.to(resolved_device)),
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
        "dataset": dataset_name,
        "encoder": encoder_name,
        "encoder_model_name": inferred_model_name or "",
        "seed": seed,
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
    parser = argparse.ArgumentParser(
        description="Evaluate the counterfactual probe after changing only the classifier head while keeping embeddings fixed."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-model-name", default=None)
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

    output = run_unimodal_classifier_head_variation(
        dataset_name=args.dataset,
        encoder_name=args.encoder,
        probe_checkpoint=args.probe_checkpoint,
        encoder_model_name=args.encoder_model_name,
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
        intervention_seeds=_parse_int_list(args.intervention_seed, []),
        intervention_probe_lrs=_parse_float_list(args.intervention_probe_lr, []),
        intervention_probe_weight_decays=_parse_float_list(args.intervention_probe_weight_decay, []),
        intervention_probe_epochs=args.intervention_probe_epochs,
    )

    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
