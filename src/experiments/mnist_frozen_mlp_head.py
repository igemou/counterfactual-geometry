from __future__ import annotations

from ..core.utils import write_json
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ..core.classifier import train_linear_probe
from ..core.datasets import build_datamodule
from ..core.encoders import build_encoder, build_processor, freeze_encoder
from ..core.geometry import dataset_density_scale
from ..core.utils import configure_runtime_paths, embedding_cache_path, set_seed
from ..counterfactuals.evaluation import evaluate_embeddings
from ..experiments.common import resolve_device
from ..experiments.unimodal_encoder_comparison import _classification_accuracy, extract_embeddings


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)


def run_mnist_frozen_mlp_head(
    encoder_name: str,
    *,
    encoder_model_name: str | None = None,
    hidden_dim: int = 256,
    dropout: float = 0.0,
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
    k: int = 20,
    step_size: float = 1e-2,
    max_steps: int = 300,
    trust_radius: float = 1.0,
    shift_weight: float = 0.0,
    tangent_dim: int = 2,
    save_checkpoint_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    embedding_cache_root: str | Path | None = None,
    hf_cache_dir: str | Path | None = None,
) -> dict[str, object]:
    configure_runtime_paths(data_dir=data_dir, embedding_cache_root=embedding_cache_root, hf_cache_dir=hf_cache_dir)
    set_seed(seed)
    resolved_device = resolve_device(device)
    datamodule = build_datamodule("mnist", batch_size=batch_size, num_workers=num_workers, normalize=encoder_name.lower() != "dinov2")
    datamodule.setup(None)

    encoder = freeze_encoder(build_encoder(encoder_name, model_name=encoder_model_name)).to(resolved_device)
    processor = build_processor(encoder_name, model_name=encoder_model_name) if getattr(encoder, "uses_processor", False) else None

    train_embeddings, train_labels = extract_embeddings(
        datamodule.train_dataloader(),
        encoder=encoder,
        device=resolved_device,
        processor=processor,
        cache_path=embedding_cache_path("mnist", f"{encoder_name}_mlp_head", encoder_model_name, "train", root=embedding_cache_root),
    )
    val_embeddings, val_labels = extract_embeddings(
        datamodule.val_dataloader(),
        encoder=encoder,
        device=resolved_device,
        processor=processor,
        cache_path=embedding_cache_path("mnist", f"{encoder_name}_mlp_head", encoder_model_name, "val", root=embedding_cache_root),
    )
    test_embeddings, test_labels = extract_embeddings(
        datamodule.test_dataloader(),
        encoder=encoder,
        device=resolved_device,
        processor=processor,
        cache_path=embedding_cache_path("mnist", f"{encoder_name}_mlp_head", encoder_model_name, "test", root=embedding_cache_root),
    )

    classifier_head = MLPHead(train_embeddings.size(1), hidden_dim=hidden_dim, num_classes=datamodule.num_classes, dropout=dropout)
    classifier_head, training_stats = train_linear_probe(
        classifier_head,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        batch_size=128,
    )

    split_to_embeddings = {"train": (train_embeddings, train_labels), "val": (val_embeddings, val_labels), "test": (test_embeddings, test_labels)}
    eval_embeddings, eval_labels = split_to_embeddings[eval_split]
    reference_embeddings, reference_labels = split_to_embeddings[reference_split]
    results, summary = evaluate_embeddings(
        embeddings=eval_embeddings,
        classifier_head=classifier_head,
        labels=eval_labels,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        max_examples=max_examples,
        same_reference_pool=eval_split == reference_split,
        k=k,
        step_size=step_size,
        max_steps=max_steps,
        trust_radius=trust_radius,
        shift_weight=shift_weight,
        tangent_dim=tangent_dim,
    )
    support_scale = dataset_density_scale(reference_embeddings, reference_labels, k=k)

    checkpoint_path_str = None
    if save_checkpoint_path is not None:
        checkpoint_path = Path(save_checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "variant": "mnist_frozen_mlp_head",
                "dataset": "mnist",
                "encoder": encoder_name,
                "encoder_model_name": encoder_model_name or "",
                "seed": seed,
                "head_state_dict": classifier_head.cpu().state_dict(),
                "head_hidden_dim": hidden_dim,
                "head_dropout": dropout,
                "head_input_dim": int(train_embeddings.size(1)),
                "num_classes": datamodule.num_classes,
                "metadata": {
                    "probe_epochs": probe_epochs,
                    "probe_lr": probe_lr,
                    "probe_weight_decay": probe_weight_decay,
                    "eval_split": eval_split,
                    "reference_split": reference_split,
                    "counterfactual_mode": "targeted",
                    "k": k,
                    "step_size": step_size,
                    "max_steps": max_steps,
                    "trust_radius": trust_radius,
                "shift_weight": shift_weight,
                "tangent_dim": tangent_dim,
                    "probe_best_epoch": int(training_stats["best_epoch"]),
                    "probe_best_score": float(training_stats["best_score"]),
                },
            },
            checkpoint_path,
        )
        checkpoint_path_str = str(checkpoint_path)
    return {
        **summary,
        "dataset": "mnist",
        "encoder": encoder_name,
        "encoder_model_name": encoder_model_name or "",
        "variant": "frozen_encoder_mlp_head",
        "mlp_hidden_dim": hidden_dim,
        "mlp_dropout": dropout,
        "seed": seed,
        "eval_split": eval_split,
        "reference_split": reference_split,
        "counterfactual_mode": "targeted",
        "k": k,
        "step_size": step_size,
        "max_steps": max_steps,
        "trust_radius": trust_radius,
        "shift_weight": shift_weight,
        "tangent_dim": tangent_dim,
        "num_train": int(train_embeddings.size(0)),
        "num_val": int(val_embeddings.size(0)),
        "num_test": int(test_embeddings.size(0)),
        "train_accuracy": _classification_accuracy(classifier_head, train_embeddings, train_labels),
        "val_accuracy": _classification_accuracy(classifier_head, val_embeddings, val_labels),
        "test_accuracy": _classification_accuracy(classifier_head, test_embeddings, test_labels),
        "reference_support_scale": support_scale,
        "probe_lr": probe_lr,
        "probe_weight_decay": probe_weight_decay,
        "probe_epochs": probe_epochs,
        "probe_selection_metric": str(training_stats["selection_metric"]),
        "probe_best_epoch": int(training_stats["best_epoch"]),
        "probe_best_score": float(training_stats["best_score"]),
        "checkpoint_path": checkpoint_path_str or "",
        "num_evaluated": len(results),
        "raw_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST frozen-encoder MLP head appendix experiment.")
    parser.add_argument("--encoder", required=True, choices=["resnet50", "vit", "dinov2"])
    parser.add_argument("--encoder-model-name", default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
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
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--trust-radius", type=float, default=1.0)
    parser.add_argument("--save-checkpoint-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--embedding-cache-root", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shift-weight", type=float, default=0.0)
    parser.add_argument("--tangent-dim", type=int, default=2)
    args = parser.parse_args()
    output = run_mnist_frozen_mlp_head(
        encoder_name=args.encoder,
        encoder_model_name=args.encoder_model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
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
        k=args.k,
        step_size=args.step_size,
        max_steps=args.max_steps,
        trust_radius=args.trust_radius,
        shift_weight=args.shift_weight,
        tangent_dim=args.tangent_dim,
        save_checkpoint_path=args.save_checkpoint_path,
        data_dir=args.data_dir,
        embedding_cache_root=args.embedding_cache_root,
        hf_cache_dir=args.hf_cache_dir,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output is not None:
        write_json(args.output, output)


if __name__ == "__main__":
    main()
