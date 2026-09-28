from __future__ import annotations

from ..core.utils import write_json
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ..core.datasets import build_datamodule
from ..core.encoders import build_encoder, build_processor, freeze_encoder, unpack_batch
from ..core.geometry import dataset_density_scale
from ..core.utils import configure_runtime_paths, set_seed, to_device
from ..counterfactuals.evaluation import evaluate_embeddings
from ..experiments.common import PROCESSOR_VISION_ENCODERS, resolve_device
from ..experiments.unimodal_encoder_comparison import _classification_accuracy, _prepare_encoder_inputs


class EncoderWithHead(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, features) -> torch.Tensor:
        if isinstance(features, dict):
            embeddings = self.encoder(**features)
        else:
            embeddings = self.encoder(features)
        return self.head(embeddings)


def _iter_trainable_blocks(encoder_name: str, encoder: nn.Module) -> list[nn.Module]:
    lowered = encoder_name.lower()
    if lowered == "resnet50":
        return [encoder.model.layer1, encoder.model.layer2, encoder.model.layer3, encoder.model.layer4]
    if lowered == "vit":
        return list(encoder.model.encoder.layers)
    if lowered == "dinov2":
        return list(encoder.model.encoder.layer)
    raise ValueError(f"Partial fine-tuning is not configured for encoder {encoder_name}")


def _unfreeze_last_blocks(encoder_name: str, encoder: nn.Module, num_blocks: int) -> list[str]:
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    blocks = _iter_trainable_blocks(encoder_name, encoder)
    selected = blocks[-min(num_blocks, len(blocks)) :]
    for block in selected:
        for parameter in block.parameters():
            parameter.requires_grad = True
    return [f"{type(block).__name__}:{index}" for index, block in enumerate(selected, start=len(blocks) - len(selected))]


def _collect_split_embeddings(dataloader, encoder, device: torch.device, processor=None) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            features, labels = unpack_batch(batch)
            labels = to_device(labels, device)
            prepared = _prepare_encoder_inputs(features, processor=processor, device=device)
            embeddings = encoder(**prepared) if isinstance(prepared, dict) else encoder(prepared)
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def _accuracy_from_loader(model: nn.Module, dataloader, device: torch.device, processor=None) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            features, labels = unpack_batch(batch)
            labels = to_device(labels, device)
            prepared = _prepare_encoder_inputs(features, processor=processor, device=device)
            logits = model(prepared)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    return correct / max(total, 1)


def run_mnist_partial_finetune(
    encoder_name: str,
    *,
    encoder_model_name: str | None = None,
    num_unfrozen_blocks: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
    seed: int = 42,
    finetune_epochs: int = 5,
    finetune_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
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
    del embedding_cache_root
    configure_runtime_paths(data_dir=data_dir, hf_cache_dir=hf_cache_dir)
    set_seed(seed)
    resolved_device = resolve_device(device)

    datamodule = build_datamodule("mnist", batch_size=batch_size, num_workers=num_workers, normalize=encoder_name.lower() not in PROCESSOR_VISION_ENCODERS)
    datamodule.setup(None)

    encoder = freeze_encoder(build_encoder(encoder_name, model_name=encoder_model_name)).to(resolved_device)
    unfrozen_blocks = _unfreeze_last_blocks(encoder_name, encoder, num_blocks=num_unfrozen_blocks)
    processor = build_processor(encoder_name, model_name=encoder_model_name) if getattr(encoder, "uses_processor", False) else None

    head = nn.Linear(int(encoder.output_dim), datamodule.num_classes).to(resolved_device)
    model = EncoderWithHead(encoder, head).to(resolved_device)

    optimizer = torch.optim.Adam(
        [
            {"params": [parameter for parameter in encoder.parameters() if parameter.requires_grad], "lr": finetune_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = float("-inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        encoder.eval()  # Frozen blocks must not update batch-normalization statistics.
        for batch in datamodule.train_dataloader():
            features, labels = unpack_batch(batch)
            labels = to_device(labels, resolved_device)
            prepared = _prepare_encoder_inputs(features, processor=processor, device=resolved_device)
            optimizer.zero_grad()
            logits = model(prepared)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_accuracy = _accuracy_from_loader(model, datamodule.val_dataloader(), resolved_device, processor=processor)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    train_embeddings, train_labels = _collect_split_embeddings(datamodule.train_dataloader(), encoder, resolved_device, processor=processor)
    val_embeddings, val_labels = _collect_split_embeddings(datamodule.val_dataloader(), encoder, resolved_device, processor=processor)
    test_embeddings, test_labels = _collect_split_embeddings(datamodule.test_dataloader(), encoder, resolved_device, processor=processor)
    split_to_embeddings = {
        "train": (train_embeddings, train_labels),
        "val": (val_embeddings, val_labels),
        "test": (test_embeddings, test_labels),
    }
    eval_embeddings, eval_labels = split_to_embeddings[eval_split]
    reference_embeddings, reference_labels = split_to_embeddings[reference_split]

    classifier_head = head.cpu().eval()
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
                "variant": "mnist_partial_finetune",
                "dataset": "mnist",
                "encoder": encoder_name,
                "encoder_model_name": encoder_model_name or "",
                "seed": seed,
                "num_unfrozen_blocks": num_unfrozen_blocks,
                "unfrozen_blocks": unfrozen_blocks,
                "encoder_state_dict": encoder.cpu().state_dict(),
                "head_state_dict": head.cpu().state_dict(),
                "head_input_dim": int(encoder.output_dim),
                "num_classes": datamodule.num_classes,
                "metadata": {
                    "finetune_epochs": finetune_epochs,
                    "finetune_lr": finetune_lr,
                    "head_lr": head_lr,
                    "weight_decay": weight_decay,
                    "eval_split": eval_split,
                    "reference_split": reference_split,
                    "counterfactual_mode": "targeted",
                    "k": k,
                    "step_size": step_size,
                    "max_steps": max_steps,
                    "trust_radius": trust_radius,
                "shift_weight": shift_weight,
                "tangent_dim": tangent_dim,
                    "best_val_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
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
        "variant": "partial_finetune",
        "num_unfrozen_blocks": num_unfrozen_blocks,
        "unfrozen_blocks": unfrozen_blocks,
        "finetune_epochs": finetune_epochs,
        "finetune_lr": finetune_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
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
        "best_val_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "checkpoint_path": checkpoint_path_str or "",
        "num_evaluated": len(results),
        "raw_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST partial fine-tuning appendix experiment.")
    parser.add_argument("--encoder", required=True, choices=["resnet50", "vit", "dinov2"])
    parser.add_argument("--encoder-model-name", default=None)
    parser.add_argument("--num-unfrozen-blocks", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    output = run_mnist_partial_finetune(
        encoder_name=args.encoder,
        encoder_model_name=args.encoder_model_name,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
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
