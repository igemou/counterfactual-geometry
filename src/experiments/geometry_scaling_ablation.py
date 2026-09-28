from __future__ import annotations

from ..core.utils import write_json

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch

from ..analysis.common import load_main_payloads, load_payload_probe, load_payload_splits, evaluation_indices
from ..analysis.common import model_label, split_cache_path
from ..core.geometry import calibrate_curvature
from ..counterfactuals.evaluation import evaluate_embeddings
from ..counterfactuals.search import build_baseline_config


def matched_configs(base, *, knn_weight, curvature_alpha, curvature_scale):
    if knn_weight <= 0 or curvature_alpha <= 0:
        raise ValueError("Ablation requires positive support weight and curvature alpha")
    base = replace(base, knn_weight=0.0, curvature_alpha=0.0, curvature_scale=curvature_scale)
    return {
        "baseline": base,
        "support_only": replace(base, knn_weight=knn_weight),
        "curvature_only": replace(base, curvature_alpha=curvature_alpha),
        "combined": replace(base, knn_weight=knn_weight, curvature_alpha=curvature_alpha),
    }


def evaluate_matched_variants(probe, embeddings, labels, reference_embeddings, reference_labels,
                              validation_embeddings, base, *, k=20, knn_weight=1.0,
                              curvature_alpha=1.0, support_quantile=0.5,
                              validation_is_reference=False, example_indices=None,
                              record_trajectory=False, target_label=None):
    scale = calibrate_curvature(validation_embeddings, reference_embeddings, k, base.tangent_dim,
                                base.curvature_eps, validation_is_reference)
    configs = matched_configs(base, knn_weight=knn_weight, curvature_alpha=curvature_alpha, curvature_scale=scale)
    variants = {}
    for name, config in configs.items():
        rows, summary = evaluate_embeddings(
            embeddings, probe, labels, reference_embeddings, reference_labels,
            config=config, k=k, example_indices=example_indices, support_quantile=support_quantile,
            record_endpoint=True, record_trajectory=record_trajectory, target_label=target_label)
        if record_trajectory:
            for row in rows:
                target_indices = torch.where(reference_labels == row['target_label'])[0]
                if len(target_indices):
                    endpoint = reference_embeddings.new_tensor(row['final_embedding'])
                    distances = torch.linalg.vector_norm(reference_embeddings[target_indices] - endpoint, dim=1)
                    nearest = distances.topk(min(5, len(distances)), largest=False).indices
                    row['target_neighbor_indices'] = target_indices[nearest].tolist()
        variants[name] = {"config": asdict(config), "summary": summary, "raw_results": rows}
    return variants


def run_geometry_scaling_ablation(compare_dir=Path("outputs"), cache_dir=Path("outputs/cache/embeddings"),
                                 output_dir=Path("outputs/analysis"), *, max_examples=None,
                                 knn_weight=1.0, curvature_alpha=1.0, tangent_dim=2,
                                 max_steps=300, support_quantile=0.5, record_trajectory=False, dataset_settings=None):
    model_rows = []
    for payload in load_main_payloads(compare_dir):
        probe, _ = load_payload_probe(payload)
        splits = load_payload_splits(payload, cache_dir)
        eval_split = payload.get("eval_split", "test")
        reference_split = payload.get("reference_split", "val")
        if eval_split != "test":
            raise ValueError("Paper ablations must evaluate held-out test examples")
        embeddings, labels = splits[eval_split]
        refs, ref_labels = splits[reference_split]
        indices = evaluation_indices(payload, len(embeddings), max_examples)
        settings = (dataset_settings or {}).get(payload["dataset"], {})
        model_tangent_dim = settings.get("tangent_dim", tangent_dim)
        base = build_baseline_config(float(payload["step_size"]), float(payload["trust_radius"]), max_steps,
                                     shift_weight=float(payload.get("shift_weight", 0.0)), tangent_dim=model_tangent_dim)
        variants = evaluate_matched_variants(
            probe, embeddings[indices], labels[indices], refs, ref_labels, splits["val"][0], base,
            k=int(payload.get("k", 20)), knn_weight=settings.get("knn_weight", knn_weight), curvature_alpha=settings.get("curvature_alpha", curvature_alpha),
            support_quantile=support_quantile, validation_is_reference=reference_split == "val",
            example_indices=indices, record_trajectory=record_trajectory)
        if record_trajectory:
            case_dir = output_dir / 'cases' / payload['dataset'] / model_label(payload) / f"seed{payload['seed']}"
            for name, variant in variants.items():
                case_payload = {key: value for key, value in payload.items() if key != 'raw_results'}
                case_payload.update(variant['summary'])
                case_payload.update(variant['config'])
                case_payload['raw_results'] = variant['raw_results']
                case_payload['search_variant'] = name
                case_payload['split_paths'] = {split: str(split_cache_path(payload, cache_dir, split).resolve())
                                               for split in ('train', 'val', 'test')}
                write_json(case_dir / f'{name}.json', case_payload)
        model_rows.append({"dataset": payload["dataset"], "model": model_label(payload),
                           "seed": payload.get("seed"), "probe_checkpoint": payload["probe_checkpoint"],
                           "variants": variants})
    metrics = ("counterfactual_success_mean", "supported_counterfactual_success_mean")
    dataset_rows = []
    for dataset in sorted({r["dataset"] for r in model_rows}):
        rows = [r for r in model_rows if r["dataset"] == dataset]
        dataset_rows.append({"dataset": dataset, "variants": {
            v: {m: float(np.mean([r["variants"][v]["summary"][m] for r in rows])) for m in metrics}
            for v in rows[0]["variants"]}})
    overall = {v: {m: float(np.mean([r["variants"][v][m] for r in dataset_rows])) for m in metrics}
               for v in model_rows[0]["variants"]}
    result = {"aggregation": "equal dataset weights; equal model/run weights within dataset",
              "model_rows": model_rows, "dataset_rows": dataset_rows, "overall": overall,
              "num_datasets": len(dataset_rows), "support_quantile": support_quantile}
    write_json(output_dir / "geometry_scaling_ablation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--knn-weight", type=float, required=True)
    parser.add_argument("--curvature-alpha", type=float, required=True)
    parser.add_argument("--tangent-dim", type=int, required=True)
    parser.add_argument("--support-quantile", type=float, default=0.5)
    parser.add_argument("--record-trajectory", action="store_true")
    run_geometry_scaling_ablation(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
