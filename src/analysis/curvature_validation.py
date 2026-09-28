from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from ..core.geometry import tangent_geometry, local_curvature
from ..core.utils import write_json
from .common import load_main_payloads, load_payload_splits, evaluation_indices
from .common import model_label, spearman, metric_value
from .geometry_prediction import _fit_and_score_ols, _held_out_split


@torch.no_grad()
def held_out_linearization(z, references, k=20, tangent_dim=2, holdout_fraction=.25, seed=0, eps=1e-8):
    if not 0 < holdout_fraction < 1 or len(references) < k:
        raise ValueError("Invalid held-out neighborhood")
    indices = torch.linalg.vector_norm(references - z, dim=1).topk(k, largest=False).indices
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(k, generator=generator).to(indices.device)
    n_test = max(1, round(k * holdout_fraction))
    offsets = references[indices] - z
    train = offsets[order[n_test:]]
    test = offsets[order[:n_test]]
    curvature, basis = tangent_geometry(train, tangent_dim, eps)
    residual = test - (test @ basis) @ basis.T
    error = residual.norm(dim=1).mean()
    return {"curvature": float(curvature), "held_out_error": float(error),
            "relative_held_out_error": float(error / (test.norm(dim=1).mean() + eps)),
            "fit_neighbor_indices": indices[order[n_test:]].tolist(),
            "held_out_neighbor_indices": indices[order[:n_test]].tolist()}


def run_curvature_validation(compare_dir, cache_dir, output_dir, *, ks=(5,10,20,50),
                             tangent_dims=(1,2,3), max_examples=None, seed=0):
    model_rows = []
    for payload in load_main_payloads(compare_dir):
        splits = load_payload_splits(payload, cache_dir)
        embeddings = splits[payload.get("eval_split", "test")][0]
        refs = splits[payload.get("reference_split", "val")][0]
        indices = evaluation_indices(payload, len(embeddings), max_examples)
        for k in ks:
            for m in tangent_dims:
                if m >= min(k-max(1,round(k*.25)), embeddings.shape[1]):
                    continue
                rows = [dict(example_index=int(i), **held_out_linearization(embeddings[i], refs, k, m, seed=seed+int(i))) for i in indices]
                outcomes = {int(row['example_index']): row for row in payload['raw_results']}
                diagnostic_rows = []
                for index in indices:
                    raw = outcomes[int(index)]
                    diagnostic_rows.append({
                        'dataset': payload['dataset'], 'model': model_label(payload),
                        'example_index': int(index),
                        'local_curvature': local_curvature(embeddings[index], refs, k, m,
                            exclude_self=payload.get('eval_split', 'test') == payload.get('reference_split', 'val')),
                        'boundary_distance': metric_value(raw, 'boundary_distance'),
                        'counterfactual_success': float(raw['counterfactual_success']),
                        'counterfactual_distance': metric_value(raw, 'counterfactual_distance'),
                    })
                train, test = _held_out_split(diagnostic_rows, .2, seed)
                associations = {
                    outcome: {'spearman': spearman(
                        [r['local_curvature'] for r in diagnostic_rows
                         if outcome != 'counterfactual_distance' or r['counterfactual_success']],
                        [r[outcome] for r in diagnostic_rows
                         if outcome != 'counterfactual_distance' or r['counterfactual_success']]),
                        'boundary_r2': _fit_and_score_ols(train, test, ('boundary_distance',), outcome),
                        'boundary_curvature_r2': _fit_and_score_ols(
                            train, test, ('boundary_distance', 'local_curvature'), outcome)}
                    for outcome in ('counterfactual_success', 'counterfactual_distance')}
                order = np.argsort([r['curvature'] for r in rows])
                quartiles = [{'quartile': q+1, 'num_examples': len(ids),
                              'mean_held_out_error': float(np.mean([rows[i]['held_out_error'] for i in ids]))}
                             for q, ids in enumerate(np.array_split(order, 4)) if len(ids)]
                model_rows.append({'dataset': payload['dataset'], 'model': model_label(payload),
                                   'seed': payload.get('seed'), 'k': k, 'tangent_dim': m, 'rows': rows,
                                   'quartiles': quartiles, 'outcome_associations': associations,
                                   'spearman': spearman([r['curvature'] for r in rows], [r['held_out_error'] for r in rows])})
    result = {'split_seed': seed, 'model_rows': model_rows}
    write_json(output_dir / 'curvature_validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--cache-dir', type=Path, default=Path('outputs/cache/embeddings'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/analysis'))
    parser.add_argument('--ks', type=int, nargs='+', default=[5,10,20,50])
    parser.add_argument('--tangent-dims', type=int, nargs='+', default=[1,2,3])
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    run_curvature_validation(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
