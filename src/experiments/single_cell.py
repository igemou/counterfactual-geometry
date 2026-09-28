from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..analysis.common import spearman
from ..core.utils import write_json, run_provenance
from ..core.utils import load_probe, set_seed
from ..counterfactuals.search import build_baseline_config
from .geometry_scaling_ablation import evaluate_matched_variants


class ExpressionDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, z):
        return self.layers(z)


def train_decoder(embeddings, expression, *, seed=0, hidden_dim=256, epochs=100, batch_size=128,
                  lr=.001, patience=10):
    if len(embeddings) < 5 or len(expression) != len(embeddings):
        raise ValueError('Decoder needs at least five aligned reference cells')
    set_seed(seed)
    order = torch.randperm(len(embeddings))
    n_val = max(1, len(order)//5)
    val, train = order[:n_val], order[n_val:]
    decoder = ExpressionDecoder(embeddings.shape[1], expression.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    best, best_state, stale = float('inf'), None, 0
    for epoch in range(epochs):
        decoder.train()
        for indices in train[torch.randperm(len(train))].split(batch_size):
            loss = nn.functional.mse_loss(decoder(embeddings[indices]), expression[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        decoder.eval()
        with torch.no_grad():
            loss = float(nn.functional.mse_loss(decoder(embeddings[val]), expression[val]))
        if loss < best:
            best, best_state, stale = loss, deepcopy(decoder.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError('Decoder training produced no finite validation loss')
    decoder.load_state_dict(best_state)
    decoder.eval().requires_grad_(False)
    return decoder, {'validation_mse':best, 'train_indices':train.tolist(), 'validation_indices':val.tolist(),
                     'epochs_run':epoch+1, 'seed':seed, 'hidden_dim':hidden_dim, 'lr':lr}


def matched_observed_change(expression, labels, groups, starting_groups):
    """Weight group-specific high-minus-low means by eligible starting-cell counts."""
    if len(groups) != len(expression) or len(labels) != len(expression):
        raise ValueError('Observed cells, labels and groups must align')
    groups = [str(g) for g in groups]
    starts = [str(g) for g in starting_groups]
    deltas, weights = [], []
    eligible_groups = set()
    for group in sorted(set(starts)):
        low = torch.tensor([g == group and int(y)==0 for g,y in zip(groups,labels)])
        high = torch.tensor([g == group and int(y)==1 for g,y in zip(groups,labels)])
        if low.any() and high.any():
            deltas.append(expression[high].mean(0)-expression[low].mean(0))
            weights.append(starts.count(group))
            eligible_groups.add(group)
    if not deltas:
        raise ValueError('No starting-cell group has both observed low- and high-response cells')
    weights = torch.tensor(weights, dtype=expression.dtype)
    delta = (torch.stack(deltas)*weights[:,None]).sum(0)/weights.sum()
    return delta, torch.tensor([g in eligible_groups for g in starts])


def gene_agreement(predicted, observed, top_k=100):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError('Gene-change vectors must be finite and aligned')
    n = min(top_k, len(predicted))
    def recovery(sign):
        observed_ids = np.flatnonzero(sign * observed > 0)
        predicted_ids = np.flatnonzero(sign * predicted > 0)
        observed_top = set(observed_ids[np.argsort(-sign * observed[observed_ids])[:n]])
        predicted_top = set(predicted_ids[np.argsort(-sign * predicted[predicted_ids])[:n]])
        return len(observed_top & predicted_top) / len(observed_top) if observed_top else float('nan')

    return {'pearson':float(np.corrcoef(predicted, observed)[0,1]) if predicted.std() and observed.std() else float('nan'),
            'spearman':spearman(predicted.tolist(), observed.tolist()),
            'direction_agreement':float(np.mean(np.sign(predicted)==np.sign(observed))),
            'top_k':n, 'upregulated_recovery':recovery(1), 'downregulated_recovery':recovery(-1)}


def _validate_cells(cells, name):
    for key in ['embeddings','labels','expression','gene_names','example_ids','groups']:
        if key not in cells:
            raise ValueError(f'{name} missing {key}')
    n = len(cells['embeddings'])
    if any(len(cells[key]) != n for key in ['labels','expression','example_ids','groups']):
        raise ValueError(f'{name} rows are not aligned')
    if len(set(map(str,cells['example_ids']))) != n:
        raise ValueError(f'{name} has duplicate cell IDs')
    if cells.get('expression_scale') != 'log_normalized':
        raise ValueError('Declare expression_scale=log_normalized after actual preprocessing')
    if cells['expression'].ndim != 2 or len(cells['gene_names']) != cells['expression'].shape[1]:
        raise ValueError('Expression columns must align with gene_names')
    if not torch.isfinite(cells['expression']).all() or not torch.isfinite(cells['embeddings']).all():
        raise ValueError('Nonfinite expression/embeddings')
    if not set(cells['labels'].tolist()) <= {0,1}:
        raise ValueError('Labels must encode low=0 and high=1')


def run_single_cell(*, reference, validation, test, observed, probe_checkpoint, output_dir,
                    step_size, trust_radius, shift_weight, knn_weight, curvature_alpha,
                    tangent_dim, k=20, max_steps=300, num_genes=4000, seed=0,
                    decoder_epochs=100, max_examples=None):
    ref = torch.load(reference, map_location='cpu')
    cells = torch.load(test, map_location='cpu')
    obs = torch.load(observed, map_location='cpu')
    validation_cells = torch.load(validation, map_location='cpu')
    if set(map(str,validation_cells['example_ids'])) & set(map(str,cells['example_ids'])):
        raise ValueError('Curvature validation cells overlap test cells')
    _validate_cells(ref, 'reference')
    _validate_cells(cells, 'test')
    _validate_cells(obs, 'observed')
    if set(map(str,ref['example_ids'])) & set(map(str,cells['example_ids'])):
        raise ValueError('Decoder reference cells overlap test cells')
    if list(ref['gene_names']) != list(cells['gene_names']) or list(ref['gene_names']) != list(obs['gene_names']):
        raise ValueError('Gene columns must have identical names and order')
    if not 1 <= num_genes <= ref['expression'].shape[1]:
        raise ValueError('Requested number of genes unavailable')
    # Select genes on reference cells only; test expression never fits or selects the decoder.
    genes = ref['expression'].var(0).topk(num_genes).indices
    decoder, decoder_stats = train_decoder(ref['embeddings'].float(), ref['expression'][:,genes].float(),
                                           seed=seed, epochs=decoder_epochs)
    probe, _ = load_probe(probe_checkpoint, map_location='cpu')
    probe.eval()
    with torch.no_grad():
        predicted = probe(cells['embeddings'].float()).argmax(1)
    starts = torch.where((cells['labels']==0) & (predicted==0))[0]
    if max_examples is not None:
        starts = starts[:max_examples]
    observed_delta, eligible = matched_observed_change(obs['expression'][:,genes], obs['labels'], obs['groups'],
                                                       [cells['groups'][int(i)] for i in starts])
    starts = starts[eligible]
    if not len(starts):
        raise ValueError('No eligible low-response starts')
    config = build_baseline_config(step_size, trust_radius, max_steps, shift_weight=shift_weight, tangent_dim=tangent_dim)
    variants = evaluate_matched_variants(probe, cells['embeddings'][starts].float(), cells['labels'][starts],
                                        ref['embeddings'].float(), ref['labels'], validation_cells['embeddings'].float(), config,
                                        k=k, knn_weight=knn_weight, curvature_alpha=curvature_alpha,
                                        validation_is_reference=Path(validation).resolve()==Path(reference).resolve(), example_indices=starts,
                                        target_label=1, record_trajectory=True)
    for name, variant in variants.items():
        endpoints = torch.tensor([r['final_embedding'] for r in variant['raw_results']])
        with torch.no_grad():
            changes = decoder(endpoints)-decoder(cells['embeddings'][starts].float())
        delta = changes.mean(0)
        variant['mean_decoded_change'] = delta.tolist()
        variant['gene_agreement'] = gene_agreement(delta.numpy(), observed_delta.numpy())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':decoder.state_dict(), 'input_dim':ref['embeddings'].shape[1],
                'gene_indices':genes, 'training':decoder_stats}, output_dir/'expression_decoder.pt')
    result = {'dataset':'papalexi', 'num_genes':num_genes,
              'gene_names':[str(ref['gene_names'][int(i)]) for i in genes],
              'starting_cell_ids':[str(cells['example_ids'][int(i)]) for i in starts],
              'observed_change':observed_delta.tolist(), 'decoder_training':decoder_stats, 'variants':variants,
              'evaluation_population':'same eligible low-response starts for every variant, including unsuccessful searches',
              'provenance':run_provenance([reference,validation,test,observed,probe_checkpoint],
                                         dict(step_size=step_size,trust_radius=trust_radius,shift_weight=shift_weight,
                                              knn_weight=knn_weight,curvature_alpha=curvature_alpha,tangent_dim=tangent_dim,
                                              k=k,max_steps=max_steps,num_genes=num_genes,seed=seed))}
    write_json(output_dir/'single_cell.json', result)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['reference','validation','test','observed','probe-checkpoint','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['step-size','trust-radius','shift-weight','knn-weight','curvature-alpha']:
        p.add_argument('--'+name,type=float,required=True)
    p.add_argument('--tangent-dim',type=int,required=True)
    for name,default in [('k',20),('max-steps',300),('num-genes',4000),('seed',0),('decoder-epochs',100)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--max-examples',type=int)
    run_single_cell(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
