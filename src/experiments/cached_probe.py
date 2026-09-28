from __future__ import annotations
import argparse
from pathlib import Path
import torch
from ..core.classifier import build_classifier, train_linear_probe
from ..core.utils import set_seed
from ..core.utils import run_provenance, write_json
from ..counterfactuals.evaluation import evaluate_embeddings


def load_splits(paths):
    splits = {name: torch.load(path, map_location='cpu') for name,path in paths.items()}
    if set(splits) != {'train','val','test'}:
        raise ValueError('Provide train, val and test splits')
    seen = set()
    dimension = None
    for name, split in splits.items():
        z, y = split['embeddings'], split['labels']
        ids = split.get('example_ids')
        if z.ndim != 2 or y.ndim != 1 or len(z) != len(y) or not len(z) or not torch.isfinite(z).all():
            raise ValueError(f'Invalid embeddings/labels in {name}')
        if ids is None or len(ids) != len(z) or len(set(map(str,ids))) != len(z):
            raise ValueError(f'{name} needs unique example_ids aligned with embedding rows')
        if seen & set(map(str,ids)):
            raise ValueError('Train, validation and test example IDs overlap')
        seen.update(map(str,ids))
        if dimension is not None and dimension != z.shape[1]:
            raise ValueError('Embedding dimensions differ across splits')
        dimension = z.shape[1]
        split['embeddings'], split['labels'] = z.float(), y.long()
    classes = splits['train']['labels'].unique().sort().values
    if not torch.equal(classes, torch.arange(len(classes))) or len(classes) < 2:
        raise ValueError('Training classes must be contiguous integers starting at zero')
    for split in splits.values():
        if not torch.isin(split['labels'], classes).all():
            raise ValueError('Evaluation labels contain an unseen class')
    return splits


def run_cached_probe(*, train, val, test, dataset, encoder, output, seed=42, step_size=.01,
                     trust_radius=1., shift_weight=0., tangent_dim=2, k=20, max_steps=300,
                     probe_epochs=100, probe_lr=.001, probe_weight_decay=.0001, max_examples=None):
    paths = {'train':train,'val':val,'test':test}
    splits = load_splits(paths)
    set_seed(seed)
    head = build_classifier(splits['train']['embeddings'].shape[1], len(splits['train']['labels'].unique()))
    head, training = train_linear_probe(head, splits['train']['embeddings'], splits['train']['labels'],
                                       splits['val']['embeddings'], splits['val']['labels'],
                                       epochs=probe_epochs, lr=probe_lr, weight_decay=probe_weight_decay)
    checkpoint = Path(output).with_suffix('.pt')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'classifier_state_dict':head.state_dict(), 'input_dim':head.input_dim,
                'num_classes':head.num_classes, 'metadata':{'seed':seed}}, checkpoint)
    config = dict(k=k, step_size=step_size, trust_radius=trust_radius, max_steps=max_steps,
                  shift_weight=shift_weight, tangent_dim=tangent_dim)
    rows, summary = evaluate_embeddings(splits['test']['embeddings'], head, splits['test']['labels'],
                                       splits['val']['embeddings'], splits['val']['labels'],
                                       max_examples=max_examples, record_endpoint=True, **config)
    metrics = {}
    with torch.no_grad():
        for name, split in splits.items():
            logits = head(split['embeddings'])
            metrics[name+'_accuracy'] = float((logits.argmax(1)==split['labels']).float().mean())
            metrics[name+'_ce'] = float(torch.nn.functional.cross_entropy(logits, split['labels']))
    for row in rows:
        row['example_id'] = str(splits['test']['example_ids'][row['example_index']])
    result = {**summary, **metrics, **config, 'dataset':dataset,
              'encoder':encoder, 'seed':seed, 'eval_split':'test', 'reference_split':'val',
              'counterfactual_mode':'targeted', 'target_strategy':'second_best',
              'split_paths':{n:str(Path(p).resolve()) for n,p in paths.items()},
              'probe_checkpoint':str(checkpoint.resolve()), 'training':training, 'raw_results':rows,
              'provenance':run_provenance(paths.values(), {**config, 'seed':seed, 'probe_epochs':probe_epochs,
                                                         'probe_lr':probe_lr, 'probe_weight_decay':probe_weight_decay})}
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ['train','val','test','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--encoder', required=True)
    for name, default in [('seed',42), ('k',20), ('tangent-dim',2), ('max-steps',300), ('probe-epochs',100)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('step-size',.01), ('trust-radius',1.), ('shift-weight',0.), ('probe-lr',.001), ('probe-weight-decay',.0001)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--max-examples', type=int)
    run_cached_probe(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
