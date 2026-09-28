from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from ..core.utils import json_safe, run_provenance, set_seed, write_json


def prepare_cells(adata, *, counts_layer, label_col, low_label, high_label,
                  split_col, group_cols, gene_name_col, reference_split,
                  num_genes, target_sum):
    counts = sparse.csr_matrix(adata.X if counts_layer is None else adata.layers[counts_layer], dtype=np.float32)
    counts.sum_duplicates()
    values = counts.data
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.round(values)):
        raise ValueError('Extraction requires raw nonnegative counts, not normalized expression')
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if (totals == 0).any() or not adata.obs_names.is_unique:
        raise ValueError('Cells must have positive total counts and unique IDs')
    metadata = adata.obs[[label_col, split_col, *group_cols]]
    if metadata.isna().any().any():
        raise ValueError('Labels, splits, and sample/perturbation groups must be supplied for every cell')
    mapping = {str(low_label): 0, str(high_label): 1}
    labels = metadata[label_col].astype(str).map(mapping)
    if low_label == high_label or labels.isna().any():
        raise ValueError('Response labels must match the distinct low/high labels supplied')
    splits = metadata[split_col].astype(str).to_numpy()
    if not {'train', 'val', 'test'} <= set(splits) or not set(splits) <= {'train', 'val', 'test', 'observed'}:
        raise ValueError('Split column must contain train, val, test, and optionally observed')
    gene_names = adata.var_names if gene_name_col == 'index' else adata.var[gene_name_col]
    if pd.isna(gene_names).any():
        raise ValueError('Gene names must be present')
    gene_names = list(map(str, gene_names))
    if len(set(gene_names)) != len(gene_names):
        raise ValueError('Gene names must be unique so decoder columns remain identifiable')
    if not 1 <= num_genes <= counts.shape[1] or target_sum <= 0:
        raise ValueError('Select an available positive gene count and normalization target')

    expression = counts.multiply((target_sum / totals)[:, None]).tocsr()
    expression.data = np.log1p(expression.data)
    reference = expression[splits == reference_split]
    # Decoder gene selection uses reference cells, never test-cell variance.
    variance = np.asarray(reference.power(2).mean(axis=0) - np.square(reference.mean(axis=0))).ravel()
    genes = np.argsort(-variance, kind='stable')[:num_genes]
    groups = [json.dumps(list(map(str, row))) for row in metadata[group_cols].itertuples(index=False, name=None)]
    return counts, {
        'labels': torch.tensor(labels.to_numpy(), dtype=torch.long),
        'example_ids': list(map(str, adata.obs_names)),
        'groups': groups,
        'gene_names': [gene_names[i] for i in genes],
        'expression': expression[:, genes],
        'expression_scale': 'log_normalized',
        'splits': splits,
    }


def extract_scgpt(counts, obs, var, model_dir, gene_name_col, batch_size, max_length, device):
    from anndata import AnnData
    from scgpt.tasks import embed_data

    data = AnnData(X=counts, obs=obs.copy(), var=var.copy())
    result = embed_data(data, model_dir, gene_col=gene_name_col, batch_size=batch_size,
                        max_length=max_length, device=device, use_fast_transformer=False,
                        return_new_adata=False)
    positions = result.obs_names.get_indexer(obs.index)
    if (positions < 0).any() or len(result) != len(obs):
        raise ValueError('scGPT output does not contain exactly the requested cells')
    return np.asarray(result.obsm['X_scGPT'][positions], dtype=np.float32)


def align_geneformer_embeddings(frame, cell_ids):
    frame = frame.set_index('cell_id', verify_integrity=True)
    frame.index = frame.index.astype(str)
    if set(frame.index) != set(cell_ids):
        raise ValueError('Geneformer dropped or added cells; check counts and Ensembl vocabulary coverage')
    return frame.loc[cell_ids].to_numpy(dtype=np.float32)


def extract_geneformer(counts, obs, var, model_dir, ensembl_col, model_version,
                       emb_layer, batch_size, work_dir):
    from anndata import AnnData
    from geneformer import TranscriptomeTokenizer, EmbExtractor

    ensembl_ids = var.index if ensembl_col == 'index' else var[ensembl_col]
    if pd.isna(ensembl_ids).any():
        raise ValueError('Geneformer requires Ensembl IDs for every input gene')
    cells = pd.DataFrame({'cell_id': obs.index.astype(str),
                          'n_counts': np.asarray(counts.sum(axis=1)).ravel()}, index=obs.index)
    genes = pd.DataFrame({'ensembl_id': list(map(str, ensembl_ids))}, index=var.index)
    data = AnnData(X=counts, obs=cells, var=genes)
    # The tokenizer needs all measured genes, not the decoder's selected subset.
    with TemporaryDirectory(prefix='geneformer-', dir=work_dir) as tmp:
        root = Path(tmp)
        inputs, tokens, outputs = root / 'input', root / 'tokens', root / 'embeddings'
        for directory in (inputs, tokens, outputs):
            directory.mkdir()
        data.write_h5ad(inputs / 'cells.h5ad')
        tokenizer = TranscriptomeTokenizer({'cell_id': 'cell_id'}, nproc=1, model_version=model_version)
        tokenizer.tokenize_data(str(inputs), str(tokens), 'cells', file_format='h5ad')
        extractor = EmbExtractor(model_type='Pretrained', emb_mode='cell',
                                 max_ncells=None, emb_layer=emb_layer, emb_label=['cell_id'],
                                 forward_batch_size=batch_size, nproc=1, model_version=model_version)
        frame = extractor.extract_embs(str(model_dir), str(tokens / 'cells.dataset'), str(outputs), 'cells')
    return align_geneformer_embeddings(frame, list(obs.index.astype(str)))


def save_splits(embeddings, prepared, output_dir, metadata):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(prepared['example_ids']) or not torch.isfinite(embeddings).all():
        raise ValueError('Encoder output must contain one finite embedding per input cell')
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ('train', 'val', 'test', 'observed'):
        indices = np.flatnonzero(prepared['splits'] == split)
        if not len(indices):
            continue
        payload = {
            'embeddings': embeddings[indices], 'labels': prepared['labels'][indices],
            'example_ids': [prepared['example_ids'][i] for i in indices],
            'groups': [prepared['groups'][i] for i in indices],
            'gene_names': prepared['gene_names'],
            'expression': torch.from_numpy(prepared['expression'][indices].toarray()),
            'expression_scale': prepared['expression_scale'], 'extraction': metadata,
        }
        path = output_dir / f'{split}.pt'
        torch.save(payload, path)
        paths[split] = str(path.resolve())
    write_json(output_dir / 'extraction.json', {'splits': paths, **metadata})
    return paths


def run_extraction(args):
    import anndata

    set_seed(args.seed)
    adata = anndata.read_h5ad(args.input)
    counts, prepared = prepare_cells(
        adata, counts_layer=args.counts_layer, label_col=args.label_col,
        low_label=args.low_label, high_label=args.high_label, split_col=args.split_col,
        group_cols=args.group_cols, gene_name_col=args.gene_name_col,
        reference_split=args.reference_split, num_genes=args.num_genes, target_sum=args.target_sum)
    if args.encoder == 'scgpt':
        embeddings = extract_scgpt(counts, adata.obs, adata.var, args.model_dir, args.gene_name_col,
                                   args.batch_size, args.max_length, args.device)
    else:
        embeddings = extract_geneformer(counts, adata.obs, adata.var, args.model_dir, args.ensembl_col,
                                        args.geneformer_version, args.emb_layer, args.batch_size, args.work_dir)
    packages = {}
    for name in ('torch', 'anndata', 'scipy', 'transformers', args.encoder):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    metadata = run_provenance([args.input, args.model_dir], json_safe(vars(args)))
    metadata['packages'] = packages
    metadata['decoder_gene_reference_split'] = args.reference_split
    return save_splits(embeddings, prepared, args.output_dir / args.encoder, metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--encoder', choices=['scgpt', 'geneformer'], required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--counts-layer', help='Raw-count layer; omit to use X')
    parser.add_argument('--label-col', required=True)
    parser.add_argument('--low-label', required=True)
    parser.add_argument('--high-label', required=True)
    parser.add_argument('--split-col', required=True)
    parser.add_argument('--group-cols', nargs='+', required=True)
    parser.add_argument('--gene-name-col', required=True, help='Gene symbols column, or index')
    parser.add_argument('--reference-split', choices=['train', 'val'], default='val')
    parser.add_argument('--num-genes', type=int, default=4000)
    parser.add_argument('--target-sum', type=float, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=1200, help='scGPT token limit')
    parser.add_argument('--device', default='cuda', help='scGPT device; Geneformer uses CUDA')
    parser.add_argument('--ensembl-col', help='Geneformer Ensembl ID column, or index')
    parser.add_argument('--geneformer-version', choices=['V1', 'V2'])
    parser.add_argument('--emb-layer', type=int, choices=[-1, 0], default=-1,
                        help='Geneformer: -1 penultimate layer, 0 final layer')
    parser.add_argument('--work-dir', type=Path, help='Existing scratch directory for Geneformer intermediates')
    args = parser.parse_args()
    if args.encoder == 'geneformer':
        if not args.ensembl_col or not args.geneformer_version:
            parser.error('Geneformer requires --ensembl-col and --geneformer-version matching the checkpoint')
        if args.device != 'cuda' or not torch.cuda.is_available():
            parser.error('The official Geneformer embedding extractor requires a CUDA job')
    print(json.dumps(run_extraction(args), indent=2))


if __name__ == '__main__':
    main()
