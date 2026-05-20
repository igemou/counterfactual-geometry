from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from ..analysis.common import load_cached_split, load_json, model_label, split_cache_path


def _projection(
    embeddings: np.ndarray,
    method: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    perplexity: float,
    learning_rate: float | str,
) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            perplexity=perplexity,
            learning_rate=learning_rate,
        )
        return reducer.fit_transform(embeddings)
    if method == "umap":
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        return reducer.fit_transform(embeddings)
    raise ValueError(f"Unsupported method: {method}")


def _example_success_map(payload: dict[str, Any]) -> dict[int, bool]:
    success_by_index: dict[int, bool] = {}
    for fallback_index, row in enumerate(payload.get("raw_results", [])):
        if not isinstance(row, dict):
            continue
        example_index = int(row.get("example_index", fallback_index))
        success_by_index[example_index] = bool(row.get("counterfactual_success", False))
    return success_by_index


def plot_embedding_projection(
    payload: dict[str, Any],
    cache_dir: Path,
    output_path: Path,
    *,
    split: str | None = None,
    method: str = "pca",
    seed: int = 0,
    max_points: int | None = 3000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    highlight_success: bool = True,
) -> dict[str, Any]:
    projection_split = split or str(payload.get("eval_split", "test"))
    embeddings, labels = load_cached_split(split_cache_path(payload, cache_dir, projection_split))

    if max_points is not None and embeddings.size(0) > max_points:
        indices = np.linspace(0, embeddings.size(0) - 1, num=max_points, dtype=int)
        embeddings = embeddings[indices]
        labels = labels[indices]
    else:
        indices = np.arange(embeddings.size(0), dtype=int)

    projected = _projection(
        embeddings.numpy(),
        method=method,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        perplexity=perplexity,
        learning_rate=learning_rate,
    )

    label_values = labels.numpy()
    unique_labels = sorted(np.unique(label_values).tolist())
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    success_by_index = _example_success_map(payload) if highlight_success else {}

    figure, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    for color_index, label in enumerate(unique_labels):
        mask = label_values == label
        axis.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=12,
            alpha=0.65,
            color=cmap(color_index),
            label=str(label),
            linewidths=0.0,
        )

    if highlight_success:
        success_mask = np.asarray([success_by_index.get(int(index), False) for index in indices], dtype=bool)
        if success_mask.any():
            axis.scatter(
                projected[success_mask, 0],
                projected[success_mask, 1],
                s=26,
                facecolors="none",
                edgecolors="black",
                linewidths=0.7,
                label="successful CF",
            )

    method_label = "t-SNE" if method == "tsne" else method.upper()
    axis.set_title(f"{model_label(payload)} {projection_split} embeddings ({method_label})")
    axis.set_xlabel(f"{method_label} 1")
    axis.set_ylabel(f"{method_label} 2")
    axis.legend(loc="best", fontsize=8, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return {
        "output": str(output_path),
        "dataset": str(payload.get("dataset", "")),
        "model": model_label(payload),
        "split": projection_split,
        "method": method,
        "num_points": int(projected.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cached embedding spaces with PCA, t-SNE, or UMAP.")
    parser.add_argument("--input", type=Path, required=True, help="Experiment JSON payload.")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/embeddings"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--no-highlight-success", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.input)
    result = plot_embedding_projection(
        payload,
        args.cache_dir,
        args.output,
        split=args.split,
        method=args.method,
        seed=args.seed,
        max_points=args.max_points,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        perplexity=args.perplexity,
        learning_rate="auto" if args.learning_rate is None else args.learning_rate,
        highlight_success=not args.no_highlight_success,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
