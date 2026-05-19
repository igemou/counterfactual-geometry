"""Counterfactual search, tuning, and evaluation utilities."""

from .evaluation import evaluate_embeddings, evaluate_single_example, summarize_metrics
from .search import (
    CounterfactualResult,
    build_baseline_config,
    build_geometry_aware_config,
    build_geometry_scales,
    evaluate_counterfactual_embeddings,
    evaluate_counterfactual_example,
    generate_counterfactual,
    target_density,
)
from .tuning import (
    GeometryScales,
    TunedCFConfig,
    optimize_counterfactual_config,
    tune_counterfactual_config,
)

__all__ = [
    "CounterfactualResult",
    "GeometryScales",
    "TunedCFConfig",
    "build_baseline_config",
    "build_geometry_aware_config",
    "build_geometry_scales",
    "evaluate_counterfactual_embeddings",
    "evaluate_counterfactual_example",
    "evaluate_embeddings",
    "evaluate_single_example",
    "generate_counterfactual",
    "optimize_counterfactual_config",
    "summarize_metrics",
    "target_density",
    "tune_counterfactual_config",
]
