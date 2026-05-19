from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Literal


OptimizationMethod = Literal["random", "optuna_tpe"]


@dataclass
class GeometryScales:
    density_scale: float
    boundary_scale: float


@dataclass
class TunedCFConfig:
    step_size: float
    trust_radius: float
    shift_weight: float
    knn_weight: float
    max_steps: int
    density_scale: float
    boundary_scale: float
    optimizer_name: str = "sgd"
    prior_scaling: str = "geometry"


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0.0 or high <= 0.0:
        raise ValueError("Log-uniform bounds must be strictly positive.")
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _sample_geometry_aware_config(
    scales: GeometryScales,
    rng: random.Random,
    max_steps: int = 500,
) -> TunedCFConfig:
    density_scale = max(scales.density_scale, 1e-8)
    boundary_scale = max(scales.boundary_scale, 1e-8)
    return TunedCFConfig(
        step_size=_log_uniform(rng, 0.1 * density_scale, 1.0 * density_scale),
        trust_radius=_log_uniform(rng, 0.1 * boundary_scale, 10.0 * boundary_scale),
        shift_weight=_log_uniform(rng, 0.1 / density_scale, 10.0 / density_scale),
        knn_weight=_log_uniform(rng, 0.1 / (density_scale**2), 10.0 / (density_scale**2)),
        max_steps=max_steps,
        density_scale=scales.density_scale,
        boundary_scale=scales.boundary_scale,
        prior_scaling="geometry",
    )


def _sample_absolute_config(
    rng: random.Random,
    max_steps: int = 500,
) -> TunedCFConfig:
    return TunedCFConfig(
        step_size=_log_uniform(rng, 1e-4, 1.0),
        trust_radius=_log_uniform(rng, 1e-2, 100.0),
        shift_weight=_log_uniform(rng, 1e-3, 100.0),
        knn_weight=_log_uniform(rng, 1e-4, 100.0),
        max_steps=max_steps,
        density_scale=1.0,
        boundary_scale=1.0,
        prior_scaling="none",
    )


def counterfactual_validation_objective(
    summary: dict[str, float],
    density_scale: float,
    failure_penalty: float = 5.0,
) -> float:
    density_scale = max(density_scale, 1e-8)
    failure_rate = 1.0 - float(summary.get("counterfactual_success_mean", 0.0))
    normalized_distance = float(summary.get("counterfactual_distance_mean", 0.0)) / density_scale
    normalized_support_radius = float(summary.get("target_support_radius_mean", summary.get("counterfactual_density_mean", 0.0))) / density_scale
    return failure_penalty * failure_rate + normalized_distance + normalized_support_radius


def _optimize_counterfactual_config_random(
    *,
    prior_scaling: str,
    scales: GeometryScales,
    evaluator,
    num_trials: int,
    seed: int,
    max_steps: int = 500,
) -> dict[str, Any]:
    rng = random.Random(seed)
    trials: list[dict[str, Any]] = []
    best_config: TunedCFConfig | None = None
    best_summary: dict[str, float] | None = None
    best_score = float("inf")

    for trial_index in range(num_trials):
        if prior_scaling == "geometry":
            config = _sample_geometry_aware_config(scales=scales, rng=rng, max_steps=max_steps)
        elif prior_scaling == "none":
            config = _sample_absolute_config(rng=rng, max_steps=max_steps)
        else:
            raise ValueError(f"Unsupported prior_scaling: {prior_scaling}")
        summary = evaluator(config)
        score = counterfactual_validation_objective(summary=summary, density_scale=scales.density_scale)
        trials.append({
            "trial_index": trial_index,
            "objective": score,
            "config": asdict(config),
            "summary": summary,
        })
        if score < best_score:
            best_score = score
            best_config = config
            best_summary = summary

    if best_config is None or best_summary is None:
        raise RuntimeError("Random search did not produce any trial results.")

    return {
        "search_method": "random_search",
        "prior_scaling": prior_scaling,
        "num_trials": num_trials,
        "best_objective": best_score,
        "best_config": asdict(best_config),
        "best_summary": best_summary,
        "trials": trials,
    }


def _sample_optuna_config(
    trial,
    prior_scaling: str,
    scales: GeometryScales,
    max_steps: int,
) -> TunedCFConfig:
    density_scale = max(scales.density_scale, 1e-8)
    boundary_scale = max(scales.boundary_scale, 1e-8)
    if prior_scaling == "geometry":
        return TunedCFConfig(
            step_size=trial.suggest_float("step_size", 0.1 * density_scale, 1.0 * density_scale, log=True),
            trust_radius=trial.suggest_float("trust_radius", 0.1 * boundary_scale, 10.0 * boundary_scale, log=True),
            shift_weight=trial.suggest_float("shift_weight", 0.1 / density_scale, 10.0 / density_scale, log=True),
            knn_weight=trial.suggest_float("knn_weight", 0.1 / (density_scale**2), 10.0 / (density_scale**2), log=True),
            max_steps=max_steps,
            density_scale=scales.density_scale,
            boundary_scale=scales.boundary_scale,
            prior_scaling="geometry",
        )
    if prior_scaling == "none":
        return TunedCFConfig(
            step_size=trial.suggest_float("step_size", 1e-4, 1.0, log=True),
            trust_radius=trial.suggest_float("trust_radius", 1e-2, 100.0, log=True),
            shift_weight=trial.suggest_float("shift_weight", 1e-3, 100.0, log=True),
            knn_weight=trial.suggest_float("knn_weight", 1e-4, 100.0, log=True),
            max_steps=max_steps,
            density_scale=1.0,
            boundary_scale=1.0,
            prior_scaling="none",
        )
    raise ValueError(f"Unsupported prior_scaling: {prior_scaling}")


def _optimize_counterfactual_config_optuna_tpe(
    *,
    prior_scaling: str,
    scales: GeometryScales,
    evaluator,
    num_trials: int,
    seed: int,
    max_steps: int = 500,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for search_method='optuna_tpe'. Install optuna first.") from exc

    trial_records: list[dict[str, Any]] = []

    def objective(trial) -> float:
        config = _sample_optuna_config(trial=trial, prior_scaling=prior_scaling, scales=scales, max_steps=max_steps)
        summary = evaluator(config)
        score = counterfactual_validation_objective(summary=summary, density_scale=scales.density_scale)
        trial.set_user_attr("config", asdict(config))
        trial.set_user_attr("summary", summary)
        trial_records.append({
            "trial_index": trial.number,
            "objective": score,
            "config": asdict(config),
            "summary": summary,
        })
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

    best_trial = study.best_trial
    return {
        "search_method": "optuna_tpe",
        "prior_scaling": prior_scaling,
        "num_trials": num_trials,
        "best_objective": float(best_trial.value),
        "best_config": best_trial.user_attrs["config"],
        "best_summary": best_trial.user_attrs["summary"],
        "trials": trial_records,
    }


def optimize_counterfactual_config(
    *,
    method: OptimizationMethod,
    prior_scaling: str,
    scales: GeometryScales,
    evaluator,
    num_trials: int,
    seed: int,
    max_steps: int = 500,
) -> dict[str, Any]:
    if method == "random":
        return _optimize_counterfactual_config_random(
            prior_scaling=prior_scaling,
            scales=scales,
            evaluator=evaluator,
            num_trials=num_trials,
            seed=seed,
            max_steps=max_steps,
        )
    if method == "optuna_tpe":
        return _optimize_counterfactual_config_optuna_tpe(
            prior_scaling=prior_scaling,
            scales=scales,
            evaluator=evaluator,
            num_trials=num_trials,
            seed=seed,
            max_steps=max_steps,
        )
    raise ValueError(f"Unsupported optimization method: {method}")


def tune_counterfactual_config(
    *,
    search_method: OptimizationMethod,
    prior_scaling: str,
    scales: GeometryScales,
    evaluator,
    num_trials: int,
    seed: int,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Backward-compatible alias for optimize_counterfactual_config."""
    return optimize_counterfactual_config(
        method=search_method,
        prior_scaling=prior_scaling,
        scales=scales,
        evaluator=evaluator,
        num_trials=num_trials,
        seed=seed,
        max_steps=max_steps,
    )
