"""API for shared core components with lazy imports."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "LinearClassifier": (".classifier", "LinearClassifier"),
    "build_classifier": (".classifier", "build_classifier"),
    "train_linear_probe": (".classifier", "train_linear_probe"),
    "build_datamodule": (".datasets", "build_datamodule"),
    "MULTIMODAL_ENCODERS": (".encoders", "MULTIMODAL_ENCODERS"),
    "TEXT_ENCODERS": (".encoders", "TEXT_ENCODERS"),
    "VISION_ENCODERS": (".encoders", "VISION_ENCODERS"),
    "build_encoder": (".encoders", "build_encoder"),
    "build_processor": (".encoders", "build_processor"),
    "encode_batch": (".encoders", "encode_batch"),
    "freeze_encoder": (".encoders", "freeze_encoder"),
    "unpack_batch": (".encoders", "unpack_batch"),
    "approx_boundary_distance": (".geometry", "approx_boundary_distance"),
    "choose_target_label": (".geometry", "choose_target_label"),
    "class_knn_radius": (".geometry", "class_knn_radius"),
    "dataset_density_scale": (".geometry", "dataset_density_scale"),
    "decision_margin": (".geometry", "decision_margin"),
    "estimate_local_geometry": (".geometry", "estimate_local_geometry"),
    "logit_gap": (".geometry", "logit_gap"),
    "project_to_l2_ball": (".geometry", "project_to_l2_ball"),
    "untargeted_decision_margin": (".geometry", "untargeted_decision_margin"),
    "data_root": (".utils", "data_root"),
    "embedding_cache_dir": (".utils", "embedding_cache_dir"),
    "embedding_cache_path": (".utils", "embedding_cache_path"),
    "ensure_2d": (".utils", "ensure_2d"),
    "hf_cache_root": (".utils", "hf_cache_root"),
    "l2_distance": (".utils", "l2_distance"),
    "load_probe": (".utils", "load_probe"),
    "load_probe_checkpoint": (".utils", "load_probe_checkpoint"),
    "mean_std": (".utils", "mean_std"),
    "project_root": (".utils", "project_root"),
    "set_seed": (".utils", "set_seed"),
    "to_device": (".utils", "to_device"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
