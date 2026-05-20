"""Figure-generation utilities aligned with the paper figures and case studies."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "extract_case_study": (".case_studies", "extract_case_study"),
    "plot_embedding_projection": (".embedding_plots", "plot_embedding_projection"),
    "select_case_studies": (".case_studies", "select_case_studies"),
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
