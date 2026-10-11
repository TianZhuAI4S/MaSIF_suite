"""MaSIF networks, imported lazily so runtime setup does not require Torch."""

from importlib import import_module

__version__ = "0.2.0"
__all__ = ["MaSIF_site", "MaSIF_ppi_search", "MaSIF_ligand", "ScoreNN", "AlignmentEvaluationNN", "build_model", "build_task"]


def __getattr__(name):
    """Expose the canonical model/task objects without importing Torch at startup."""
    if name not in __all__:
        raise AttributeError(name)
    module = import_module('.tasks' if name == 'build_task' else '.models', __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
