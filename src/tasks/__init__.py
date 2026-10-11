"""Lazy task registry; each task owns its model or molecular workflow."""

from collections.abc import Callable
from importlib import import_module

_TASKS: dict[str, Callable] = {}
_BUILTINS = {
    "site": "site", "ppi_search": "search", "search": "search", "ligand": "ligand",
    "score": "score", "alignment": "alignment", "seed": "seed", "masif-seed": "seed", "masif_seed": "seed",
    "neosurf": "neosurface", "neosurface": "neosurface", "masif-neosurf": "neosurface", "masif-neosurface": "neosurface",
    "preparation": "preparation", "rosetta": "rosetta", "benchmark": "benchmark",
}


def register_task(name: str, factory: Callable) -> None:
    """Register a user-owned application without replacing existing names."""
    if name in _TASKS or name in _BUILTINS:
        raise ValueError(f"Task already registered: {name}")
    _TASKS[name] = factory


def build_task(config: dict):
    """Lazily import the selected task and dispatch its run contract."""
    name = config["task"]
    if name in _TASKS:
        return _TASKS[name](config)
    if name not in _BUILTINS:
        raise ValueError(f"Unknown task: {name}; available: {sorted(set(_TASKS) | set(_BUILTINS))}")
    config = dict(config)
    if name == "search":
        config["task"] = "ppi_search"
    return import_module(f"{__name__}.{_BUILTINS[name]}").build_task(config)
