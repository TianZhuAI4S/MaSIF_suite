"""Public MaSIF API; all objects retain their canonical checkpoint module names."""

import masif_torch as _implementation

__version__ = _implementation.__version__
__all__ = _implementation.__all__


def __getattr__(name):
    """Return the original implementation object, preserving type and state identity."""
    if name not in __all__:
        raise AttributeError(name)
    return getattr(_implementation, name)
