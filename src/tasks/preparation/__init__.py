"""Original protein/small-molecule geometry with modern PyTorch predictions."""

from .pipeline import PreparationTask


def build_task(config):
    """Construct the staged, externally configured raw structure pipeline."""
    return PreparationTask(config)
