"""Search model and task entry point."""

from .model import MaSIF_ppi_search


def build_task(config):
    """Construct this task using the shared prepared-packet execution contract."""
    from ...shared.task import NeuralTask
    return NeuralTask(config)
