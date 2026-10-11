"""MaSIF-seed retrieval and docking application."""

from .workflow import SeedWorkflow


def build_task(config):
    """Construct the seed workflow with composed geometry, scoring and pose providers."""
    return SeedWorkflow(config)
