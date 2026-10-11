"""MaSIF-neosurface protein/ligand design application."""

from .workflow import NeoSurfaceWorkflow


def build_task(config):
    """Construct neosurface search or already-aligned complex scoring."""
    return NeoSurfaceWorkflow(config)
