"""External Rosetta refinement with the original neosurface protocols."""

from .workflow import RosettaWorkflow


def build_task(config: dict) -> RosettaWorkflow:
    """Construct a configured ligand-preserving Rosetta refinement task."""
    return RosettaWorkflow(config)
