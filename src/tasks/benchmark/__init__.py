"""Public small-decoy MaSIF-neosurf benchmark task."""

from .workflow import BenchmarkTask


def build_task(config: dict) -> BenchmarkTask:
    """Build a complete benchmark from explicit dataset and protocol locations."""
    return BenchmarkTask(config)
