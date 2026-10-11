"""Compatibility entry point for the task-owned Rosetta preparation worker."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/tasks/rosetta/worker.py"), run_name="__main__")
