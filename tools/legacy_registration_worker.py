"""Compatibility entry point for the packaged Open3D0.5 worker (Python3.6)."""

from pathlib import Path
import runpy


if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/tasks/benchmark/worker_registration.py'), run_name='__main__')
