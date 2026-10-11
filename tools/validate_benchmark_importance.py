"""Verify deferred importance against original eager evaluation on real captured poses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from masif_torch.configuration import load_config_file
from masif_torch.checkpoints import load_checkpoint
from masif_torch.shared.design.features import AlignmentScorer
from masif_torch.tasks.benchmark.workflow import BenchmarkWorkflow


def same_rng(first: tuple, second: tuple) -> bool:
    """Compare every component of NumPy's legacy MT19937 state exactly."""
    return (first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:])


def validate(config: dict, fixtures: Path) -> dict:
    """Compare eager/deferred probabilities and per-point importance without changing sampling."""
    torch.set_num_threads(1)
    model, _ = load_checkpoint(config['checkpoint'], 'cpu')
    workflow = object.__new__(BenchmarkWorkflow)
    workflow.scorer = AlignmentScorer(model)
    rows = []
    start = time.monotonic()
    for path in sorted(fixtures.glob('site_*.npz')):
        with np.load(path, allow_pickle=False) as packet:
            features = packet['features']
        before = workflow.scorer.rng.get_state()
        eager_probability, eager_importance = model.eval_model(features, 0.9, workflow.scorer.rng)
        eager_after = workflow.scorer.rng.get_state()
        workflow.scorer.rng.set_state(before)
        probability, importance = model.eval_model(features, 1.0, workflow.scorer.rng)
        if np.any(importance):
            raise AssertionError('Rejected diagnostics should omit expensive importance')
        row = dict(features=features, score=float(probability[0, 0]), point_importance=importance)
        workflow._finalize_score(row)
        equal = (np.array_equal(probability, eager_probability) and
                 np.array_equal(row['point_importance'], eager_importance, equal_nan=True) and
                 same_rng(eager_after, workflow.scorer.rng.get_state()))
        if not equal:
            raise AssertionError(f'Deferred importance differs for {path}')
        rows.append(dict(packet=path.name, points=len(features), exact=True))
    if len(rows) != 53:
        raise AssertionError('Expected all 53 accepted original Open3D benchmark poses')
    # Exercise the >200 boundary: both paths sample once and importance remains zero.
    with np.load(next(fixtures.glob('site_*.npz')), allow_pickle=False) as packet:
        original = packet['features']
        features = np.tile(original, (model.config['max_npoints'] // len(original) + 1, 1))
    assert len(features) > model.config['max_npoints']
    before = workflow.scorer.rng.get_state()
    eager_probability, eager_importance = model.eval_model(features, 0.9, workflow.scorer.rng)
    eager_after = workflow.scorer.rng.get_state()
    workflow.scorer.rng.set_state(before)
    probability, importance = model.eval_model(features, 1.0, workflow.scorer.rng)
    row = dict(features=features, score=float(probability[0, 0]), point_importance=importance)
    workflow._finalize_score(row)
    boundary = (np.array_equal(probability, eager_probability) and
                np.array_equal(row['point_importance'], eager_importance) and
                same_rng(eager_after, workflow.scorer.rng.get_state()))
    if not boundary:
        raise AssertionError('Deferred importance changed >200 sampling')
    return dict(passed=True, accepted_real_poses=len(rows), exact_probability=True,
                exact_point_importance=True, equal_nan=True, unchanged_rng=True,
                oversized_sampling_boundary=boundary, initial_deferred_importance_exactly_zero=True, elapsed_seconds=time.monotonic()-start, cases=rows)


def main() -> None:
    """Run the numerical regression and save its inspectable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples/benchmark.yaml')
    parser.add_argument('--fixtures', default='artifacts/benchmark_full/legacy_comparison/with_ligand/targets/1A7X_A')
    parser.add_argument('--report', default='docs/torch_migration/numerical_benchmark_importance.json')
    args = parser.parse_args()
    report = validate(load_config_file(args.config), Path(args.fixtures))
    Path(args.report).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='cases'}), flush=True)


if __name__ == '__main__':
    main()
