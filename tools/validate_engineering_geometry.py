"""Compare exact geometry caches and model outputs against preserved pre-optimization source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from masif_torch.checkpoints import load_checkpoint
from masif_torch.configuration import load_config_file
from masif_torch.shared.design.data import SurfaceRecord, transform_points
from masif_torch.shared.design.features import AlignmentScorer, alignment_features, count_clashes
from masif_torch.validation import deterministic


def baseline_namespace(path: Path) -> dict:
    """Execute the byte-preserved feature module under its original relative-import package."""
    namespace = dict(__name__='masif_torch.shared.design._engineering_baseline',
                     __package__='masif_torch.shared.design')
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    return namespace


def same_rng(first: tuple, second: tuple) -> bool:
    """Require every NumPy MT19937 state component to match exactly."""
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


def real_poses(root: Path, fixtures: Path) -> tuple[SurfaceRecord, list[dict]]:
    """Reconstruct all 53 real old-registration poses using unchanged patch-transform ordering."""
    target = SurfaceRecord.from_npz(root/'with_ligand/1A7X_A_target.npz')
    report = json.loads((fixtures/'results.json').read_text())
    surfaces, poses = {}, []
    for row in report['candidates']:
        if not row.get('accepted'):
            continue
        if row['source'] not in surfaces:
            surfaces[row['source']] = SurfaceRecord.from_npz(root/'without_ligand'/(row['source']+'_source.npz'))
        surface = surfaces[row['source']]
        source = surface.patch(row['source_vertex'], .25)
        with np.load(fixtures/row['packet'], allow_pickle=False) as packet:
            matrix = packet['transformation']
            original_features = packet['features']
        source['points'] = transform_points(source['points'], matrix)
        source['normals'] = source['normals'] @ matrix[:3,:3].T
        poses.append(dict(key=row['key'], target=target.patch(row['target_vertex'], .25, flipped=True),
                          source=source, surface=surface, matrix=matrix, original_features=original_features))
    if len(poses) != 53:
        raise AssertionError('Expected all 53 captured original-registration poses')
    return target, poses


def validate(config: dict, devices: list[str]) -> dict:
    """Check raw features, clashes, probabilities, importance and sampling state exactly."""
    deterministic(42)
    torch.set_num_threads(1)
    original_path = Path('artifacts/engineering_acceleration/baseline_source/features.py')
    original = baseline_namespace(original_path)
    root = Path('artifacts/benchmark_full/run/surface_cache')
    target, poses = real_poses(root, Path('artifacts/benchmark_full/legacy_comparison/with_ligand/targets/1A7X_A'))
    device_reports = []
    for device in devices:
        model, _ = load_checkpoint(config['checkpoint'], device)
        baseline = original['AlignmentScorer'](model, device, 42)
        cached = AlignmentScorer(model, device, 42)
        for scorer in [baseline,cached]:
            scorer.score(poses[0]['target'],poses[0]['source'],target.atoms,importance_cutoff=1.)
            scorer.rng = np.random.RandomState(42)
        durations = dict(baseline=0., optimized=0.)
        for pose in poses:
            results = {}
            for name, scorer in [('baseline',baseline),('optimized',cached)]:
                start = time.perf_counter()
                results[name] = scorer.score(pose['target'], pose['source'], target.atoms, importance_cutoff=.9)
                if device == 'cuda':
                    torch.cuda.synchronize()
                durations[name] += time.perf_counter()-start
            first, second = results['baseline'], results['optimized']
            for name in ['score','descriptor_score','importance_finite']:
                if first[name] != second[name]:
                    raise AssertionError(f'{device}: {name} changed for {pose["key"]}')
            for name in ['features','point_importance']:
                if not np.array_equal(first[name],second[name],equal_nan=True):
                    raise AssertionError(f'{device}: {name} changed for {pose["key"]}')
            if not np.array_equal(second['features'],pose['original_features']):
                raise AssertionError('Reconstructed raw features differ from the original captured packet')
            old_clashes = original['count_clashes'](pose['surface'], target.atoms, pose['matrix'])
            new_clashes = count_clashes(pose['surface'], target.atoms, pose['matrix'],
                                        target_tree=cached.coordinate_tree(target.atoms))
            if old_clashes != new_clashes:
                raise AssertionError('Cached clash counts changed')
        # A real-pose duplication guarantees sampling above the architecture's 200-point limit.
        oversized = {name:np.tile(poses[0]['source'][name],(2,1)) for name in ['points','normals','descriptors']}
        if len(oversized['points']) <= model.config['max_npoints']:
            raise AssertionError('Oversized sampling gate is not above max_npoints')
        old_sample = baseline.score(poses[0]['target'],oversized,target.atoms)
        new_sample = cached.score(poses[0]['target'],oversized,target.atoms)
        if old_sample['score'] != new_sample['score'] or not np.array_equal(old_sample['point_importance'],new_sample['point_importance']):
            raise AssertionError('Geometry caching changed oversized point sampling')
        if not same_rng(baseline.rng.get_state(),cached.rng.get_state()):
            raise AssertionError('Geometry reuse changed the IPA sampling stream')
        device_reports.append(dict(device=device,poses=len(poses),exact_features=True,exact_score=True,
                                   exact_importance_equal_nan=True,exact_clashes=True,exact_rng=True,oversized_sampling_exact=True,
                                   elapsed_seconds=durations))
    # Measure geometry independently; the unchanged NN dominates end-to-end inference.
    model, _ = load_checkpoint(config['checkpoint'],'cpu')
    cached = AlignmentScorer(model)
    durations = {}
    for name in ['baseline','optimized']:
        start = time.perf_counter()
        for _ in range(20):
            for pose in poses:
                if name == 'baseline':
                    original['alignment_features'](pose['target'],pose['source'],target.atoms)
                else:
                    alignment_features(pose['target'],pose['source'],target.atoms,
                        point_tree=cached.coordinate_tree(pose['target']['points']),
                        atom_tree=cached.coordinate_tree(target.atoms))
        durations[name] = time.perf_counter()-start
    return dict(passed=True,baseline_source_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                optimized_source_sha256=hashlib.sha256(Path('src/shared/design/features.py').read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest(),
                original_registered_poses=53,device_reports=device_reports,geometry_calls=20*len(poses),
                geometry_elapsed_seconds=durations,geometry_speedup=durations['baseline']/durations['optimized'],
                comparison_scope='Before/after engineering changes on the same PyTorch model; exact, not a new TensorFlow-bitwise claim.')


def main() -> None:
    """Run the real-data cache regression and retain numerical/performance evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='examples/benchmark.yaml')
    parser.add_argument('--devices',nargs='+',default=['cpu','cuda'])
    parser.add_argument('--report',default='docs/torch_migration/numerical_engineering_geometry.json')
    args = parser.parse_args()
    if 'cuda' in args.devices and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA gate is unavailable')
    report = validate(load_config_file(args.config),args.devices)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
