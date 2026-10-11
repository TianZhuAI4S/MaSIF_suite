"""Measure an optional correspondence cache without altering the full benchmark backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from contextlib import nullcontext
from unittest.mock import patch as mock_patch

import numpy as np
import open3d as o3d

from masif_torch.shared.design.data import SurfaceRecord, transform_points
from masif_torch.shared.design.registration import Open3DRegistration
from masif_torch.tasks.benchmark.workflow import BenchmarkRegistration
from masif_torch.shared.design.features import match_descriptors


def cloud(patch: dict):
    """Create the exact numeric Open3D point cloud and double-precision feature matrix."""
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(patch['points'])
    result.normals = o3d.utility.Vector3dVector(patch['normals'])
    feature = o3d.pipelines.registration.Feature()
    feature.data = patch['descriptors'].T
    return result, feature


def align(source, target, source_feature, target_feature, correspondences=None):
    """Compare both official entry points with identical RANSAC and ICP parameters."""
    registration = o3d.pipelines.registration
    estimation = registration.TransformationEstimationPointToPoint(False)
    checkers = [registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                registration.CorrespondenceCheckerBasedOnDistance(1.),
                registration.CorrespondenceCheckerBasedOnNormal(np.pi/2)]
    criteria = registration.RANSACConvergenceCriteria(2000, .999)
    if correspondences is None:
        pose = registration.registration_ransac_based_on_feature_matching(
            source, target, source_feature, target_feature, False, 1.5, estimation, 3, checkers, criteria)
    else:
        pose = registration.registration_ransac_based_on_correspondence(
            source, target, correspondences, 1.5, estimation, 3, checkers, criteria)
    refined = registration.registration_icp(source, target, 1., pose.transformation,
                                          registration.TransformationEstimationPointToPlane())
    return np.asarray(refined.transformation), float(refined.fitness)


def identity_pose(self, key: str, source: dict, target: dict):
    """Capture the actual benchmark initial source transform without fitting a pose."""
    return np.eye(4), 1.0


def duplicate_feature_ties() -> bool:
    """Verify target descriptor ties resolve identically for full and indexed source features."""
    registration = o3d.pipelines.registration
    target, source, subset = registration.Feature(), registration.Feature(), registration.Feature()
    values = np.random.RandomState(7).normal(size=(80, 5))
    target.data = np.repeat(values[:, :3], 2, axis=1)
    source.data = values
    indices = np.array([4, 0, 2])
    subset.data = values[:, indices]
    full = np.asarray(registration.correspondences_from_features(source, target, False))
    actual = np.asarray(registration.correspondences_from_features(subset, target, False))
    return bool(np.array_equal(actual, np.column_stack([np.arange(len(indices)), full[indices, 1]])))


def cached_registration(self, key: str, source: dict, target: dict):
    """Test a cached inner registration only inside a temporary diagnostic patch."""
    group = ':'.join(key.split(':')[:2])
    registration = o3d.pipelines.registration
    source_cloud, source_feature = cloud(source)
    target_cloud, target_feature = cloud(target)
    entry = self._profile_correspondences.get(group)
    if entry is None or not np.array_equal(entry[0], target['descriptors']):
        features = registration.Feature()
        features.data = np.asarray(self.surfaces[key.split(':')[1]].descriptors, dtype=np.float64).T
        full = np.asarray(registration.correspondences_from_features(features, target_feature, False))
        if not np.array_equal(full[:, 0], np.arange(len(full))):
            raise AssertionError('Global correspondence rows differ from vertex indices')
        entry = (target['descriptors'].copy(), full[:, 1].copy())
        self._profile_correspondences[group] = entry
    pairs = np.column_stack([np.arange(len(source['points'])), entry[1][source['indices']]])
    return align(source_cloud, target_cloud, source_feature, target_feature, o3d.utility.Vector2iVector(pairs))


def profile(cache: Path) -> dict:
    """Check exact correspondence/pose equality and measure uncapped-backend optimization potential."""
    registration = o3d.pipelines.registration
    target = SurfaceRecord.from_npz(cache/'with_ligand/1A7X_A_target.npz')
    packets, requests, build_seconds = [], [], 0.
    surfaces = [SurfaceRecord.from_npz(path) for path in sorted((cache/'decoys').glob('*.npz'))[:5]]
    backend = BenchmarkRegistration({surface.name: surface for surface in surfaces}, np.random.RandomState(42),
                                    iterations=2000, radius=1.5, seed=42)
    for surface in surfaces:
        all_features = registration.Feature()
        all_features.data = np.asarray(surface.descriptors, dtype=np.float64).T
        for site in [104, 475, 136]:
            target_patch = target.patch(site, .25, flipped=True)
            target_cloud, target_feature = cloud(target_patch)
            start = time.perf_counter()
            all_pairs = np.asarray(registration.correspondences_from_features(all_features, target_feature, False))
            build_seconds += time.perf_counter()-start
            if not np.array_equal(all_pairs[:,0], np.arange(len(surface.points))):
                raise AssertionError('Global correspondence order differs from source vertex order')
            with mock_patch.object(Open3DRegistration, 'align', identity_pose):
                initial, _ = backend.align(f'{site}:{surface.name}:0', surface.patch(0, .25), target_patch)
            vertices, _ = match_descriptors(surface, target.flipped_descriptors[site], 0., 3.)
            if not len(vertices):
                continue
            for vertex in vertices[np.linspace(0,len(vertices)-1,min(len(vertices),40),dtype=int)]:
                patch = surface.patch(int(vertex), .25)
                patch['points'] = transform_points(patch['points'], initial)
                patch['normals'] = patch['normals'] @ initial[:3,:3].T
                source_cloud, source_feature = cloud(patch)
                actual = np.asarray(registration.correspondences_from_features(source_feature, target_feature, False))
                cached = np.column_stack([np.arange(len(patch['points'])), all_pairs[patch['indices'],1]])
                if not np.array_equal(actual,cached):
                    raise AssertionError('Cached source→target correspondence differs, including order')
                packets.append((source_cloud,target_cloud,source_feature,target_feature,o3d.utility.Vector2iVector(cached)))
                requests.append((str(site)+':'+surface.name+':'+str(vertex), surface, int(vertex), site))
    ties = duplicate_feature_ties()
    unique_keys = len({request[0] for request in requests})
    if not ties or unique_keys < 500:
        raise AssertionError('Regression must include ties and at least 500 real randomized patches')
    outputs, timings = {}, {}
    for mode in ['original','cached']:
        o3d.utility.random.seed(42)
        start=time.perf_counter()
        outputs[mode]=[align(*packet[:4],None if mode=='original' else packet[4]) for packet in packets]
        timings[mode]=time.perf_counter()-start
    exact=all(np.array_equal(first[0],second[0]) and first[1]==second[1]
              for first,second in zip(outputs['original'],outputs['cached'],strict=True))
    if not exact:
        raise AssertionError('Cached entry point changes seeded sequential RANSAC/ICP output')
    direct_outputs, direct_states, direct_timings = {}, {}, {}
    for enabled in [False, True]:
        direct = BenchmarkRegistration({surface.name: surface for surface in surfaces}, np.random.RandomState(42),
                                       iterations=2000, radius=1.5, seed=42)
        direct._profile_correspondences = {}
        context = mock_patch.object(Open3DRegistration, 'align', cached_registration) if enabled else nullcontext()
        start = time.perf_counter()
        with context:
            direct_outputs[enabled] = [direct.align(key, surface.patch(vertex, .25), target.patch(site, .25, flipped=True))
                                       for key, surface, vertex, site in requests]
        direct_timings[str(enabled)] = time.perf_counter()-start
        direct_states[enabled] = direct.rng.get_state()
    direct_exact = all(np.array_equal(first[0], second[0]) and first[1] == second[1]
                       for first, second in zip(direct_outputs[False], direct_outputs[True], strict=True))
    first, second = direct_states[False], direct_states[True]
    rng_exact = first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]
    if not direct_exact or not rng_exact:
        raise AssertionError('Actual sequential BenchmarkRegistration API or RNG changed')
    return dict(diagnostic_only=True,active_backend_changed=False,real_candidate_patches=len(packets),unique_candidate_keys=unique_keys,
                exact_correspondences_and_order=True,exact_sequential_transforms_and_fitness=True,
                duplicate_descriptor_ties_exact=ties,initial_poses_from_actual_benchmark=True,
                direct_benchmark_api_transforms_fitness_exact=direct_exact,direct_numpy_rng_state_exact=rng_exact,
                direct_api_elapsed_seconds=direct_timings,
                seed=42,cache_build_seconds=build_seconds,elapsed_seconds=timings,
                speedup_excluding_cache_build=timings['original']/timings['cached'],
                decision='Not activated: expanded measured benefit does not justify discarding accumulated full-run work.',
                source_urls=['https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/pipelines/registration/Registration.cpp',
                             'https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/pipelines/registration/Feature.cpp'])


def main() -> None:
    """Save the standalone real-patch performance and numerical experiment."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('artifacts/benchmark_full/run/surface_cache'))
    parser.add_argument('--report',type=Path,default=Path('docs/torch_migration/numerical_benchmark_registration_profile.json'))
    args=parser.parse_args()
    report=profile(args.cache)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
