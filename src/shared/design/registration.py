"""Replaceable pose providers: exact saved poses or explicitly modern Open3D docking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .data import transform_points


class RegistrationBackend(ABC):
    """Minimal pose-provider contract, independent of the IPA neural network."""

    @abstractmethod
    def align(self, key: str, source: dict, target: dict) -> tuple[np.ndarray, float]:
        """Return source-to-target rigid transform and correspondence fraction."""


class ReplayRegistration(RegistrationBackend):
    """Replay final original poses; do not rerun a different RANSAC sampler."""

    def __init__(self, path: str | Path):
        """Read explicit candidate keys and finite rigid transforms from safe NPZ."""
        with np.load(path, allow_pickle=False) as archive:
            keys, matrices = archive["keys"].astype(str), archive["transforms"]
            fractions = archive["fitness"] if "fitness" in archive else None
        if keys.ndim != 1 or matrices.shape != (len(keys), 4, 4) or len(set(keys)) != len(keys):
            raise ValueError("Replay requires unique keys [K] and transforms [K,4,4]")
        self.poses, self.fractions = {}, {}
        if fractions is not None and (fractions.shape != (len(keys),) or
                                     not np.isfinite(fractions).all() or (fractions < 0).any() or (fractions > 1).any()):
            raise ValueError("Replay fitness must lie in [0,1] for every candidate")
        for index, (key, matrix) in enumerate(zip(keys, matrices, strict=True)):
            transform_points(np.zeros((1, 3)), matrix)
            self.poses[key] = matrix.copy()
            if fractions is not None:
                self.fractions[key] = float(fractions[index])

    def align(self, key: str, source: dict, target: dict) -> tuple[np.ndarray, float]:
        """Require a supplied pose for every matched candidate; never use silent identity."""
        if key not in self.poses:
            raise ValueError(f"Missing original candidate pose: {key}")
        matrix = self.poses[key]
        distances, _ = cKDTree(target["points"]).query(transform_points(source["points"], matrix))
        fraction = self.fractions.get(key, float(np.mean(distances < 1.0)))
        return matrix, fraction


class Open3DRegistration(RegistrationBackend):
    """Modern RANSAC/ICP backend; pose results are not claimed identical to Open3D 0.5."""

    def __init__(self, iterations: int = 2000, radius: float = 1.5, confidence: float = 0.999, seed: int = 42):
        """Use explicit modern stopping semantics rather than reinterpret old max_validation=500."""
        if iterations < 1 or radius <= 0 or not 0 <= confidence <= 1:
            raise ValueError("Invalid modern registration settings")
        import open3d as o3d
        self.o3d, self.iterations, self.radius = o3d, iterations, radius
        self.confidence = confidence
        o3d.utility.random.seed(seed)

    def align(self, key: str, source: dict, target: dict) -> tuple[np.ndarray, float]:
        """Run feature RANSAC followed by original point-to-plane ICP threshold 1 Å."""
        o3d = self.o3d
        def cloud(data):
            """Copy numeric points/normals into the optional geometry backend."""
            result = o3d.geometry.PointCloud()
            result.points = o3d.utility.Vector3dVector(data["points"])
            result.normals = o3d.utility.Vector3dVector(data["normals"])
            feature = o3d.pipelines.registration.Feature()
            feature.data = data["descriptors"].T
            return result, feature
        source_cloud, source_feature = cloud(source)
        target_cloud, target_feature = cloud(target)
        registration = o3d.pipelines.registration
        result = registration.registration_ransac_based_on_feature_matching(
            source_cloud, target_cloud, source_feature, target_feature, False, self.radius,
            registration.TransformationEstimationPointToPoint(False), 3,
            [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             registration.CorrespondenceCheckerBasedOnDistance(1.0),
             registration.CorrespondenceCheckerBasedOnNormal(np.pi / 2)],
            registration.RANSACConvergenceCriteria(self.iterations, self.confidence))
        refined = registration.registration_icp(source_cloud, target_cloud, 1.0, result.transformation,
                                               registration.TransformationEstimationPointToPlane())
        return np.asarray(refined.transformation), float(refined.fitness)


def build_registration(config: dict) -> RegistrationBackend:
    """Construct a pose provider from explicit mode and mode-owned parameters."""
    config = dict(config)
    mode = config.pop("mode")
    providers = {"replay": ReplayRegistration, "open3d": Open3DRegistration}
    if mode not in providers:
        raise ValueError(f"Unknown registration mode: {mode}")
    return providers[mode](**config)
