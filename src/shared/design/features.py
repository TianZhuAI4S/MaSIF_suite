"""Original seed/neosurf site selection, descriptor cutoffs and IPA feature arithmetic."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np
from scipy.spatial import cKDTree

from ...tasks.alignment.model import AlignmentEvaluationNN
from .data import SurfaceRecord


def target_sites(surface: SurfaceRecord, num_sites: int = 1, selected: np.ndarray | None = None) -> list[int]:
    """Choose highest patch mean, break ties by input order, halve selected neighbors."""
    if num_sites < 1:
        raise ValueError("num_sites must be positive")
    selected = np.arange(len(surface.points)) if selected is None else np.asarray(selected)
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected) or (selected < 0).any() or (selected >= len(surface.points)).any():
        raise ValueError("Selected target centers must be nonempty and inside the surface")
    interface, result = surface.interface.copy(), []
    for _ in range(num_sites):
        best, best_value = -1, float("-inf")
        for center in selected:
            value = np.mean(interface[surface.neighbors[center]])
            if value > best_value:
                best, best_value = int(center), value
        interface[surface.neighbors[best]] *= 0.5
        result.append(best)
    return result


def match_descriptors(surface: SurfaceRecord, target: np.ndarray, iface_cutoff: float = 0.75,
                      descriptor_cutoff: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Use strict > interface and < Euclidean descriptor thresholds in vertex order."""
    target = np.asarray(target)
    if surface.descriptors is None:
        raise ValueError("Database surface requires straight descriptors")
    if target.shape != surface.descriptors.shape[1:]:
        raise ValueError("Target and database descriptor widths must match")
    distance = np.sqrt(np.sum(np.square(surface.descriptors - target), axis=1))
    # Both masks index the same vertex order; intersect1d would sort these indices again.
    selected = np.flatnonzero((surface.interface > iface_cutoff) & (distance < descriptor_cutoff))
    return selected, distance[selected]


def alignment_features(target: dict, source: dict, target_atoms: np.ndarray, *,
                       point_tree: cKDTree | None = None, atom_tree: cKDTree | None = None) -> tuple[np.ndarray, float]:
    """Return [inverse point distance, inverse descriptor distance, normal dot, inverse atom distance]."""
    if target_atoms is None or not len(target_atoms):
        raise ValueError("IPA requires target heavy-atom coordinates")
    point_tree = cKDTree(target["points"]) if point_tree is None else point_tree
    distance, nearest = point_tree.query(source["points"])
    distance[distance < 0.5] = 0.5
    descriptor_distance = np.sqrt(np.sum(np.square(
        target["descriptors"][nearest] - source["descriptors"]), axis=1))
    # No epsilon was used for descriptor distance in the original workflow.
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_descriptor = 1.0 / descriptor_distance
    descriptor_score = np.sum(np.square(inverse_descriptor[np.where(distance < 1.5)[0]]))
    normal_dot = np.multiply(source["normals"], target["normals"][nearest]).sum(1)
    atom_tree = cKDTree(target_atoms) if atom_tree is None else atom_tree
    atom_distance, _ = atom_tree.query(source["points"])
    atom_distance[atom_distance < 0.25] = 0.25
    features = np.vstack([1.0 / distance, inverse_descriptor, normal_dot, 1.0 / atom_distance]).T
    return features, float(descriptor_score)


class AlignmentScorer:
    """Own the IPA model and legacy RandomState stream while geometry stays numeric."""

    def __init__(self, model: AlignmentEvaluationNN, device: str = "cpu", seed: int = 42,
                 cache_geometry: bool = True):
        """Require the IPA architecture, move it once, and create the original RNG kind."""
        if not isinstance(model, AlignmentEvaluationNN):
            raise ValueError("Seed/neosurf scoring requires an alignment checkpoint")
        self.model, self.rng = model.to(device), np.random.RandomState(seed)
        self.cache_geometry = cache_geometry
        self._coordinate_trees = OrderedDict()

    def coordinate_tree(self, coordinates: np.ndarray) -> cKDTree:
        """Reuse up to eight exact-coordinate trees; changed bytes invalidate cached entries."""
        if not self.cache_geometry:
            return cKDTree(coordinates)
        values = np.ascontiguousarray(coordinates)
        key = (values.shape, values.dtype.str, hashlib.sha256(values.tobytes()).digest())
        if key not in self._coordinate_trees:
            # Own the indexed values so in-place caller edits cannot corrupt a cached tree.
            self._coordinate_trees[key] = cKDTree(values, copy_data=True)
            if len(self._coordinate_trees) > 8:
                self._coordinate_trees.popitem(last=False)
        self._coordinate_trees.move_to_end(key)
        return self._coordinate_trees[key]

    def clear_geometry_cache(self) -> None:
        """Release coordinate trees without changing model state or sampling RNG."""
        self._coordinate_trees.clear()

    def score(self, target: dict, source: dict, target_atoms: np.ndarray,
              importance_cutoff: float = 0.9) -> dict:
        """Compute features, positive probability and signed point importance for one pose."""
        if target_atoms is None or not len(target_atoms):
            raise ValueError("IPA requires target heavy-atom coordinates")
        features, descriptor_score = alignment_features(target, source, target_atoms,
            point_tree=self.coordinate_tree(target["points"]), atom_tree=self.coordinate_tree(target_atoms))
        probability, importance = self.model.eval_model(features, importance_cutoff, self.rng)
        return dict(score=float(probability[0, 0]), descriptor_score=descriptor_score,
                    features=features, point_importance=importance,
                    importance_finite=bool(np.isfinite(importance).all()))


def count_clashes(source: SurfaceRecord, target_atoms: np.ndarray, transformation: np.ndarray,
                  radius: float = 2.0, ca_threshold: int = 0, *,
                  target_tree: cKDTree | None = None) -> tuple[int, int]:
    """Count transformed source CA/heavy atoms against target heavy atoms, preserving early abort."""
    from .data import transform_points
    if source.atoms is None or source.ca_atoms is None or target_atoms is None or not len(target_atoms):
        raise ValueError("Clash filtering requires source CA/heavy and target heavy atoms")
    if radius <= 0 or ca_threshold < 0:
        raise ValueError("Clash radius must be positive and threshold nonnegative")
    tree = cKDTree(target_atoms) if target_tree is None else target_tree
    # The official implementation queries target heavy atoms for both counts.
    distance, _ = tree.query(transform_points(source.ca_atoms, transformation), distance_upper_bound=radius)
    ca_count = int(np.sum(distance <= radius))
    if ca_count > ca_threshold:
        return ca_count, 0
    distance, _ = tree.query(transform_points(source.atoms, transformation), distance_upper_bound=radius)
    return ca_count, int(np.sum(distance <= radius))
