"""Compare fixed-pose geometry with unchanged AST-extracted official functions."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import open3d as o3d
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from plyfile import PlyData
from scipy.spatial import cKDTree

from masif_torch.shared.design.data import SurfaceRecord, transform_points
from masif_torch.shared.design import features as modern
from masif_torch.validation import compare


def upstream_functions(path: Path) -> dict:
    """Execute unchanged function ASTs, injecting only optional dependency aliases."""
    names = {"get_target_vix", "get_patch_geo", "match_descriptors", "compute_nn_score", "count_clashes"}
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Official source lacks one of the audited geometry functions")
    namespace = dict(np=np, os=os, copy=copy, cKDTree=cKDTree,
                     PointCloud=o3d.geometry.PointCloud, Vector3dVector=o3d.utility.Vector3dVector,
                     Feature=o3d.pipelines.registration.Feature)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def upstream_equivalence(path: Path) -> dict:
    """Record both official file hashes and require equal ASTs for the five audited functions."""
    repo = path.parents[2]
    other_name = "masif-neosurf" if repo.name == "masif_seed" else "masif_seed"
    other = repo.parent / other_name / "masif_seed_search/source/alignment_utils.py"
    if repo.name not in ("masif_seed", "masif-neosurf") or not other.exists():
        return dict(compared=False)
    names = {"get_target_vix", "get_patch_geo", "match_descriptors", "compute_nn_score", "count_clashes"}
    functions = []
    for source in (path, other):
        functions.append({node.name: ast.dump(node, include_attributes=False)
                          for node in ast.parse(source.read_text()).body
                          if isinstance(node, ast.FunctionDef) and node.name in names})
    equal = {name: functions[0].get(name) == functions[1].get(name) for name in sorted(names)}
    return dict(compared=True, other_source=str(other),
                other_source_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
                functions_ast_equal=equal, passed=all(equal.values()))


def compare_exact(actual, expected) -> dict:
    """Require exact values and matching NaN/Inf masks, reporting finite-value errors."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, reason=f"Shape mismatch: {actual.shape} vs {expected.shape}")
    finite = np.isfinite(expected)
    result = (compare(actual[finite], expected[finite], atol=0, rtol=0) if finite.any() else
              dict(passed=True, max_abs=0.0, max_rel=0.0, relative_l2=0.0, rmse=0.0, atol=0, rtol=0))
    result.update(passed=bool(np.array_equal(actual, expected, equal_nan=True)),
                  actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
                  shape=list(actual.shape), finite_count=int(finite.sum()),
                  nonfinite_masks_equal=bool(np.array_equal(np.isnan(actual), np.isnan(expected)) and
                      np.array_equal(np.isposinf(actual), np.isposinf(expected)) and
                      np.array_equal(np.isneginf(actual), np.isneginf(expected))))
    return result


def as_cloud(points, normals=None):
    """Use the double Open3D vector representation used by the original functions."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def load_record(root: Path, name: str, target: bool = False):
    """Read genuine fixtures directly and keep descriptor screening dtype unchanged."""
    # Original load_protein_pcd uses the original molecular PLY for geometric coordinates.
    cloud = o3d.io.read_point_cloud(str(root / "data_preparation/01-benchmark_surfaces" / (name + ".ply")))
    base = root / "descriptors/sc05/all_feat" / name
    straight = None if target else np.load(base / "p1_desc_straight.npy", allow_pickle=False)
    flipped = np.load(base / "p1_desc_flipped.npy", allow_pickle=False)
    neighbors = np.load(root / "data_preparation/04b-precomputation_12A/precomputation" /
                        name / "p1_list_indices.npy", allow_pickle=True)
    if target:
        # The original main script chooses target sites from the colored PLY attribute.
        interface = np.asarray(PlyData.read(root / "output/all_feat_3l/pred_surfaces" /
            (name + ".ply"))["vertex"]["iface"], dtype=np.float64)
    else:
        # The original source matching function reads the unrounded numeric prediction.
        interface = np.load(root / "output/all_feat_3l/pred_data" / ("pred_" + name + ".npy"),
                            allow_pickle=False).reshape(-1)
    pdb = root / "data_preparation/01-benchmark_pdbs" / (name + ".pdb")
    structure = PDBParser(QUIET=True).get_structure(name, pdb)
    heavy = [atom for atom in structure.get_atoms() if not atom.get_name().startswith("H")]
    atoms = np.asarray([atom.get_coord() for atom in heavy], dtype=np.float64)
    ca = np.asarray([atom.get_coord() for atom in heavy if atom.get_id() == "CA"], dtype=np.float64)
    # Targets legitimately contain flipped descriptors only; no straight values are invented.
    record = SimpleNamespace(name=name, points=np.asarray(cloud.points), normals=np.asarray(cloud.normals),
                             descriptors=straight, flipped_descriptors=flipped, interface=interface,
                             neighbors=[np.asarray(row, dtype=np.int64) for row in neighbors],
                             atoms=atoms, ca_atoms=ca, pdb_path=pdb)
    return record, structure


def compare_patch(oracle, record, center, flipped, results, prefix):
    """Compare shift-before-flip geometry and Open3D double descriptor storage."""
    raw = record.flipped_descriptors if flipped else record.descriptors
    cloud, desc, indices = oracle["get_patch_geo"](as_cloud(record.points, record.normals),
        record.neighbors, center, raw[None], outward_shift=0.25, flip_normals=flipped)
    actual = SurfaceRecord.patch(record, center, shift=0.25, flipped=flipped)
    for field, expected in (("points", cloud.points), ("normals", cloud.normals),
                            ("descriptors", desc[0].data.T), ("indices", indices)):
        results[prefix + "/" + field] = compare_exact(actual[field], expected)
    return actual


def compare_matches(oracle, record, target, results, prefix, iface_cutoff=.75, descriptor_cutoff=2.):
    """Run the original disk reader and observe original local distance outputs."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        desc, iface = root / "descriptors" / record.name, root / "iface"
        desc.mkdir(parents=True)
        iface.mkdir()
        np.save(desc / "p1_desc_straight.npy", record.descriptors)
        np.save(iface / ("pred_" + record.name + ".npy"), record.interface[None])
        params = dict(seed_desc_dir=str(root / "descriptors"), seed_iface_dir=str(iface),
                      iface_cutoff=iface_cutoff, desc_dist_cutoff=descriptor_cutoff)
        captured, function = {}, oracle["match_descriptors"]
        def capture(frame, event, value):
            """Observe locals without modifying original function AST or arithmetic."""
            if event == "return" and frame.f_code is function.__code__:
                captured.update(frame.f_locals)
        previous = sys.getprofile()
        try:
            sys.setprofile(capture)
            with contextlib.redirect_stdout(io.StringIO()):
                selected = function([record.name], ["p1"], target, params)
        finally:
            sys.setprofile(previous)
        expected = np.asarray(selected.get((record.name, "p1"), []), dtype=np.int64)
        actual, distance = modern.match_descriptors(record, target, iface_cutoff, descriptor_cutoff)
        results[prefix + "/indices"] = compare_exact(actual, expected)
        results[prefix + "/distance"] = compare_exact(distance,
            captured.get("matched_desc_dist", np.empty(0, dtype=record.descriptors.dtype)))


class FeatureCapture:
    """Capture raw upstream features while preserving all original feature arithmetic."""

    def eval_model(self, features, cutoff):
        """Record features and return neutral predictions unused by geometry comparisons."""
        self.features, self.cutoff = features.copy(), cutoff
        return np.array([[0.5]]), np.zeros(len(features))


def compare_features(oracle, target, source, atoms, results, prefix):
    """Execute unchanged upstream compute_nn_score and compare all four feature columns."""
    descriptors = []
    for patch in (target, source):
        feature = o3d.pipelines.registration.Feature()
        feature.data = patch["descriptors"].T
        descriptors.append([feature])
    scorer = FeatureCapture()
    atom_distance, _ = cKDTree(atoms).query(source["points"])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected, _ = oracle["compute_nn_score"](as_cloud(target["points"], target["normals"]),
            as_cloud(source["points"], source["normals"]), np.empty((0, 2), dtype=np.int64),
            descriptors[0], descriptors[1], cKDTree(target["points"]), scorer, atom_distance.copy(),
            nn_score_cutoff=0.1)
        actual, score = modern.alignment_features(target, source, atoms)
    results[prefix + "/features"] = compare_exact(actual, scorer.features)
    results[prefix + "/descriptor_score"] = compare_exact(score, expected[1])
    results[prefix + "/legacy_importance_cutoff"] = dict(passed=scorer.cutoff == 0.9)


def compare_clashes(oracle, source, structure, target, matrix, results, prefix, ca_threshold=1):
    """Compare CA early-abort and heavy counts without propagating legacy PDB mutation."""
    expected = oracle["count_clashes"](matrix, source.points, copy.deepcopy(structure),
        cKDTree(target.ca_atoms), cKDTree(target.atoms), radius=2.0,
        clashing_ca_thresh=ca_threshold, clashing_thresh=10**9)
    actual = modern.count_clashes(source, target.atoms, matrix, radius=2., ca_threshold=ca_threshold)
    results[prefix] = compare_exact(actual, expected)


def run(source: Path, fixture: Path) -> dict:
    """Exercise genuine protein poses plus clipping, strict thresholds and tied sites."""
    oracle, results = upstream_functions(source), {}
    target_a, _ = load_record(fixture / "with_ligand", "1A7X_A", target=True)
    target_b, _ = load_record(fixture / "with_ligand", "1A7X_B", target=True)
    decoy, structure_decoy = load_record(fixture / "decoys", "2GKW_B")
    for record, root, role in ((target_a, fixture / "with_ligand", "target"),
                               (target_b, fixture / "with_ligand", "target"),
                               (decoy, fixture / "decoys", "source")):
        loaded = SurfaceRecord.from_processed(root, record.name, trusted_indices=True, role=role)
        for field in ("points", "normals", "interface", "flipped_descriptors", "atoms", "ca_atoms"):
            results[record.name + "/adapter/" + field] = compare_exact(getattr(loaded, field), getattr(record, field))
        if role == "source":
            results[record.name + "/adapter/straight_descriptors"] = compare_exact(loaded.descriptors, record.descriptors)
        else:
            results[record.name + "/adapter/target_only_flipped"] = dict(passed=loaded.descriptors is None)
        with tempfile.TemporaryDirectory() as temporary:
            packet = Path(temporary) / "surface.npz"
            loaded.save(packet)
            restored = SurfaceRecord.from_npz(packet)
            results[record.name + "/adapter/npz_optional_straight"] = dict(
                passed=(restored.descriptors is None) if loaded.descriptors is None else
                np.array_equal(restored.descriptors, loaded.descriptors))
    for target in (target_a, target_b):
        selected = np.arange(0, len(target.points), 7, dtype=np.int64)
        for subset in (None, selected):
            suffix = "all" if subset is None else "restricted"
            expected = oracle["get_target_vix"](target.neighbors, target.interface, 3, subset)
            results[target.name + "/sites/" + suffix] = compare_exact(modern.target_sites(target, 3, subset), expected)
        center = modern.target_sites(target)[0]
        target_patch = compare_patch(oracle, target, center, True, results, target.name + "/patch")
        compare_matches(oracle, decoy, target.flipped_descriptors[center], results, target.name + "/match")
        source_center = int(np.argmin(np.linalg.norm(decoy.points - target.points[center], axis=1)))
        source_patch = compare_patch(oracle, decoy, source_center, False, results, target.name + "/decoy_patch")
        compare_features(oracle, target_patch, source_patch, target.atoms, results, target.name + "/native_pose")
        matrix, angle = np.eye(4), .3
        matrix[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        matrix[:3, 3] = target_patch["points"].mean(0) - source_patch["points"].mean(0) @ matrix[:3, :3].T
        transformed = dict(source_patch, points=transform_points(source_patch["points"], matrix),
                           normals=source_patch["normals"] @ matrix[:3, :3].T)
        compare_features(oracle, target_patch, transformed, target.atoms, results, target.name + "/fixed_rigid_pose")
        compare_clashes(oracle, decoy, structure_decoy, target, matrix, results, target.name + "/clash_early")
        compare_clashes(oracle, decoy, structure_decoy, target, matrix, results,
                        target.name + "/clash_complete", ca_threshold=10**9)
    synthetic = SimpleNamespace(name="TEST_A", points=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
        normals=np.tile([0., 0., 1.], (3, 1)), descriptors=np.array([[0., 0.], [1., 0.], [2., 0.]], dtype=np.float32),
        flipped_descriptors=np.array([[2., 0.], [3., 0.], [4., 0.]], dtype=np.float32),
        interface=np.array([.75, .7500001, 1.]), neighbors=[np.array([i]) for i in range(3)])
    compare_matches(oracle, synthetic, np.zeros(2, dtype=np.float32), results, "boundary/strict_match",
                    iface_cutoff=.75, descriptor_cutoff=2.)
    tied = copy.deepcopy(synthetic)
    tied.interface[:] = 1
    expected = oracle["get_target_vix"](tied.neighbors, tied.interface, 3, np.array([2, 1, 0]))
    results["boundary/site_ties"] = compare_exact(modern.target_sites(tied, 3, np.array([2, 1, 0])), expected)
    for flipped in (False, True):
        compare_patch(oracle, synthetic, 1, flipped, results, "boundary/patch_" + str(flipped))
    patch = dict(points=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
                 normals=np.tile([0., 0., 1.], (3, 1)), descriptors=np.ones((3, 2)))
    probe = dict(points=np.array([[0., 0., 0.], [1.5, 0., 0.], [3.5, 0., 0.]]),
                 normals=np.tile([0., 0., -1.], (3, 1)), descriptors=np.zeros((3, 2)))
    compare_features(oracle, patch, probe, np.array([[0., 0., 0.], [1.25, 0., 0.]]), results, "boundary/clips")
    probe["descriptors"][:] = 1
    compare_features(oracle, patch, probe, np.array([[0., 0., 0.]]), results, "boundary/identical_descriptor_inf")
    coordinates = np.array([[2., 0., 0.], [np.nextafter(2., 0.), 0., 0.],
                            [np.nextafter(2., np.inf), 0., 0.], [1., 0., 0.]])
    atoms = [Atom("CA" if index < 3 else "O", coordinate.copy(), 0., 1., " ",
                  " CA " if index < 3 else " O  ", index, element="C" if index < 3 else "O")
             for index, coordinate in enumerate(coordinates)]
    structure = SimpleNamespace(get_atoms=lambda: iter(atoms))
    clash_source = SimpleNamespace(points=coordinates, atoms=coordinates, ca_atoms=coordinates[:3])
    clash_target = SimpleNamespace(atoms=np.zeros((1, 3)), ca_atoms=np.array([[50., 0., 0.]]))
    compare_clashes(oracle, clash_source, structure, clash_target, np.eye(4), results,
                    "boundary/clash_exact_radius_and_target_heavy", ca_threshold=10**9)
    compare_clashes(oracle, clash_source, structure, clash_target, np.eye(4), results,
                    "boundary/clash_ca_early_abort", ca_threshold=0)
    equivalence = upstream_equivalence(source)
    manifest = fixture / "manifest.json"
    fixture_hashes = {str(path.relative_to(fixture)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(fixture.rglob("*"))
                      if path.is_file() and path.suffix.lower() in (".npy", ".npz", ".ply", ".pdb")}
    return dict(passed=all(item["passed"] for item in results.values()) and equivalence.get("passed", True),
                oracle="unchanged_ast_source_functions", upstream_equivalence=equivalence,
                source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                fixture=str(fixture), fixture_file_sha256=fixture_hashes,
                fixture_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
                numpy=np.__version__, open3d=o3d.__version__, exact_geometry=True, stochastic_ransac_tested=False,
                legacy_clash_pdb_mutation_tested=False, tensors=results)


def main():
    """Write the auditable report and fail when any numerical comparison fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/upstream_repos/masif_seed/masif_seed_search/source/alignment_utils.py"))
    parser.add_argument("--fixture", type=Path, default=Path("artifacts/real_fixtures/neosurf_1a7x"))
    parser.add_argument("--report", type=Path, default=Path("docs/torch_migration/numerical_design_geometry.json"))
    args = parser.parse_args()
    report = run(args.source, args.fixture)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], tensors=len(report["tensors"]), report=str(args.report),
                         failed={key: value for key, value in report["tensors"].items() if not value["passed"]}), indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
