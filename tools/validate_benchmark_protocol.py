"""Compare benchmark randomization and COM against unchanged official expressions."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from masif_torch.tasks.benchmark.workflow import interface_center, BenchmarkRegistration
from masif_torch.shared.design.registration import Open3DRegistration


def identity_registration(self, key, source, target):
    """Isolate the original source randomization from stochastic pose fitting."""
    return np.eye(4), 1.0


def main():
    """Write exact-array comparisons against independent original source calculations."""
    root = Path(__file__).resolve().parents[1]
    module = ast.parse((root / 'artifacts/upstream_repos/masif-neosurf/masif_seed_search/source/alignment_utils.py').read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and
                 node.name in ('rand_rotation_matrix', 'get_center_and_random_rotate')]
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'official_rotation', 'exec'), namespace)
    points = np.random.RandomState(7).normal(size=(20, 3))
    np.random.seed(42)
    expected_rotation = namespace['get_center_and_random_rotate'](SimpleNamespace(points=points))
    with patch.object(Open3DRegistration, 'align', identity_registration):
        backend = BenchmarkRegistration({'protein': SimpleNamespace(points=points)}, np.random.RandomState(42),
                                        iterations=2000, radius=1.5, seed=42)
        actual_rotation, _ = backend.align('0:protein:0',
                                          dict(points=points, normals=points, descriptors=np.zeros((20, 80))), {})
    assert np.array_equal(expected_rotation, actual_rotation)
    pdb = root / 'artifacts/benchmark_full/run/original_pdbs/1a7x.pdb'
    model = PDBParser(QUIET=True).get_structure('1a7x', pdb)[0]
    ligand = [residue for residue in model['B'] if residue.resname == 'FKA'][0]
    target = np.stack([atom.coord for atom in model['A'].get_atoms() if is_aa(atom.parent)] +
                      [atom.coord for atom in ligand.get_atoms()])
    binder = np.stack([atom.coord for atom in model['B'].get_atoms() if is_aa(atom.parent)])
    distance = np.sqrt(np.sum((target[:, None] - binder[None]) ** 2, axis=-1))
    expected_center = np.mean(target[np.any(distance < 4.0, axis=1)], axis=0)
    actual_center = interface_center(pdb, 'A', 'B', 'FKA_B')
    assert np.array_equal(expected_center, actual_center)
    report = dict(original_random_transform_array_equal=True, original_float32_interface_COM_array_equal=True,
                  COM=actual_center.tolist(), coordinate_units='angstrom', all_finite=bool(np.isfinite(actual_center).all()))
    (root / 'docs/torch_migration/numerical_benchmark_protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
