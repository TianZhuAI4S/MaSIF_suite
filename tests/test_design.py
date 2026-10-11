"""Input contracts and scientific edge cases for extensible seed/neosurf workflows."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from masif_torch.shared.design.data import SurfaceRecord, transform_points
from masif_torch.shared.design.features import alignment_features, count_clashes, match_descriptors, target_sites
from masif_torch.shared.design.registration import ReplayRegistration
from masif_torch.tasks.seed.workflow import write_aligned_pdb
from masif_torch.models import AlignmentEvaluationNN
from masif_torch.tasks import build_task, register_task


def surface():
    """Create two indexed vertices with deliberately asymmetric interface/atom coordinates."""
    return SurfaceRecord("example", np.array([[0., 0., 0.], [3., 0., 0.]]),
                         np.array([[0., 0., 1.], [0., 0., 1.]]), np.array([[0., 0.], [2., 0.]]),
                         np.array([0.75, 1.]), [np.array([0]), np.array([1])],
                         flipped_descriptors=np.array([[1., 0.], [3., 0.]]),
                         atoms=np.array([[0., 0., 0.], [5., 0., 0.]]), ca_atoms=np.array([[0., 0., 0.]]))


class DesignTests(unittest.TestCase):
    """Exercise boundary behavior whose accidental changes alter selected scientific candidates."""

    def test_cutoffs_and_site_ties(self):
        """Equality fails both strict matching cutoffs and site ties retain supplied order."""
        record = surface()
        indices, _ = match_descriptors(record, np.zeros(2), 0.75, 2.)
        self.assertEqual(indices.tolist(), [])
        indices, _ = match_descriptors(record, np.zeros(2), 0.74, 2.01)
        self.assertEqual(indices.tolist(), [0, 1])
        record.interface[:] = 1
        self.assertEqual(target_sites(record, 2, np.array([1, 0])), [1, 0])
        with self.assertRaises(ValueError):
            target_sites(record, selected=np.array([0.5]))

    def test_target_only_roundtrip(self):
        """A flipped-only target remains target-only after safe NPZ and shifts before normal flip."""
        record = surface()
        record.descriptors = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npz"
            record.save(path)
            restored = SurfaceRecord.from_npz(path)
            self.assertIsNone(restored.descriptors)
            patch = restored.patch(0, flipped=True)
            np.testing.assert_array_equal(patch["points"], [[0., 0., .25]])
            np.testing.assert_array_equal(patch["normals"], [[0., 0., -1.]])
            with self.assertRaises(ValueError):
                restored.patch(0)
            with self.assertRaises(ValueError):
                match_descriptors(restored, np.zeros(2))
            with np.load(path) as values:
                data = dict(values)
            data["neighbor_indices"] = np.array([0.5, 1.])
            np.savez(path, **data)
            with self.assertRaises(ValueError):
                SurfaceRecord.from_npz(path)

    def test_clash_semantics_and_rigid_validation(self):
        """CA queries target heavy atoms and early abort returns zero heavy count."""
        record = surface()
        target = np.array([[.1, 0., 0.], [5.1, 0., 0.]])
        self.assertEqual(count_clashes(record, target, np.eye(4), ca_threshold=0), (1, 0))
        self.assertEqual(count_clashes(record, target, np.eye(4), ca_threshold=1), (1, 2))
        scale = np.eye(4)
        scale[0, 0] = 2
        with self.assertRaises(ValueError):
            transform_points(record.points, scale)

    def test_features_degenerate_descriptor(self):
        """Identical descriptors retain the legacy undefined inverse; NN rejects nonfinite inputs."""
        patch = surface().patch(0)
        raw, descriptor_score = alignment_features(patch, patch, np.zeros((1, 3)))
        self.assertTrue(np.isinf(raw[0, 1]))
        self.assertTrue(np.isinf(descriptor_score))
        with self.assertRaises(ValueError):
            AlignmentEvaluationNN().prepare(raw, np.random.RandomState(42))

    def test_replay_requires_pose(self):
        """Replay cannot fabricate a missing candidate pose or accept a nonrigid pose."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.npz"
            np.savez(path, keys=np.array(["0:example:0"]), transforms=np.eye(4)[None])
            provider = ReplayRegistration(path)
            patch = surface().patch(0)
            matrix, fitness = provider.align("0:example:0", patch, patch)
            np.testing.assert_array_equal(matrix, np.eye(4))
            self.assertEqual(fitness, 1.)
            moved = dict(patch, points=patch["points"] + np.array([1., 0., 0.]))
            self.assertEqual(provider.align("0:example:0", patch, moved)[1], 0.)
            with self.assertRaises(ValueError):
                provider.align("missing", patch, patch)
            np.savez(path, keys=np.array(["same", "same"]), transforms=np.repeat(np.eye(4)[None], 2, axis=0))
            with self.assertRaises(ValueError):
                ReplayRegistration(path)

    def test_sampling_stream_and_extension(self):
        """Oversized IPA packets consume consecutive legacy RNG choices; tasks admit new owners."""
        model = AlignmentEvaluationNN(max_npoints=100)
        raw = np.arange(440, dtype=np.float64).reshape(110, 4)
        expected_rng = np.random.RandomState(42)
        for _ in range(2):
            _, rows = model.prepare(raw, model._sampling_rng)
            np.testing.assert_array_equal(rows, expected_rng.choice(110, 100, replace=False))
        class CustomTask:
            """Example third-party owner implementing only the required run interface."""
            def __init__(self, config):
                """Store extension-specific configuration."""
                self.value = config["value"]
            def run(self):
                """Return the requested application value."""
                return self.value
        register_task("test_custom_design_owner", CustomTask)
        self.assertEqual(build_task(dict(task="test_custom_design_owner", value=17)).run(), 17)
        with self.assertRaises(ValueError):
            register_task("test_custom_design_owner", CustomTask)

    @unittest.skipUnless(importlib.util.find_spec("Bio"), "Install the design extra for PDB output tests")
    def test_seed_pdb_hydrogens(self):
        """Seed output removes H by original atom-name rule and applies the pose only once."""
        from Bio.PDB import PDBParser
        with tempfile.TemporaryDirectory() as directory:
            source = surface()
            source.pdb_path = Path(directory) / "input.pdb"
            source.pdb_path.write_text(
                "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N  \n"
                "ATOM      2  H   ALA A   1       1.100   2.000   3.000  1.00  0.00           H  \nEND\n")
            matrix = np.eye(4)
            matrix[:3, 3] = [2., 0., 0.]
            output = Path(directory) / "aligned.pdb"
            write_aligned_pdb(source, matrix, output)
            atoms = list(PDBParser(QUIET=True).get_structure("result", output).get_atoms())
            self.assertEqual([atom.get_name() for atom in atoms], ["N"])
            np.testing.assert_array_equal(atoms[0].coord, [3., 2., 3.])


if __name__ == "__main__":
    unittest.main()
