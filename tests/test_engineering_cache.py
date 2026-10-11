"""Regression contracts for exact geometry reuse and mutable-input invalidation."""

import unittest

import numpy as np
from scipy.spatial import cKDTree

from masif_torch.models import AlignmentEvaluationNN
from masif_torch.shared.design.features import AlignmentScorer, alignment_features, match_descriptors
from masif_torch.tasks.seed.workflow import SeedWorkflow
from masif_torch.shared.design.data import SurfaceRecord


class GeometryCacheTests(unittest.TestCase):
    """Check coordinate identity, duplicate-point ties and bounded owned tree storage."""

    def test_mutation_invalidates_owned_tree(self):
        """In-place input edits must neither reuse stale trees nor corrupt previous indexes."""
        scorer = AlignmentScorer(AlignmentEvaluationNN())
        values = np.array([[0., 0., 0.], [2., 0., 0.]])
        old = scorer.coordinate_tree(values)
        self.assertIs(old, scorer.coordinate_tree(values.copy()))
        values[0, 0] = 10.
        new = scorer.coordinate_tree(values)
        self.assertIsNot(old, new)
        self.assertEqual(old.query([[0., 0., 0.]])[0][0], 0.)
        np.testing.assert_array_equal(new.query([[0., 0., 0.]])[0], cKDTree(values).query([[0., 0., 0.]])[0])

    def test_duplicate_ties_and_dtype(self):
        """Cached construction preserves exact nearest indices and feature arithmetic with ties."""
        scorer = AlignmentScorer(AlignmentEvaluationNN())
        target = dict(points=np.array([[0., 0., 0.], [0., 0., 0.], [2., 0., 0.]]),
                      descriptors=np.array([[1., 0.], [2., 0.], [3., 0.]]),
                      normals=np.array([[0., 0., 1.]]*3))
        source = dict(points=np.array([[.1, 0., 0.], [1., 0., 0.]]),
                      descriptors=np.array([[4., 0.], [5., 0.]]), normals=np.array([[0., 0., -1.]]*2))
        for dtype in [np.float32, np.float64]:
            atoms = np.array([[0., 0., 0.], [3., 0., 0.]], dtype=dtype)
            baseline = alignment_features(target, source, atoms)
            cached = alignment_features(target, source, atoms,
                point_tree=scorer.coordinate_tree(target['points']), atom_tree=scorer.coordinate_tree(atoms))
            np.testing.assert_array_equal(cached[0], baseline[0])
            self.assertEqual(cached[1], baseline[1])

    def test_patch_template_isolates_provider_mutation(self):
        """Writable per-pose copies prevent a custom provider from corrupting later target patches."""
        workflow = object.__new__(SeedWorkflow)
        workflow.config, workflow.shift, workflow._target_patches = {}, .25, {}
        workflow.target = SurfaceRecord('target', np.array([[0.,0.,0.]]), np.array([[0.,0.,1.]]),
            np.array([[1.,2.]]), np.ones(1), [np.array([0])], flipped_descriptors=np.array([[3.,4.]]))
        first = workflow._target_patch(0)
        first['points'][:] = 50.
        first['descriptors'][:] = 80.
        first['normals'][:] = -50.
        second = workflow._target_patch(0)
        expected = workflow.target.patch(0,.25,flipped=True)
        for name in ['points','normals','descriptors']:
            np.testing.assert_array_equal(second[name], expected[name])
            self.assertTrue(second[name].flags.writeable)

    def test_descriptor_thresholds_and_vertex_order(self):
        """Keep strict cutoff boundaries and ascending vertices without intersection sorting."""
        surface = SurfaceRecord('source', np.zeros((4,3)), np.ones((4,3)),
            np.array([[1.,0.],[2.,0.],[.5,0.],[.25,0.]], dtype=np.float32),
            np.array([.75,1.,1.,1.]), [np.array([index]) for index in range(4)])
        vertices, distances = match_descriptors(surface,np.array([0.,0.],dtype=np.float32),.75,2.)
        np.testing.assert_array_equal(vertices,np.array([2,3]))
        np.testing.assert_array_equal(distances,np.array([.5,.25],dtype=np.float32))

    def test_cache_bound_and_clear_preserve_rng(self):
        """Variable inputs cannot grow storage indefinitely or advance the sampling stream."""
        scorer = AlignmentScorer(AlignmentEvaluationNN())
        before = scorer.rng.get_state()
        for index in range(20):
            scorer.coordinate_tree(np.array([[float(index), 0., 0.]]))
        self.assertLessEqual(len(scorer._coordinate_trees), 8)
        scorer.clear_geometry_cache()
        self.assertEqual(len(scorer._coordinate_trees), 0)
        after = scorer.rng.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == '__main__':
    unittest.main()
