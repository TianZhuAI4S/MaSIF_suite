"""Preserve the original reader's ASCII attributes and binary PLY semantics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement

from masif_torch.shared.design.data import SurfaceRecord


class SurfacePrecisionTests(unittest.TestCase):
    """Guard the reproduced ASCII float-header narrowing regression."""

    def test_ascii_interface_reads_decimal_as_float64(self):
        """A float header must not narrow ASCII iface decimals before site selection."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'surface.ply'
            path.write_text('ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n'
                            'property float z\nproperty float iface\nend_header\n'
                            '0 0 0 0.148389\n1 2 3 0.189981\n')
            parsed = PlyData.read(path)
            actual = SurfaceRecord._interface_values(path, parsed)
            expected = np.array([float('0.148389'), float('0.189981')], dtype=np.float64)
            self.assertTrue(np.array_equal(actual, expected))
            self.assertFalse(np.array_equal(actual, np.asarray(parsed['vertex']['iface'], dtype=np.float64)))

    def test_binary_interface_keeps_stored_float32_values(self):
        """Binary float32 properties must preserve their actual bytes when widened."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'surface.ply'
            vertices = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('iface', 'f4')])
            vertices['iface'] = [0.148389, 0.189981]
            PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(path)
            actual = SurfaceRecord._interface_values(path, PlyData.read(path))
            self.assertTrue(np.array_equal(actual, vertices['iface'].astype(np.float64)))


if __name__ == '__main__':
    unittest.main()
