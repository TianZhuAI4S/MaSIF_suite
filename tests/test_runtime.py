"""Validate hardware selection without installing wheels or changing model settings."""

import contextlib
import io
import unittest
from unittest.mock import patch

from masif_torch import runtime


def gpu(capability='8.9', driver='560.35.03'):
    """Represent one visible GPU using the same normalized metadata as detection."""
    return dict(capability=runtime.version_tuple(capability), driver=runtime.version_tuple(driver))


class RuntimeTests(unittest.TestCase):
    """Check supported architectures, drivers, and explicit nonmutating planning."""

    def test_selection(self):
        """Cover CPU, validated CUDA12, Blackwell, and short version normalization."""
        self.assertEqual(runtime.select_backend('auto', []), 'cpu')
        self.assertEqual(runtime.select_backend('auto', [gpu()]), 'cu126')
        self.assertEqual(runtime.select_backend('auto', [gpu('5.0')]), 'cu126')
        self.assertEqual(runtime.select_backend('auto', [gpu('10.0','580')]), 'cu130')
        self.assertEqual(runtime.select_backend('cu130', [gpu('7.5','580')]), 'cu130')

    def test_incompatible_hardware(self):
        """Reject old drivers and mixed architectures before a package mutation."""
        for requested, devices in [('cu126',[gpu(driver='520')]), ('cu130',[gpu('7.0','580')]),
                                   ('auto',[gpu('10.0','580'),gpu('7.0','580')]),
                                   ('cu126',[gpu('10.0','580')]),('auto',[gpu('3.5')])]:
            with self.subTest(requested=requested, devices=devices), self.assertRaises(ValueError):
                runtime.select_backend(requested, devices)

    def test_dry_run(self):
        """Ensure a plan performs neither Torch inspection nor pip installation."""
        with patch.object(runtime,'detect_gpus',return_value=[gpu()]), patch.object(runtime,'inspect_runtime') as inspect, patch.object(runtime.subprocess,'run') as run, contextlib.redirect_stdout(io.StringIO()):
            report=runtime.main(['--dry-run'])
        self.assertEqual(report['backend'],'cu126')
        self.assertIn('torch==2.14.1+cu126', report['command'])
        inspect.assert_not_called()
        run.assert_not_called()

    def test_matching_runtime(self):
        """Verify a matching runtime without reinstalling or retaining imported Torch."""
        with patch.object(runtime,'detect_gpus',return_value=[gpu()]), patch.object(runtime,'inspect_runtime',return_value=dict(version='2.14.1+cu126',cuda='12.6')), patch.object(runtime,'verify_runtime',return_value={'kernel_checks_passed':True}), patch.object(runtime.subprocess,'run') as run, contextlib.redirect_stdout(io.StringIO()):
            report=runtime.main([])
        self.assertFalse(report['installed'])
        run.assert_not_called()

    def test_console_exit_status(self):
        """Console wrappers must return None rather than pass a report to sys.exit."""
        from masif_torch.entrypoint import main
        with patch.object(runtime, 'main', return_value={'dry_run': True}) as setup:
            self.assertIsNone(main(['install-runtime', '--backend', 'cpu', '--dry-run']))
        setup.assert_called_once_with(['--backend', 'cpu', '--dry-run'])

    def test_hidden_devices(self):
        """Respect CPU-only job allocation before attempting to inspect the driver."""
        with patch.dict(runtime.os.environ,{'CUDA_VISIBLE_DEVICES':'-1'}), patch.object(runtime.shutil,'which') as which:
            self.assertEqual(runtime.detect_gpus(),[])
        which.assert_not_called()


if __name__ == '__main__':
    unittest.main()
