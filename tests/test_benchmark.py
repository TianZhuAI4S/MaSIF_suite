"""Protect scientific benchmark resumption from mixing different numerical inputs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from masif_torch.tasks.benchmark.workflow import BenchmarkTask


class BenchmarkResumeTests(unittest.TestCase):
    """Check provenance gates before any expensive geometry process is started."""

    def test_changed_checkpoint_rejects_resume(self):
        """Replacing checkpoint bytes must reject existing case summaries."""
        self._assert_changed_input_rejected('checkpoint')

    def test_changed_definition_rejects_resume(self):
        """Changing complex definitions must reject existing case summaries."""
        self._assert_changed_input_rejected('definition')

    def test_changed_surface_rejects_resume(self):
        """A changed source surface fingerprint must reject existing case summaries."""
        self._assert_changed_input_rejected('surface')

    def _assert_changed_input_rejected(self, changed):
        """Construct genuine input hashes and verify the resume guard blocks execution."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, definition = root / 'weights.pt', root / 'definition.txt'
            checkpoint.write_bytes(b'original checkpoint bytes')
            definition.write_text('1A7X,A,B,FKA_B,1a7x_C_FKA\n')
            output = root / 'output'
            (output / 'case').mkdir(parents=True)
            task = BenchmarkTask(dict(checkpoint=str(checkpoint), definition=str(definition), output=str(output)))
            task.input_fingerprints = {'with_ligand/1A7X_A': 'original surface SHA256'}
            cases = [dict(output=str(output / 'case'))]
            (output / 'provenance.json').write_text(json.dumps(task.provenance(cases)))
            (output / 'case/summary.json').write_text('{}')
            if changed == 'checkpoint':
                checkpoint.write_bytes(b'a different checkpoint')
            elif changed == 'definition':
                definition.write_text('1A7X,B,A,FKA_B,1a7x_C_FKA\n')
            else:
                task.input_fingerprints['with_ligand/1A7X_A'] = 'changed surface SHA256'
            with patch.object(task, 'prepare', return_value=cases):
                with self.assertRaisesRegex(ValueError, 'inputs or protocol changed'):
                    task.run()

    def test_unverified_summary_rejects_resume(self):
        """Existence of a summary alone cannot certify a previous experiment."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, definition = root / 'weights.pt', root / 'definition.txt'
            checkpoint.write_bytes(b'checkpoint')
            definition.write_text('definition')
            (root / 'case').mkdir()
            (root / 'case/summary.json').write_text('{}')
            task = BenchmarkTask(dict(checkpoint=str(checkpoint), definition=str(definition), output=str(root)))
            task.input_fingerprints = {}
            with patch.object(task, 'prepare', return_value=[dict(output=str(root / 'case'))]):
                with self.assertRaisesRegex(ValueError, 'lack verified provenance'):
                    task.run()


if __name__ == '__main__':
    unittest.main()
