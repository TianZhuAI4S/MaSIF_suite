"""Configuration composition and exact native-training frontend regressions."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra.errors import ConfigCompositionException
from omegaconf import MissingMandatoryValue
import numpy as np
import torch

from masif_torch.cli import main
from masif_torch.configuration import compose_config, configuration_dict, load_config_file

ROOT = Path(__file__).resolve().parents[1]


def assert_identical(test, left, right):
    """Compare complete checkpoint state recursively, including optimizer and RNG tensors."""
    test.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_identical(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_identical(test, a, b)
    else:
        test.assertEqual(left, right)


class ConfigurationTests(unittest.TestCase):
    """Exercise actual Hydra defaults and actual ScoreNN training/resumption."""

    def test_training_groups_and_overrides(self):
        """Every supported training selector composes its matching model group."""
        for name in ('site', 'search', 'ligand', 'score', 'alignment'):
            config = configuration_dict(compose_config('train', [f'task={name}',
                'dataset.input=/tmp/packets', 'trainer.output=/tmp/train', 'trainer.seed=7']))
            expected = 'ppi_search' if name == 'search' else name
            self.assertEqual(config['task'], expected)
            self.assertEqual(config['model']['task'], expected)
            self.assertEqual(config['trainer']['seed'], 7)
        cache = compose_config('train', ['task=search', 'dataset=search_cache'])
        self.assertEqual(cache.dataset.kind, 'search_cache')
        with self.assertRaises(ConfigCompositionException):
            compose_config('train', ['trainer.unknown_option=1'])
        with self.assertRaises(MissingMandatoryValue):
            configuration_dict(compose_config('site'))

    def test_examples_and_relative_paths(self):
        """User examples compose and relative values retain the calling directory semantics."""
        cwd = Path.cwd()
        for path in (ROOT / 'examples').glob('*.yaml'):
            self.assertIsInstance(load_config_file(path), dict, path)
        self.assertEqual(Path.cwd(), cwd)
        config = compose_config('site', ['input=relative/packet.npz'])
        self.assertEqual(config.input, 'relative/packet.npz')
        self.assertFalse(list((ROOT / 'examples').glob('*.json')))

    def test_dry_run_and_legacy_mapping(self):
        """Dry runs tolerate missing paths and never build a task; old JSON remains readable."""
        with patch('sys.argv', ['masif', 'site', '--cfg', 'job']), redirect_stdout(io.StringIO()):
            main()
        with patch('masif_torch.tasks.build_task', side_effect=AssertionError('Unexpected execution')):
            output = io.StringIO()
            with redirect_stdout(output):
                main(['site', '--cfg', 'job'])
            self.assertIn('checkpoint: ???', output.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.json'
            path.write_text(json.dumps({'task': 'score'}))
            self.assertEqual(load_config_file(path), {'task': 'score'})
            with self.assertRaisesRegex(ValueError, 'command and configured task'):
                main(['site', '--config', str(path)])
            with self.assertRaisesRegex(ValueError, 'training configuration'):
                main(['train', '--config', str(path)])
            with self.assertRaises(ValueError):
                load_config_file(path, ['task=site'])
            short_suffix = Path(directory) / 'config.yml'
            short_suffix.write_text('task: score\n')
            with self.assertRaisesRegex(ValueError, '.yaml extension'):
                load_config_file(short_suffix)

    def test_training_and_resume_exact(self):
        """YAML versus legacy frontend preserves every parameter, Adam state and RNG byte."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet = root / 'packet.npz'
            rng = np.random.default_rng(18)
            np.savez(packet, features=rng.normal(size=(4, 6, 3)).astype(np.float32),
                     labels=np.array([0, 1, 0, 1], dtype=np.int64))
            model = root / 'model.yaml'
            model.write_text('task: score\n')
            legacy, hydra = root / 'legacy', root / 'hydra'
            with redirect_stdout(io.StringIO()):
                main(['train', '--input', str(packet), '--output', str(legacy),
                      '--config', str(model), '--epochs', '1', '--seed', '5',
                      '--class-weights', '0.1', '2', '--learning-rate', '0.0002'])
                main(['train', 'task=score', f'dataset.input={packet}', f'trainer.output={hydra}',
                      'trainer.epochs=1', 'trainer.seed=5', 'trainer.class_weights=[0.1,2]',
                      'trainer.learning_rate=0.0002'])
            first_a = torch.load(legacy / 'last.pt', weights_only=False)
            first_b = torch.load(hydra / 'last.pt', weights_only=False)
            assert_identical(self, first_a, first_b)
            self.training_evidence = dict(fresh_checkpoint_exact=True, model_tensors=len(first_a["state_dict"]),
                fresh_history=first_a["history"], seed=first_a["seed"])
            self.assertTrue((hydra / "resolved_config.yaml").is_file())
            checkpoint = root / 'initial.pt'
            checkpoint.write_bytes((legacy / 'last.pt').read_bytes())
            with redirect_stdout(io.StringIO()):
                main(['train', '--input', str(packet), '--output', str(legacy),
                      '--checkpoint', str(checkpoint), '--resume', '--epochs', '1',
                      '--class-weights', '0.2', '1', '--learning-rate', '0.0003'])
                main(['train', 'task=score', f'dataset.input={packet}', f'trainer.output={hydra}',
                      f'trainer.checkpoint={checkpoint}', 'trainer.resume=true', 'trainer.epochs=1',
                      'trainer.class_weights=[0.2,1]', 'trainer.learning_rate=0.0003'])
            assert_identical(self, torch.load(legacy / 'last.pt', weights_only=False),
                             torch.load(hydra / 'last.pt', weights_only=False))
            self.training_evidence.update(resumed_checkpoint_exact=True,
                resumed_history=json.loads((hydra / "history.json").read_text()))
            self.assertEqual(json.loads((hydra / 'history.json').read_text())[-1]['epoch'], 1)
            with self.assertRaisesRegex(ValueError, 'checkpoint task'):
                main(['train', 'task=site', f'dataset.input={packet}', f'trainer.output={hydra}',
                      f'trainer.checkpoint={checkpoint}', 'trainer.resume=true', 'trainer.epochs=1'])


if __name__ == '__main__':
    unittest.main()
