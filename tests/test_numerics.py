"""Meaningful oracle, malformed input, conversion, and resume regression tests."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from masif_torch.checkpoints import convert, load_checkpoint, save_checkpoint
from masif_torch.cli import predict
from masif_torch.data import PacketDataset, load_npz, pad_indices, flip_binder
from masif_torch.models import KerasBatchNorm1d, MaSIF_site, MaSIF_ppi_search, MaSIF_ligand, ScoreNN
from masif_torch.optim import TFAdam
from masif_torch.training import Trainer
from masif_torch.validation import compare, deterministic, validate, validate_adam

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / "artifacts/source_oracles"


class NumericalTests(unittest.TestCase):
    """Compare against original TensorFlow arithmetic instead of mirrored tests."""

    def setUp(self):
        """Fix CPU arithmetic for repeatable numerical regression tests."""
        deterministic()

    def test_source_oracles(self):
        """Cover site 1/2/3, search, ligand and every parameter gradient."""
        if not (ORACLES / "ligand_1/config.json").exists():
            self.skipTest("Generate tools/create_source_oracles.py references first")
        for name in ("site_1", "site_2", "site_3", "ppi_search_1", "ligand_1"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                checkpoint = Path(temporary) / "model.pt"
                convert(ORACLES / name, checkpoint, include_optimizer=True)
                report = validate(ORACLES / name, checkpoint)
                self.assertTrue(report["passed"], {k: v for k, v in report["tensors"].items() if not v["passed"]})

    def test_adam_oracle(self):
        """Tiny-gradient steps distinguish TF epsilon semantics from torch Adam."""
        path = ORACLES / "adam_reference.npz"
        if not path.exists():
            self.skipTest("TensorFlow Adam reference not generated")
        self.assertTrue(validate_adam(path)["passed"])
        score = ORACLES / "score_adam_reference.npz"
        if score.exists():
            self.assertTrue(validate_adam(score)["passed"])

    def test_bn_train_eval_oracle(self):
        """Check training gradients, population moving variance and subsequent inference."""
        path = ORACLES / "bn_reference.npz"
        if not path.exists():
            self.skipTest("TensorFlow BN reference not generated")
        with np.load(path, allow_pickle=False) as reference:
            module = KerasBatchNorm1d(2, eps=1e-3, momentum=0.01)
            x = torch.tensor(reference["input"], requires_grad=True)
            y = module(x)
            y.square().sum().backward()
            for actual, key in ((y.detach().numpy(), "train"), (x.grad.numpy(), "gradient"),
                                (module.running_mean.numpy(), "moving_mean"),
                                (module.running_var.numpy(), "moving_variance")):
                self.assertTrue(compare(actual, reference[key])["passed"], key)
            module.eval()
            self.assertTrue(compare(module(x).detach().numpy(), reference["evaluation"])["passed"])

    def test_resume_next_update(self):
        """Saving and restoring model plus Adam moments must preserve the next update."""
        model = MaSIF_ligand(n_thetas=2, n_rhos=2, n_rotations=2, n_ligands=3,
                             feat_mask=(1, 1), dropout=0.25)
        rng = np.random.default_rng(4)
        packet = dict(features=rng.normal(size=(4, 3, 2)).astype(np.float32),
                      rho=rng.uniform(0, 12, (4, 3)).astype(np.float32),
                      theta=rng.uniform(0, 6, (4, 3)).astype(np.float32),
                      mask=np.ones((4, 3), dtype=np.float32), labels=np.array([0, 1, 0], dtype=np.float32))
        trainer = Trainer(model)
        trainer.train_step(packet)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_checkpoint(model, path, trainer.optimizer)
            trainer.train_step(packet)
            restored, payload = load_checkpoint(path)
            other = Trainer(restored)
            other.restore(payload)
            other.train_step(packet)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_bad_indices_and_nonfinite(self):
        """Reject negative neighbors, NaN features, and unsupported site shapes."""
        model = MaSIF_site(n_thetas=2, n_rhos=2, n_rotations=2, n_conv_layers=2, feat_mask=(1,))
        x, coordinate = torch.ones(2, 2, 1), torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            model(x, coordinate, coordinate, torch.ones_like(coordinate), torch.full((2, 2), -1))
        with self.assertRaises(ValueError):
            model.loss(torch.zeros(2, 2), torch.eye(2), torch.tensor([-1]), torch.tensor([0]))
        x[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(x, coordinate, coordinate, torch.ones_like(coordinate), torch.zeros(2, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            MaSIF_site(n_conv_layers=4)
        with self.assertRaises(ValueError):
            MaSIF_ppi_search().loss(torch.zeros(3, 80))

    def test_feature_reflection_and_padding(self):
        """Retain hydrophobicity by original column identity, including partial masks."""
        mask = [0, 1, 0, 0, 1]
        packet = dict(features=np.array([[[2.0, 3.0]]]), theta=np.array([[1.0]]))
        flipped = flip_binder(packet, mask)
        np.testing.assert_array_equal(flipped["features"], [[[-2, 3]]])
        np.testing.assert_array_equal(packet["features"], [[[2, 3]]])
        np.testing.assert_array_equal(pad_indices([[1], [0, 1]], 3), [[1, 0, 0], [0, 1, 1]])

    def test_custom_adam_group(self):
        """Parameter groups must initialize their own beta powers."""
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = TFAdam([dict(params=[parameter], betas=(0.8, 0.9))])
        self.assertEqual(float(optimizer.param_groups[0]["beta1_power"]), float(torch.tensor(0.8)))

    def test_dataset_filter_and_score_batches(self):
        """Bound score batches, visit each alignment once, and report skipped pockets."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path / "score.npz", features=np.arange(10)[:, None, None], labels=np.arange(10) % 2)
            dataset = PacketDataset(path / "score.npz", {"task": "score"}, batch_size=4)
            self.assertEqual([len(dataset[i]["features"]) for i in range(len(dataset))], [4, 4, 2])
            np.testing.assert_array_equal(np.sort(np.concatenate([dataset[i]["features"].ravel()
                                                                   for i in range(len(dataset))])), np.arange(10))
            np.savez(path / "good.npz", pocket_labels=np.ones((32, 1)), features=np.zeros((32, 2, 1)))
            np.savez(path / "bad.npz", pocket_labels=np.ones((31, 1)), features=np.zeros((31, 2, 1)))
            ligand = path / "ligand"
            ligand.mkdir()
            (path / "good.npz").rename(ligand / "good.npz")
            (path / "bad.npz").rename(ligand / "bad.npz")
            dataset = PacketDataset(ligand, {"task": "ligand"}, batch_size=32)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataset.excluded), 1)

    def test_score_weighted_train_unweighted_eval(self):
        """Match Keras class-weight reduction in training and unweighted validation."""
        model = ScoreNN()
        trainer = Trainer(model, class_weights=(0.1, 2.0))
        packet = dict(features=np.zeros((3, 8, 3), dtype=np.float32), labels=np.array([0, 0, 1]))
        model.train()
        output, loss = trainer.compute(packet)
        per_sample = torch.nn.functional.cross_entropy(output, torch.tensor([0, 0, 1]), reduction="none")
        self.assertTrue(torch.allclose(loss, (per_sample * torch.tensor([0.1, 0.1, 2.0])).mean()))
        model.eval()
        output, loss = trainer.compute(packet)
        self.assertTrue(torch.equal(loss, torch.nn.functional.cross_entropy(output, torch.tensor([0, 0, 1]))))
        self.assertEqual(trainer.optimizer.param_groups[0]["eps"], 1e-7)

    def test_resume_history_best(self):
        """Keep old history and best metric when resuming into a new output directory."""
        model = ScoreNN()
        trainer = Trainer(model)
        packet = dict(features=np.zeros((2, 8, 3), dtype=np.float32), labels=np.array([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path / "packet.npz", **packet)
            dataset = PacketDataset(path / "packet.npz", model.config, batch_size=2)
            trainer.best_metric = -1.0
            trainer.history = [dict(epoch=0, step=0, training_loss=0.0)]
            trainer.fit(dataset, path / "run", epochs=1, start_epoch=1)
            restored, payload = load_checkpoint(path / "run/last.pt")
            other = Trainer(restored)
            self.assertEqual(other.restore(payload), 2)
            self.assertEqual(other.best_metric, -1.0)
            self.assertEqual(len(other.history), 2)
            self.assertFalse((path / "run/best.pt").exists())

    def test_record_conversion(self):
        """One-time TFRecord conversion must preserve every raw array exactly."""
        source = ORACLES / "record_expected.npz"
        converted = ROOT / "artifacts/records/SYNTHETIC_A.npz"
        if not source.exists() or not converted.exists():
            self.skipTest("Generate and convert the synthetic TFRecord first")
        expected, actual = load_npz(source), load_npz(converted)
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


if __name__ == "__main__":
    unittest.main()
