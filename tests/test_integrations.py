"""Guard numerical provenance and external-stage failure reporting without expensive jobs."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("Bio"), "Install the design extra for integration coverage")
class IntegrationTests(unittest.TestCase):
    """Exercise recurring failure paths using isolated files and checked task contracts."""

    def test_rosetta_requested_repeat_failure(self):
        """A failed required repeat must retain success=False after successful earlier stages."""
        from masif_torch.tasks.rosetta.workflow import RosettaWorkflow
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol"
            protocol.mkdir()
            for name in ("relax.xml", "interface_design.xml", "no_ref.rosettacon2018.beta_nov16.txt"):
                (protocol / name).write_text("original source")
            task = RosettaWorkflow.__new__(RosettaWorkflow)
            task.output, task.protocol_dir, task.database = root, protocol, str(root / "database")
            task.report = dict(task="rosetta", success=False, stages=[], seed=42)
            task.config = dict(ligand="FKA", legacy_python="python", verify_repeat=True,
                               ligand_sdf=str(root / "ligand_input.sdf"))
            (root / "ligand_input.sdf").write_text("input")
            outcomes = [dict(pdb=str(root / "relaxed.pdb"), scores={}),
                        dict(pdb=str(root / "designed.pdb"), scores={}),
                        RuntimeError("required repeat failed")]
            with patch.object(task, "_prepare_phase", return_value={"fix": "82A"}), \
                    patch.object(task, "_execute"), \
                    patch.object(task, "_protocol", side_effect=outcomes), \
                    patch("masif_torch.tasks.rosetta.workflow.pdb_coordinates"):
                with self.assertRaisesRegex(RuntimeError, "required repeat failed"):
                    task.run()
            self.assertFalse(task.report["success"])

    def test_mol2_substructure_idempotent(self):
        """Restore a missing residue table without changing atom/bond/charge text."""
        from masif_torch.tasks.preparation.worker_chemistry import complete_mol2
        original = ("@<TRIPOS>MOLECULE\nFKA\n2 1 0 0 0\nSMALL\nUSER_CHARGES\n\n"
                    "@<TRIPOS>ATOM\n1 C1 1.0 2.0 3.0 C.3 1 FKA -0.125\n"
                    "2 O1 2.0 2.0 3.0 O.3 1 FKA 0.125\n@<TRIPOS>BOND\n1 1 2 1\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ligand.mol2"
            path.write_text(original)
            complete_mol2(path)
            completed = path.read_text()
            before = original.split("@<TRIPOS>ATOM\n", 1)[1]
            after = completed.split("@<TRIPOS>ATOM\n", 1)[1].split("@<TRIPOS>SUBSTRUCTURE", 1)[0]
            self.assertEqual(before, after)
            self.assertEqual(completed.splitlines()[2], "2 1 1 0 0")
            self.assertIn("1 FKA 1 GROUP 0 **** FKA 0 ROOT", completed)
            complete_mol2(path)
            self.assertEqual(path.read_text(), completed)


if __name__ == "__main__":
    unittest.main()
