"""Regression checks for molecule format adapters and interpreter boundaries."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from masif_torch.tasks.preparation import PreparationTask
from masif_torch.tasks.preparation.worker_chemistry import protonate_ligand


class PreparationTests(unittest.TestCase):
    """Catch input changes that silently alter the scientific surface pipeline."""

    @unittest.skipUnless(importlib.util.find_spec('rdkit'), 'Install the chemistry extra')
    def test_partial_ligand_hydrogens(self):
        """Complete a partially hydrated ligand and preserve all original PDB atom lines."""
        from rdkit import Chem
        molecule = Chem.MolFromSmiles('CO')
        conformer = Chem.Conformer(2)
        conformer.SetAtomPosition(0, (0., 0., 0.))
        conformer.SetAtomPosition(1, (1.4, 0., 0.))
        molecule.AddConformer(conformer)
        molecule = Chem.AddHs(molecule, addCoords=True)
        for index, atom in enumerate(molecule.GetAtoms()):
            info = Chem.AtomPDBResidueInfo()
            info.SetName((atom.GetSymbol() + str(index + 1)).rjust(4))
            info.SetResidueName('LIG')
            info.SetChainId('B')
            info.SetResidueNumber(201)
            info.SetIsHeteroAtom(True)
            atom.SetMonomerInfo(info)
        records = [line for line in Chem.MolToPDBBlock(molecule).splitlines() if line.startswith('HETATM')]
        original = records[:3]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tmp').mkdir()
            pdb = root / 'tmp/TEST_protonated.pdb'
            pdb.write_text('\n'.join(original + ['END']) + '\n')
            sdf = root / 'template.sdf'
            writer = Chem.SDWriter(str(sdf))
            writer.write(molecule)
            writer.close()
            protonate_ligand(dict(output=str(root), identifier='TEST_A', ligand='LIG_B', sdf=str(sdf)))
            added = [line for line in pdb.read_text().splitlines() if line.startswith('HETATM')]
            self.assertEqual(added[:3], original)
            self.assertEqual(len(added), molecule.GetNumAtoms())
            names = [line[12:16].strip() for line in added]
            self.assertEqual(len(names), len(set(names)))
            saved = pdb.read_text()
            protonate_ligand(dict(output=str(root), identifier='TEST_A', ligand='LIG_B', sdf=str(sdf)))
            self.assertEqual(pdb.read_text(), saved)

    def test_precompute_checkpoint_mismatch(self):
        """Reject missing requested preprocessing before invoking external programs."""
        with self.assertRaisesRegex(ValueError, 'search_checkpoint requires ppi_search'):
            PreparationTask(dict(pdb='missing.pdb', source_root='.', legacy_python='python',
                                 output='out', precompute=['site'], search_checkpoint='model.pt'))


if __name__ == '__main__':
    unittest.main()
