"""Run original neosurf chemical functions with modern RDKit/OpenBabel, without PyMesh."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def protonate_ligand(config):
    """Add explicit ligand hydrogens while preserving every original heavy-atom line.

    Reduce does not protonate FKA in this input. RDKit uses the supplied bond-order
    template and keeps heavy coordinates in the original crystallographic Å frame.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    code, chain = config['ligand'].split('_')
    pdb_id = config['identifier'].split('_')[0]
    path = Path(config['output']) / 'tmp' / (pdb_id + '_protonated.pdb')
    lines = path.read_text().splitlines()
    selected = [i for i, line in enumerate(lines) if line.startswith('HETATM')
                and line[17:20].strip() == code and line[21] == chain]
    if not selected:
        raise ValueError('Requested ligand is absent from the reduced PDB')
    block = '\n'.join(lines[i] for i in selected) + '\nEND\n'
    molecule = Chem.MolFromPDBBlock(block, removeHs=False)
    if molecule is None:
        raise ValueError('RDKit could not parse the ligand')
    heavy = molecule.GetNumHeavyAtoms()
    original_count = molecule.GetNumAtoms()
    template = config.get('sdf')
    if not template:
        raise ValueError('Unprotonated ligand requires an SDF connectivity template')
    reference = Chem.SDMolSupplier(template, removeHs=False)[0]
    if reference is None:
        raise ValueError('Invalid SDF connectivity template')
    molecule = AllChem.AssignBondOrdersFromTemplate(Chem.RemoveHs(reference), molecule)
    coordinates = np.asarray(molecule.GetConformer().GetPositions()).copy()
    hydrated = Chem.AddHs(molecule, addCoords=True)
    if not np.array_equal(coordinates, hydrated.GetConformer().GetPositions()[:original_count]):
        raise ValueError('Ligand protonation changed heavy coordinates')
    residue = molecule.GetAtomWithIdx(0).GetPDBResidueInfo()
    existing_names = {lines[i][12:16].strip() for i in selected}
    hydrogen_number = 1
    for index in range(original_count, hydrated.GetNumAtoms()):
        info = Chem.AtomPDBResidueInfo()
        while 'H' + str(hydrogen_number) in existing_names:
            hydrogen_number += 1
        info.SetName(('H' + str(hydrogen_number)).rjust(4))
        hydrogen_number += 1
        info.SetResidueName(code)
        info.SetChainId(chain)
        info.SetResidueNumber(residue.GetResidueNumber())
        info.SetInsertionCode(residue.GetInsertionCode())
        info.SetIsHeteroAtom(True)
        hydrated.GetAtomWithIdx(index).SetMonomerInfo(info)
    records = [line for line in Chem.MolToPDBBlock(hydrated).splitlines()
               if line.startswith('HETATM')]
    extra = records[original_count:]
    last = selected[-1] + 1
    path.write_text('\n'.join(lines[:last] + extra + lines[last:]) + '\n')
    report = dict(added_hydrogens=len(extra), heavy_atoms=heavy, heavy_coordinates_exact=True,
                  method='RDKit AddHs(addCoords=True), original SDF template')
    (path.parent / 'ligand_protonation.json').write_text(json.dumps(report, indent=2) + '\n')


def complete_mol2(path):
    """Restore a MOL2 residue table omitted by OpenBabel3, retaining atoms/bonds/charges."""
    text = path.read_text()
    if '@<TRIPOS>SUBSTRUCTURE' in text:
        return
    lines = text.splitlines()
    atom_section = lines.index('@<TRIPOS>ATOM') + 1
    bond_section = lines.index('@<TRIPOS>BOND')
    residues = {}
    for line in lines[atom_section:bond_section]:
        fields = line.split()
        residues.setdefault(int(fields[6]), (fields[7], int(fields[0])))
    counts = lines[2].split()
    counts[2] = str(len(residues))
    lines[2] = ' '.join(counts)
    lines.append('@<TRIPOS>SUBSTRUCTURE')
    lines.extend('{} {} {} GROUP 0 **** {} 0 ROOT'.format(index, name, root, name)
                 for index, (name, root) in sorted(residues.items()))
    path.write_text('\n'.join(lines) + '\n')


def chemistry(config):
    """Compute original H-bond/hydrophobic features on exact mesh arrays from the legacy stage."""
    from triangulation.computeCharges import computeCharges
    from triangulation.computeHydrophobicity import computeHydrophobicity
    from triangulation.ligand_utils import extract_ligand
    temporary = Path(config['output']) / 'tmp'
    with np.load(temporary / 'mesh.npz', allow_pickle=False) as archive:
        data = dict(archive)
    ligand = config.get('ligand')
    code, chain = ligand.split('_') if ligand else (None, None)
    molecule = None
    if ligand:
        from rdkit import Chem
        pdb_id = config['identifier'].split('_')[0]
        molecule = extract_ligand(str(temporary / (pdb_id + '_protonated.pdb')), code, chain,
                                 str(temporary / 'ligand.mol2'), sdf_template=config.get('sdf'),
                                 patched_mol2_file=config.get('mol2'))
        writer = Chem.SDWriter(str(temporary / 'ligand.sdf'))
        writer.write(molecule)
        writer.close()
        complete_mol2(temporary / 'ligand.mol2')
    stem = str(temporary / config['identifier'])
    hbond = computeCharges(stem, data['raw_vertices'], data['names'], code, molecule)
    hphob = computeHydrophobicity(data['names'], code, molecule)
    if not all(np.isfinite(value).all() for value in (hbond, hphob)):
        raise ValueError('Nonfinite raw chemical features')
    # The geometry interpreter performs interpolation with the original KDTree.
    np.savez_compressed(temporary / 'raw_chemistry.npz', hbond=hbond, hphob=hphob)
    print('CHEMISTRY', len(data['vertices']), flush=True)


def main():
    """Load explicit source and pipeline inputs; keep the modern environment TensorFlow-free."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['chemistry', 'protonate-ligand'], default='chemistry')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, config['source_root'])
    {'chemistry': chemistry, 'protonate-ligand': protonate_ligand}[args.stage](config)


if __name__ == '__main__':
    main()
