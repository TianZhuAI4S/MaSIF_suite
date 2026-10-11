"""Prepare Rosetta ligand inputs without importing the neural-network package."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from Bio import PDB


def make_sdf(config: dict, output: Path) -> dict:
    """Complete ligand hydrogens while preserving all existing atomic coordinates."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    ligand = config["ligand"]
    lines = [line for line in Path(config["target"]).read_text().splitlines()
             if line.startswith("HETATM") and line[17:20].strip() == ligand
             and (config.get("ligand_chain") is None or line[21] == config["ligand_chain"])]
    if not lines:
        raise ValueError("The target PDB does not contain the requested ligand")
    molecule = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", removeHs=False)
    if molecule is None:
        raise ValueError("RDKit could not read the target ligand")
    template = config.get("template_sdf")
    if template:
        reference = Chem.SDMolSupplier(template, removeHs=False)[0]
        if reference is None:
            raise ValueError("The ligand connectivity template is invalid")
        molecule = AllChem.AssignBondOrdersFromTemplate(Chem.RemoveHs(reference), molecule)
    previous_atoms = molecule.GetNumAtoms()
    previous_coordinates = np.array(molecule.GetConformer().GetPositions(), copy=True)
    # PDB ligands may contain only some hydrogens; complete implicit valences as well.
    molecule = Chem.AddHs(molecule, addCoords=True)
    added = molecule.GetNumAtoms() > previous_atoms
    coordinates_preserved = np.array_equal(previous_coordinates,
        molecule.GetConformer().GetPositions()[:previous_atoms])
    if not coordinates_preserved:
        raise AssertionError("Ligand protonation changed existing atomic coordinates")
    writer = Chem.SDWriter(str(output / "ligand.sdf"))
    writer.write(molecule)
    writer.close()
    return dict(added_hydrogens=added, existing_coordinates_bitwise_equal=coordinates_preserved, ligand_atoms=molecule.GetNumAtoms(),
                ligand_heavy_atoms=molecule.GetNumHeavyAtoms())


def assemble_complex(config: dict, output: Path) -> dict:
    """Keep coordinates and put target/ligand in A and the binder in B."""
    target = [line for line in Path(config["target"]).read_text().splitlines()
              if line.startswith("ATOM") and line[21] in config.get("target_chains", "A")]
    binder = [line for line in Path(config["binder"]).read_text().splitlines()
              if line.startswith("ATOM") and line[21] in config.get("binder_chains", "B")]
    ligand = [line for line in (output / (config["ligand"] + ".pdb")).read_text().splitlines()
              if line.startswith("HETATM")]
    if not target or not binder or not ligand:
        raise ValueError("Target, binder and parameterized ligand must all contain atoms")
    if len(set(line[21] for line in target)) != 1 or len(set(line[21] for line in binder)) != 1:
        raise ValueError("This refinement adapter supports one protein chain per partner")
    residue = max(int(line[22:26]) for line in target) + 1
    target = [line[:21] + "A" + line[22:] for line in target]
    ligand = [line[:21] + "A" + f"{residue:4d}" + line[26:] for line in ligand]
    binder = [line[:21] + "B" + line[22:] for line in binder]
    (output / "complex.pdb").write_text("\n".join(target + ligand + ["TER"] + binder + ["TER", "END"]) + "\n")
    params = output / (config["ligand"] + ".params")
    # Original preparation removes PDB_ROTAMERS to avoid relative conformer paths.
    params.write_text("\n".join(line for line in params.read_text().splitlines()
                               if not line.startswith("PDB_ROTAMERS")) + "\n")
    return dict(target_atoms=len(target), binder_atoms=len(binder), ligand_atoms=len(ligand),
                ligand_residue=residue)


def fixed_residues(config: dict, pdb_path: Path) -> str:
    """Execute the original ligand-contact selector with its radii and side-chain rules."""
    source = Path(config["protocol_dir"]).parent / "utils/other/get_mol_targetres.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "contact_res")
    namespace = {"PDB": PDB, "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    contacts = namespace["contact_res"](str(pdb_path), "A", config["ligand"])
    return ",".join(f"{residue}A" for residue in contacts)


def main() -> None:
    """Run one preparation phase and write its machine-readable result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sdf", "assemble", "contacts"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdb", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "sdf":
        result = make_sdf(config, args.output)
    elif args.phase == "assemble":
        result = assemble_complex(config, args.output)
    else:
        result = {"fix": fixed_residues(config, args.pdb)}
    (args.output / (args.phase + "_result.json")).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
