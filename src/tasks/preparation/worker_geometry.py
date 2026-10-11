"""Run unchanged upstream PyMesh/MSMS/APBS routines in a compatible legacy interpreter."""

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def protonate_input(config):
    """Run the original Reduce protein protonation before optional ligand hydration."""
    from input_output.protonate import protonate
    code = config['identifier'].split('_')[0]
    protonate(config['pdb'], str(Path(config['output']) / 'tmp' / (code + '_protonated.pdb')))


def mesh(config):
    """Protonate/extract, triangulate and regularize original protein/ligand Å geometry."""
    import pymesh
    from sklearn.neighbors import KDTree
    from input_output.protonate import protonate
    from input_output.extractPDB import extractPDB
    from triangulation.computeMSMS import computeMSMS
    from triangulation.fixmesh import fix_mesh
    from triangulation.compute_normal import compute_normal
    from default_config.masif_opts import masif_opts
    output = Path(config['output'])
    temporary = output / 'tmp'
    code, chains = config['identifier'].split('_', 1)
    ligand = config.get('ligand')
    ligand_code, ligand_chain = ligand.split('_') if ligand else (None, None)
    protonated = temporary / (code + '_protonated.pdb')
    if not protonated.is_file():
        raise FileNotFoundError('Run the protonation stage before triangulation')
    stem = temporary / config['identifier']
    extractPDB(str(protonated), str(stem) + '.pdb', chains, ligand_code, ligand_chain)
    raw, faces, _, names, _ = computeMSMS(str(stem) + '.pdb', protonate=True, ligand_code=ligand_code)
    regular = fix_mesh(pymesh.form_mesh(raw, faces), masif_opts['mesh_res'])
    normals = compute_normal(regular.vertices, regular.faces)
    complete, _, _, _, _ = computeMSMS(str(protonated), protonate=True, ligand_code=ligand_code)
    distance, _ = KDTree(complete).query(regular.vertices)
    interface = np.zeros(len(regular.vertices))
    interface[np.where(np.square(distance) >= 2.0)[0]] = 1.
    np.savez_compressed(temporary / 'mesh.npz', raw_vertices=raw, names=np.asarray(names),
                        vertices=regular.vertices, faces=regular.faces, normals=normals, iface=interface)
    print('MESH', regular.num_vertices, regular.num_faces, flush=True)


def interpolate(config):
    """Run the exact upstream interpolation in the legacy KDTree numerical environment.

    Equidistant MSMS vertices can be ordered differently by modern sklearn.
    The standalone original function is loaded without importing RDKit in Python3.6.
    """
    from sklearn.neighbors import KDTree
    temporary = Path(config['output']) / 'tmp'
    path = Path(config['source_root']) / 'triangulation/computeCharges.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'assignChargesToNewMesh')
    namespace = {'np': np, 'KDTree': KDTree}
    exec(compile(ast.Module(body=[function]), str(path), 'exec'), namespace)
    from default_config.masif_opts import masif_opts
    with np.load(temporary / 'mesh.npz', allow_pickle=False) as archive:
        mesh_data = dict(archive)
    with np.load(temporary / 'raw_chemistry.npz', allow_pickle=False) as archive:
        raw = dict(archive)
    features = {key: namespace['assignChargesToNewMesh'](mesh_data['vertices'], mesh_data['raw_vertices'],
                                                       values, masif_opts) for key, values in raw.items()}
    if not all(np.isfinite(value).all() for value in features.values()):
        raise ValueError('Nonfinite interpolated chemical features')
    np.savez_compressed(temporary / 'chemistry.npz', **features)


def apbs(config, vertices, stem, mol2):
    """Run the original APBS commands with checked exit codes and explicit CLI dialect.

    PDB2PQR 2.1 accepts bare --apbs-input; the neosurf wrapper assumes a later
    version accepting --apbs-input=filename. Both request the same PB calculation.
    """
    from default_config.global_vars import apbs_bin, pdb2pqr_bin, multivalue_bin
    import csv
    dialect = config.get('pdb2pqr_apbs_input', 'flag')
    if dialect not in ('flag', 'filename'):
        raise ValueError('pdb2pqr_apbs_input must be flag or filename')
    flag = '--apbs-input' if dialect == 'flag' else '--apbs-input=' + stem.name + '.in'
    args = [pdb2pqr_bin, stem.name + '.pdb', stem.name, '--ff=PARSE', '--whitespace', '--noopt', flag]
    if mol2:
        args.append('--ligand=' + mol2)
    subprocess.run(args, cwd=str(stem.parent), check=True)
    subprocess.run([apbs_bin, stem.name + '.in'], cwd=str(stem.parent), check=True)
    np.savetxt(str(stem) + '.csv', vertices, delimiter=',', fmt='%.18e')
    subprocess.run([multivalue_bin, stem.name + '.csv', stem.name + '.dx', stem.name + '_out.csv'],
                   cwd=str(stem.parent), check=True)
    with open(str(stem) + '_out.csv') as handle:
        rows = list(csv.reader(handle))
    if len(rows) != len(vertices) or any(len(row) != 4 for row in rows):
        raise ValueError('APBS output rows must correspond to every surface vertex')
    charges = np.asarray([float(row[3]) for row in rows])
    if not np.isfinite(charges).all():
        raise ValueError('Nonfinite APBS potential')
    np.save(str(stem) + '_apbs.npy', charges)
    return charges


def color_site(config):
    """Write site probabilities with original PyMesh serialization and preserved geometry.

    Reading an ASCII PLY through its nominal float32 schema narrows decimal
    coordinates. PyMesh retains original decimal float64 values before output.
    """
    import pymesh
    output = Path(config['output'])
    identifier = config['identifier']
    mesh_data = pymesh.load_mesh(str(output / 'data_preparation/01-benchmark_surfaces' / (identifier + '.ply')))
    values = np.load(output / 'output/all_feat_3l/pred_data' / ('pred_' + identifier + '.npy'), allow_pickle=False)
    if values.shape != (1, mesh_data.num_vertices) or not np.isfinite(values).all():
        raise ValueError('Site probabilities must correspond to every surface vertex')
    mesh_data.remove_attribute('vertex_iface')
    mesh_data.add_attribute('iface')
    mesh_data.set_attribute('iface', values[0])
    # Structural attributes are written automatically; retaining them duplicates PLY fields.
    for name in ('vertex_x', 'vertex_y', 'vertex_z', 'face_vertex_indices'):
        mesh_data.remove_attribute(name)
    path = output / 'output/all_feat_3l/pred_surfaces' / (identifier + '.ply')
    path.parent.mkdir(parents=True, exist_ok=True)
    pymesh.save_mesh(str(path), mesh_data, *mesh_data.get_attribute_names(), use_float=True, ascii=True)
    reloaded = pymesh.load_mesh(str(path))
    if not np.array_equal(mesh_data.vertices, reloaded.vertices) or not np.array_equal(mesh_data.faces, reloaded.faces):
        raise ValueError('Prediction coloring changed the geometry')


def finalize(config):
    """Compute original APBS and write the upstream PLY including supplied chemical arrays."""
    from input_output.save_ply import save_ply
    output = Path(config['output'])
    temporary = output / 'tmp'
    stem = temporary / config['identifier']
    with np.load(temporary / 'mesh.npz', allow_pickle=False) as archive:
        mesh_data = dict(archive)
    with np.load(temporary / 'chemistry.npz', allow_pickle=False) as archive:
        chemistry = dict(archive)
    mol2 = str(temporary / 'ligand.mol2') if config.get('ligand') else None
    charges = apbs(config, mesh_data['vertices'], stem, mol2)
    if not np.isfinite(charges).all():
        raise ValueError('Nonfinite APBS charges')
    save_ply(str(stem) + '.ply', mesh_data['vertices'], mesh_data['faces'], normals=mesh_data['normals'],
             charges=charges, normalize_charges=True, hbond=chemistry['hbond'],
             hphob=chemistry['hphob'], iface=mesh_data['iface'])
    for suffix, folder in [('.pdb', '01-benchmark_pdbs'), ('.ply', '01-benchmark_surfaces')]:
        destination = output / 'data_preparation' / folder
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(stem) + suffix, str(destination / (config['identifier'] + suffix)))
    print('SURFACE', len(charges), flush=True)


def main():
    """Resolve the unchanged source tree and dispatch one isolated geometry stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['protonate', 'mesh', 'interpolate', 'finalize', 'color-site'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, config['source_root'])
    {'protonate': protonate_input, 'mesh': mesh, 'interpolate': interpolate, 'finalize': finalize, 'color-site': color_site}[args.stage](config)


if __name__ == '__main__':
    main()
