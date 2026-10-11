"""Capture original Open3D0.5 geometry and PyMesh target scores (Python3.6)."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pymesh


def main():
    """Export all 256 published surfaces using the actual original file readers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = 0
    for condition in ('with_ligand', 'without_ligand', 'decoys'):
        root = args.dataset / condition
        target = args.output / condition
        target.mkdir(parents=True, exist_ok=True)
        for ply in sorted((root / 'output/all_feat_3l/pred_surfaces').glob('*.ply')):
            raw = root / 'data_preparation/01-benchmark_surfaces' / ply.name
            original = o3d.read_point_cloud(str(raw))
            colored = o3d.read_point_cloud(str(ply))
            mesh = pymesh.load_mesh(str(ply))
            np.savez_compressed(target / (ply.stem + '.npz'),
                                points=np.asarray(original.points), normals=np.asarray(original.normals),
                                colored_points=np.asarray(colored.points), colored_normals=np.asarray(colored.normals),
                                interface=mesh.get_attribute('vertex_iface'), pymesh_points=mesh.vertices)
            count += 1
        print('Captured original readers', condition, count, flush=True)
    report = dict(surfaces=count, open3d=o3d.__version__, numpy=np.__version__, pymesh=pymesh.__version__)
    (args.output / 'runtime.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
