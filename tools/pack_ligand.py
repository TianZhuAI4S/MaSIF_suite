"""Create ligand NPZ packets directly from the original precomputed surface arrays."""

import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from masif_torch.data import load_surface


def pack_ligand(precomputed, coords, output, split_list, trusted_coordinates=False):
    """Assign one-based ligand classes within the original 3 Å Euclidean cutoff."""
    precomputed, coords, output = Path(precomputed), Path(coords), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    structures = np.load(split_list, allow_pickle=False).astype(str)
    labels = dict(ADP=1, COA=2, FAD=3, HEM=4, NAD=5, NAP=6, SAM=7)
    for pdb in structures:
        directory = precomputed / (pdb + "_")
        data = load_surface(directory, require_indices=False)
        xyz = np.column_stack([np.load(directory / f"p1_{axis}.npy", allow_pickle=False) for axis in "XYZ"])
        tree = KDTree(xyz)
        pdb_id = pdb.split("_")[0]
        coordinates = np.load(coords / f"{pdb_id}_ligand_coords.npy", allow_pickle=trusted_coordinates)
        types = np.load(coords / f"{pdb_id}_ligand_types.npy", allow_pickle=False).astype(str)
        if len(coordinates) != len(types):
            raise ValueError(f"Ligand coordinate/type count mismatch: {pdb}")
        pocket_labels = np.zeros((len(xyz), len(types)), dtype=np.int64)
        for i, ligand in enumerate(types):
            if ligand not in labels:
                raise ValueError(f"Unknown ligand class: {ligand}")
            neighbors = tree.query_ball_point(coordinates[i], 3.0)
            indices = sorted({point for row in neighbors for point in row})
            pocket_labels[indices, i] = labels[ligand]
        data.update(pocket_labels=pocket_labels, pdb=np.asarray(pdb))
        np.savez_compressed(output / f"{pdb}.npz", **data)
        print(f"Packed {pdb}", flush=True)


def main():
    """Pack an explicit original sequence split without creating TensorFlow records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--split-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-coordinates", action="store_true")
    args = parser.parse_args()
    pack_ligand(args.precomputed, args.coordinates, args.output, args.split_list, args.trusted_coordinates)


if __name__ == "__main__":
    main()
