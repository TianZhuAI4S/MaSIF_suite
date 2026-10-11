"""Typed molecular surfaces and explicit original precomputation layout adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a source-to-target homogeneous rigid transform to Å coordinates."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Transform must be finite [4,4]")
    if (not np.allclose(matrix[3], [0, 0, 0, 1]) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Transform must contain a proper rigid rotation")
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass
class SurfaceRecord:
    """One immutable-input surface with shared indexing across all numeric arrays."""

    name: str
    points: np.ndarray
    normals: np.ndarray
    descriptors: np.ndarray | None
    interface: np.ndarray
    neighbors: list[np.ndarray]
    flipped_descriptors: np.ndarray | None = None
    atoms: np.ndarray | None = None
    ca_atoms: np.ndarray | None = None
    pdb_path: Path | None = None

    def __post_init__(self) -> None:
        """Check geometry, descriptor widths and every unpadded geodesic index row."""
        count = len(self.points)
        if not count or self.points.shape != (count, 3) or self.normals.shape != (count, 3):
            raise ValueError("Surface points and normals must be nonempty [N,3]")
        available = [value for value in (self.descriptors, self.flipped_descriptors) if value is not None]
        if not available or any(value.ndim != 2 or value.shape[0] != count for value in available):
            raise ValueError("At least one descriptor orientation with shape [N,G] is required")
        self.interface = np.asarray(self.interface).reshape(-1)
        if len(self.interface) != count or len(self.neighbors) != count:
            raise ValueError("Interface and neighbor rows must match surface vertices")
        for row in self.neighbors:
            if row.ndim != 1 or not len(row) or row.dtype.kind not in "iu" or (row < 0).any() or (row >= count).any():
                raise ValueError("Each patch must contain valid nonempty integer surface indices")
        if len(available) == 2 and available[0].shape != available[1].shape:
            raise ValueError("Straight/flipped descriptor shapes must match")
        for value in (self.points, self.normals, self.descriptors, self.interface,
                      self.flipped_descriptors, self.atoms, self.ca_atoms):
            if value is not None and not np.isfinite(value).all():
                raise ValueError("Surface arrays must be finite")
        for value in (self.atoms, self.ca_atoms):
            if value is not None and (value.ndim != 2 or value.shape[1] != 3):
                raise ValueError("Atomic coordinates must have shape [atoms,3] in Å")

    @classmethod
    def from_npz(cls, path: str | Path) -> SurfaceRecord:
        """Load safe numeric arrays with flat/offset neighbors, without optional geometry libs."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        offsets, flat = data.pop("neighbor_offsets"), data.pop("neighbor_indices")
        if flat.ndim != 1 or flat.dtype.kind not in "iu":
            raise ValueError("Neighbor indices must be a one-dimensional integer array")
        if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != len(data["points"]) + 1 or
                offsets[0] != 0 or offsets[-1] != len(flat) or (np.diff(offsets) < 0).any()):
            raise ValueError("Invalid flat-neighbor offsets")
        rows = [flat[start:stop].astype(np.int64) for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]
        data["name"] = str(data.get("name", path.stem))
        data.setdefault("descriptors", None)
        if "pdb_path" in data:
            data["pdb_path"] = Path(str(data["pdb_path"]))
        return cls(neighbors=rows, **data)

    @staticmethod
    def _interface_values(path: Path, parsed) -> np.ndarray:
        """Preserve PyMesh ASCII decimal precision; binary properties preserve their stored numeric values."""
        vertices = parsed['vertex']
        if not parsed.text:
            return np.asarray(vertices['iface'], dtype=np.float64)
        column = vertices.data.dtype.names.index('iface')
        with path.open() as stream:
            for line in stream:
                if line.strip() == 'end_header':
                    break
            else:
                raise ValueError('Incomplete ASCII PLY header')
            # PyMesh reads ASCII attribute text directly into doubles, despite a float header.
            return np.loadtxt(stream, dtype=np.float64, usecols=column, max_rows=len(vertices.data), ndmin=1)

    @classmethod
    def from_processed(cls, root: str | Path, identifier: str, pid: str = "p1",
                       trusted_indices: bool = False, role: str = "target") -> SurfaceRecord:
        """Read seed/neosurf processed root including ligand-complex surface geometry."""
        # Optional file readers are kept outside the shared NN import path.
        import open3d as o3d
        from Bio.PDB import PDBParser
        from plyfile import PlyData

        root = Path(root)
        if role not in ("target", "source"):
            raise ValueError("Surface role must be target or source")
        fields = identifier.split("_")
        chain_index = 1 if pid == "p1" else 2
        if len(fields) <= chain_index:
            raise ValueError("Identifier must contain the selected chain")
        chain_name = fields[0] + "_" + fields[chain_index]
        preparation = root / "data_preparation"
        ply = root / "output/all_feat_3l/pred_surfaces" / (chain_name + ".ply")
        cloud = o3d.io.read_point_cloud(str(ply))
        parsed = PlyData.read(ply)
        attributes = parsed["vertex"]
        if "iface" not in attributes.data.dtype.names:
            raise ValueError("Colored surface PLY must contain the original iface attribute")
        interface = (cls._interface_values(ply, parsed) if role == "target" else
                     np.load(root / "output/all_feat_3l/pred_data" / ("pred_" + chain_name + ".npy"),
                             allow_pickle=False)[0])
        neighbors = np.load(preparation / "04b-precomputation_12A/precomputation" / identifier /
                            (pid + "_list_indices.npy"), allow_pickle=trusted_indices)
        descriptors = root / "descriptors/sc05/all_feat" / identifier
        straight_path = descriptors / (pid + "_desc_straight.npy")
        straight = np.load(straight_path, allow_pickle=False) if straight_path.exists() else None
        flipped_path = descriptors / (pid + "_desc_flipped.npy")
        flipped = np.load(flipped_path, allow_pickle=False) if flipped_path.exists() else None
        pdb = preparation / "01-benchmark_pdbs" / (chain_name + ".pdb")
        atoms = ca = None
        if pdb.exists():
            structure = PDBParser(QUIET=True).get_structure(chain_name, pdb)
            heavy = [atom for atom in structure.get_atoms() if not atom.get_name().startswith("H")]
            atoms = np.asarray([atom.get_coord() for atom in heavy], dtype=np.float64)
            ca = np.asarray([atom.get_coord() for atom in heavy if atom.get_id() == "CA"], dtype=np.float64).reshape(-1, 3)
        return cls(chain_name, np.asarray(cloud.points), np.asarray(cloud.normals), straight,
                   interface,
                   [np.asarray(row, dtype=np.int64) for row in neighbors], flipped, atoms, ca,
                   pdb if pdb.exists() else None)

    def save(self, path: str | Path) -> None:
        """Store primitive NPZ arrays with exact descriptor values and safe neighbor encoding."""
        offsets = np.concatenate(([0], np.cumsum([len(row) for row in self.neighbors])))
        data = dict(name=np.asarray(self.name), points=self.points, normals=self.normals,
                    interface=self.interface,
                    neighbor_indices=np.concatenate(self.neighbors), neighbor_offsets=offsets)
        for name in ("descriptors", "flipped_descriptors", "atoms", "ca_atoms", "pdb_path"):
            value = getattr(self, name)
            if value is not None:
                data[name] = np.asarray(str(value)) if name == "pdb_path" else value
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)

    def patch(self, center: int, shift: float = 0.25, flipped: bool = False) -> dict:
        """Shift outward along original normals before flipping target normals/descriptor."""
        if not 0 <= center < len(self.points):
            raise ValueError("Patch center outside the surface")
        indices = self.neighbors[center]
        descriptors = self.flipped_descriptors if flipped else self.descriptors
        if descriptors is None:
            raise ValueError("Target requires explicit flipped descriptors")
        normals = np.asarray(self.normals[indices], dtype=np.float64)
        return dict(points=np.asarray(self.points[indices], dtype=np.float64) + shift * normals,
                    normals=-normals if flipped else normals,
                    descriptors=np.asarray(descriptors[indices], dtype=np.float64), indices=indices)
