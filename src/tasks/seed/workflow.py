"""Composable seed retrieval/docking and neosurf scoring over processed surfaces."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from ...checkpoints import load_checkpoint
from ...shared.design.data import SurfaceRecord, transform_points
from ...shared.design.features import AlignmentScorer, count_clashes, match_descriptors, target_sites
from ...shared.design.registration import RegistrationBackend, build_registration


def read_surface(spec: str | dict, role: str) -> SurfaceRecord:
    """Load a safe NPZ or an explicit original processed-root adapter configuration."""
    return (SurfaceRecord.from_npz(spec) if isinstance(spec, str) else
            SurfaceRecord.from_processed(**dict(spec, role=role)))


def write_aligned_pdb(source: SurfaceRecord, matrix: np.ndarray, path: Path) -> None:
    """Remove name-prefixed hydrogens as in seed align_and_save, then transform remaining atoms once."""
    from Bio.PDB import PDBIO, PDBParser
    structure = PDBParser(QUIET=True).get_structure(source.name, source.pdb_path)
    for atom in list(structure.get_atoms()):
        if atom.get_name().startswith("H"):
            atom.get_parent().detach_child(atom.get_id())
    atoms = list(structure.get_atoms())
    coordinates = transform_points(np.asarray([atom.coord for atom in atoms]), matrix)
    for atom, coordinate in zip(atoms, coordinates, strict=True):
        atom.set_coord(coordinate)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))


class SeedWorkflow:
    """Share matching, IPA and output logic; compose the selected pose provider."""

    def __init__(self, config: dict, registration: RegistrationBackend | None = None):
        """Resolve inputs once and maintain one legacy sampling stream per workflow."""
        self.config = dict(config)
        self.target = read_surface(config["target"], "target")
        self.database = [read_surface(spec, "source") for spec in config["database"]]
        names = [surface.name for surface in self.database]
        if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or
                                               name in (".", "..") for name in names):
            raise ValueError("Database names must be unique filename-safe identifiers")
        if self.target.flipped_descriptors is None:
            raise ValueError("Target requires flipped descriptors")
        if any(surface.descriptors is None for surface in self.database):
            raise ValueError("Database requires straight descriptors")
        self.centers = target_sites(self.target, config.get("num_sites", 1),
                                   config.get("selected_vertices"))
        self.shift = config.get("surface_outward_shift", 0.25)
        if not np.isfinite(self.shift):
            raise ValueError("Surface outward shift must be finite (Å)")
        self.registration = registration
        self.scorer = None

    def matches(self) -> list[dict]:
        """Retrieve candidates in original site, database and vertex order with strict cutoffs."""
        result = []
        for site, center in enumerate(self.centers):
            for surface in self.database:
                vertices, distances = match_descriptors(
                    surface, self.target.flipped_descriptors[center],
                    self.config.get("iface_cutoff", 0.75), self.config.get("descriptor_cutoff", 2.0))
                for vertex, distance in zip(vertices, distances, strict=True):
                    result.append(dict(key=f"{site}:{surface.name}:{vertex}", site=site,
                                       target_vertex=center, source=surface.name,
                                       source_vertex=int(vertex), center_desc_dist=float(distance)))
        return result

    def _score(self, target: dict, source: dict) -> dict:
        """Lazily load IPA so descriptor-only retrieval needs no checkpoint."""
        if self.scorer is None:
            model, _ = load_checkpoint(self.config["checkpoint"], self.config.get("device", "cpu"))
            self.scorer = AlignmentScorer(model, self.config.get("device", "cpu"), self.config.get("seed", 42),
                                          cache_geometry=self.config.get("cache_geometry", True))
        # Upstream compute_nn_score always passes 0.9, independent of hit cutoff.
        return self.scorer.score(target, source, self.target.atoms, importance_cutoff=0.9)


    def _target_patch(self, center: int) -> dict:
        """Reuse exact patch templates while giving each pose writable coordinate/feature copies."""
        if not self.config.get("cache_geometry", True):
            return self.target.patch(center, self.shift, flipped=True)
        key = (center, self.shift)
        if key not in self._target_patches:
            self._target_patches[key] = self.target.patch(center, self.shift, flipped=True)
        patch = self._target_patches[key]
        # A custom pose provider may edit its input; keep later candidate templates intact.
        return dict(patch, **{name: patch[name].copy() for name in ("points", "normals", "descriptors")})

    def _finalize_score(self, row: dict) -> None:
        """Provide an extension point for accepted-pose diagnostics after clash filtering."""

    def search(self) -> list[dict]:
        """Align matches, filter original fitness/IPA/clash thresholds and keep rejected diagnostics."""
        if self.registration is None:
            self.registration = build_registration(self.config["registration"])
        surfaces = {surface.name: surface for surface in self.database}
        self._target_patches = {}
        result = []
        for candidate in self.matches():
            surface = surfaces[candidate["source"]]
            target = self._target_patch(candidate["target_vertex"])
            source = surface.patch(candidate["source_vertex"], self.shift)
            matrix, fitness = self.registration.align(candidate["key"], source, target)
            source["points"] = transform_points(source["points"], matrix)
            source["normals"] = source["normals"] @ matrix[:3, :3].T
            row = dict(candidate, transformation=matrix, fitness=fitness, accepted=False)
            if fitness < self.config.get("fitness_cutoff", 0.3):
                row["rejection"] = "fitness"
            else:
                row.update(self._score(target, source))
                if row["score"] <= self.config.get("nn_score_cutoff", 0.9):
                    row["rejection"] = "score"
                else:
                    ca, heavy = count_clashes(surface, self.target.atoms, matrix,
                        self.config.get("clash_radius", 2.0), self.config.get("allowed_CA_clashes", 0),
                        target_tree=(self.scorer.coordinate_tree(self.target.atoms)
                                     if isinstance(self.scorer, AlignmentScorer) else None))
                    row.update(clashing_ca=ca, clashing_heavy=heavy)
                    row["accepted"] = (ca <= self.config.get("allowed_CA_clashes", 0) and
                                       heavy <= self.config.get("allowed_heavy_atom_clashes", 5))
                    if not row["accepted"]:
                        row["rejection"] = "clashes"
                    else:
                        self._finalize_score(row)
            row["aligned_points"] = source["points"]
            result.append(row)
        return result

    def actions(self) -> dict:
        """Expose seed retrieval and docking actions for task-owned extensions."""
        return {"match": self.matches, "search": self.search}

    def run(self) -> dict:
        """Execute match/search/score and save JSON diagnostics plus safe numeric hit packets."""
        action = self.config.get("action", "search")
        actions = self.actions()
        if action not in actions:
            raise ValueError(f"Unknown design action: {action}")
        rows = actions[action]()
        output = Path(self.config["output"])
        output.mkdir(parents=True, exist_ok=True)
        sources = {surface.name: surface for surface in self.database}
        summary = []
        for row in rows:
            numeric = {key: value for key, value in row.items() if isinstance(value, np.ndarray)}
            record = {key: value for key, value in row.items() if not isinstance(value, np.ndarray)}
            if row.get("accepted"):
                stem = f"site_{row['site']}_{row['source']}_{row['source_vertex']}"
                np.savez_compressed(output / (stem + ".npz"), **numeric)
                record["packet"] = stem + ".npz"
                source = sources[row["source"]]
                if source.pdb_path is not None and self.config.get("write_pdb", True):
                    pdb_output = output / (stem + ".pdb")
                    if action == "score":
                        shutil.copyfile(source.pdb_path, pdb_output)
                    else:
                        write_aligned_pdb(source, row["transformation"], pdb_output)
            if "transformation" in row:
                record["transformation"] = row["transformation"].tolist()
            summary.append(record)
        report = dict(task=self.config.get("task", "seed"), action=action, target=self.target.name,
                      registration=self.config.get("registration"), sites=self.centers,
                      candidates=summary, accepted=sum(row.get("accepted", False) for row in rows))
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
