"""Auditable external Rosetta execution; protocols and scores stay task-owned."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


def read_scores(path: Path) -> dict[str, float]:
    """Read the final Rosetta SCORE row and reject any nonfinite numeric metric."""
    rows = [line.split()[1:] for line in path.read_text().splitlines() if line.startswith("SCORE:")]
    if len(rows) < 2 or "description" not in rows[0]:
        raise ValueError(f"No completed Rosetta score row: {path}")
    header = rows[0]
    last = rows[-1]
    if len(header) != len(last):
        raise ValueError("Rosetta score header and row lengths differ")
    values = {name: float(value) for name, value in zip(header, last, strict=True) if name != "description"}
    if not values or not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Rosetta returned nonfinite scores")
    return values


def pdb_coordinates(path: Path, ligand: str) -> np.ndarray:
    """Validate a ligand-preserving two-chain PDB and return its coordinates in file order."""
    lines = [line for line in path.read_text().splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    if not any(line[17:20].strip() == ligand for line in lines):
        raise ValueError("Rosetta output lost the ligand")
    protein_chains = {line[21] for line in lines if line.startswith("ATOM")}
    if not {"A", "B"} <= protein_chains:
        raise ValueError("Rosetta output lost one of the partner protein chains")
    coordinates = np.asarray([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])
    if not np.isfinite(coordinates).all():
        raise ValueError("Rosetta output contains nonfinite atomic coordinates")
    return coordinates


def ligand_atom_names(path: Path, ligand: str) -> set[str]:
    """Read the complete ligand atom-name set to detect silent atom loss during refinement."""
    names = [line[12:16].strip() for line in path.read_text().splitlines()
             if line.startswith(("ATOM  ", "HETATM")) and line[17:20].strip() == ligand]
    if not names or len(names) != len(set(names)):
        raise ValueError("The ligand must have a nonempty, unique atom-name set")
    return set(names)


class RosettaWorkflow:
    """Prepare a complex, run original relax/design XML and validate real outputs."""

    def __init__(self, config: dict):
        """Resolve external tool/input paths and reserve a separate output directory."""
        self.config = dict(config)
        self.output = Path(config["output"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.protocol_dir = Path(config["protocol_dir"]).resolve()
        self.binary = str(Path(config["binary"]).resolve())
        self.database = str(Path(config["database"]).resolve())
        self.worker = Path(__file__).resolve().parent / "worker.py"
        for name in ("target", "binder", "chemistry_python", "legacy_python"):
            self.config[name] = str(Path(config[name]).resolve())
            if not Path(self.config[name]).is_file():
                raise FileNotFoundError(f"Missing {name}: {self.config[name]}")
        for name in ("ligand_sdf", "template_sdf"):
            if config.get(name):
                self.config[name] = str(Path(config[name]).resolve())
        self.config["protocol_dir"] = str(self.protocol_dir)
        ligand = config["ligand"]
        if len(ligand) != 3 or not ligand.isalnum():
            raise ValueError("The Rosetta ligand identifier must have three alphanumeric characters")
        if not Path(self.binary).is_file() or not Path(self.database).is_dir():
            raise FileNotFoundError("Rosetta binary and database must exist")
        self.timeout = float(config.get("timeout_seconds", 7200))
        if self.timeout <= 0 or not np.isfinite(self.timeout):
            raise ValueError("timeout_seconds must be positive and finite")
        self.report = dict(task="rosetta", success=False, original_protocols=True, stages=[], seed=int(config.get("seed", 42)))
        self.config_path = self.output / "resolved_config.json"
        self.config_path.write_text(json.dumps(self.config, indent=2) + "\n")
        self._save_report()

    def _save_report(self) -> None:
        """Persist completed and failed stages so unsuccessful runs remain inspectable."""
        (self.output / "results.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    def _execute(self, name: str, command: list[str], cwd: Path) -> None:
        """Run one external command with a dedicated log, explicit timeout and checked exit."""
        log = self.output / (name + ".log")
        started = time.monotonic()
        stage = dict(name=name, command=command, cwd=str(cwd), log=str(log), success=False)
        self.report["stages"].append(stage)
        try:
            with log.open("w") as stream:
                process = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=self.timeout, check=False)
            stage["exit_code"] = process.returncode
            stage["elapsed_seconds"] = time.monotonic() - started
            stage["success"] = process.returncode == 0
            if process.returncode:
                raise RuntimeError(f"External stage {name} failed ({process.returncode}); see {log}")
        except subprocess.TimeoutExpired:
            stage["elapsed_seconds"] = time.monotonic() - started
            stage["error"] = "timeout"
            raise
        finally:
            self._save_report()

    def _prepare_phase(self, phase: str, pdb: Path | None = None) -> dict:
        """Use the chemistry environment for molecular preparation and original contact selection."""
        command = [self.config["chemistry_python"], str(self.worker), phase,
                   "--config", str(self.config_path), "--output", str(self.output)]
        if pdb is not None:
            command.extend(["--pdb", str(pdb)])
        self._execute(phase, command, self.output)
        return json.loads((self.output / (phase + "_result.json")).read_text())

    def _protocol(self, name: str, pdb: Path, fix: str, suffix: str = "") -> dict:
        """Run an unchanged upstream XML with fixed RNG seed and one real output structure."""
        folder = self.output / (name + suffix)
        folder.mkdir(parents=True, exist_ok=True)
        score = folder / "score.sc"
        command = [self.binary, "-database", self.database,
                   "-parser:protocol", str(self.output / "src" / (name + ".xml")),
                   "-s", str(pdb), "-nstruct", "1", "-constant_seed", "-jran", str(self.report["seed"]),
                   "-no_nstruct_label", "-out:path:all", str(folder), "-out:file:scorefile", str(score),
                   "-overwrite", "-renumber_pdb", "-parser:script_vars", "mol=" + self.config["ligand"], "fix=" + fix,
                   "-in:file:extra_res_fa", str(self.output / (self.config["ligand"] + ".params"))]
        if name == "interface_design":
            command.extend(["-holes:dalphaball", str(Path(self.config["dalphaball"]).resolve())])
        started_ns = time.time_ns()
        self._execute(name + suffix, command, self.output)
        structures = list(folder.glob("*.pdb"))
        if len(structures) != 1:
            raise ValueError(f"Expected one completed {name} structure, got {len(structures)}")
        if structures[0].stat().st_mtime_ns < started_ns or score.stat().st_mtime_ns < started_ns:
            raise ValueError("Rosetta did not produce fresh structure and score outputs")
        pdb_coordinates(structures[0], self.config["ligand"])
        names = ligand_atom_names(structures[0], self.config["ligand"])
        if names != ligand_atom_names(self.output / "complex.pdb", self.config["ligand"]):
            raise ValueError("Rosetta changed or removed parameterized ligand atoms")
        return dict(ligand_atoms=len(names), pdb=str(structures[0]), scorefile=str(score), scores=read_scores(score))

    def run(self) -> dict:
        """Execute ligand parameterization, original relaxation/design and optional exact repeats."""
        destination = self.output / "src"
        destination.mkdir(exist_ok=True)
        hashes = {}
        for name in ("relax.xml", "interface_design.xml", "no_ref.rosettacon2018.beta_nov16.txt"):
            source = self.protocol_dir / name
            shutil.copyfile(source, destination / name)
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        self.report["protocol_sha256"] = hashes
        if self.config.get("ligand_sdf"):
            shutil.copyfile(self.config["ligand_sdf"], self.output / "ligand.sdf")
            self.report["ligand_preparation"] = dict(provided_sdf=True)
        else:
            self.report["ligand_preparation"] = self._prepare_phase("sdf")
        parameters = Path(self.database).parent / "source/scripts/python/public/molfile_to_params.py"
        self._execute("parameterization", [
            self.config["legacy_python"], str(parameters), "-n", self.config["ligand"], "-p", self.config["ligand"],
            "--chain", "A", "--clobber", "--conformers-in-one-file", str(self.output / "ligand.sdf")], self.output)
        self.report["prepared_complex"] = self._prepare_phase("assemble")
        pdb_coordinates(self.output / "complex.pdb", self.config["ligand"])
        fix = self._prepare_phase("contacts", self.output / "complex.pdb")["fix"]
        relaxed = self._protocol("relax", self.output / "complex.pdb", fix)
        fix = self._prepare_phase("contacts", Path(relaxed["pdb"]))["fix"]
        designed = self._protocol("interface_design", Path(relaxed["pdb"]), fix)
        self.report.update(relax=relaxed, interface_design=designed)
        if self.config.get("verify_repeat", False):
            repeat = self._protocol("interface_design", Path(relaxed["pdb"]), fix, "_repeat")
            reference_coords = pdb_coordinates(Path(designed["pdb"]), self.config["ligand"])
            repeat_coords = pdb_coordinates(Path(repeat["pdb"]), self.config["ligand"])
            coordinates_equal = np.array_equal(reference_coords, repeat_coords)
            scores_equal = designed["scores"] == repeat["scores"]
            self.report["repeat"] = dict(result=repeat, coordinates_bitwise_equal=coordinates_equal,
                                        scores_bitwise_equal=scores_equal)
            if not coordinates_equal or not scores_equal:
                self.report["success"] = False
                self._save_report()
                raise AssertionError("Repeated Rosetta design was not numerically identical")
        self.report["success"] = True
        self._save_report()
        return self.report
