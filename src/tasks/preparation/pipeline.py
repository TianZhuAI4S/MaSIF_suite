"""Keep original PyMesh geometry isolated from modern chemistry and neural inference."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


class PreparationTask:
    """Prepare an Å-coordinate PDB and optional ligand using explicit external tools.

    Primitive NPZ arrays cross interpreter boundaries. Original MSMS, PyMesh,
    charge interpolation and geodesic routines retain their existing definitions.
    """

    def __init__(self, config: dict):
        """Resolve inputs before running stages; never modify either source tree."""
        self.config = dict(config)
        for name in ("pdb", "source_root", "legacy_python", "output"):
            self.config[name] = str(Path(config[name]).expanduser().resolve())
        for name in ("sdf", "mol2", "site_checkpoint", "search_checkpoint"):
            if config.get(name):
                self.config[name] = str(Path(config[name]).expanduser().resolve())
        tasks = self.config.get("precompute", ["site", "ppi_search"])
        if not isinstance(tasks, list) or set(tasks) - {"site", "ppi_search", "ligand"}:
            raise ValueError("precompute must list site, ppi_search or ligand")
        for task, key in (("site", "site_checkpoint"), ("ppi_search", "search_checkpoint")):
            if config.get(key) and task not in tasks:
                raise ValueError(f"{key} requires {task} in precompute")
        if not Path(self.config["pdb"]).is_file():
            raise FileNotFoundError(self.config["pdb"])
        if not os.access(self.config["legacy_python"], os.X_OK):
            raise ValueError("legacy_python must be an executable compatible geometry interpreter")
        self.output = Path(self.config["output"])
        self.source = Path(self.config["source_root"])
        self.stages = []
        self.env = os.environ.copy()
        self.env.update(config.get("environment", {}))
        self.env.update(PYTHONPATH=str(self.source), TMPDIR=str(self.output / "tmp"),
                        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for name in ("MSMS_BIN", "PDB2PQR_BIN", "APBS_BIN", "MULTIVALUE_BIN"):
            executable = self.env.get(name)
            if not executable or not os.access(executable, os.X_OK):
                raise ValueError(f"Set {name} to an executable file in environment")
        if shutil.which("reduce", path=self.env["PATH"]) is None:
            raise ValueError("Add the Reduce executable's directory to environment.PATH")

    def _stage(self, name: str, command: list[str]) -> None:
        """Run a checked subprocess and preserve its complete diagnostic output."""
        log_path = self.output / f"{name}.log"
        start = time.monotonic()
        print(f"Preparation: {name}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=self.output, env=self.env,
                                    stdout=log, stderr=subprocess.STDOUT)
        self.stages.append(dict(name=name, command=command, log=str(log_path),
                                seconds=time.monotonic() - start, returncode=result.returncode))
        (self.output / "stages.json").write_text(json.dumps(self.stages, indent=2) + "\n")
        if result.returncode:
            (self.output / "result.json").write_text(json.dumps(dict(
                task="preparation", success=False, status="failed", failed_stage=name,
                stages=self.stages), indent=2) + "\n")
            raise subprocess.CalledProcessError(result.returncode, command)

    def _predict(self) -> dict:
        """Predict site and both descriptor orientations on the generated full surface."""
        from ...checkpoints import load_checkpoint
        from ...cli import predict
        from ...data import flip_binder, load_surface
        results = {}
        identifier = self.config["identifier"]
        for task, key, directory in (
                ("site", "site_checkpoint", "04a-precomputation_9A"),
                ("ppi_search", "search_checkpoint", "04b-precomputation_12A")):
            if not self.config.get(key):
                continue
            model, _ = load_checkpoint(self.config[key], self.config.get("device", "cpu"))
            if model.config["task"] != task:
                raise ValueError(f"{key} must contain a {task} model")
            data = load_surface(self.output / "data_preparation" / directory / "precomputation" / identifier,
                                feat_mask=model.config["feat_mask"], trusted_indices=True,
                                require_indices=task == "site")
            # Only indices written locally by the immediately preceding checked stage are unpickled.
            predictions = {"site": predict(model, data, self.config.get("device", "cpu"))} if task == "site" else {
                orientation: predict(model, packet, self.config.get("device", "cpu"))
                for orientation, packet in (("straight", data), ("flipped", flip_binder(data, model.config["feat_mask"])))
            }
            for orientation, values in predictions.items():
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {task}/{orientation} predictions")
                if task == "site":
                    path = self.output / "output/all_feat_3l/pred_data" / f"pred_{identifier}.npy"
                else:
                    path = self.output / "descriptors/sc05/all_feat" / identifier / f"p1_desc_{orientation}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values)
                results[f"{task}_{orientation}"] = dict(path=str(path), shape=list(values.shape))
            if task == "site":
                self._stage("color_site", [self.config["legacy_python"], "-u",
                            str(Path(__file__).parent / "worker_geometry.py"), "--config",
                            str(self.output / "resolved_config.json"), "--stage", "color-site"])
        return results

    def run(self) -> dict:
        """Execute geometry, chemistry, APBS, original precomputation and Torch inference."""
        temporary = self.output / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        # Invalidate any prior successful report before executing a new configuration.
        (self.output / "result.json").write_text(json.dumps(dict(
            task="preparation", success=False, status="running"), indent=2) + "\n")
        config_path = self.output / "resolved_config.json"
        config_path.write_text(json.dumps(self.config, indent=2) + "\n")
        workers = Path(__file__).parent
        geometry = str(workers / "worker_geometry.py")
        self._stage("protonate", [self.config["legacy_python"], "-u", geometry,
                                  "--config", str(config_path), "--stage", "protonate"])
        if self.config.get("ligand"):
            self._stage("ligand_protonate", [self.config.get("chemistry_python", sys.executable), "-u",
                        str(workers / "worker_chemistry.py"), "--config", str(config_path),
                        "--stage", "protonate-ligand"])
        self._stage("mesh", [self.config["legacy_python"], "-u", geometry,
                             "--config", str(config_path), "--stage", "mesh"])
        self._stage("chemistry", [self.config.get("chemistry_python", sys.executable), "-u",
                                  str(workers / "worker_chemistry.py"), "--config", str(config_path)])
        self._stage("interpolate", [self.config["legacy_python"], "-u", geometry,
                                   "--config", str(config_path), "--stage", "interpolate"])
        self._stage("apbs", [self.config["legacy_python"], "-u", geometry,
                             "--config", str(config_path), "--stage", "finalize"])
        for task in self.config.get("precompute", ["site", "ppi_search"]):
            if task not in ("site", "ppi_search", "ligand"):
                raise ValueError(f"Unknown preprocessing task: {task}")
            self._stage(f"precompute_{task}", [self.config["legacy_python"], "-u",
                        str(self.source / "data_preparation/04-masif_precompute.py"),
                        f"masif_{task}", self.config["identifier"]])
        result = dict(task="preparation", success=True, status="complete", identifier=self.config["identifier"],
                      ligand=self.config.get("ligand"), stages=self.stages, predictions=self._predict())
        (self.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
