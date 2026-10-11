"""Exercise real precomputed design inputs and the optional modern registration backend."""

import argparse
import json
from pathlib import Path

import numpy as np

from masif_torch.shared.design.data import SurfaceRecord, transform_points
from masif_torch.shared.design.features import target_sites
from masif_torch.shared.design.registration import Open3DRegistration
from masif_torch.tasks import build_task
from masif_torch.validation import deterministic, write_report


def modern_registration() -> dict:
    """Recover a known synthetic rigid translation; this is a functionality gate, not legacy parity."""
    rng = np.random.RandomState(17)
    points = rng.normal(size=(40, 3))
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    descriptors = rng.normal(size=(40, 8))
    known = np.eye(4)
    known[:3, 3] = [0.3, -0.2, 0.1]
    source = dict(points=points, normals=normals, descriptors=descriptors)
    target = dict(points=transform_points(points, known), normals=normals, descriptors=descriptors)
    matrix, fitness = Open3DRegistration(iterations=500).align("synthetic", source, target)
    error = float(np.max(np.abs(transform_points(points, matrix) - target["points"])))
    return dict(passed=error < 1e-6 and fitness == 1., max_coordinate_error=error,
                fitness=fitness, scope="modern Open3D functional test; no legacy random pose equivalence")


def real_workflows(root: Path, checkpoint: Path, output: Path, device: str) -> dict:
    """Use real surfaces and a constructed fixed pose to test seed/neosurf output contracts."""
    target = SurfaceRecord.from_processed(root / "with_ligand", "1A7X_A", trusted_indices=True)
    source = SurfaceRecord.from_processed(root / "decoys", "2GKW_B", trusted_indices=True, role="source")
    output.mkdir(parents=True, exist_ok=True)
    target.save(output / "target.npz")
    source.save(output / "source.npz")
    vertex = target_sites(target)[0]
    distances = np.sqrt(np.sum(np.square(source.descriptors - target.flipped_descriptors[vertex]), axis=1))
    order = np.argsort(distances)
    cutoff = float((distances[order[0]] + distances[order[1]]) / 2)
    config = dict(task="seed", target=str(output / "target.npz"), database=[str(output / "source.npz")],
                  checkpoint=str(checkpoint), output=str(output / "match"), action="match", device=device,
                  selected_vertices=[vertex], iface_cutoff=-1., descriptor_cutoff=cutoff, nn_score_cutoff=0.,
                  allowed_CA_clashes=10000, allowed_heavy_atom_clashes=10000)
    matches = build_task(config).run()
    if len(matches["candidates"]) != 1:
        raise AssertionError("Fixture must produce exactly one nearest descriptor candidate")
    matrix = np.eye(4)
    matrix[:3, 3] = target.points[vertex] - source.points[order[0]]
    poses = output / "poses.npz"
    np.savez(poses, keys=np.asarray([matches["candidates"][0]["key"]]), transforms=matrix[None], fitness=np.array([1.]))
    config.update(action="search", output=str(output / "seed"), registration=dict(mode="replay", path=str(poses)))
    (output / "seed.json").write_text(json.dumps(config, indent=2) + "\n")
    seed = build_task(config).run()
    config.update(task="neosurf", action="score", output=str(output / "neosurf"))
    (output / "neosurf.json").write_text(json.dumps(config, indent=2) + "\n")
    neosurf = build_task(config).run()
    saved = []
    for name, report in (("seed", seed), ("neosurf", neosurf)):
        for candidate in report["candidates"]:
            if candidate.get("accepted"):
                path = output / name / candidate["packet"]
                with np.load(path, allow_pickle=False) as data:
                    if not np.isfinite(data["features"]).all():
                        raise AssertionError("Nonfinite real IPA features")
                saved.append(dict(packet=str(path), pdb_exists=path.with_suffix(".pdb").exists()))
    return dict(passed=seed["accepted"] == 1 and neosurf["accepted"] == 1 and all(row["pdb_exists"] for row in saved),
                device=device, seed_hits=seed["accepted"], neosurf_pairs=neosurf["accepted"], outputs=saved,
                pose_provenance="constructed translation, not a saved legacy RANSAC result",
                thresholds="relaxed only for smoke output coverage; no benchmark performance claim")


def main():
    """Run real fixture CPU/GPU workflows and separately label the modern pose functionality gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    deterministic()
    reports = [real_workflows(args.fixtures, args.checkpoint, args.output / device, device)
               for device in (["cpu", "cuda"] if args.cuda else ["cpu"])]
    modern = modern_registration()
    report = dict(passed=all(row["passed"] for row in reports) and modern["passed"],
                  workflows=reports, modern_registration=modern)
    write_report(report, args.report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
