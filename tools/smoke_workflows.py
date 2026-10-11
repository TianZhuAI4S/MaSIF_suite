"""Exercise installed CLI training, resume and inference on explicit synthetic packets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from masif_torch.checkpoints import convert, load_checkpoint
from masif_torch.data import load_npz
from masif_torch.validation import compare, write_report


def run(arguments):
    """Run the installed native CLI and fail immediately with its original error."""
    subprocess.run([sys.executable, "-m", "masif_torch", *map(str, arguments)], check=True)


def smoke(oracles, score_export, output, devices):
    """Compare saved NPY predictions and exercise one epoch plus native resumption."""
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    cases = [("site", oracles / "site_3"), ("ppi_search", oracles / "ppi_search_1"),
             ("ligand", oracles / "ligand_1"), ("score", score_export)]
    for task, export in cases:
        directory = output / task
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "converted.pt"
        convert(export, checkpoint, include_optimizer=task != "score")
        packet = load_npz(export / "fixture.npz")
        np.savez_compressed(directory / "packet.npz", **packet)
        with np.load(export / "reference.npz", allow_pickle=False) as reference:
            expected = reference["probability"]
        if task == "site":
            expected = expected[None]
        for device in devices:
            prediction = directory / f"prediction-{device}.npy"
            run(["predict", "--checkpoint", checkpoint, "--input", directory / "packet.npz",
                 "--output", prediction, "--device", device, "--batch-size", 4])
            result = compare(np.load(prediction, allow_pickle=False), expected)
            training = directory / f"training-{device}"
            run(["train", "--checkpoint", checkpoint, "--input", directory / "packet.npz",
                 "--validation", directory / "packet.npz", "--output", training,
                 "--epochs", 1, "--device", device, "--batch-size", 4])
            run(["train", "--checkpoint", training / "last.pt", "--resume", "--input",
                 directory / "packet.npz", "--validation", directory / "packet.npz",
                 "--output", training, "--epochs", 1, "--device", device, "--batch-size", 4])
            model, payload = load_checkpoint(training / "last.pt")
            history = json.loads((training / "history.json").read_text())
            finite = all(torch.isfinite(v).all().item() for v in model.state_dict().values())
            result.update(task=task, device=device, resumed_epoch=payload["epoch"],
                          history_rows=len(history), finite_checkpoint=finite)
            result["passed"] &= finite and payload["epoch"] == 1 and len(history) == 2
            reports.append(result)
        if task == "site":
            surface = directory / "surface"
            surface.mkdir(exist_ok=True)
            names = dict(features="input_feat", rho="rho_wrt_center", theta="theta_wrt_center", mask="mask",
                         indices="list_indices")
            for key, suffix in names.items():
                np.save(surface / f"p1_{suffix}.npy", packet[key])
            np.save(surface / "p1_iface_labels.npy", packet["labels"][:, 0])
            run(["pack-surface", "--input", surface, "--output", directory / "packed.npz",
                 "--feat-mask", 1, 1, 1])
            run(["surface", "--input", surface, "--output", directory / "surface.npy", "--checkpoint", checkpoint])
            reports.append(dict(task="legacy-surface", **compare(np.load(directory / "surface.npy"), expected)))
    # Evaluate a complete structure packet with an eligible 32-point pocket.
    pocket = load_npz(output / "ligand/packet.npz")
    pocket["pocket_labels"] = np.full((len(pocket["features"]), 1), 3, dtype=np.int64)
    np.savez_compressed(output / "ligand/pocket.npz", **pocket)
    ligand = output / "record-evaluation"
    run(["evaluate-ligand", "--checkpoint", output / "ligand/converted.pt", "--input",
         output / "ligand/pocket.npz", "--output", ligand, "--samples", 2])
    probability = np.load(ligand / "pocket_logits.npy")
    reports.append(dict(task="ligand-evaluation", shape=list(probability.shape),
                        passed=probability.shape == (1, 2, 1, 1, 3) and
                        bool(np.isfinite(probability).all()) and
                        bool(np.allclose(probability.sum(-1), 1))))
    cache = output / "search-cache"
    cache.mkdir(exist_ok=True)
    packet = load_npz(output / "ppi_search/packet.npz")
    for group in ("pos", "binder", "neg"):
        for key, suffix in dict(features="input_feat", rho="rho_wrt_center",
                               theta="theta_wrt_center", mask="mask").items():
            np.save(cache / f"{group}_{suffix}.npy", packet[key])
    for group in ("pos", "neg"):
        np.save(cache / f"{group}_training_idx.npy", np.arange(len(packet["features"])))
    run(["train", "--checkpoint", output / "ppi_search/converted.pt", "--search-cache",
         "--input", cache, "--output", output / "search-cache-training", "--epochs", 1,
         "--steps", 2, "--batch-size", 4])
    _, payload = load_checkpoint(output / "search-cache-training/last.pt")
    reports.append(dict(task="legacy-search-cache", step=payload["step"], passed=payload["step"] == 2))
    return dict(synthetic_inputs=True, workflows=reports,
                passed=all(row["passed"] for row in reports))


def main():
    """Require generated independent references; report synthetic coverage honestly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--score-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    report = smoke(args.oracles, args.score_export, args.output, ["cpu", "cuda"] if args.cuda else ["cpu"])
    write_report(report, args.report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
