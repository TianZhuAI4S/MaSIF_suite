"""TF-only regression for real NPZ geometry ranks and custom reference exports."""

import argparse
import json
from pathlib import Path
import numpy as np
from export_tf import export, export_adam, graph_reference


def main():
    """Compare all source outputs/gradients with 2D and singleton-channel geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("site_3", "ppi_search_1", "ligand_1"):
        source = args.oracles / name
        config = json.loads((source / "config.json").read_text())
        with np.load(source / "fixture.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        first, _ = graph_reference(source / "original", config, data)
        for key in ("rho", "theta", "mask"):
            data[key] = data[key][..., None]
        second, _ = graph_reference(source / "original", config, data)
        exact = all(np.array_equal(first[key], second[key]) for key in first)
        if not exact:
            raise AssertionError(f"Singleton geometry axis changed TF arithmetic: {name}")
        # Mimic real packets: no unused neighbor array or preselected site loss indices.
        for key in (("pos_idx", "neg_idx") if config["task"] == "site" else ("indices",)):
            data.pop(key, None)
        packet = args.output / f"{name}.npz"
        data["pdb"] = np.asarray("SYNTHETIC_A")
        np.savez_compressed(packet, **data)
        export(source / "original", args.output / name, config, packet)
        rows.append(dict(task=config["task"], exact_rank_invariance=exact, exported=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    export_adam(args.output)
    args.report.write_text(json.dumps(dict(passed=True, models=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
