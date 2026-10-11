"""Export every finite IPA checkpoint/HDF5 in the two pinned official design repositories."""

import argparse
from pathlib import Path

from export_tf import export, export_adam


def export_repository(repository: Path, output: Path) -> None:
    """Keep upstream layout and select 100/200 input rows according to 9/12 Å model name."""
    root = repository / "masif_seed_search/data/scoring_nn"
    prefixes = [path.with_suffix("") for path in sorted(root.rglob("*.index"))]
    prefixes += [path for path in sorted(root.rglob("*.hdf5")) if path.is_file()]
    if not prefixes:
        raise ValueError(f"No IPA checkpoints found: {root}")
    for prefix in prefixes:
        overrides = dict(max_npoints=100 if "9A" in prefix.name else 200)
        if not export(prefix, output / prefix.relative_to(root), overrides):
            raise ValueError(f"Nonfinite IPA weights rejected: {prefix}")


def main():
    """Export cloned seed and neosurf weights in the isolated TensorFlow environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-repo", type=Path, required=True)
    parser.add_argument("--neosurf-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_repository(args.seed_repo, args.output / "masif_seed")
    export_repository(args.neosurf_repo, args.output / "masif-neosurf")
    export_adam(args.output)


if __name__ == "__main__":
    main()
