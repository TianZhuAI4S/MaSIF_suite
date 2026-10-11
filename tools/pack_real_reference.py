"""Prepare fixed real NN packets from the checksum-verified public 2GKW fixture."""

import argparse
from pathlib import Path

import numpy as np

from masif_torch.data import load_surface


def pack_references(fixtures: Path, output: Path) -> None:
    """Keep full 9 Å site indexing and select 32 real 12 Å search patches in stored order."""
    preparation = fixtures / "decoys/data_preparation"
    site = load_surface(preparation / "04a-precomputation_9A/precomputation/2GKW_B", trusted_indices=True)
    site["pos_idx"] = np.where(site["labels"][:, 0] == 1)[0][:20]
    site["neg_idx"] = np.where(site["labels"][:, 1] == 1)[0][:20]
    if not len(site["pos_idx"]) or not len(site["neg_idx"]):
        raise ValueError("Real site reference requires both label classes")
    search = load_surface(preparation / "04b-precomputation_12A/precomputation/2GKW_B", require_indices=False)
    search = {key: value[:32] for key, value in search.items() if key in ("features", "rho", "theta", "mask")}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "site_real.npz", **site)
    np.savez_compressed(output / "search_real.npz", **search)


def main():
    """Pack verified trusted fixtures; this command reads legacy object indices explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack_references(args.fixtures, args.output)


if __name__ == "__main__":
    main()
