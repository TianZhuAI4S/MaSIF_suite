"""Freeze actual raw-prepared protein/ligand surfaces for independent TF/Torch checks."""

import argparse
from pathlib import Path
import numpy as np
from masif_torch.data import load_surface


def pack(root: Path, identifier: str, output: Path) -> None:
    """Preserve full site indexing and choose 32 real search patches in stored order."""
    root = root / 'data_preparation'
    site = load_surface(root / '04a-precomputation_9A/precomputation' / identifier, trusted_indices=True)
    count = min(20, np.count_nonzero(site['labels'][:, 0] == 1), np.count_nonzero(site['labels'][:, 1] == 1))
    if count < 1:
        raise ValueError('The reference surface needs both site label classes')
    site['pos_idx'] = np.flatnonzero(site['labels'][:, 0] == 1)[:count]
    site['neg_idx'] = np.flatnonzero(site['labels'][:, 1] == 1)[:count]
    search = load_surface(root / '04b-precomputation_12A/precomputation' / identifier, require_indices=False)
    if len(search['features']) < 32:
        raise ValueError('Need at least 32 search patches for four equal loss blocks')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'site.npz', **site)
    np.savez_compressed(output / 'search.npz', **{key: value[:32] for key, value in search.items()
                                             if key in ('features', 'rho', 'theta', 'mask')})


def main() -> None:
    """Require explicitly trusted local preprocessing output before reading ragged indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--identifier', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pack(args.root, args.identifier, args.output)


if __name__ == '__main__':
    main()
