"""Fetch a finite official historical search checkpoint with pinned SHA256 hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "51e5422fc6c5f25628dcfb64c2c521347c7d0569"
CHECKSUMS = {
    ".index": "98c00fbe7b9b46efdee00179f2f996be7cdda8ef30985a0dfed1a187f896dcc0",
    ".meta": "49126028c5c8552ad0d18336c6f7b19855e88942406799d18dbca663d4a61a92",
    ".data-00000-of-00001": "4541e79570cdefe0e741359a0601e9b8391a136544c7c26070cde1ec59fa7133",
}


def main():
    """Download only into an explicit new directory and verify each published file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/LPDI-EPFL/masif/{REVISION}/data/masif_ppi_search/nn_models/sc05/all_feat/model_data/model"
    for suffix, checksum in CHECKSUMS.items():
        with urllib.request.urlopen(base + suffix, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"Official weight checksum mismatch for {suffix}")
        (args.output / ("model" + suffix)).write_bytes(data)
    (args.output / "origin.json").write_text(json.dumps(dict(repository="LPDI-EPFL/masif",
                                                           revision=REVISION, sha256=CHECKSUMS), indent=2) + "\n")
    print(f"Fetched verified historical search checkpoint at {args.output / 'model'}")


if __name__ == "__main__":
    main()
