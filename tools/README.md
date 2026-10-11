# User tools

Application workflows run through `masif`. These scripts prepare input data,
read legacy weights or download optional datasets. Full commands are in
[USAGE.md](../USAGE.md).

| Script | Purpose | Environment |
|---|---|---|
| `convert_records.py` | Convert original ligand TFRecords to numeric NPZ | `masif_nn` |
| `pack_ligand.py` | Build ligand packets from original precomputed arrays | `masif_torch` |
| `export_tf.py` | Export finite TensorFlow/Keras weights and fixed numerical references | `masif_nn` |
| `export_design_weights.py` | Export the seed/neosurface IPA weight families | `masif_nn` |
| `legacy_weights.py` | Shared old-format reader used by the export tools | Imported by export tools |
| `run_validation.py` | Batch-convert exported weights and check their reference outputs | `masif_torch` |
| `organize_weights.py` | Collect verified weights under task/purpose filenames | `masif_torch` |
| `fetch_reference_weights.py` | Fetch the pinned finite historical search weight | Either environment |
| `fetch_design_fixtures.py` | Download or verify a small public prepared-surface subset | Either environment |
| `download_benchmark.py` | Download and verify the optional full benchmark archive | Either environment |
| `extract_benchmark.py` | Verify and extract that archive | Either environment |

`data/design_members.json` supplies file locations and checksums for the small
prepared-surface download. Binary data and weights are generated locally and are
not bundled in Git or the wheel. External geometry and Rosetta workers ship under
`src/tasks/` and are called by the corresponding application workflow.
