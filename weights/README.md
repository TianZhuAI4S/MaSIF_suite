# Pretrained weights

Checkpoints are grouped by application and named by their purpose. Run commands
from the repository root, or use absolute checkpoint paths. All files are native
Torch `.pt` checkpoints and load without TensorFlow. `manifest.yaml` records each
source artifact, file SHA256 and embedded model configuration.

## Neural tasks

| File | Application | Scope |
|---|---|---|
| `site/interface_prediction_backup.pt` | Protein interface prediction | Finite three-layer peptides backup; different from the damaged primary site model |
| `search/surface_descriptors_historical.pt` | Surface descriptors | Official historical revision `51e5422`; current preprocessing/paper metrics are not certified |
| `ligand/pocket_classification.pt` | Ligand pocket classification | Original all-feature, seven-class ligand network |
| `score/second_stage_interface_scoring.pt` | Second-stage interface scoring | Original six-layer ScoreNN; distinct from five-layer IPA |

## Seed and neosurface IPA scoring

Both `seed/` and `neosurface/` contain the following five filenames. They retain
separate upstream provenance even where corresponding model tensors are equal.
IPA scores an already prepared interaction; it does not compute site predictions
or surface descriptors.

| Filename in either task directory | Features | Points | Use |
|---|---|---:|---|
| `ipa_12A_surface_atom.pt` | Surface + atom distance, columns `[0,1,2,3]` | 200 | Standard 12 Å seed/neosurface scoring; selected in the official computational benchmark |
| `ipa_12A_surface.pt` | Surface only, columns `[0,1,2]` | 200 | Explicit three-feature 12 Å scoring alternative |
| `ipa_9A_surface_atom.pt` | Surface + atom distance, columns `[0,1,2,3]` | 100 | Explicit 9 Å scoring alternative; supply matching patches |
| `ipa_12A_benchmark.pt` | Surface + atom distance, columns `[0,1,2,3]` | 200 | Benchmark-training snapshot from upstream `models_benchmark` |
| `ipa_12A_benchmark_alternative.pt` | Surface + atom distance, columns `[0,1,2,3]` | 200 | Distinct benchmark-training snapshot from upstream standard HDF5 weights |

The last two files contain different parameters. The `benchmark` names identify
upstream training snapshots; the official benchmark workflow uses
`ipa_12A_surface_atom.pt`. No accuracy ordering between variants is implied.

Surface features are inverse nearest-point distance, inverse descriptor distance
and normal dot product. The fourth feature is inverse nearest target-heavy-atom
distance, including ligand atoms when present. `12A`/`9A` denote upstream patch
radii in Angstroms. Keep patch preparation and the checkpoint's feature/point
configuration consistent; filenames do not modify data automatically.

## Use

```bash
masif site checkpoint=weights/site/interface_prediction_backup.pt \
  input=data/surface.npz output=results/site.npy
masif search checkpoint=weights/search/surface_descriptors_historical.pt \
  input=data/surface.npz output=results/descriptors.npy
masif seed --config examples/seed.yaml
masif neosurface --config examples/neosurface.yaml
```

The YAML examples select purpose-named checkpoints. Supply your own input data.
Installed packages used outside this checkout require explicit paths to downloaded
or generated weights; the weights are not bundled in the wheel.

## Recreate the layout

After generating the validated source artifacts described in the
[weight guide](../USAGE.md#convert-tensorflow-weights), run:

```bash
python tools/organize_weights.py
# Or collect selected tasks:
python tools/organize_weights.py --task seed --task neosurface
```

The tool copies existing checkpoint bytes, verifies hashes and finite tensors,
checks task compatibility and IPA feature/point configuration, and writes the
manifest. Base source names are `artifacts/checkpoints/site_backup.pt`,
`ppi_search_historical.pt`, `ligand.pt` and `score.pt` (all in that directory);
the usage guide generates these before organizing. For ScoreNN, export the finite
original score checkpoint and convert it with `--output artifacts/checkpoints/score.pt`
without `--include-optimizer`. Different existing destination
weights are never overwritten. Original artifact files remain available for
historical validation records.

Binary weights remain excluded from Git. This catalog and its YAML manifest are
included so a clean source checkout exposes the expected filenames and provenance.
The manifest describes the currently organized files; regeneration can change
serialization hashes and must produce a corresponding updated manifest.

## Numerical scope

Converted finite weights retain exact parameter values. The recorded TF/Torch
CPU/GPU inference comparisons pass `atol=1e-5, rtol=1e-4`; they do not establish
universal bitwise inference equality or full paper benchmark performance.
Historical search gradients and independent IPA training trajectories retain
strict failures. The seven original NaN-containing site/search groups are excluded.
See the [numerical scope](../README.md#validation-and-scientific-scope).
