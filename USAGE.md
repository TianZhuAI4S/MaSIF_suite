# Detailed MaSIF usage

This guide preserves the detailed preparation, checkpoint conversion, numerical verification, and compatibility commands. Run these commands from the standalone project root. The primary application interface is now `masif` with Hydra configuration; start with the [application README](README.md) and [installation instructions](README.md#installation). The `masif_torch` Python API and existing compatibility commands remain available.

MaSIF (Molecular Surface Interaction Fingerprints) learns interaction patterns from molecular surface geometry and chemistry. This standalone project provides PyTorch MaSIF-site, MaSIF-search, MaSIF-ligand, the second-stage ScoreNN, and the IPA alignment network used by MaSIF-seed and MaSIF-neosurf. It includes training, recovery, inference, legacy-weight conversion, prepared-surface retrieval, registration, scoring, and numerical audits.

The implementation lives directly in `src/`, with each task in `src/tasks/<task>/`. Installation exposes the `masif` API and command, with `masif_torch` and `masif-torch` compatibility names. The original TensorFlow repository is a separate checkout, called `../masif` in examples; adjust this path to your installation.

**Validated:** finite legacy weights, CPU/GPU inference and losses, original-source gradient comparisons, native Torch training recovery, real public precomputed surfaces, fixed-pose seed/neosurf geometry/scoring, complete raw protein/FKA preparation, and real Rosetta relaxation/interface design with exact same-seed output repeat. **Limits:** cross-framework operations are not generally bitwise identical; independent IPA training parameter updates fail the strict tolerance; modern Open3D docking is not an exact reproduction of historical random docking. Raw preparation, Rosetta, and full benchmarks are evaluated separately below.

## Contents

- [Methods and numerical contract](#methods-and-numerical-contract)
- [Requirements and installation](#requirements-and-installation)
- [Input data and preparation](#input-data-and-preparation)
- [Convert TensorFlow weights](#convert-tensorflow-weights)
- [Neural-network applications](#neural-network-applications)
- [MaSIF-seed and MaSIF-neosurface](#masif-seed-and-masif-neosurface)
- [Raw preparation, Rosetta, and benchmarks](#raw-preparation-rosetta-and-benchmarks)
- [Architecture and extension](#architecture-and-extension)
- [Documentation and references](#documentation-and-references)

## Methods and numerical contract

| Task | Method / input | Main commands |
|---|---|---|
| `site` | Gaussian patch convolutions, whole-surface neighbor aggregation | `surface`, `predict`, `train` |
| `ppi_search` / `search` | Unnormalized straight/flipped surface descriptors | `surface`, `predict`, `train` |
| `ligand` | Pocket classification from an uncentered descriptor Gram matrix | `predict`, `evaluate-ligand`, `train` |
| `score` | Six-convolution ScoreNN, `[B,N,3]` features | `predict`, `train` |
| `alignment` | Five-convolution IPA network, `[B,100 or 200,F]` | `predict`, `train` |
| `seed` | Descriptor retrieval, registration, IPA, clash filtering | `run`, action `match` / `search` |
| `neosurface` / `neosurf` | Seed search or scoring an aligned protein/ligand complex | `run`, action `search` / `score` |

The migration preserves operation order, tensor layouts, the actual mathematics behind swapped site layer-2/layer-3 parameter names, search population-variance loss, Keras BN statistics, and TensorFlow Adam epsilon placement. ScoreNN and IPA have different architectures and cannot share checkpoints.

Converted weights must round-trip **exactly** with `array_equal`, including transpositions. FP32 forward/loss/gradient acceptance is:

```text
abs(torch - reference) <= 1e-5 + 1e-4 * abs(reference)
```

This is not a bitwise-equality claim. Validation disables TF32/mixed precision and requests deterministic Torch arithmetic. The pinned historical search checkpoint has high-scale synthetic gradient cancellation errors: its explicit optional gradient policy is `atol=1e-2, rtol=1e-3`, while forward/loss tolerances remain unchanged and strict failures remain recorded.

The original-source IPA audit passes inference, loss, single-step gradients, BN statistics, optimizer updates with identical replayed gradients, and evaluation probabilities after three updates. Independent TensorFlow/Torch parameter updates fail for some pre-BN biases (maximum differences about `1.61e-4` CPU / `1.34e-4` GPU). The theoretical zero bias gradient has framework-specific FP32 residuals amplified by Adam. Original gradients/epsilon are retained; this full audit intentionally reports `passed=false`.

## Requirements and installation

Follow the [README installation instructions](README.md#installation) for the
Torch environment, CPU/CUDA selection, optional molecular dependencies and the
separate TensorFlow reader. The commands in this guide use the Conda names
`masif_torch` and `masif_nn`; substitute your environment activation if using venv.
Run application commands from the repository root. Keep the original MaSIF checkout
separately when reading its weights/source or using its geometry routines:

```bash
git clone https://github.com/LPDI-EPFL/masif.git ../masif
```

## Input data and preparation

### Surface arrays

| Array | Shape / dtype | Meaning |
|---|---|---|
| `features` | `[N,V,F]`, FP32 | N patches, V neighborhood vertices, F selected features |
| `rho` | `[N,V]` or `[N,V,1]`, FP32 | Geodesic distance from patch center, Å |
| `theta` | Same ranks as rho, FP32 | Patch angle, radians |
| `mask` | Same ranks as rho, FP32 | Validity mask in `[0,1]` |
| `indices` | `[N,V]`, integer | Neighbor rows in the **complete surface**, for multilayer site |

Five features in original order: shape index, distance-dependent curvature (DDC), hydrogen bonding, normalized electrostatics, hydrophobicity. `feat_mask` selects columns in that order. Default radius is 9 Å for site, 12 Å for search/ligand. Cartesian point distance cannot replace geodesic rho.

Original folders contain `p1_input_feat.npy`, `p1_rho_wrt_center.npy`, `p1_theta_wrt_center.npy`, and `p1_mask.npy`; site also needs `p1_list_indices.npy`, and training needs `p1_iface_labels.npy`. `p2` uses corresponding names.

```bash
python -m masif_torch pack-surface \
  --input /path/to/precomputation/1AKJ_AB_DE --pid p1 --trusted-indices \
  --output /path/to/site/train/1AKJ_AB.npz
```

Legacy ragged indices may be pickled; enable `--trusted-indices` only for trusted files. Padding repeats the center index. Keep train/validation/test directories separate using the existing split.

### Ligand records and coordinates

Convert each split once, then use numeric NPZ entirely in Torch:

```bash
conda activate masif_nn
python tools/convert_records.py --input /path/to/train.tfrecord --output /path/to/ligand/train
```

`conversion.json` records counts and exact per-array read-back. Alternatively, label original precomputed patches with ligand coordinates and the original 3 Å pocket rule:

```bash
conda activate masif_torch
python tools/pack_ligand.py \
  --precomputed /path/to/04-precomputation_12A --coordinates /path/to/ligand_coords \
  --split-list ../masif/data/masif_ligand/lists/train_pdbs_sequence.npy \
  --output /path/to/ligand/train --trusted-coordinates
```

Coordinate files are `<PDB>_ligand_coords.npy` / `<PDB>_ligand_types.npy`; precomputed subdirectories are `<PDB_CHAIN>_`. `--trusted-coordinates` is for trusted old object arrays.

### Training labels

- Site: `labels=[N,2]`, interface in column 0, non-interface in column 1.
- Ligand: full-structure `pocket_labels=[N,pockets]`, 0 for no label and 1–7 for class; a fixed pocket instead has one-hot `labels=[C]`.
- ScoreNN / IPA: integer `labels=[B]`, classes 0/1.
- Search: four equal blocks, positive / binder / negative / negative2.

Site records lacking either class and ligand structures with no pocket of at least 32 points are excluded in `excluded.json`; an entirely unusable dataset raises an error. Score/IPA batches visit each alignment once per epoch.

## Convert TensorFlow weights

The [weight catalog](weights/README.md) lists task/purpose filenames and their
limits. Local inference examples use `weights/`; original validation artifacts
remain in `artifacts/` to preserve their recorded provenance. To organize existing
validated artifacts without altering checkpoint bytes:

```bash
python tools/organize_weights.py
# Select only the tasks whose source artifacts you have generated:
python tools/organize_weights.py --task seed --task neosurface
```

The tool verifies finite model tensors, task/configuration and SHA256, refuses to
overwrite different existing weights, and writes `weights/manifest.yaml`.
The commands below first generate source artifacts and then organize their task
weights. Conversion can instead write directly to a purpose-named destination;
this alternative does not create organizer source artifacts or update its manifest.
If a destination already contains different serialized weights, the organizer
refuses to overwrite it. Regenerate in a clean destination and retain its own
manifest rather than treating different checkpoint files as the same snapshot.


```bash
conda activate masif_nn
python tools/export_tf.py \
  --checkpoint ../masif/data/masif_ligand/nn_models/all_feat/model \
  --output artifacts/tf_exports/ligand
conda activate masif_torch
python -m masif_torch convert --export artifacts/tf_exports/ligand \
  --output artifacts/checkpoints/ligand.pt --include-optimizer
python tools/organize_weights.py --task ligand
python -m masif_torch validate --export artifacts/tf_exports/ligand \
  --checkpoint weights/ligand/pocket_classification.pt --report artifacts/checkpoints/ligand.validation.json
```

Pass the checkpoint **prefix**, without `.index`. Keep `.index`, `.data-*`, and original site/search/ligand `.meta` graphs. Exports contain weights, config, fixed input, original graph results, and hashes. Portable `.pt` files contain primitive metadata/config and `state_dict`, loading without TensorFlow.

`--include-optimizer` converts original TF1 Adam moments, beta powers, and graph learning rate for site/search/ligand. It is unsupported for ScoreNN/IPA Keras optimizer slots; omit it there. Native Torch checkpoints support full recovery. For custom radii/rotations/feature masks use `--config config.yaml`; shapes alone cannot infer every setting.

`--input packet.npz` generates references for supplied data. Site derives positive/negative loss indices from two-column labels. Ligand reference data must be a fixed sampled 32-patch pocket with one-hot `labels`, not a whole structure's `pocket_labels`. Search references need the four-block layout.

| Original path under `../masif` | Status |
|---|---|
| `data/masif_ligand/nn_models/all_feat/model` | Finite, convertible, CPU/GPU validated |
| `data/masif_peptides/nn_models/all_feat_3l/model_data.bak/model` | Finite three-layer site **backup**, validated |
| `data/masif_pdl1_benchmark/models/nn_score/trained_model.hdf5` | TF object checkpoint prefix, convertible |
| `data/masif_ppi_search/transformation_training_data/models/nn_score/trained_model.hdf5` | TF object checkpoint prefix, convertible |
| Seven other original checkpoint groups | NaN parameters/slots, rejected |

Nonfinite exports write `rejected.json`; NaNs are not replaced with zeros. The site backup is not equivalent to the damaged main model. See the known limitations in the [weight catalog](weights/README.md).

A finite official historical search checkpoint is pinned at `51e5422fc6c5f25628dcfb64c2c521347c7d0569`:

```bash
conda activate masif_torch
python tools/fetch_reference_weights.py --output artifacts/upstream/51e5422
conda activate masif_nn
python tools/export_tf.py --checkpoint artifacts/upstream/51e5422/model \
  --output artifacts/tf_exports/historical_search
conda activate masif_torch
python -m masif_torch convert --export artifacts/tf_exports/historical_search \
  --output artifacts/checkpoints/ppi_search_historical.pt --include-optimizer
python tools/organize_weights.py --task search
```

All three files have pinned SHA256 checks. Compatibility with current preprocessing or paper metrics has not been established. This explicit demonstration model never overwrites original weights.

## Neural-network applications

### MaSIF-site

```bash
conda activate masif_nn
python tools/export_tf.py \
  --checkpoint ../masif/data/masif_peptides/nn_models/all_feat_3l/model_data.bak/model \
  --output artifacts/tf_exports/site_backup
conda activate masif_torch
python -m masif_torch convert --export artifacts/tf_exports/site_backup \
  --output artifacts/checkpoints/site_backup.pt --include-optimizer
python tools/organize_weights.py --task site
python -m masif_torch surface --checkpoint weights/site/interface_prediction_backup.pt \
  --input /path/to/precomputation/1AKJ_AB_DE --pid p1 --trusted-indices \
  --output /path/to/predictions/1AKJ_AB.npy --device cuda
python -m masif_torch train --checkpoint weights/site/interface_prediction_backup.pt \
  --input /path/to/site/train --validation /path/to/site/validation \
  --output /path/to/site/run --epochs 100 --device cuda --seed 0
```

Output `[1,N]` preserves `sigmoid(logits[:,0])`. Multilayer site requires the whole surface, because neighbors reference its rows. One/two/three-layer models are supported; the original four-layer layout is unsupported.

### MaSIF-search

```bash
python -m masif_torch surface --checkpoint weights/search/surface_descriptors_historical.pt \
  --input /path/to/precomputation/1AKJ_AB_DE --pid p1 \
  --output /path/to/descriptors/1AKJ_AB_DE --batch-size 100 --device cuda
python -m masif_torch train --checkpoint weights/search/surface_descriptors_historical.pt \
  --input /path/to/search/cache --search-cache --steps 1000 --batch-size 32 \
  --output /path/to/search/run --epochs 100 --device cuda
```

Writes `p1_desc_straight.npy` / `p1_desc_flipped.npy`, each `[N,G]`, default G=80, without L2 normalization. Binder reflection uses `2π-theta`, negates geometry/hydrogen-bond/electrostatics, and preserves hydrophobicity by original column identity.

Original caches need `pos/binder/neg_input_feat.npy`, corresponding rho/theta/mask arrays, and `pos_training_idx.npy` / `neg_training_idx.npy`. Batch size must be divisible by four. Explicit validation uses independently prepared four-block NPZ. Successful descriptors alone do not reproduce PD-L1 second-stage paper metrics.

### MaSIF-ligand

```bash
python -m masif_torch train --checkpoint weights/ligand/pocket_classification.pt \
  --input /path/to/ligand/train --validation /path/to/ligand/validation \
  --output /path/to/ligand/run --epochs 100 --device cuda --seed 0
python -m masif_torch evaluate-ligand --checkpoint /path/to/ligand/run/best.pt \
  --input /path/to/ligand/test --output /path/to/ligand/test_set_predictions \
  --samples 100 --seed 0 --device cuda
```

Every eligible pocket draws 32 patches; evaluation defaults to 100 draws. `<structure>_labels.npy` contains zero-based classes. `<structure>_logits.npy` retains the old filename but contains **softmax probabilities**, `[pockets,samples,1,1,C]`. Single-pocket `predict` returns `[1,1,C]`.

Sampling fixes the historical patch/mask mismatch and selects eligible pockets instead of skipping random undersized pockets. Fixed-packet mathematics is validated; the historical sampling schedule/trajectory is not claimed identical. Independent training trajectories can therefore differ.

### ScoreNN and IPA

Export either finite ScoreNN prefix above and convert without `--include-optimizer`:

```bash
conda activate masif_nn
python tools/export_tf.py \
  --checkpoint ../masif/data/masif_pdl1_benchmark/models/nn_score/trained_model.hdf5 \
  --output artifacts/tf_exports/score
conda activate masif_torch
python -m masif_torch convert --export artifacts/tf_exports/score \
  --output artifacts/checkpoints/score.pt
python tools/organize_weights.py --task score
python -m masif_torch predict --checkpoint weights/score/second_stage_interface_scoring.pt \
  --input /path/to/alignment_features.npz --output /path/to/score_probability.npy \
  --batch-size 32 --device cuda
python -m masif_torch train --checkpoint weights/score/second_stage_interface_scoring.pt \
  --input /path/to/score/train --validation /path/to/score/validation \
  --output /path/to/score/run --batch-size 32 --epochs 100 --device cuda
python -m masif_torch train --config examples/alignment.yaml \
  --input /path/to/ipa/train --validation /path/to/ipa/validation --output results/ipa --epochs 10
```

Both return `[B,2]` probabilities. BN uses population variance, epsilon `.001`, moving factor `.01`; Adam epsilon is `1e-7`. ScoreNN class weights default to `(.005,1)`; IPA uses `(1/n_negatives,1/n_positives)`, normally `(.01,1)`. Override with `--class-weights w0 w1`. Validation loss is unweighted. IPA input is FP32 `[B,100 or 200,F]`; `AlignmentEvaluationNN.prepare(raw,rng)` selects four-feature columns, pads, or samples.

### Save, resume, Python API

Every epoch writes `last.pt`, `history.json`, and `best.pt` when loss improves. Native checkpoints preserve optimizer, epoch, step, seed, class weights, best metric, history, and Torch/CUDA RNG.

```bash
python -m masif_torch train --checkpoint /path/to/run/last.pt --resume \
  --input /path/to/train --validation /path/to/validation --output /path/to/run --epochs 10 --device cuda
```

`--epochs 10` runs ten additional epochs. Without `--resume`, a checkpoint loads weights and starts a new optimizer. Keep the same data/validation definition on recovery. Old TF files lack original sampler/RNG/history, so Adam slots do not restore an old trajectory. Best selection uses loss rather than old site/search AUC/early stopping.

```python
from masif_torch.checkpoints import load_checkpoint
from masif_torch.cli import predict
from masif_torch.data import load_npz
from masif_torch.validation import deterministic

# Match the validated FP32 policy, including disabling TF32 on CUDA.
deterministic(seed=0)
model, metadata = load_checkpoint("model.pt", device="cpu")
prediction = predict(model, load_npz("packet.npz"))
```

Fresh training replaces `--checkpoint` with `--config config.yaml`, for example:

```yaml
task: site
max_rho: 9.0
n_thetas: 4
n_rhos: 3
n_rotations: 4
feat_mask:
- 1
- 1
- 1
- 1
- 1
n_conv_layers: 3
```

## MaSIF-seed and MaSIF-neosurface

### Official IPA weights

```bash
mkdir -p artifacts/upstream_repos
git clone https://github.com/LPDI-EPFL/masif_seed.git artifacts/upstream_repos/masif_seed
git -C artifacts/upstream_repos/masif_seed checkout 77153911d8252d02193a6d1d6d0539bf02c5faba
git clone https://github.com/LPDI-EPFL/masif-neosurf.git artifacts/upstream_repos/masif-neosurf
git -C artifacts/upstream_repos/masif-neosurf checkout b522fdc537e0724e7c574753f9068f88e687c736
conda activate masif_nn
python tools/export_design_weights.py --seed-repo artifacts/upstream_repos/masif_seed \
  --neosurf-repo artifacts/upstream_repos/masif-neosurf --output artifacts/design_exports
conda activate masif_torch
python tools/run_validation.py --exports artifacts/design_exports \
  --output artifacts/design_checkpoints --report artifacts/design_weights.json --cuda
python tools/organize_weights.py --task seed --task neosurface
```

Omit `--cuda` for CPU. Ten files across both upstream repositories, including 3/4-feature, 9/12 Å, standard/benchmark variants, pass default forward/loss/all-gradient tolerances on CPU/GPU (20 combinations). Example output: `weights/seed/ipa_12A_surface_atom.pt`.

`.hdf5.index` means a TF checkpoint prefix; a standalone `.hdf5` file is real HDF5. Both are supported. Conversion preserves inference weights/BN buffers, records ignored Keras slots, and rejects Keras `--include-optimizer`.

IPA raw columns: inverse point distance, inverse descriptor distance, normal dot product, inverse atom distance. In the name `0129`, atom feature is actual column 3; three-feature models select `[0,1,2]`. A 9 Å model uses 100 points, 12 Å uses 200. Zero padding participates in pooling/BN. Oversized patches consume a continuous `RandomState(42)` stream.

### Processed surfaces and real fixtures

```text
processed_root/
  data_preparation/01-benchmark_pdbs/CHAIN.pdb
  data_preparation/04b-precomputation_12A/precomputation/ID/p1_list_indices.npy
  descriptors/sc05/all_feat/ID/p1_desc_straight.npy
  descriptors/sc05/all_feat/ID/p1_desc_flipped.npy
  output/all_feat_3l/pred_surfaces/CHAIN.ply
  output/all_feat_3l/pred_data/pred_CHAIN.npy
```

Target geometry/interface comes from PLY; source screening uses original NPY interface precision. Targets need flipped descriptors and may lack straight; sources need straight. Points, normals, surfaces, PDB atoms share one Å coordinate frame. A protein-only surface cannot substitute for a protein/ligand target.

```bash
python -m masif_torch pack-design --root processed_target --identifier 1A7X_A \
  --role target --trusted-indices --output target.npz
python -m masif_torch pack-design --root processed_database --identifier 2GKW_B \
  --role source --trusted-indices --output source.npz
```

Safe NPZ encodes neighbors as `neighbor_indices` + `neighbor_offsets`; stores points/normals, available descriptors, interface, heavy/CA atoms, optional `pdb_path`. Keep PDB paths valid when moving packets if writing PDB. JSON paths are relative to the **command working directory**, not config location.

A reproducible 43-member subset of the [official Zenodo dataset](https://zenodo.org/records/14225758):

```bash
python tools/fetch_design_fixtures.py --output artifacts/real_fixtures/neosurf_1a7x --download
python tools/fetch_design_fixtures.py --output artifacts/real_fixtures/neosurf_1a7x
```

The second command verifies local files without networking. Selected members are checked by size/CRC32/SHA256; this subset fetch does not verify the full archive MD5. The separate full-benchmark download below was subsequently verified against the official full MD5. HTTP 429 stops downloading. Fixtures include with-ligand 1A7X A/B and decoy 2GKW_B. Targets lack raw NN arrays; decoy has complete 9/12 Å arrays.

### Seed retrieval and docking

Copy [examples/seed.yaml](examples/seed.yaml), replacing paths:

```yaml
task: seed
action: search
device: cpu
seed: 42
target: target.npz
database:
- source.npz
checkpoint: weights/seed/ipa_12A_surface_atom.pt
output: results/seed
num_sites: 1
registration:
  mode: replay
  path: original_poses.npz
```

```bash
python -m masif_torch run --config seed.yaml
```

Defaults: outward shift `.25 Å`, interface `>.75`, descriptor distance `<2`, fitness `>=.3`, IPA `>.9`, CA clashes `<=0`, heavy clashes `<=5`, clash radius `2 Å`. Target normals flip after outward shift. Site selection uses mean patch interface, halves selected-patch values, retains supplied order on ties; `selected_vertices` restricts eligible centers.

- `action:"match"` needs no checkpoint or registration.
- Replay NPZ: `keys:[K]`, `transforms:[K,4,4]`, optional `fitness:[K]`. Keys are `site_index:source_name:source_vertex`; transforms map original source to target, including historical random initialization. Every candidate needs a pose.
- Modern docking: `"registration":{"mode":"open3d","iterations":2000,"radius":1.5,"confidence":0.999,"seed":42}`. Confidence differs from old `max_validation=500`; poses are not guaranteed identical to historical Open3D.

`results.json` records ordered candidates/rejections/probabilities/distances/clashes/transforms. Accepted hits save numeric NPZ features, importance, and aligned patches; valid `pdb_path` enables PDB output. Seed removes atoms whose names begin `H` and transforms remaining atoms once. Clash semantics retain source CA queries against target heavy atoms and heavy-count zero on CA early abort.

### Neosurface aligned-pair scoring

```yaml
task: neosurface
action: score
target: target_with_ligand.npz
database:
- already_aligned_binder.npz
checkpoint: weights/neosurface/ipa_12A_surface_atom.pt
num_sites: 1
output: results/neosurface
device: cpu
```

Run the same `run --config` command. Binder must already be aligned; `score` does not dock. It chooses source vertex nearest target center and reports center/average-patch descriptor distances, IPA, and signed importance. Score-only output copies original binder PDB text/atoms. To dock, use `search` with registration.

Importance is `P(occluded)-P(original)` with upstream hardcoded `.9` importance cutoff. All-zero changes yield original NaNs: NPZ retains them, JSON flags `importance_finite:false`. Identical descriptors give undefined inverse-descriptor Inf; scoring rejects this input instead of changing mathematics with epsilon.

## Raw preparation, Rosetta, and benchmarks

These task implementations live in `src/tasks/{preparation,rosetta,benchmark}`. Install their modern extras:

```bash
conda activate masif_torch
python -m pip install -e '.[design,chemistry]' --index-url https://pypi.org/simple
```

### Raw protein and protein/ligand surfaces

The original geometry algorithms remain isolated in a compatible legacy interpreter. The validated existing interpreter is Python 3.6.5, PyMesh 0.2.1, NumPy 1.19.4, SciPy 1.1.0, scikit-learn 0.24.0, Biopython 1.73, NetworkX 2.3, and Open3D 0.5.0.0. It was reused, not newly rebuilt. The pinned neosurf repository's `Dockerfile` is an upstream starting point for restoring these dependencies and reduce/MSMS/APBS/PDB2PQR/multivalue; a fresh build of that Dockerfile has not been validated in this migration. A compatible local interpreter or container wrapper must expose the configured source/data paths and executable tools.

Copy [examples/preparation_protein.yaml](examples/preparation_protein.yaml) or [examples/preparation_ligand.yaml](examples/preparation_ligand.yaml), replacing machine-specific paths. Important configuration keys:

| Key | Required value |
|---|---|
| `pdb`, `identifier` | Raw PDB and selected `PDB_CHAIN` identifier |
| `source_root` | Pinned neosurf `masif/source`, containing original geometry/precomputation routines |
| `legacy_python` | Compatible original PyMesh Python executable |
| `chemistry_python` | Optional modern interpreter, default current Torch Python |
| `output` | Separate preparation output directory |
| `ligand`, `sdf`, `mol2` | Optional residue/chain such as `FKA_B`, SDF bond-order template, optional patched MOL2 |
| `site_checkpoint`, `search_checkpoint` | Optional finite converted models for generated-surface predictions |
| `precompute` | Defaults to `["site","ppi_search"]`; also supports `ligand` |
| `pdb2pqr_apbs_input` | `flag` for PDB2PQR 2.1, `filename` for the upstream later CLI dialect |
| `environment` | Absolute `MSMS_BIN`, `PDB2PQR_BIN`, `APBS_BIN`, `MULTIVALUE_BIN`; `PATH` including reduce; optional `REDUCE_HET_DICT` |

```bash
python -m masif_torch run --config examples/preparation_protein.yaml
python -m masif_torch run --config examples/preparation_ligand.yaml
```

The preparation task runs checked external stages and writes per-stage logs, prepared PDB/PLY, original 9/12 Å patch arrays and optional Torch site/straight/flipped predictions. Outputs retain the processed-root layout used by `pack-design`.



Ligand preparation preserves supplied heavy-atom coordinates and adds missing hydrogens. The MOL2 adapter supplies the substructure table required by old PDB2PQR. Hydrogen coordinates can differ from historical tools, so independently rebuilt ligand surfaces and downstream scores need not be identical. Select the matching PDB2PQR dialect in the configuration; external stage failures are reported.

### Rosetta relaxation and interface design

Copy [examples/rosetta_1a7x.yaml](examples/rosetta_1a7x.yaml), replacing external tool/database/Python paths:

```bash
python -m masif_torch run --config examples/rosetta_1a7x.yaml
```

Provide `binary` (RosettaScripts), `database`, `dalphaball`, `protocol_dir` (upstream `rosetta_scripts/seed_refine/src`), `legacy_python` for original `molfile_to_params.py`, modern `chemistry_python`, target/binder paths/chains, ligand name/chain, and a template SDF or prepared `ligand_sdf`. Optional `verify_repeat:true` runs interface design twice from the same relaxed structure and fixed seed.

Rosetta uses the configured original protocols, parameterizes the ligand, places target/ligand in chain A and binder in chain B, then runs relaxation and interface design. It checks fresh output files, finite scores/coordinates and retained partners. Logs and `results.json` record stage failures. Optional same-seed repetition checks the same binary/database/input at output-file precision; it does not establish equivalence across versions or rebuilt hydrogen inputs. Scaffold grafting remains a separate upstream workflow.

### Public small-decoy benchmark and numerical comparison

The optional public benchmark covers 112 searches over the original complexes and
small-decoy database. Download the official archive and extract it before running:

```bash
python tools/download_benchmark.py --output artifacts/benchmark_full
python tools/extract_benchmark.py --archive artifacts/benchmark_full/dataset.zip \
  --output artifacts/benchmark_full/dataset
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m masif_torch run --config examples/benchmark.yaml
```

The extraction tool verifies member CRC32 and handles the archive's wrapped ZIP32
offsets. The benchmark uses `weights/neosurface/ipa_12A_surface_atom.pt`; different
upstream benchmark-training snapshots are not interchangeable defaults. Resume
requires matching model, input, protocol and cache provenance.

Modern Open3D registration can produce different poses and rankings from historical
Open3D. The migration's full benchmark stopped after 29/112 searches; full paper
recovery rates are not certified. Run the complete benchmark only if needed.

## Architecture and extension

```text
masif_torch/
  src/                         # installed as package masif_torch
    tasks/
      site/model.py
      search/model.py
      ligand/model.py
      score/model.py
      alignment/model.py
      seed/workflow.py
      neosurface/workflow.py
      preparation/             # legacy geometry / modern chemistry bridge
      rosetta/                 # external refinement and packaged worker
      benchmark/               # full public small-decoy protocol
    shared/layers.py            # Gaussian convolutions / common math
    shared/task.py              # prepared-packet execution adapter
    shared/design/             # surfaces, features, registration
    models.py                  # compatible exports and model factory
    checkpoints.py, optim.py, data.py, training.py, validation.py, cli.py
  tools/                       # input conversion, weight export/collection, data downloads
  environments/                # recipes, dependency pins, snapshots
  examples/                    # task configurations
  USAGE.md                     # application instructions
  artifacts/                   # generated/downloaded local data and outputs
```

Task registration lazily loads the selected owner. Neural tasks use `NeuralTask`, shared trainer/checkpoint contracts. `NeoSurfaceWorkflow` extends `SeedWorkflow` with aligned-pair scoring; both compose `SurfaceRecord`, `AlignmentScorer`, `RegistrationBackend`. Implementations preserve model parameter names/order for old converted state dictionaries and optimizer states. No duplicate SeedNN/NeoSurfNN is introduced.

Custom applications provide `run()` and register with `register_task("my_task", MyTask)`; existing names cannot be overwritten. Custom registration implements `RegistrationBackend.align(key,source,target)`, returning a source-to-target rigid transform and fitness; pass it to `SeedWorkflow(config, registration=my_backend)`.

Neural tasks also support `run` JSON: `{"task":"site","checkpoint":"site.pt","input":"surface.npz","output":"site.npy"}`. Training uses `Trainer`. Aliases include `masif-seed`, `masif_seed`, `masif-neosurf`, `masif-neosurface`, `neosurf`. Optional registration backends import only when used.

## Documentation and references

Common issues:

- **NaN checkpoint:** inspect `rejected.json`; select an explicit finite alternative or train anew. There is no exact repair of damaged weights.
- **Site index out of bounds:** retain the complete surface; do not independently batch/filter indexed vertices.
- **Custom features:** specify the original five-column mask; input F equals selected-column count.
- **TensorFlow in runtime:** keep environments separate; TF reader/export tools run in `masif_nn`.
- **Different training experiment:** initialization RNG, corrected ligand sampling, best selection, and cross-framework trajectories differ as documented.

Original MaSIF is [Apache 2.0 licensed](LICENSE); retain original license/attribution when distributing derivative code. Seed/neosurf code and data retain upstream terms. See the original [citation file](citation.bib).

### Original methods used by this repository

1. **MaSIF-site, MaSIF-search, MaSIF-ligand and second-stage scoring:** Gainza, P., et al. *Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning.* **Nature Methods 17**, 184–192 (2020). [DOI: 10.1038/s41592-019-0666-6](https://doi.org/10.1038/s41592-019-0666-6).
2. **MaSIF-seed and protein binder design:** Gainza, P., et al. *De novo design of protein interactions with learned surface fingerprints.* **Nature 617**, 176–184 (2023). [DOI: 10.1038/s41586-023-05993-x](https://doi.org/10.1038/s41586-023-05993-x).
3. **MaSIF-neosurf / MaSIF-neosurface:** Marchand, A., et al. *Targeting protein–ligand neosurfaces with a generalizable deep learning tool.* **Nature 639**, 522–531 (2025). [DOI: 10.1038/s41586-024-08435-4](https://doi.org/10.1038/s41586-024-08435-4).

### Related MaSIF methods and applications

4. **dMaSIF:** Sverrisson, F., Feydy, J., Correia, B. E., and Bronstein, M. M. *Fast End-to-End Learning on Protein Surfaces.* **IEEE/CVF CVPR**, 15272–15281 (2021). [Paper and official BibTeX](https://openaccess.thecvf.com/content/CVPR2021/html/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.html).
5. **Physics-informed docking with dMaSIF:** Sverrisson, F., Feydy, J., Southern, J., Bronstein, M. M., and Correia, B. *Physics-informed deep neural network for rigid-body protein docking.* **Machine Learning for Drug Discovery Workshop at ICLR (MLDD)** (2022). [Paper](https://openreview.net/forum?id=5yn5shS6wN).
6. **DiffMaSIF:** Sverrisson, F., et al. *DiffMaSIF: Surface-based Protein-Protein Docking with Diffusion Models.* **Machine Learning for Structural Biology Workshop at NeurIPS (MLSB)** (2023). [Paper](https://www.mlsb.io/papers_2023/DiffMaSIF_Surface-based_Protein-Protein_Docking_with_Diffusion_Models.pdf).
7. **Human surfaceome / SURFACE-Bind application:** Balbi, P. E. M., et al. *Mapping targetable sites on the human surfaceome for the design of novel binders.* **PNAS 123**(5), e2506269123 (2026). [DOI: 10.1073/pnas.2506269123](https://doi.org/10.1073/pnas.2506269123).

The related methods and applications above provide further reading; dMaSIF, physics-informed docking, DiffMaSIF, and SURFACE-Bind are not implemented by this repository.

Full BibTeX entries are in [citation.bib](citation.bib). Cite the original method for each application you use.

Sources: [official Torch CUDA wheels](https://download.pytorch.org/whl/cu126/torch/), [TensorFlow installation](https://www.tensorflow.org/install/pip), [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
