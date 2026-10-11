# Detailed MaSIF usage and numerical validation

This guide preserves the detailed preparation, checkpoint conversion, numerical verification, and compatibility commands. Run these commands from the standalone project root. The primary application interface is now `masif` with Hydra configuration; start with the [application README](../README.md) and [installation guide](../INSTALL.md). The `masif_torch` Python API and existing compatibility commands remain available.

MaSIF (Molecular Surface Interaction Fingerprints) learns interaction patterns from molecular surface geometry and chemistry. This standalone project provides PyTorch MaSIF-site, MaSIF-search, MaSIF-ligand, the second-stage ScoreNN, and the IPA alignment network used by MaSIF-seed and MaSIF-neosurf. It includes training, recovery, inference, legacy-weight conversion, prepared-surface retrieval, registration, scoring, and numerical audits.

The implementation lives directly in `src/`, with each task in `src/tasks/<task>/`. Installation exposes the stable `masif_torch` package and `masif-torch` command. The original TensorFlow repository is a separate checkout, called `../masif` in examples; adjust this path to your installation.

**Validated:** finite legacy weights, CPU/GPU inference and losses, original-source gradient comparisons, native Torch training recovery, real public precomputed surfaces, fixed-pose seed/neosurf geometry/scoring, complete raw protein/FKA preparation, and real Rosetta relaxation/interface design with exact same-seed output repeat. **Limits:** cross-framework operations are not generally bitwise identical; independent IPA training parameter updates fail the strict tolerance; modern Open3D docking is not an exact reproduction of historical random docking. Raw preparation, Rosetta, and full benchmarks are evaluated separately below. See the [overall report](torch_migration/REPORT.md).

## Contents

- [Methods and numerical contract](#methods-and-numerical-contract)
- [Requirements and installation](#requirements-and-installation)
- [Input data and preparation](#input-data-and-preparation)
- [Convert TensorFlow weights](#convert-tensorflow-weights)
- [Neural-network applications](#neural-network-applications)
- [MaSIF-seed and MaSIF-neosurface](#masif-seed-and-masif-neosurface)
- [Raw preparation, Rosetta, and benchmarks](#raw-preparation-rosetta-and-benchmarks)
- [Reproduce numerical validation](#reproduce-numerical-validation)
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

Validated on 2026-10-10: Linux x86_64, Python 3.12.15, NVIDIA L40S, driver 560.35.03.

| Environment | Validated versions | Purpose |
|---|---|---|
| `masif_torch` | Torch 2.14.1+cu126, NumPy 2.5.3, SciPy 1.18.1 | Train, predict, convert exported NPZ weights, validate |
| `masif_nn` | Python 3.12, tensorflow-cpu 2.21.0 | Read TF checkpoints/TFRecords and generate references |
| Optional design extra | Open3D 0.19.0, Biopython 1.88, plyfile 1.1.5 | Modern registration, PLY input, PDB output |
| Optional chemistry extra | RDKit 2026.9.1, openbabel-wheel 3.1.1.23, ProDy 2.6.1, scikit-learn 1.9.1, Biopython 1.88 | Raw ligand / Rosetta input preparation |

Prepared-array NN execution needs no TensorFlow, PyMesh, APBS, or system CUDA toolkit. GPU wheels include CUDA runtime libraries and require a compatible driver. CPU execution was tested on the CUDA wheel's CPU backend; a separate pure CPU wheel environment and other operating systems have not been fully audited.

### Get the project and install

Obtain this standalone project directory and run every command from its root:

```bash
cd /path/to/masif_torch
```

This migration is a local workspace project, not a published release of official MaSIF. Keep an original checkout separately for weights/source/geometry:

```bash
git clone https://github.com/LPDI-EPFL/masif.git ../masif
```

GPU runtime:

```bash
conda env create -f environments/masif_torch.yml
conda activate masif_torch
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r environments/runtime-requirements.txt
python -m pip install -e .
python -m masif_torch doctor
```

For CPU, replace the Torch line with:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

Alternatively, the installation helper accepts `cpu` or `cu126` and an optional **new** environment prefix:

```bash
bash scripts/install.sh cu126 /path/to/new/masif_torch
```

`doctor` reports versions, GPU availability, and `tensorflow_installed` (expected `false` in the target environment). Use `--device cuda` for GPU; NN commands otherwise default to CPU.

Legacy conversion and optional design setup:

```bash
conda env create -f environments/masif_nn.yml
conda activate masif_nn
python -c "import tensorflow as tf; print(tf.__version__)"
conda activate masif_torch
python -m pip install -e '.[design]' --index-url https://pypi.org/simple
```

TF `compat.v1` executes saved `.meta` graphs. TF2 has no `tf.contrib`; use these tools rather than invoking all original TF1 training scripts directly. Portable installation recipes are included in `environments/*.yml` and `environments/runtime-requirements.txt`. Optional machine-specific audit snapshots (`environments/*.lock.txt`) are kept locally and excluded from Git publication. Install the chosen Torch wheel first and local project separately; CUDA snapshots describe Linux CUDA, not CPU/cross-platform recipes. If inherited pip mirrors interfere, prefix the affected command with `PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL=''` and specify the intended index. See [INSTALL.md](../INSTALL.md).

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

The [weight catalog](../weights/README.md) lists task/purpose filenames and their
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

Nonfinite exports write `rejected.json`; NaNs are not replaced with zeros. The site backup is not equivalent to the damaged main model. Details are in [stage 03](torch_migration/03_checkpoints.md).

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

Sampling fixes the historical patch/mask mismatch and selects eligible pockets instead of skipping random undersized pockets. Fixed-packet mathematics is validated; the historical sampling schedule/trajectory is not claimed identical. See [stage 04](torch_migration/04_workflows.md).

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

Copy [examples/seed.yaml](../examples/seed.yaml), replacing paths:

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

Copy [examples/preparation_protein.yaml](../examples/preparation_protein.yaml) or [examples/preparation_ligand.yaml](../examples/preparation_ligand.yaml), replacing machine-specific paths. Important configuration keys:

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

The task executes checked subprocess stages and exchanges primitive NPZ arrays. It writes per-stage logs, `stages.json`, prepared PDB/PLY, 9/12 Å original patch arrays, and optional Torch site/straight/flipped predictions. Outputs keep the processed-root layout used by `pack-design`. Original PyMesh comparisons confirm that predicted PLY files retain coordinates, faces, and all unchanged vertex attributes exactly for both tested raw inputs; the new `iface` probabilities preserve original PyMesh serialization ([prediction-surface geometry](torch_migration/numerical_prediction_surface_geometry.json)).

The raw **protein** pipeline actually completed for 1A7X A: 2,253 vertices, 4,502 faces, site output `[1,2253]`, and straight/flipped descriptors `[2253,80]`. Direct original execution and the bridge agree exactly in every PLY field (including H-bond/hydrophobicity) and face indices. Initial modern near-neighbor interpolation differed on distance ties; executing the original interpolation in the legacy geometry interpreter restored exact equality. This result covers the tested protein, not all ligand or geometry-version combinations. Evidence: [raw geometry comparison](torch_migration/numerical_raw_geometry.json), [18 exact precomputation arrays](torch_migration/numerical_raw_precomputation.json), and [stage 12](torch_migration/12_raw_preparation.md).

For raw FKA, the bridge completes 76 missing hydrogens around 67 heavy atoms while keeping original heavy coordinates exactly. OpenBabel 3 can omit the MOL2 `SUBSTRUCTURE` table needed by PDB2PQR 2.1; the worker restores this table without changing atom/bond/charge entries. PDB2PQR CLI dialect selection is explicit and every external exit code is checked. The complete raw protein/ligand pipeline also finished: 2,543 vertices, site `[1,2543]`, and both descriptor orientations `[2543,80]`. New RDKit hydrogen coordinates are not claimed identical to historical hydrogen placement; fixed-input NN comparison is audited separately from hydrogen-generation equivalence. Both raw inputs pass independent TF/Torch validation on CPU and CUDA: full site surfaces of 2,253/2,543 vertices and 32 fixed search patches per input, eight groups / 400 forward, trace, loss, and gradient tensors. Maximum site probability error is below `3e-7`; maximum search descriptor error is `1.67e-6`. See [raw neural evidence](torch_migration/numerical_raw_neural.json).

### Rosetta relaxation and interface design

Copy [examples/rosetta_1a7x.yaml](../examples/rosetta_1a7x.yaml), replacing external tool/database/Python paths:

```bash
python -m masif_torch run --config examples/rosetta_1a7x.yaml
```

Provide `binary` (RosettaScripts), `database`, `dalphaball`, `protocol_dir` (upstream `rosetta_scripts/seed_refine/src`), `legacy_python` for original `molfile_to_params.py`, modern `chemistry_python`, target/binder paths/chains, ligand name/chain, and a template SDF or prepared `ligand_sdf`. Optional `verify_repeat:true` runs interface design twice from the same relaxed structure and fixed seed.

The task copies original XML/design schedules byte for byte with SHA256, parameterizes the ligand, places target/ligand in A and binder in B, executes the original contact selector, then runs FastRelax and interface design. It checks fresh output files, finite scores/coordinates, retained ligand/partners, and optional exact repeat equality. Logs and `results.json` record failures; success requires all requested stages to finish. The real 1A7X run completed all eight external phases with Rosetta 2021.16: relaxation (209.98 s), interface design (145.54 s), and a same-seed design repeat (143.93 s). All 143 FKA atoms were retained; the 3,466 exported design coordinate rows and all 36 exported score values are exactly equal to the repeat. This verifies the same binary/database/input/seed at file output precision, not internal state or cross-version equivalence. A second integration run consumed the newly prepared raw protein/FKA inputs and completed all six requested stages: relaxation and interface design retained all 143 FKA atoms, produced finite scores, and used the original XML bytes ([prepared-input results](torch_migration/numerical_rosetta_prepared.json)). Its rebuilt hydrogen coordinates differ from the standalone input, so the two runs are separate validations. Scaffold grafting remains a separate upstream workflow. See [stage 13](torch_migration/13_rosetta.md) and [standalone repeat results](torch_migration/numerical_rosetta.json).

### Public small-decoy benchmark and numerical comparison

The benchmark task covers all 14 complexes, both target directions (28 subunits), with/without ligand, against the 28 subunits and 200 decoys: **112 searches**. The separate optional 8,879-decoy PDBbind dataset and the paper's seed design database are different datasets. Full preparation needs approximately 10 GB archive plus 17.3 GB extracted files and additional cache/results space.

```bash
python tools/download_benchmark.py --output artifacts/benchmark_full
python tools/extract_benchmark.py --archive artifacts/benchmark_full/dataset.zip \
  --output artifacts/benchmark_full/dataset
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m masif_torch run --config examples/benchmark.yaml
```

Download/extraction actually completed: 9,961,569,979 archive bytes, official MD5 `2b45977254f45c6e229390cc62f3685d`, 5,339 extracted files / 17,289,741,138 bytes, all member CRC32 checked. The extraction tool handles the archive's wrapped ZIP32 offsets.

The protocol uses the actual upstream benchmark config: `models_std/weights_12A_0129.hdf5`, descriptor `<3`, interface `>0`, IPA `>.8`, three target sites, fitness `>=.3`, original clash limits, and modern 2,000-iteration RANSAC. Independent AST comparisons verify all 28 interface COMs and five random-transform/composition checks exactly ([33-check review](torch_migration/numerical_benchmark_protocol_review.json)). Benchmark input/cache/resume provenance is tied to checkpoint, definitions, consumed dataset files, original PDBs, code, cases, and cache hashes; incompatible resumed runs are rejected. Benchmark point importance is computed after acceptance; rejected poses explicitly mark this diagnostic as omitted. The eager and deferred paths preserve exact probabilities, accepted point importance (including NaN semantics), and RNG states for all 53 captured real poses, plus the oversized sampling boundary ([optimization regression](torch_migration/numerical_benchmark_importance.json)). Default seed scoring retains eager diagnostics.

A complete 1A7X A comparison case also ran through the original Open3D 0.5 registration worker: 30,810 candidates and 53 accepted poses versus 67 accepted by the modern backend. Both recover the true partner at group rank 1, but pose coordinates differ. Ranking retains the upstream best pose per `(site,source)` group. RMSD/iRMSD summaries now read exported PDB coordinates as float32 and preserve the original protein-heavy atom selection and arithmetic. Independent original-source AST comparisons agree exactly for all 28 coordinate selections, the 18 captured native-partner pose metrics, and their best-group summary. The earlier unrounded-coordinate difference remains recorded in the [metric review](torch_migration/numerical_benchmark_metrics_review.json). For all 53 captured legacy-pose feature packets, original Keras logits/probabilities and Torch CPU/CUDA inference pass strict FP32 tolerances ([fixed-pose scores](torch_migration/numerical_benchmark_legacy_scores.json)). TF32 was initially detected causing CUDA strict failure; that failure remains in a [separate report](torch_migration/numerical_benchmark_legacy_scores_tf32.json), and the corrected test disables TF32 rather than loosening tolerance.

At the user's request, the full evaluation was stopped after **29 of 112 searches** completed, retaining **1,925,440 candidates and 1,738 accepted poses**. The final task scope is fixed-input numerical comparison. The original 53-pose TF/Torch score comparison passes strict FP32 tolerance, and protocol/reader/metric/importance audits establish the exact comparisons described above. Full benchmark success and paper-wide recovery rates are not claimed. See [stage 14](torch_migration/14_benchmark.md) and the [stopped-run evidence](torch_migration/numerical_benchmark_partial_run.json) for retained partial execution. The complete-run commands above remain optional. After a future run completes all 112 cases, combine target/decoy rankings and analyze all 228 source proteins:

```bash
python tools/analyze_benchmark.py --run artifacts/benchmark_full/run \
  --original-analysis artifacts/upstream_repos/masif-neosurf/computational_benchmark/analysis_utils.py \
  --output docs/torch_migration/numerical_benchmark_combined.json
```

The analysis reports any-rank recovery and separate Top-K curves. `original_source_topk` executes the unchanged original function, including its `len(result_list)<=k` skip; `inclusive_topk` includes shorter result lists. These definitions are kept separate.

Modern Open3D 0.19 random/stopping semantics and the explicit sorted database order differ from historical Open3D/filesystem order; successful completion does not prove identical poses, ranks, or paper recovery rates.

## Reproduce numerical validation

Existing evidence is in `docs/torch_migration/numerical_*.json`. Regeneration uses `masif_nn` for TF tools and `masif_torch` for validation/inference. Omit `--cuda` without GPU.

### Original models and runtime tests

```bash
conda activate masif_nn
python tools/create_source_oracles.py --source ../masif/source --output artifacts/source_oracles
python tools/convert_records.py --input artifacts/source_oracles/synthetic.tfrecord --output artifacts/records
python tools/check_input_shapes.py --oracles artifacts/source_oracles \
  --output artifacts/input_shapes --report artifacts/numerical_input_shapes.json
python tools/export_tf.py --repo ../masif --output artifacts/tf_exports
conda activate masif_torch
python -m unittest discover -s tests -v
python tools/run_validation.py --exports artifacts/source_oracles \
  --output artifacts/source_validated --report artifacts/numerical_source.json --cuda
python tools/run_validation.py --exports artifacts/tf_exports \
  --output artifacts/validated_checkpoints --report artifacts/numerical_checkpoints.json --cuda
python tools/run_validation.py --exports artifacts/input_shapes \
  --output artifacts/input_shapes_validated --report artifacts/numerical_input_shapes_torch.json --cuda
python tools/smoke_workflows.py --oracles artifacts/source_oracles \
  --score-export artifacts/tf_exports/data/masif_pdl1_benchmark/models/nn_score/trained_model.hdf5 \
  --output artifacts/workflows --report artifacts/numerical_workflows.json --cuda
```

The original 18-test suite passed without skips after relocation. Final standalone installation verification passes **28 tests with no skips** and confirms **41 wheel source files are byte for byte identical** to the project sources, including all ten task packages and four external workers ([installation evidence](torch_migration/numerical_standalone_install.json)). The expanded integration suite includes source-oracle forward/loss/all-gradient tests, exact native dropout recovery, benchmark provenance rejection, partial-ligand hydrogen preservation, MOL2 compatibility, and Rosetta repeat-failure reporting. Missing generated oracles cause skips; skips are not numerical acceptance. Core-only setup can run `python -m unittest discover -s tests -p test_numerics.py -v`; design extra provides full geometry/PDB coverage.

### Real surfaces and geometry

After fetching fixtures/IPA weights:

```bash
conda activate masif_torch
python tools/validate_design_geometry.py \
  --source artifacts/upstream_repos/masif_seed/masif_seed_search/source/alignment_utils.py \
  --fixture artifacts/real_fixtures/neosurf_1a7x --report artifacts/design_geometry.json
python tools/smoke_design.py --fixtures artifacts/real_fixtures/neosurf_1a7x \
  --checkpoint weights/seed/ipa_12A_surface_atom.pt \
  --output artifacts/design_smoke --report artifacts/design_workflows.json --cuda
python tools/pack_real_reference.py --fixtures artifacts/real_fixtures/neosurf_1a7x --output artifacts/real_fixtures
conda activate masif_nn
python tools/export_tf.py \
  --checkpoint ../masif/data/masif_peptides/nn_models/all_feat_3l/model_data.bak/model \
  --input artifacts/real_fixtures/site_real.npz --output artifacts/real_exports/site
python tools/export_tf.py --checkpoint artifacts/upstream/51e5422/model \
  --input artifacts/real_fixtures/search_real.npz --output artifacts/real_exports/search
cp artifacts/real_exports/site/adam_reference.npz artifacts/real_exports/adam_reference.npz
conda activate masif_torch
python tools/run_validation.py --exports artifacts/real_exports \
  --output artifacts/real_checkpoints --report artifacts/real_surfaces.json --cuda
```

The geometry audit extracts original functions via AST and records source hashes: 83 real/synthetic boundary comparisons pass exactly. Real 2GKW_B site/search passes strict CPU/GPU tolerance. Design smoke uses a constructed translation and relaxed thresholds for output coverage; it does not reproduce old random docking or paper benchmarks.

### Raw-prepared protein/ligand neural references

Use the checked preparation output directly, preserving full site indices and choosing 32 search patches for the four-block loss:

```bash
conda activate masif_torch
python tools/pack_preparation_references.py --root artifacts/raw_pipeline/protein_bridge \
  --identifier 1A7X_A --output artifacts/raw_pipeline/inputs/protein
conda activate masif_nn
python tools/export_tf.py \
  --checkpoint ../masif/data/masif_peptides/nn_models/all_feat_3l/model_data.bak/model \
  --input artifacts/raw_pipeline/inputs/protein/site.npz --output artifacts/raw_pipeline/exports/protein_site
python tools/export_tf.py --checkpoint artifacts/upstream/51e5422/model \
  --input artifacts/raw_pipeline/inputs/protein/search.npz --output artifacts/raw_pipeline/exports/protein_search
conda activate masif_torch
python -m masif_torch validate --export artifacts/raw_pipeline/exports/protein_site \
  --checkpoint weights/site/interface_prediction_backup.pt --report artifacts/raw_pipeline/protein_site.json
python -m masif_torch validate --export artifacts/raw_pipeline/exports/protein_search \
  --checkpoint weights/search/surface_descriptors_historical.pt --report artifacts/raw_pipeline/protein_search.json
```

For the ligand-containing surface, replace preparation root with `artifacts/raw_pipeline/ligand` and use separate input/export/report paths. Both raw protein/FKA inputs pass the full-site audits (2,253/2,543 vertices, 60 tensors each) and 32-patch search audits (40 tensors each) on CPU and CUDA: eight groups, 400 tensors. See [summary](torch_migration/numerical_raw_neural.json). Add `--device cuda` to the validate commands to regenerate GPU results. These are fixed-input NN comparisons; independently generated hydrogen coordinates and historical random docking are separate comparisons.

### Original Keras IPA source audit

```bash
conda activate masif_nn
python tools/export_ipa_source.py \
  --source artifacts/upstream_repos/masif_seed/masif_seed_search/source/alignment_evaluation_nn.py \
  --checkpoint artifacts/upstream_repos/masif_seed/masif_seed_search/data/scoring_nn/models_std/weights_12A_0129.hdf5 \
  --config artifacts/design_exports/masif_seed/models_std/weights_12A_0129.hdf5/config.json \
  --real-features artifacts/design_smoke/cpu/seed/site_0_2GKW_B_35.npz \
  --output artifacts/ipa_source/standard
conda activate masif_torch
python tools/validate_ipa_source.py --reference artifacts/ipa_source/standard \
  --checkpoint weights/seed/ipa_12A_surface_atom.pt \
  --report artifacts/ipa_source_real.json --cuda
```

**The last command is expected to exit 1.** `groups` preserves failing independent three-step parameter updates; source inference/loss/gradients/BN, same-gradient optimizer updates, and updated evaluation probabilities pass. Do not interpret the full audit as passing.

Evidence: [source](torch_migration/numerical_source.json), [checkpoints](torch_migration/numerical_checkpoints.json), [20 IPA combinations](torch_migration/numerical_design_weights.json), [83 exact geometry checks](torch_migration/numerical_design_geometry.json), [real site/search](torch_migration/numerical_real_surfaces.json), [IPA grouped audit](torch_migration/numerical_ipa_source_real.json), [design workflows](torch_migration/numerical_design_workflows.json).

### Original-registration pose scores

The captured real comparison batch is `artifacts/benchmark_full/legacy_comparison/prepared_scores.npz`. It contains the 53 fixed legacy-pose `[200,4]` IPA packets; it must already exist from the completed comparison case. Re-export official-source scores and compare:

```bash
conda activate masif_nn
python tools/export_benchmark_scores.py \
  --input artifacts/benchmark_full/legacy_comparison/prepared_scores.npz \
  --config artifacts/benchmark_full/legacy_comparison/config.json \
  --source artifacts/upstream_repos/masif-neosurf/masif_seed_search/source/alignment_evaluation_nn.py \
  --checkpoint artifacts/upstream_repos/masif-neosurf/masif_seed_search/data/scoring_nn/models_std/weights_12A_0129.hdf5 \
  --output artifacts/benchmark_full/legacy_comparison/tensorflow_scores.npz
conda activate masif_torch
python tools/validate_benchmark_scores.py \
  --reference artifacts/benchmark_full/legacy_comparison/tensorflow_scores.npz \
  --checkpoint weights/neosurface/ipa_12A_surface_atom.pt \
  --output artifacts/benchmark_full/legacy_comparison/score_comparison.json
```

The validator disables TF32 and tests CPU plus CUDA when available. This compares scores for the same captured inputs, not independent stochastic docking trajectories.

Recheck eager versus deferred accepted-pose importance and sampling in Torch:

```bash
python tools/validate_benchmark_importance.py --config examples/benchmark.yaml \
  --fixtures artifacts/benchmark_full/legacy_comparison/with_ligand/targets/1A7X_A \
  --report artifacts/benchmark_full/legacy_comparison/importance_comparison.json
```

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
  tools/                       # reference export, conversion, fetch, audit/smoke
  tests/                       # original-source and recovery regressions
  environments/                # recipes, dependency pins, snapshots
  examples/                    # task configurations
  docs/torch_migration/        # stage logs and numerical reports
  artifacts/                   # generated/downloaded local data and outputs
```

Task registration lazily loads the selected owner. Neural tasks use `NeuralTask`, shared trainer/checkpoint contracts. `NeoSurfaceWorkflow` extends `SeedWorkflow` with aligned-pair scoring; both compose `SurfaceRecord`, `AlignmentScorer`, `RegistrationBackend`. Implementations preserve model parameter names/order for old converted state dictionaries and optimizer states. No duplicate SeedNN/NeoSurfNN is introduced.

Custom applications provide `run()` and register with `register_task("my_task", MyTask)`; existing names cannot be overwritten. Custom registration implements `RegistrationBackend.align(key,source,target)`, returning a source-to-target rigid transform and fitness; pass it to `SeedWorkflow(config, registration=my_backend)`.

Neural tasks also support `run` JSON: `{"task":"site","checkpoint":"site.pt","input":"surface.npz","output":"site.npy"}`. Training uses `Trainer`. Aliases include `masif-seed`, `masif_seed`, `masif-neosurf`, `masif-neosurface`, `neosurf`. Optional registration backends import only when used.

## Documentation and references

Stage records describe purpose, planned work, actual changes, numerical tests, and limits:

[01 Baseline](torch_migration/01_baseline.md) · [02 Models](torch_migration/02_models.md) · [03 Checkpoints](torch_migration/03_checkpoints.md) · [04 Workflows](torch_migration/04_workflows.md) · [05 Installation/verification](torch_migration/05_end_to_end.md) · [06 Review](torch_migration/06_review.md) · [07 Design plan](torch_migration/07_seed_neosurf_plan.md) · [08 IPA](torch_migration/08_alignment_models.md) · [09 Design workflows](torch_migration/09_design_workflows.md) · [10 Design review](torch_migration/10_design_review.md) · [11 Standalone structure](torch_migration/11_standalone_structure.md) · [12 Raw preparation](torch_migration/12_raw_preparation.md) · [13 Rosetta](torch_migration/13_rosetta.md) · [14 Benchmark](torch_migration/14_benchmark.md) · [Overall report](torch_migration/REPORT.md).

Earlier logs preserve historical paths and evidence; use this README and [INSTALL.md](../INSTALL.md) for current commands. No commit/publication is implied. Original PyMOL plugin/coloring and Docker documentation remain in the [official repository](https://github.com/LPDI-EPFL/masif); a new Torch Docker image and end-to-end PyMOL coloring were not validated. Prediction/descriptor array formats are preserved.

Common issues:

- **NaN checkpoint:** inspect `rejected.json`; select an explicit finite alternative or train anew. There is no exact repair of damaged weights.
- **Site index out of bounds:** retain the complete surface; do not independently batch/filter indexed vertices.
- **Custom features:** specify the original five-column mask; input F equals selected-column count.
- **TensorFlow in runtime:** keep environments separate; TF reader/export tools run in `masif_nn`.
- **Different training experiment:** initialization RNG, corrected ligand sampling, best selection, and cross-framework trajectories differ as documented.

Original MaSIF is [Apache 2.0 licensed](../LICENSE); retain original license/attribution when distributing derivative code. Seed/neosurf code and data retain upstream terms. See the original [citation file](../citation.bib).

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

Full BibTeX entries are in [citation.bib](../citation.bib). Cite the original method for each application you use.

Sources: [official Torch CUDA wheels](https://download.pytorch.org/whl/cu126/torch/), [TensorFlow installation](https://www.tensorflow.org/install/pip), [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
