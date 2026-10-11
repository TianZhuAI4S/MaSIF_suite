# MaSIF user guide

This guide walks you through using MaSIF with your own structures and datasets.
Complete the [installation](README.md#installation) first, then choose the task
that matches your goal.

## Contents

- [Getting started](#getting-started)
- [Notebook tutorials](#notebook-tutorials)
- [Input data and preparation](#input-data-and-preparation)
- [Neural-network applications](#neural-network-applications)
- [MaSIF-seed and MaSIF-neosurface](#masif-seed-and-masif-neosurface)
- [Rosetta relaxation and interface design](#rosetta-relaxation-and-interface-design)
- [Training and continuing a run](#training-and-continuing-a-run)
- [Convert TensorFlow weights](#convert-tensorflow-weights)
- [Run the public benchmark](#run-the-public-benchmark)
- [Common questions](#common-questions)

## Getting started

Activate your installation environment and run the examples from the repository
root. If you installed with venv, activate that environment instead.

```bash
conda activate masif_torch
masif --help
```

Prepare the input files and the checkpoint for your task. Checkpoint files are not
included in the Git checkout or wheel. Put supplied Torch weights under `weights/`
using the [weight catalog](weights/README.md), or follow
[TensorFlow weight conversion](#convert-tensorflow-weights) below.

| I want to… | Task | Starting example |
|---|---|---|
| Find protein interaction sites | `site` | [site.yaml](examples/site.yaml) |
| Compute surface fingerprints | `search` | [search.yaml](examples/search.yaml) |
| Classify a ligand-binding pocket | `ligand` | [ligand.yaml](examples/ligand.yaml) |
| Score prepared candidate interactions | `score` | [score.yaml](examples/score.yaml) |
| Score prepared alignment features | `alignment` | [inference_alignment.yaml](examples/inference_alignment.yaml) |
| Find and align candidate binders | `seed` | [seed.yaml](examples/seed.yaml) |
| Work with a protein–ligand target surface | `neosurface` | [neosurface.yaml](examples/neosurface.yaml) |
| Prepare a structure for MaSIF | `prepare` | [preparation_protein.yaml](examples/preparation_protein.yaml) |
| Refine a protein–ligand–binder complex | `rosetta` | [rosetta_1a7x.yaml](examples/rosetta_1a7x.yaml) |

### Choose your settings

For a single prediction, provide the checkpoint, input and destination directly:

```bash
masif site checkpoint=weights/site/interface_prediction_backup.pt \
  input=data/surfaces/1A7X_A.npz output=results/site/1A7X_A.npy
```

For repeated runs, copy an example YAML file and edit its paths:

```bash
mkdir -p configs
cp examples/site.yaml configs/site.yaml
# Edit configs/site.yaml to use your checkpoint, input and output paths.
masif site --config configs/site.yaml
```

Settings on the command line override those in YAML. Use `device=cuda` for a
configured NVIDIA GPU or `device=cpu` for CPU:

```bash
masif site --config configs/site.yaml device=cuda
```

Preview settings before starting a task:

```bash
masif site --config configs/site.yaml --cfg job
```

Paths are relative to the directory where you launch the command. Replace example
paths with your own files. Preparation and Rosetta examples also contain external
tool paths that must be changed for your machine.

## Notebook tutorials

If you prefer working through a notebook, choose your task below. Follow the
[opening instructions](tutorials/README.md#open-a-tutorial), then edit the input,
checkpoint and destination settings in the notebook. Run the cells in order;
set `RUN_TASK = True` when your files are ready.

| Tutorial | What you will do |
|---|---|
| [MaSIF-site](tutorials/site.ipynb) | Predict interaction sites from a prepared surface |
| [MaSIF-search](tutorials/search.ipynb) | Compute surface fingerprints |
| [MaSIF-ligand](tutorials/ligand.ipynb) | Classify a prepared binding pocket |
| [ScoreNN](tutorials/score.ipynb) | Score prepared candidate interactions |
| [Alignment](tutorials/alignment.ipynb) | Score prepared alignment features |
| [MaSIF-seed](tutorials/seed.ipynb) | Retrieve and dock candidate binders |
| [MaSIF-neosurface](tutorials/neosurface.ipynb) | Score an aligned protein–ligand target and binder |
| [Protein preparation](tutorials/prepare_protein.ipynb) | Prepare a protein PDB for MaSIF |
| [Protein–ligand preparation](tutorials/prepare_ligand.ipynb) | Prepare a ligand-containing target |
| [Rosetta](tutorials/rosetta.ipynb) | Refine a positioned target–ligand–binder complex |
| [Training](tutorials/training.ipynb) | Train, fine-tune and continue neural tasks |
| [Public benchmark](tutorials/benchmark.ipynb) | Configure the optional full dataset workflow |

The notebook commands use the environment selected by the Jupyter kernel. Choose
`Python (masif_torch)` so the notebook runs in your MaSIF installation.

## Input data and preparation

Choose the starting point you already have:

- **A PDB structure:** follow [raw structure preparation](#raw-protein-and-proteinligand-surfaces).
- **Original MaSIF surface arrays:** pack them as described below, or use
  `masif surface` directly for site and search.
- **Prepared NPZ files:** use them directly with the corresponding task.
- **Processed seed/neosurface data:** follow [prepare target and candidate files](#prepare-target-and-candidate-files).

### Original surface arrays

An original precomputation folder contains files such as `p1_input_feat.npy`,
`p1_rho_wrt_center.npy`, `p1_theta_wrt_center.npy` and `p1_mask.npy`. Site also needs
`p1_list_indices.npy`. Pack one surface for use with the YAML examples:

```bash
masif pack-surface \
  --input data/precomputation/1AKJ_AB_DE --pid p1 --trusted-indices \
  --output data/surfaces/1AKJ_AB.npz
```

Use `--pid p2` for the second partner. Only use `--trusted-indices` for files from a
source you trust. Keep the complete surface when preparing site inputs.

### Ligand records and coordinates

To reuse original MaSIF-ligand TFRecords, convert each dataset split separately:

```bash
conda activate masif_nn
python tools/convert_records.py \
  --input data/ligand/train.tfrecord --output data/ligand/train
conda activate masif_torch
```

Repeat for validation and test records with their respective destinations.
These structure packets can be used for training and dataset evaluation.
For a single `masif ligand` prediction, supply a prepared pocket packet.

If you have original surface arrays and ligand-coordinate files instead, use:

```bash
python tools/pack_ligand.py \
  --precomputed data/04-precomputation_12A --coordinates data/ligand_coords \
  --split-list data/lists/train_pdbs_sequence.npy \
  --output data/ligand/train --trusted-coordinates
```

The coordinate directory should contain `<PDB>_ligand_coords.npy` and
`<PDB>_ligand_types.npy`. Only enable `--trusted-coordinates` for trusted files.

### Raw protein and protein/ligand surfaces

Raw structure preparation requires the external geometry tools listed in the
[installation guide](README.md#optional-molecular-components).
Copy the appropriate example:

```bash
mkdir -p configs
cp examples/preparation_protein.yaml configs/prepare_protein.yaml
cp examples/preparation_ligand.yaml configs/prepare_ligand.yaml
```

Edit the settings before running:

| Setting | What to provide |
|---|---|
| `pdb` | Input PDB file |
| `identifier` | Protein and selected chains, such as `1A7X_A` |
| `output` | Destination directory |
| `source_root` | Original neosurf `masif/source` directory |
| `legacy_python` | Python executable with the original geometry dependencies |
| `environment` | Paths to MSMS, PDB2PQR, APBS and multivalue; `PATH` must include Reduce |
| `site_checkpoint`, `search_checkpoint` | Checkpoints if you also want site predictions and descriptors; otherwise set them to `null` |
| `ligand`, `sdf` | For a ligand-containing target: residue and chain, such as `FKA_B`, and its SDF template |

Use `pdb2pqr_apbs_input: flag` for PDB2PQR 2.1, or `filename` for the later command
format. Replace the example `environment.PATH` with your own tool directories and
system path.

Prepare either a protein surface or a protein–ligand surface:

```bash
masif prepare --config configs/prepare_protein.yaml
masif prepare --config configs/prepare_ligand.yaml
```

The resulting preparation folder can be used with the surface commands below.
For seed/neosurface, include site predictions and search descriptors, then pack
the processed surfaces as described in the next section.

## Neural-network applications

Each command below uses an existing checkpoint and input file. Choose the matching
example YAML from [Getting started](#getting-started) if you prefer saved settings.

### MaSIF-site

Run a prepared surface:

```bash
masif site checkpoint=weights/site/interface_prediction_backup.pt \
  input=data/surfaces/1A7X_A.npz output=results/site/1A7X_A.npy
```

Or use an original precomputation directory directly:

```bash
masif surface --checkpoint weights/site/interface_prediction_backup.pt \
  --input data/precomputation/1AKJ_AB_DE --pid p1 --trusted-indices \
  --output results/site/1AKJ_AB.npy --device cpu
```

### MaSIF-search

Run a prepared surface:

```bash
masif search checkpoint=weights/search/surface_descriptors_historical.pt \
  input=data/surfaces/1A7X_A.npz output=results/search/1A7X_A.npy
```

For seed/neosurface preparation, generate both descriptor orientations from the
original precomputation folder:

```bash
masif surface --checkpoint weights/search/surface_descriptors_historical.pt \
  --input data/precomputation/1AKJ_AB_DE --pid p1 \
  --output results/descriptors/1AKJ_AB_DE --device cpu
```

### MaSIF-ligand

Run a prepared binding pocket:

```bash
masif ligand checkpoint=weights/ligand/pocket_classification.pt \
  input=data/pockets/example.npz output=results/ligand/example.npy
```

To evaluate a directory of converted structure packets:

```bash
masif evaluate-ligand --checkpoint weights/ligand/pocket_classification.pt \
  --input data/ligand/test --output results/ligand/test --device cpu
```

### ScoreNN and alignment scoring

These tasks take prepared interaction features, rather than raw PDB files. Use
ScoreNN inputs with `score` and seed/neosurface alignment inputs with `alignment`:

```bash
masif score checkpoint=weights/score/second_stage_interface_scoring.pt \
  input=data/score/example.npz output=results/score/example.npy
masif alignment checkpoint=weights/seed/ipa_12A_surface_atom.pt \
  input=data/alignment/example.npz output=results/alignment/example.npy
```

Use the complete seed or neosurface workflow below when starting from surfaces.

## MaSIF-seed and MaSIF-neosurface

These workflows use a processed target and one or more processed candidate binders.
Install the `design` dependencies as described in the
[installation guide](README.md#optional-molecular-components).

### Prepare target and candidate files

Use processed upstream datasets, or prepare your structures with site predictions
and search descriptors. A processed folder should have this layout:

```text
processed_root/
  data_preparation/01-benchmark_pdbs/CHAIN.pdb
  data_preparation/04b-precomputation_12A/precomputation/ID/p1_list_indices.npy
  descriptors/sc05/all_feat/ID/p1_desc_straight.npy
  descriptors/sc05/all_feat/ID/p1_desc_flipped.npy
  output/all_feat_3l/pred_surfaces/CHAIN.ply
  output/all_feat_3l/pred_data/pred_CHAIN.npy
```

Pack the target and each candidate:

```bash
masif pack-design --root data/processed_target --identifier 1A7X_A \
  --role target --trusted-indices --output data/design/target.npz
masif pack-design --root data/processed_database --identifier 2GKW_B \
  --role source --trusted-indices --output data/design/source.npz
```

Targets need flipped descriptors; candidate sources need straight descriptors.
Keep structures and their surfaces in the same coordinate system. For neosurface,
prepare the target with its ligand included. Keep referenced PDB files available.

You can download a small processed example dataset before trying these commands:

```bash
python tools/fetch_design_fixtures.py --output data/neosurf_example --download
```

Pack the ligand-containing target and the example candidate from their respective
subdirectories:

```bash
masif pack-design --root data/neosurf_example/with_ligand --identifier 1A7X_A \
  --role target --trusted-indices --output data/design/target_with_ligand.npz
masif pack-design --root data/neosurf_example/decoys --identifier 2GKW_B \
  --role source --trusted-indices --output data/design/source.npz
```

Use `data/design/target_with_ligand.npz` as the target in your search configuration.
You still need the checkpoints for the workflow you choose.

### Seed retrieval and docking

Create `configs/seed.yaml` with your target, candidates and checkpoint:

```yaml
task: seed
action: search
target: data/design/target.npz
database:
  - data/design/source.npz
checkpoint: weights/seed/ipa_12A_surface_atom.pt
output: results/seed
device: cpu
seed: 42
num_sites: 1
registration:
  mode: open3d
  iterations: 2000
  radius: 1.5
  confidence: 0.999
  seed: 42
```

Run it:

```bash
masif seed --config configs/seed.yaml
```

Add more candidate NPZ paths to `database` for a larger search. Set `num_sites`
to search more target sites. To retrieve matches without docking, run:

```bash
masif seed --config configs/seed.yaml action=match
```

The repository's `examples/seed.yaml` uses saved poses (`mode: replay`). Use the
configuration above for new docking; use replay only when you have the matching
pose file.

### Neosurface aligned-pair scoring

Use this workflow when the binder is already positioned against a protein–ligand
target. Create `configs/neosurface.yaml`:

```yaml
task: neosurface
action: score
target: data/design/target_with_ligand.npz
database:
  - data/design/already_aligned_binder.npz
checkpoint: weights/neosurface/ipa_12A_surface_atom.pt
output: results/neosurface
device: cpu
seed: 42
num_sites: 1
```

```bash
masif neosurface --config configs/neosurface.yaml
```

Aligned scoring does not reposition the binder. To search and dock new candidates,
copy the seed search configuration above, change `task` to `neosurface`, and use
the neosurface checkpoint, ligand-containing target and a separate output folder.

## Rosetta relaxation and interface design

Install the optional molecular dependencies and provide the external Rosetta tools
listed in the [installation guide](README.md#optional-molecular-components).
Copy [the Rosetta example](examples/rosetta_1a7x.yaml):

```bash
cp examples/rosetta_1a7x.yaml configs/rosetta.yaml
```

Edit the following settings:

- `target`, `target_chains`: target PDB and protein chains.
- `binder`, `binder_chains`: positioned binder PDB and chains.
- `ligand`, `ligand_chain`: ligand residue name and chain.
- `template_sdf`: ligand SDF template, or `ligand_sdf` for an already prepared ligand.
- `binary`, `database`, `dalphaball`: local RosettaScripts, database and DAlphaBall paths.
- `protocol_dir`: upstream neosurf `rosetta_scripts/seed_refine/src` directory.
- `legacy_python`, `chemistry_python`: interpreters for ligand parameterization and chemistry.
- `output`: destination directory. Set `verify_repeat: false` for a single design run.

Then run:

```bash
masif rosetta --config configs/rosetta.yaml
```

For a target produced by the preparation workflow, start from
[rosetta_prepared.yaml](examples/rosetta_prepared.yaml) instead and update its paths.

## Training and continuing a run

Prepare separate training and validation directories. Use the example for your task:

| Task | Training example | Data to prepare |
|---|---|---|
| Site | [train_site.yaml](examples/train_site.yaml) | Packed surfaces with original interface labels |
| Search | [train_search.yaml](examples/train_search.yaml) | Original positive/binder/negative search cache; validation uses separate labeled packets |
| Ligand | [train_ligand.yaml](examples/train_ligand.yaml) | Converted ligand structure packets |
| ScoreNN | [train_score.yaml](examples/train_score.yaml) | Labeled ScoreNN feature packets |
| Alignment | [train_alignment.yaml](examples/train_alignment.yaml) | Labeled alignment feature packets |

For site, `pack-surface` includes interface labels when they are present in the
original precomputation folder. Repeat packing for each training/validation
structure. For search, retain the original cache files together in one directory.
For ligand, convert or pack each split using the commands above. ScoreNN and
alignment require feature datasets prepared for their respective tasks.

Start a site run:

```bash
masif train --config examples/train_site.yaml \
  dataset.input=data/site/train dataset.validation=data/site/validation \
  trainer.output=runs/site trainer.epochs=10 trainer.device=cpu
```

Select another training example for another task. Change `trainer.device` to
`cuda` to use a configured GPU. Set `trainer.learning_rate` or `dataset.batch_size`
when needed; a search batch size must be divisible by four.

To train from an existing task checkpoint, add:

```bash
masif train --config examples/train_site.yaml \
  dataset.input=data/site/train dataset.validation=data/site/validation \
  trainer.checkpoint=weights/site/interface_prediction_backup.pt \
  trainer.output=runs/site_finetune trainer.epochs=10 trainer.device=cpu
```

Continue a saved training run:

```bash
masif train --config examples/train_site.yaml \
  dataset.input=data/site/train dataset.validation=data/site/validation \
  trainer.output=runs/site trainer.checkpoint=runs/site/last.pt \
  trainer.resume=true trainer.epochs=10 trainer.device=cpu
```

This runs ten additional epochs. Keep the same task and data splits when continuing.
Use `last.pt` to continue training and `best.pt` for prediction. Set the prediction
command's `checkpoint` to the desired file.

## Convert TensorFlow weights

Skip this section if you already have Torch checkpoints. Conversion requires the
separate reader environment from the
[installation guide](README.md#read-legacy-tensorflow-weights-or-records).

For example, convert the original ligand model:

```bash
git clone https://github.com/LPDI-EPFL/masif.git ../masif
conda activate masif_nn
python tools/export_tf.py \
  --checkpoint ../masif/data/masif_ligand/nn_models/all_feat/model \
  --output artifacts/tf_exports/ligand
conda activate masif_torch
masif convert --export artifacts/tf_exports/ligand \
  --output weights/ligand/pocket_classification.pt
```

Pass the checkpoint prefix without `.index`; keep its companion files together.
For another model, change `--checkpoint`, `--export` and `--output`:

| Task | Original checkpoint under `../masif/` | Torch destination |
|---|---|---|
| Site | `data/masif_peptides/nn_models/all_feat_3l/model_data.bak/model` | `weights/site/interface_prediction_backup.pt` |
| Ligand | `data/masif_ligand/nn_models/all_feat/model` | `weights/ligand/pocket_classification.pt` |
| ScoreNN | `data/masif_pdl1_benchmark/models/nn_score/trained_model.hdf5` | `weights/score/second_stage_interface_scoring.pt` |

For the historical search model, download its source files before exporting:

```bash
python tools/fetch_reference_weights.py --output artifacts/upstream/search
conda activate masif_nn
python tools/export_tf.py --checkpoint artifacts/upstream/search/model \
  --output artifacts/tf_exports/search
conda activate masif_torch
masif convert --export artifacts/tf_exports/search \
  --output weights/search/surface_descriptors_historical.pt
```

To obtain the seed and neosurface checkpoints from their upstream repositories:

```bash
mkdir -p artifacts/upstream_repos
git clone https://github.com/LPDI-EPFL/masif_seed.git artifacts/upstream_repos/masif_seed
git clone https://github.com/LPDI-EPFL/masif-neosurf.git artifacts/upstream_repos/masif-neosurf
conda activate masif_nn
python tools/export_design_weights.py \
  --seed-repo artifacts/upstream_repos/masif_seed \
  --neosurf-repo artifacts/upstream_repos/masif-neosurf \
  --output artifacts/design_exports
conda activate masif_torch
python tools/run_validation.py --exports artifacts/design_exports \
  --output artifacts/design_checkpoints --report artifacts/design_weights.json
python tools/organize_weights.py --task seed --task neosurface
```

Use the task-specific filenames in the [weight catalog](weights/README.md) when
running the applications above.

## Run the public benchmark

This optional workflow requires the full public dataset and the upstream neosurf
checkout from the previous section. Download and extract the data:

```bash
python tools/download_benchmark.py --output data/benchmark
python tools/extract_benchmark.py --archive data/benchmark/dataset.zip \
  --output data/benchmark/dataset
```

Copy [benchmark.yaml](examples/benchmark.yaml) to `configs/benchmark.yaml`.
Set `dataset` to the extracted directory, `definition` to the upstream
`computational_benchmark/benchmark_pdbs.txt`, and choose your checkpoint,
`output` directory and number of `workers`.

```bash
masif benchmark --config configs/benchmark.yaml
```

Use `resume: true` to continue an interrupted run with the same inputs and settings.

## Common questions

- **Where do I install dependencies?** Follow [README installation](README.md#installation).
- **Can I start with a PDB?** Yes; run preparation before the surface tasks.
- **Where do I put weights?** Under the task directory in `weights/`, or provide an absolute checkpoint path.
- **Why does a file path fail?** Check it from the directory where you launched the command, including paths copied from example YAML files.
- **How do I switch between CPU and GPU?** Use `device=cpu` or `device=cuda`; training uses `trainer.device`.
- **Why does seed ask for a pose file?** The shipped seed example uses replay. Use the Open3D configuration in this guide for new docking.
- **Can I use an unaligned binder with neosurface scoring?** Use `action: search` with a registration configuration to dock it first.

For the original methods and citations, see [README references](README.md#reference)
and [citation.bib](citation.bib).
