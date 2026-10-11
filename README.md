![MaSIF concept](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/img/Concept-01.png)

# MaSIF: learning molecular surface interactions with PyTorch

MaSIF (Molecular Surface Interaction Fingerprints) learns how local protein surface geometry and chemistry relate to molecular recognition. It describes surface neighborhoods with geometric neural networks and uses these fingerprints to predict interaction sites, compare complementary surfaces, and classify ligand-binding pockets.

This standalone PyTorch implementation provides the original MaSIF-site, MaSIF-search, and MaSIF-ligand applications, together with ScoreNN, MaSIF-seed, and MaSIF-neosurface workflows. The scientific approach follows the [official MaSIF project](https://github.com/LPDI-EPFL/masif); the application commands below use a modern PyTorch environment and Hydra configuration.

## What changed from the original implementation

- One PyTorch environment runs site, search, ligand, ScoreNN, and the seed/neosurface networks.
- Finite TensorFlow/Keras weights can be converted to native Torch checkpoints with numerical checks.
- Hydra YAML configurations select application settings, model/dataset groups, training, and recovery.
- A single `masif` command provides prediction, molecular workflows, and training.

## Method overview

Starting from a protein structure, MaSIF builds a solvent-excluded surface and assigns geometric and chemical features to its vertices. Overlapping neighborhoods are extracted using geodesic radii of 9 or 12 Å. A surface convolution combines patch coordinates with shape, hydrogen-bonding, electrostatic, and hydrophobic features to learn local fingerprints. Task-specific objectives turn these fingerprints into predictions or complementary-surface matches.

![MaSIF method](https://raw.githubusercontent.com/LPDI-EPFL/masif/master/img/Method-01.png)

| Application | Scientific question | Output |
|---|---|---|
| MaSIF-site | Where might another protein bind? | Interaction probability at each surface vertex |
| MaSIF-search | Which surface patches are complementary? | Straight/flipped fingerprints for surface retrieval |
| MaSIF-ligand | Which ligand class fits a binding pocket? | Pocket class probabilities |
| ScoreNN | Which candidate protein interactions are plausible? | Second-stage interaction scores |
| MaSIF-seed | Which candidate binders match a target surface? | Ranked, aligned candidates and interaction scores |
| MaSIF-neosurface | How does a protein/ligand surface match a binding partner? | Candidate or aligned-complex scores |

## Installation

Python 3.12 and Linux are the validated platform. A GPU is useful for neural inference and training; CPU execution is also supported. From this checkout:

```bash
python -m pip install .
masif install-runtime --backend auto
masif --help
masif doctor
```

The installed distribution and command are named `masif`. This checkout has not been published to PyPI: `pip install masif` from the public index is not an installation of this implementation. For an externally supplied wheel, use `pip install /path/to/masif-*.whl`. See [INSTALL.md](INSTALL.md) for CUDA/CPU selection, optional molecular tools, and environment recipes. The runtime installer chooses a wheel from the detected GPU/driver and verifies it in the current Python environment; a local CUDA toolkit is not required.

Surface retrieval and molecular preparation use optional dependencies:

```bash
python -m pip install '.[design,chemistry]'
```

Neural execution on prepared data does not require TensorFlow. Raw surface generation additionally requires the original geometry tools; Rosetta requires separately provided executables and database. Their setup is described in [INSTALL.md](INSTALL.md).

## Data preparation and pretrained models

Use the original training/validation/test splits for your application. Protein identifiers such as `1A7X_A` select chain A; paired identifiers describe two interacting domains. Preserve coordinates in Å and use geodesic patch coordinates from preparation.

There are two starting points:

- **Prepared surfaces or pockets:** use the original precomputed arrays, descriptors, or converted numeric NPZ packets. [Data formats and conversion](docs/USAGE.md#input-data-and-preparation) describe how to create them.
- **Raw protein or protein/ligand structures:** run the preparation workflow below to generate surfaces, chemical features, and 9/12 Å patches.

Original TensorFlow/Keras checkpoints can be converted to native Torch checkpoints. Conversion and pretrained-model provenance are documented in the [weight guide](docs/USAGE.md#convert-tensorflow-weights). Ready-to-use weights are grouped by task under `weights/`, with filenames describing their purpose. See the [weight catalog](weights/README.md) for variants, provenance and known limits. Provide the appropriate task checkpoint in the commands below; ScoreNN and alignment checkpoints represent different models.

## Applications

All applications use `masif <application> key=value`. Paths resolve from the directory where the command is launched. Inspect the effective settings without running a task:

```bash
masif site --cfg job
```

You can select your own YAML file with `--config /path/to/config.yaml`, or use `--config-path /path/to/configs --config-name experiment`. Examples below use illustrative data/checkpoint paths; replace them with your files.

### MaSIF-site

Predict protein interaction sites from a prepared surface packet:

```bash
masif site checkpoint=weights/site/interface_prediction_backup.pt \
  input=data/surfaces/1A7X_A.npz output=results/site/1A7X_A.npy
```

The output contains one probability per surface vertex. For PLY visualization and dataset preparation, see the [detailed site workflow](docs/USAGE.md#masif-site).

### MaSIF-search

Compute surface fingerprints for the orientation stored in a prepared packet:

```bash
masif search checkpoint=weights/search/surface_descriptors_historical.pt \
  input=data/surfaces/1A7X_A.npz output=results/descriptors/1A7X_A.npy
```

Use these descriptors for complementary-patch retrieval and subsequent alignment. The detailed guide also shows `masif surface` for producing both straight/flipped orientations from original precomputation directories. The [search guide](docs/USAGE.md#masif-search) includes the original cache format and training inputs.

### MaSIF-ligand

Classify a prepared binding-pocket packet:

```bash
masif ligand checkpoint=weights/ligand/pocket_classification.pt \
  input=data/pockets/example.npz output=results/ligand/example.npy
```

Pocket generation and original TFRecord conversion are covered in the [ligand guide](docs/USAGE.md#masif-ligand).

### Second-stage ScoreNN and alignment scoring

Evaluate prepared interaction features using the corresponding network:

```bash
masif score checkpoint=weights/score/second_stage_interface_scoring.pt \
  input=data/score/example.npz output=results/score/example.npy
masif alignment checkpoint=weights/seed/ipa_12A_surface_atom.pt \
  input=data/alignment/example.npz output=results/alignment/example.npy
```

ScoreNN scores the original second-stage search features. The alignment network supplies the IPA scores used by seed and neosurface workflows. See the [neural workflow guide](docs/USAGE.md#neural-network-applications) for their input formats.

### MaSIF-seed

Retrieve complementary surface patches, align candidate binders, score their interfaces, and filter clashes:

```bash
masif seed --config examples/seed.yaml
```

Configure a target surface, candidate surface database, and alignment checkpoint. Results include candidate diagnostics and accepted aligned structures. [Seed usage](docs/USAGE.md#masif-seed-and-masif-neosurface) describes the public fixtures and input orientations.

### MaSIF-neosurface

Score a previously aligned protein/ligand complex or search candidate partners:

```bash
masif neosurface --config examples/neosurface.yaml
```

The configuration chooses `action=score` or `action=search`; aligned scoring requires the target and partner in a common coordinate frame. See the [neosurface guide](docs/USAGE.md#neosurface-aligned-pair-scoring).

### Raw protein and protein/ligand preparation

Generate surface geometry, chemical features, and patches:

```bash
masif prepare --config examples/preparation_protein.yaml
masif prepare --config examples/preparation_ligand.yaml
```

Configure the original geometry interpreter, Reduce/MSMS/electrostatics tools, selected chains, and optional ligand template. A site/search checkpoint can also generate predictions on the prepared surface. [Preparation details](docs/USAGE.md#raw-protein-and-proteinligand-surfaces) explain the external dependencies.

### Rosetta relaxation and interface design

Refine an assembled protein/ligand/binder complex using the original protocols:

```bash
masif rosetta --config examples/rosetta_1a7x.yaml
```

Supply RosettaScripts, its database, DAlphaBall, protocol files, and partner/ligand inputs. The workflow runs relaxation and interface design; optional same-seed repetition checks output reproducibility. See the [Rosetta guide](docs/USAGE.md#rosetta-relaxation-and-interface-design).

## Train a model

Hydra groups select the task model, dataset, and trainer. Prepare separate training and validation NPZ directories, then train:

```bash
masif train task=site dataset.input=data/site/train \
  dataset.validation=data/site/validation trainer.output=runs/site \
  trainer.epochs=10 trainer.device=cuda
```

Use `task=search`, `ligand`, `score`, or `alignment` for another neural task. Their dataset formats differ; [training data and recovery](docs/USAGE.md#save-resume-python-api) provide the details. Override model/trainer settings with Hydra keys rather than editing source code.

Continue native Torch training from a saved state:

```bash
masif train task=site dataset.input=data/site/train \
  dataset.validation=data/site/validation trainer.output=runs/site \
  trainer.checkpoint=runs/site/last.pt trainer.resume=true \
  trainer.epochs=10 trainer.device=cuda
```

The epoch count requests additional epochs. A converted legacy checkpoint supplies model weights; it does not reconstruct an unavailable historical training state.

## Validation and scientific scope

Finite pretrained weights convert reversibly. Fixed-input TF/Torch forward, loss, and gradient comparisons use `atol=1e-5, rtol=1e-4`; cross-framework arithmetic is not generally bitwise equal. The evidence includes eight CPU/GPU groups and 400 tensors on real newly prepared protein/FKA surfaces, and 53 real poses scored with the original Keras model and Torch.

Some original checkpoints contain NaN and are rejected. Independent IPA training parameter trajectories retain a strict numerical failure. Modern Open3D docking differs from historical random registration. The full benchmark was stopped at the user's request after 29/112 searches; fixed-input comparisons were completed. These limits and every test are recorded in the [scientific migration report](docs/torch_migration/REPORT.md) and [detailed numerical guide](docs/USAGE.md#reproduce-numerical-validation).

## Documentation

- [Installation](INSTALL.md)
- [Detailed usage, weight conversion, and numerical reruns](docs/USAGE.md)
- [Implementation stages and overall report](docs/torch_migration/REPORT.md)
- [License](LICENSE) and [BibTeX citation](citation.bib)

## Reference

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

Please cite the original methods for the applications you use. Full author lists and BibTeX entries for all seven papers are available in [citation.bib](citation.bib). This implementation retains the upstream Apache 2.0 license.
