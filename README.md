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

Linux with Python 3.12 is the validated platform. The runtime helper pins Torch
2.14.1 and supports CPU or a compatible NVIDIA GPU. Prepared neural inputs do not
require TensorFlow or the original geometry tools.

### Install MaSIF

Clone the repository and create a separate environment:

```bash
git clone https://github.com/TianZhuAI4S/MaSIF_torch.git
cd MaSIF_torch
conda env create -f environments/masif_torch.yml
conda activate masif_torch
python -m pip install .
masif install-runtime --backend auto
masif doctor
python -m pip check
```

Without Conda, replace the two environment commands with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Then run the same pip/runtime/check commands. The installed distribution and
primary command are `masif`; `masif_torch` and `masif-torch` compatibility names
remain available. This implementation has not been published to PyPI:
`pip install masif` from the public index does not install this repository.
For a supplied wheel, replace `pip install .` with
`python -m pip install /path/to/masif-0.2.0-py3-none-any.whl`.
Use `python -m pip install -e .` for editable development.

### Choose CPU or CUDA

Preview automatic selection without installing anything:

```bash
masif install-runtime --backend auto --dry-run
```

Automatic selection uses the visible GPU architecture and NVIDIA driver. It selects
CPU when no GPU is accessible, CUDA 12.6 for compatible earlier GPU generations,
or CUDA 13.0 for Blackwell with a sufficiently new driver. Unsupported or mixed
incompatible GPUs and insufficient drivers produce an error. Explicit choices:

```bash
masif install-runtime --backend cpu
masif install-runtime --backend cu126
masif install-runtime --backend cu130
```

Choose one backend. CUDA 12.6 requires driver 525.60.13 or newer; CUDA 13.0 requires
580 or newer. GPU architecture must also be supported. A CUDA toolkit version above
12.2 alone does not establish compatibility; prebuilt Torch wheels include the
CUDA runtime and do not require a system toolkit. The helper checks actual device
calculations after selecting/installing the wheel.

Ordinary pip cannot select dependencies from GPU hardware, so `pip install .` may
first install its default Torch wheel; the runtime step selects/replaces that
wheel in the same environment. The validated numerical environment uses
Torch 2.14.1+cu126, including its CPU backend; other runtime combinations require
the helper's machine-specific verification.

The optional Conda bootstrap combines environment creation and runtime setup:

```bash
bash scripts/install.sh auto /path/to/new/masif_torch
```

The prefix must be new; `cpu`, `cu126` and `cu130` can replace `auto`.

### Optional molecular components

Surface retrieval, PLY/PDB adapters and registration use the `design` extra.
Ligand preparation and Rosetta input chemistry also use `chemistry`:

```bash
python -m pip install '.[design,chemistry]'
```

Use `'.[design]'` if only retrieval/registration is needed. For a supplied wheel,
use `python -m pip install '/path/to/masif-0.2.0-py3-none-any.whl[design,chemistry]'`.
The pinned optional versions are declared in `pyproject.toml`.

Raw surface generation additionally needs Reduce, MSMS, PDB2PQR, APBS, multivalue
and a compatible original PyMesh/geodesic interpreter. Set `legacy_python`,
`source_root` and external binary paths in the preparation YAML. The validated
geometry bridge uses an isolated Python 3.6.5/PyMesh 0.2.1 environment; its tools
are external to the modern Torch environment. PDB2PQR 2.1 uses
`pdb2pqr_apbs_input: flag`; the later dialect uses `filename`.
See [raw preparation](USAGE.md#raw-protein-and-proteinligand-surfaces).

Rosetta requires separately provided RosettaScripts, its database, DAlphaBall and
original protocol files. Configure these and partner/ligand inputs in the Rosetta
YAML. Python package installation does not install Rosetta.
See [the Rosetta workflow](USAGE.md#rosetta-relaxation-and-interface-design).

### Read legacy TensorFlow weights or records

Only weight/TFRecord export needs an independent TensorFlow environment:

```bash
conda env create -f environments/masif_nn.yml
conda activate masif_nn
python -c "import tensorflow as tf; print(tf.__version__)"
python -m pip check
conda activate masif_torch
```

The reader uses Python 3.12 and tensorflow-cpu 2.21.0. Saved TF1 graphs run through
`compat.v1`; the export adapters handle original APIs. The final command returns
Conda users to Torch for conversion and inference. If using venv instead,
deactivate the reader environment and run `source .venv/bin/activate` from the
repository root. The
[weight guide](USAGE.md#convert-tensorflow-weights) and
[data conversion guide](USAGE.md#ligand-records-and-coordinates) contain the commands.

### Verify and start

```bash
masif --help
masif doctor
python -m pip check
masif site --cfg job
```

These commands inspect installation/configuration. Then follow the application
examples below with your own weights and input data. Installation checks do not
certify full benchmark performance or eliminate the numerical limits below.

## Data preparation and pretrained models

Use the original training/validation/test splits for your application. Protein identifiers such as `1A7X_A` select chain A; paired identifiers describe two interacting domains. Preserve coordinates in Å and use geodesic patch coordinates from preparation.

There are two starting points:

- **Prepared surfaces or pockets:** use the original precomputed arrays, descriptors, or converted numeric NPZ packets. [Data formats and conversion](USAGE.md#input-data-and-preparation) describe how to create them.
- **Raw protein or protein/ligand structures:** run the preparation workflow below to generate surfaces, chemical features, and 9/12 Å patches.

Original TensorFlow/Keras checkpoints can be converted to native Torch checkpoints. Conversion and pretrained-model provenance are documented in the [weight guide](USAGE.md#convert-tensorflow-weights). Ready-to-use weights are grouped by task under `weights/`, with filenames describing their purpose. See the [weight catalog](weights/README.md) for variants, provenance and known limits. Provide the appropriate task checkpoint in the commands below; ScoreNN and alignment checkpoints represent different models.

## Tutorials

Start with a notebook for your task. Each tutorial introduces the inputs, lets you
edit settings, and previews the command before you run it with your own data.

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

See [opening and running notebooks](tutorials/README.md#open-a-tutorial) for Jupyter
setup. Choose the `Python (masif_torch)` kernel. Provide the task's input files and
checkpoint, then set `RUN_TASK = True` to start the configured calculation.

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

The output contains one probability per surface vertex. For prepared-surface usage, see the [site workflow](USAGE.md#masif-site).

### MaSIF-search

Compute surface fingerprints for the orientation stored in a prepared packet:

```bash
masif search checkpoint=weights/search/surface_descriptors_historical.pt \
  input=data/surfaces/1A7X_A.npz output=results/descriptors/1A7X_A.npy
```

Use these descriptors for complementary-patch retrieval and subsequent alignment. The detailed guide also shows `masif surface` for producing both straight/flipped orientations from original precomputation directories. See the [search guide](USAGE.md#masif-search) and [training instructions](USAGE.md#training-and-continuing-a-run).

### MaSIF-ligand

Classify a prepared binding-pocket packet:

```bash
masif ligand checkpoint=weights/ligand/pocket_classification.pt \
  input=data/pockets/example.npz output=results/ligand/example.npy
```

Pocket generation and original TFRecord conversion are covered in the [ligand guide](USAGE.md#masif-ligand).

### Second-stage ScoreNN and alignment scoring

Evaluate prepared interaction features using the corresponding network:

```bash
masif score checkpoint=weights/score/second_stage_interface_scoring.pt \
  input=data/score/example.npz output=results/score/example.npy
masif alignment checkpoint=weights/seed/ipa_12A_surface_atom.pt \
  input=data/alignment/example.npz output=results/alignment/example.npy
```

ScoreNN scores the original second-stage search features. The alignment network supplies the IPA scores used by seed and neosurface workflows. See the [neural workflow guide](USAGE.md#neural-network-applications) for commands.

### MaSIF-seed

Retrieve complementary surface patches, align candidate binders, score their interfaces, and filter clashes:

```bash
masif seed --config examples/seed.yaml
```

Configure a target surface, candidate surface database, and alignment checkpoint. Results include candidate diagnostics and accepted aligned structures. [Seed usage](USAGE.md#masif-seed-and-masif-neosurface) describes the public fixtures and input orientations.

### MaSIF-neosurface

Score a previously aligned protein/ligand complex or search candidate partners:

```bash
masif neosurface --config examples/neosurface.yaml
```

The configuration chooses `action=score` or `action=search`; aligned scoring requires the target and partner in a common coordinate frame. See the [neosurface guide](USAGE.md#neosurface-aligned-pair-scoring).

### Raw protein and protein/ligand preparation

Generate surface geometry, chemical features, and patches:

```bash
masif prepare --config examples/preparation_protein.yaml
masif prepare --config examples/preparation_ligand.yaml
```

Configure the original geometry interpreter, Reduce/MSMS/electrostatics tools, selected chains, and optional ligand template. A site/search checkpoint can also generate predictions on the prepared surface. [Preparation details](USAGE.md#raw-protein-and-proteinligand-surfaces) explain the external dependencies.

### Rosetta relaxation and interface design

Refine an assembled protein/ligand/binder complex using the original protocols:

```bash
masif rosetta --config examples/rosetta_1a7x.yaml
```

Supply RosettaScripts, its database, DAlphaBall, protocol files, and partner/ligand inputs. The workflow runs relaxation and interface design; optional same-seed repetition checks output reproducibility. See the [Rosetta guide](USAGE.md#rosetta-relaxation-and-interface-design).

## Train a model

Hydra groups select the task model, dataset, and trainer. Prepare separate training and validation NPZ directories, then train:

```bash
masif train task=site dataset.input=data/site/train \
  dataset.validation=data/site/validation trainer.output=runs/site \
  trainer.epochs=10 trainer.device=cuda
```

Use `task=search`, `ligand`, `score`, or `alignment` for another neural task. Their dataset formats differ; [training data and recovery](USAGE.md#training-and-continuing-a-run) provide the details. Override model/trainer settings with Hydra keys rather than editing source code.

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

Some original checkpoints contain NaN and are rejected. Independent IPA training parameter trajectories retain a strict numerical failure. Modern Open3D docking differs from historical random registration. The full benchmark was stopped at the user's request after 29/112 searches; fixed-input comparisons were completed. The [weight catalog](weights/README.md) explains model-specific limits.

## Documentation

- [Installation](#installation)
- [User guide](USAGE.md)
- [Task notebooks](tutorials/README.md)
- [License](LICENSE) and [BibTeX citation](citation.bib)

## Reference

If you use MaSIF in your research, please cite the paper for the application you
use. Copy the relevant BibTeX entry below, or download [citation.bib](citation.bib).

#### MaSIF-site, MaSIF-search, MaSIF-ligand and ScoreNN

[Paper](https://doi.org/10.1038/s41592-019-0666-6)

```bibtex
@article{gainza2020deciphering,
  title = {Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning},
  author = {Gainza, P. and Sverrisson, F. and Monti, F. and Rodolà, E. and Boscaini, D. and Bronstein, M. M. and Correia, B. E.},
  journal = {Nature Methods},
  volume = {17},
  number = {2},
  year = {2020},
  doi = {10.1038/s41592-019-0666-6},
  url = {https://doi.org/10.1038/s41592-019-0666-6},
  pages = {184--192}
}
```

#### MaSIF-seed and protein binder design

[Paper](https://doi.org/10.1038/s41586-023-05993-x)

```bibtex
@article{gainza2023design,
  title = {De novo design of protein interactions with learned surface fingerprints},
  author = {Gainza, Pablo and Wehrle, Sarah and Van Hall-Beauvais, Alexandra and Marchand, Anthony and Scheck, Andreas and Harteveld, Zander and Buckley, Stephen and Ni, Dongchun and Tan, Shuguang and Sverrisson, Freyr and Goverde, Casper and Turelli, Priscilla and Raclot, Charlène and Teslenko, Alexandra and Pacesa, Martin and Rosset, Stéphane and Georgeon, Sandrine and Marsden, Jane and Petruzzella, Aaron and Liu, Kefang and Xu, Zepeng and Chai, Yan and Han, Pu and Gao, George F. and Oricchio, Elisa and Fierz, Beat and Trono, Didier and Stahlberg, Henning and Bronstein, Michael and Correia, Bruno E.},
  journal = {Nature},
  volume = {617},
  number = {7959},
  year = {2023},
  doi = {10.1038/s41586-023-05993-x},
  url = {https://doi.org/10.1038/s41586-023-05993-x},
  pages = {176--184}
}
```

#### MaSIF-neosurface

[Paper](https://doi.org/10.1038/s41586-024-08435-4)

```bibtex
@article{marchand2025neosurfaces,
  title = {Targeting protein--ligand neosurfaces with a generalizable deep learning tool},
  author = {Marchand, Anthony and Buckley, Stephen and Schneuing, Arne and Pacesa, Martin and Elia, Maddalena and Gainza, Pablo and Elizarova, Evgenia and Neeser, Rebecca M. and Lee, Pao-Wan and Reymond, Luc and Miao, Yangyang and Scheller, Leo and Georgeon, Sandrine and Schmidt, Joseph and Schwaller, Philippe and Maerkl, Sebastian J. and Bronstein, Michael and Correia, Bruno E.},
  journal = {Nature},
  volume = {639},
  number = {8054},
  year = {2025},
  doi = {10.1038/s41586-024-08435-4},
  url = {https://doi.org/10.1038/s41586-024-08435-4},
  pages = {522--531}
}
```

### Related MaSIF methods and applications

The papers below provide further reading. MaSIF-mimicry, dMaSIF,
physics-informed docking, DiffMaSIF and SURFACE-Bind are separate methods or
applications; their workflows are not included in this repository.

#### MaSIF-mimicry and the CRBN interactome

[Paper](https://www.nature.com/articles/s41587-026-03237-7)

```bibtex
@article{galli2026crbn,
  title = {Proteome-wide identification of the druggable {CRBN} interactome},
  author = {Galli, Pius and Xiao, Shuhao and Meng, Yanxiang and Hanzl, Alexander and Bendel, Alexandra M. and Baur, Regina and Aguirre, Jacob D. and Diaz-Rovira, Anna M. and Stammnitz, Maximilian R. and Kempf, Georg and Kater, Lukas and Shimada, Kenji and Wei, Ye and Scheck, Andreas and Klein, Dominique and Cavadini, Simone and Lehner, Ben and Diss, Guillaume and Correia, Bruno E. and Thomä, Nicolas H.},
  journal = {Nature Biotechnology},
  year = {2026},
  doi = {10.1038/s41587-026-03237-7},
  url = {https://www.nature.com/articles/s41587-026-03237-7}
}
```

#### dMaSIF

[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.html)

```bibtex
@inproceedings{sverrisson2021surfaces,
  title = {Fast End-to-End Learning on Protein Surfaces},
  author = {Sverrisson, Freyr and Feydy, Jean and Correia, Bruno E. and Bronstein, Michael M.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {15272--15281},
  year = {2021},
  url = {https://openaccess.thecvf.com/content/CVPR2021/html/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.html}
}
```

#### Physics-informed docking with dMaSIF

[Paper](https://openreview.net/forum?id=5yn5shS6wN)

```bibtex
@inproceedings{sverrisson2022physics,
  title = {Physics-informed deep neural network for rigid-body protein docking},
  author = {Sverrisson, Freyr and Feydy, Jean and Southern, Joshua and Bronstein, Michael M. and Correia, Bruno},
  booktitle = {Machine Learning for Drug Discovery Workshop at ICLR (MLDD)},
  year = {2022},
  url = {https://openreview.net/forum?id=5yn5shS6wN}
}
```

#### DiffMaSIF

[Paper](https://www.mlsb.io/papers_2023/DiffMaSIF_Surface-based_Protein-Protein_Docking_with_Diffusion_Models.pdf)

```bibtex
@inproceedings{sverrisson2023diffmasif,
  title = {{DiffMaSIF}: Surface-based Protein-Protein Docking with Diffusion Models},
  author = {Sverrisson, Freyr and Akdel, Mehmet and Abramson, Dylan and Feydy, Jean and Goncearenco, Alexander and Adeshina, Yusuf and Kovtun, Daniel and Marquet, C{\'e}line and Zhang, Xuejin and Baugher, David and Carpenter, Zachary and Naef, Luca and Bronstein, Michael M. and Correia, Bruno},
  booktitle = {Machine Learning for Structural Biology Workshop at NeurIPS (MLSB)},
  year = {2023},
  url = {https://www.mlsb.io/papers_2023/DiffMaSIF_Surface-based_Protein-Protein_Docking_with_Diffusion_Models.pdf}
}
```

#### Human surfaceome and SURFACE-Bind

[Paper](https://doi.org/10.1073/pnas.2506269123)

```bibtex
@article{balbi2026surfaceome,
  title = {Mapping targetable sites on the human surfaceome for the design of novel binders},
  author = {Balbi, Petra E. M. and Sadek, Ahmed and Marchand, Anthony and Yu, Ta-Yi and Damjanovic, Jovan and Georgeon, Sandrine and Schmidt, Joseph and Fulle, Simone and Yang, Che and Khakzad, Hamed and Correia, Bruno E.},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {123},
  number = {5},
  year = {2026},
  doi = {10.1073/pnas.2506269123},
  url = {https://doi.org/10.1073/pnas.2506269123},
  pages = {e2506269123}
}
```

This implementation retains the upstream [Apache 2.0 license](LICENSE).
