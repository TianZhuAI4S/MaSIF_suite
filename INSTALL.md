# Install MaSIF

These instructions install this standalone PyTorch implementation. Use Python 3.12 on Linux, the validated platform. The project directory may still be called `masif_torch`; the installed distribution and primary command are `masif`.

## Install from this checkout

A separate Python environment is recommended:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
masif install-runtime --backend auto
masif doctor
python -m pip check
```

`pip install .` installs the local `masif` distribution and its Python dependencies. This implementation has not been published to PyPI; public `pip install masif` is not currently a way to obtain it. To install a supplied wheel, replace the source-install line with:

```bash
python -m pip install /path/to/masif-0.2.0-py3-none-any.whl
```

For editable development, use `python -m pip install -e .`. The existing `masif_torch` Python API and `masif-torch` compatibility command are retained.

## Choose CPU or CUDA

The runtime installer works in the active Python environment and pins the validated Torch version, 2.14.1. Preview its choice before downloading:

```bash
masif install-runtime --backend auto --dry-run
```

Automatic selection uses the NVIDIA GPU and driver: it selects CPU when no NVIDIA GPU is accessible, the validated CUDA 12.6 wheel for supported earlier GPU generations, or CUDA 13.0 for Blackwell with a sufficiently new driver. Unsupported GPU architectures, incompatible mixed allocations, and insufficient drivers are rejected before installation; select compatible devices or explicitly choose CPU. Explicit choices are available:

```bash
masif install-runtime --backend cpu
masif install-runtime --backend cu126
masif install-runtime --backend cu130
```

The helper verifies the selected runtime with an actual import and device calculation. A prebuilt Torch wheel includes its CUDA runtime; `nvcc` and the installed system CUDA toolkit do not select the wheel. A reported CUDA version above 12.2 alone does not guarantee compatibility: GPU architecture and the driver must also support the chosen runtime.

Ordinary pip dependency resolution cannot choose a wheel from GPU hardware. Installing this package first may install pip's default Torch wheel; the explicit runtime step selects/replaces it in the same environment. A fresh pure CPU wheel environment and other operating systems are not covered by the earlier CUDA-wheel CPU audit; inspect the runtime helper's verification on your machine.

## Optional molecular applications

Surface retrieval, PLY/PDB adapters, and modern registration:

```bash
python -m pip install '.[design]'
```

Raw ligand preparation and Rosetta input chemistry:

```bash
python -m pip install '.[design,chemistry]'
```

When using an installed wheel rather than a checkout, pass the wheel path with the extra, for example `pip install '/path/to/masif-0.2.0-py3-none-any.whl[design,chemistry]'`.

Validated optional packages include Open3D 0.19.0, Biopython 1.88, plyfile 1.1.5, RDKit 2026.9.1, openbabel-wheel 3.1.1.23, ProDy 2.6.1, and scikit-learn 1.9.1. The main neural runtime remains TensorFlow-free. Native Hydra YAML configurations are installed with the package; they do not require a checkout to compose.

## Run an application

Inspect a built-in configuration without executing the workflow:

```bash
masif site --cfg job
masif train task=site --cfg job
```

Then provide your data/model paths:

```bash
masif site checkpoint=/path/to/site.pt \
  input=/path/to/surface.npz output=/path/to/site.npy
masif seed --config examples/seed.yaml
```

Use `--config /path/to/config.yaml`, or `--config-path /path/to/configs --config-name experiment`, for your own configurations. Relative data/output paths use the calling directory; Hydra does not change that directory. See the [application README](README.md) for inference, search, training, and recovery.

## Convert TensorFlow weights or records

TensorFlow is needed only in an independent reader/reference environment:

```bash
conda env create -f environments/masif_nn.yml
conda activate masif_nn
python -c "import tensorflow as tf; print(tf.__version__)"
python -m pip check
```

The validated reader uses tensorflow-cpu 2.21.0 on Python 3.12. Run the original export/conversion tools there, then perform NPZ-to-Torch conversion and inference in your Torch environment. Saved TF1 graphs use `compat.v1`; supplied adapters handle original source APIs. Original nonfinite checkpoint families are rejected. Finite alternative models retain explicit provenance.

[The detailed guide](USAGE.md#convert-tensorflow-weights) contains the complete commands and numerical validation recipes. Existing `environments/masif_torch.yml` and `scripts/install.sh` remain compatibility environment recipes; the primary installation above uses pip and `masif install-runtime`.

## Raw geometry and Rosetta prerequisites

Prepared neural/design inputs can be used entirely in the modern environment. Raw preparation additionally needs Reduce, MSMS, PDB2PQR, APBS, multivalue, and the original PyMesh/geodesic routines. The validated bridge reused an isolated Python 3.6.5/PyMesh 0.2.1 interpreter with NumPy 1.19.4, SciPy 1.1.0, scikit-learn 0.24.0, Biopython 1.73, NetworkX 2.3, and Open3D 0.5.0.0. A fresh installation of every historical tool is not claimed.

Use the pinned neosurf Dockerfile as an upstream restoration reference, or configure an existing compatible interpreter. In your preparation YAML, set `legacy_python`, `source_root`, input/ligand paths, and absolute external-tool paths. Put Reduce on the subprocess `environment.PATH`. PDB2PQR 2.1 needs `pdb2pqr_apbs_input: flag`; the later upstream dialect uses `filename`. [Raw preparation details](USAGE.md#raw-protein-and-proteinligand-surfaces) describe the tested bridge.

Rosetta executables, database, and DAlphaBall are separately provided dependencies. Configure these, original protocol files, partner/ligand inputs, and parameterization interpreters in your Rosetta YAML. Python package installation does not install Rosetta. See the [Rosetta workflow](USAGE.md#rosetta-relaxation-and-interface-design).

## Verify the installation

```bash
masif doctor
python -m pip check
masif site --cfg job
```

These commands check the environment and configuration without running a full
benchmark. Validate a newly exported checkpoint against its fixed TensorFlow
reference with `masif validate`; conversion commands are in the
[weight guide](USAGE.md#convert-tensorflow-weights).

The [numerical contract](USAGE.md#methods-and-numerical-contract) distinguishes
exact weight conversion from FP32 TF/Torch inference tolerance. Independent IPA
training retains a strict parameter-update failure, and modern Open3D poses can
differ from historical registration. Full paper benchmark performance is not
certified by installation checks.
