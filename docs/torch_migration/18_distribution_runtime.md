# Stage 18 — MaSIF distribution, public API and CUDA runtime setup

## Purpose and intended work

Install this repository as the `masif` distribution, expose `import masif` and the
`masif` command, and make runtime selection an explicit, inspectable operation.
Preserve the canonical model classes, parameter names and checkpoint format.
Validate a regular wheel outside the checkout, including packaged Hydra resources.

## Implemented changes

- `pyproject.toml` names the distribution `masif` version `0.2.0`, declares the
  README/license, pins the validated Torch release and includes Hydra. Explicit
  package mapping preserves the flat `src` implementation and ships YAML resources.
- `src/api/` supplies the public `masif` import and module command. Model/task
  objects delegate to the existing `masif_torch` implementation, preserving object
  identity and class module names. Existing checkpoint state names do not change.
- `src/__init__.py` lazily exports models/tasks. `src/entrypoint.py` dispatches
  runtime setup using standard-library imports before application imports load
  Torch. Both `masif` and the compatibility `masif-torch` command use this launcher.
- `src/runtime.py` detects driver/GPU capabilities, selects an official CPU/CUDA
  wheel, supports a nonmutating plan, installs only when necessary, and checks
  FP32 matrix and convolution calculations in a fresh interpreter. Every visible
  Torch device is checked. No download or installation happens on normal import.
- `scripts/install.sh` bootstraps Hydra/NumPy/SciPy and the editable package, then
  invokes the same runtime selector; it supports auto/CPU/cu126/cu130. The pinned
  runtime requirements include Hydra; shell syntax is checked. No new conda
  environment or complete dependency download is claimed by that syntax check.
- Modern benchmark audit tools read YAML through `configuration.load_config_file`.
  Original TF oracle/export metadata remains JSON in its independent environment.
- `tests/test_runtime.py` covers driver/architecture rejection, CPU visibility,
  dry-run nonmutation, and avoiding reinstall of a matching runtime.
- `tools/validate_distribution.py` checks every packaged Python/YAML resource byte
  against source, installs a regular wheel into an isolated prefix, validates
  public object/state identity and all task imports, composes all presets, checks
  both module/console commands outside checkout, and runs the regression suite.

All added functions document their purpose; no model equation, feature dtype,
checkpoint conversion, optimizer update or training sampler is changed here.

## Installation

```bash
python -m pip install .
masif install-runtime --backend auto --dry-run
masif install-runtime --backend auto
masif doctor
```

For the generated regular wheel:

```bash
python -m pip install artifacts/distribution/masif-0.2.0-py3-none-any.whl
masif install-runtime --backend auto
```

The build/install operations above are local. This implementation has **not been
published to PyPI**. Public `pip install masif` becomes applicable after publication;
no upload or account ownership is claimed by these tests.

## Hardware selection and its limits

Ordinary pip cannot choose dependencies according to local GPU hardware: isolated
builds and reusable wheels are not target-hardware installation hooks. The helper
therefore runs explicitly in the active environment after package installation.
A default Torch dependency may be replaced by the selected official wheel.

For the validated Torch `2.14.1` release:

| Choice | Selection/checks |
|---|---|
| CPU | No visible NVIDIA GPUs, or explicit `--backend cpu` |
| `cu126` | Ordinary supported GPU generations; compute capability at least 5.0, driver at least 525.60.13 |
| `cu130` | Blackwell or explicit selection; compute capability at least 7.5 for every visible GPU, driver at least 580 |

A mixed Blackwell/Volta allocation is rejected before installation. Restrict the
allocation to compatible devices or use CPU. CUDA13 libraries removed support for
Maxwell, Pascal and Volta; the CUDA12.6 build still lists these architectures.
New hardware, operating systems and driver-specific features require actual
runtime verification. Kernel smoke checks are not full scientific parity tests
on an untested CUDA13 or pure CPU wheel.

The helper checks driver and GPU architecture, rather than requiring `nvcc`.
Torch wheels supply their CUDA runtime. A system toolkit version greater than
12.2 alone is insufficient: it does not prove adequate driver or architecture
support. CUDA minor-version compatibility also has feature/PTX limitations;
a passing version gate is followed by real execution checks.

Sources: [PyTorch installation](https://pytorch.org/get-started/locally/),
[NVIDIA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[CUDA13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/),
[pip build-system behavior](https://pip.pypa.io/en/stable/reference/build-system/).

## Verification and numerical scope

- Six smallest runtime regressions pass. The regular-wheel gate caught a console
  return-value bug (`sys.exit(report_dict)`); the launcher now prints the report
  and returns `None`. A regression checks the successful console exit contract.
- On the actual L40S, driver560.35.03 selects `cu126`; already installed
  `2.14.1+cu126` is retained. Exact FP32 matrix and convolution kernel checks pass.
  See [runtime evidence](numerical_runtime.json).
- [Distribution evidence](numerical_distribution.json) records the final regular
  wheel source/resource checks (47 Python files, 30 YAML resources), isolated
  installation, all 10 task imports, 12 presets and 43 passing tests without skips.
  Both console/module commands run outside checkout; runtime setup also passes
  with third-party site packages disabled. The active environment has an editable
  `masif0.2.0` install.
- [Hydra evidence](numerical_hydra_configuration.json) retains exact complete
  checkpoint equality for native fresh/resumed training through both frontends.
- [Engineering evidence](numerical_engineering_geometry.json) distinguishes exact
  before/after Torch checks from earlier TF/Torch tolerance comparisons.

No complete benchmark is required or restarted. Historical precision limitations,
nonfinite checkpoints, IPA independent-training differences, and old/modern
registration differences remain documented in the overall report.
