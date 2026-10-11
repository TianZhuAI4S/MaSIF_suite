# Stage 11 — Standalone project and task ownership

Date: 2026-10-10.

## Purpose

Make the PyTorch implementation an independent, easily installed project, with code directly under `src/` and task-specific directories. Keep the scientific mathematics, converted checkpoints, and native recovery states compatible while improving code ownership and English installation/application documentation.

## Intended work

1. Move the entire former `masif/torch` project to the sibling `masif_torch` directory.
2. Flatten `src/masif_torch` into `src` while preserving installed `import masif_torch` and CLI names.
3. Place model implementations in `src/tasks/{site,search,ligand,score,alignment}` and molecular workflows in `src/tasks/{seed,neosurface}`.
4. Separate shared numerical operations from task ownership, retaining compatibility exports.
5. Rewrite the detailed README/installation process in English with standalone paths and independent review/numerical evidence.

## Changes implemented

- Project moved to `/home/tian/git_repo/masif_torch` (physical path `/work/lpdi/users/tian/git_repo/masif_torch`). Original `masif/torch` no longer exists; legacy TF/geometry files remain in the original repository.
- `pyproject.toml` maps installed package `masif_torch` directly to `src` and explicitly lists subpackages. This supports editable and regular wheel installation without exposing bare `tasks`/`shared` packages.
- Real model classes moved to task-owned `model.py` modules. `src/models.py` preserves public exports and the existing `build_model` factory.
- Common Gaussian layers/helpers moved to `src/shared/layers.py`. Model attribute names, parameter registration order, and state keys remain intact.
- Former task registry moved into `src/tasks/__init__.py`, with lazy built-in imports and explicit custom-task duplicate-name rejection.
- Prepared-packet application code lives in `src/shared/task.py`.
- `SeedWorkflow` owns descriptor search/registration/scoring/output; `NeoSurfaceWorkflow` inherits it and owns aligned-pair scoring. Shared surface/feature/registration implementations live in `src/shared/design`.
- Examples/tool defaults use standalone paths. Saved smoke NPZ PDB paths resolve to the relocated genuine fixture files.
- `README.md` is rewritten in English with all NN/design task recipes, conversion, recovery, numerical regeneration, architecture, and explicit limitations. `INSTALL.md` is the current installation guide; `INSTALL_TORCH.md` redirects existing readers.

Original numerical JSON reports were moved intact. Earlier stage logs preserve the historical project paths; current commands are in the new README/INSTALL.

## Independent review and numerical checks

The independent reviewer read the installation log, registry, task packages, model/checkpoint code, and workflow ownership, then ran the following checks in the relocated project:

| Check | Result |
|---|---|
| Full AST body comparison for all original 11 model/helper/factory classes/functions | Exact equality before/after split |
| `python -m unittest discover -s tests -v` | 18/18 passed, no skips |
| Real converted site checkpoint | 60 reference tensors passed strict default tolerance |
| Real converted search checkpoint | 40 reference tensors passed strict default tolerance |
| Original converted ligand checkpoint | 44 reference tensors passed strict default tolerance |
| Official converted IPA checkpoint | 42 reference tensors passed strict default tolerance |
| Package and all seven task imports | Resolve to new `src` tree |
| Seven seed/neosurface task aliases | Construct their correct workflow classes |
| Smoke packet `pdb_path` | All four source/target CPU/GPU paths exist at the new location |
| Regular wheel build (`pip wheel --no-deps --no-build-isolation`) | Succeeded, 31 Python modules under `masif_torch/` |
| Import root package and all seven tasks directly from wheel in `/tmp` | All succeeded, independent of editable source resolution |

The 18-test suite includes original TF source forward/loss/all-parameter gradients, BN population-statistics arithmetic, TFAdam epsilon checks, malformed-input rejection, full next-step recovery with dropout, class weights, batching/filtering, TFRecord equality, strict geometric cutoff/tie cases, and PDB hydrogen/transform semantics. Default numerical acceptance remains `atol=1e-5, rtol=1e-4`; exact AST/weight equality does not imply every FP32 runtime output is bitwise equal.

## Review findings and fixes

The reviewer localized an installation-path bug: the standalone helper originally used `scripts/../..` and resolved the sibling parent rather than project root. The root was corrected to `scripts/..` by the implementation owner and the reviewer confirmed the current script. Historical saved smoke JSON still had old `torch/artifacts` prefixes; regeneration with current tools supplies new paths. These are path/install concerns, not changes to model mathematics.

## Limits and subsequent work

Existing numerical exceptions remain unchanged: seven original weight groups are nonfinite, historical high-scale search gradients need an explicitly recorded separate policy, and independent Keras/Torch IPA training updates fail strict tolerance. Native Torch recovery and fixed-packet forward/loss/gradient evidence remain valid after relocation.

Subsequent [stage 12](12_raw_preparation.md) records successful real protein/FKA preparation and strict neural comparisons; [stage 13](13_rosetta.md) records successful original Rosetta protocols and prepared-input integration. The full benchmark was stopped after 29 of 112 cases at the user's request; [stage 14](14_benchmark.md) records retained partial execution and completed fixed-input numerical comparisons. These application results are separate from the unchanged model mathematics established by this structural review.

## Final installation verification

The final expanded suite passes 28 tests with no skips. All 41 packaged Python sources are exactly equal to the standalone project sources; all ten task packages and four workers are included, and the Torch runtime has no TensorFlow installed. The dependency and function-documentation checks pass. Evidence: [standalone installation report](numerical_standalone_install.json). Raw protein/FKA TF/Torch audits cover eight CPU/CUDA groups and 400 reference tensors at the strict default tolerances. The original-registration 53-pose comparison also passes these tolerances; it does not assert bitwise neural equality or complete benchmark recovery.
