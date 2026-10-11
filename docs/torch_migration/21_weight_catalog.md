# Stage 21 — Task and purpose names for pretrained weights

## Purpose

Expose usable pretrained checkpoints in one `weights/` directory with names that
explain their application. Avoid requiring users to navigate historical export
layouts or guess the meaning of a `converted.pt` filename.

## Intended work

Group the four base neural models and ten IPA snapshots by application. Preserve
their bytes, configuration, tensors, optimizer metadata and provenance. Update
inference examples and user documentation; retain historical validation inputs.

## Implemented changes

- `weights/site/interface_prediction_backup.pt` predicts protein interfaces with
  the finite three-layer peptides backup.
- `weights/search/surface_descriptors_historical.pt` computes descriptors with
  the explicit historical upstream revision.
- `weights/ligand/pocket_classification.pt` classifies seven ligand pocket types.
- `weights/score/second_stage_interface_scoring.pt` supplies the six-layer ScoreNN.
- `weights/seed/` and `weights/neosurface/` each expose five IPA snapshots:
  `ipa_12A_surface.pt`, `ipa_12A_surface_atom.pt`, `ipa_9A_surface_atom.pt`,
  `ipa_12A_benchmark.pt` and `ipa_12A_benchmark_alternative.pt`.
- `weights/README.md` explains features, point counts, upstream snapshot distinctions,
  default selection and numerical limitations. `weights/manifest.yaml` records
  source paths, hashes and model configurations.
- `tools/organize_weights.py` validates finite tensors, model tasks and IPA feature
  and point counts before any copy. It copies checkpoint bytes and checks SHA256.
  It supports task selection and refuses to overwrite different destination files.
- README, detailed usage and ten YAML examples now use the descriptive paths.
  The benchmark continues to select the standard four-feature 12 Å model;
  `benchmark` filenames describe upstream training snapshots, not a changed
  computational benchmark default.
- Binary weights stay ignored. The weight catalog and YAML manifest are explicitly
  eligible for Git publication. Installed wheels do not bundle pretrained weights.

The original source artifacts are retained so previous numerical evidence remains
inspectable. Model implementation, public interfaces, builtin mandatory checkpoint
selection, optimizer mathematics, geometry and sampling are unchanged. A new
checkpoint filename does not change embedded configuration or prepare matching
surface patches automatically.

## Changed files

| Files | Change |
|---|---|
| `weights/{site,search,ligand,score,seed,neosurface}/*.pt` | Fourteen byte-identical copies with task/purpose names |
| `weights/README.md`, `weights/manifest.yaml` | Usage catalog and inspectable provenance |
| `tools/organize_weights.py` | Repeatable collection of existing validated artifacts |
| `examples/site.yaml`, `search.yaml`, `ligand.yaml`, `score.yaml`, `inference_alignment.yaml` | Named neural checkpoints |
| `examples/seed.yaml`, `neosurface.yaml`, `benchmark.yaml` | Named IPA checkpoints with existing default parameter choices |
| `examples/preparation_protein.yaml`, `preparation_ligand.yaml` | Named site/search checkpoints |
| `README.md`, `docs/USAGE.md` | User commands, catalog link and consistent regeneration flow |
| `.gitignore` | Publish catalog/manifest while keeping weights local |
| `docs/torch_migration/REPORT.md` | Overall report update |
| This document and `numerical_weight_layout.json` | Step record and verification evidence |

## Numerical verification

All **14 checkpoint files** retain exactly equal source/destination SHA256 and
complete safely loaded payloads. Strict state loading succeeds with finite tensors.
For **28 fixed-input CPU/CUDA groups**, original and named-copy inference outputs
are bitwise equal with `max_abs=0`. Original export fixtures supply inputs; these
checks compare the same Torch computation before and after reorganizing files.

The previous neosurface example selected the seed repository's standard IPA copy.
The new neosurface-named standard copy has exactly equal model state. Other IPA
variants keep separate parameters; the two benchmark-training snapshots differ
and are not equated.

All 12 weight fields in the ten updated YAML examples resolve to existing files.
The actual `masif alignment --config examples/inference_alignment.yaml` command
loads the named weight, exits successfully and saves probabilities exactly equal
to the original artifact checkpoint on the same fixture.
Actual Git ignore checks retain catalog/manifest/tool and exclude all 14 binaries.
A repeated organization run preserves weight files and manifest bytes. A mismatched
IPA point count is rejected before creating destination weights; a different
existing destination file is preserved and no manifest is written for that failed
operation.

Existing regressions pass:

- Source-oracle numerical suite: **11 tests**, zero failures or skips, 12.546 s.
- Hydra configuration/training/resume suite: **4 tests**, zero failures or skips,
  9.199 s.

Independent subagent review checked mappings, snapshot distinctions, byte equality,
finite strict loading, IPA fixed-input inference and Git visibility. Its findings
led to explicit IPA configuration guards and consistent source-artifact
regeneration instructions. See [the numerical record](numerical_weight_layout.json).

## Scientific limits

File organization does not upgrade TF/Torch tolerances to universal bitwise equality,
repair NaN checkpoints, certify historical preprocessing/paper metrics, resolve
independent IPA training differences, or complete the benchmark. Existing scientific
limits remain recorded in the overall report and weight catalog.
