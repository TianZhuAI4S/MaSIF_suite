# Stage 20 — Repository publication hygiene

## Purpose

Keep a source checkout suitable for GitHub publication without uploading downloaded
references, large molecular data, checkpoints, run output or machine-specific files.
Retain everything needed to install MaSIF, configure tasks, generate inputs,
convert weights, run examples and inspect the numerical comparisons.

## Intended work

Extend the standalone `.gitignore` using the current directory layout and example
output paths. Check both excluded artifacts and retained application files with
Git's actual ignore engine. Preserve the existing source architecture and model
arithmetic.

## Implemented changes

- Root data, datasets, downloads, artifacts, checkpoints and weights are ignored.
- Root preparation/search/training/Rosetta results, runs, outputs, logs, caches,
  temporary directories and backups are ignored. Hydra, TensorBoard, W&B and
  Lightning run metadata are ignored wherever generated.
- Generated neural weights, molecular structures, surface arrays and downloaded
  archives are ignored by extension. Intentional small files in `tests/fixtures`
  and `examples/fixtures` remain eligible for publication. Large fixtures should
  use the existing download workflow instead.
- Python/test/notebook caches, build output, package metadata, local virtual
  environments, editor settings, environment secrets and OS files are ignored.
  `.env.example` and `.env.template` remain eligible.
- `environments/*.lock.txt` are optional machine-specific audit snapshots. Portable
  environment YAML, `runtime-requirements.txt` and installer scripts remain included.
  The usage guide and overall report now make this distinction explicit.
- Source, packaged Hydra configuration, example YAML, tools, tests, license,
  citations, installation instructions and all numerical migration documentation
  remain eligible. Root-scoped directory rules preserve source folders with the
  same names as runtime output folders.

## Changed files

| File | Change |
|---|---|
| `.gitignore` | Grouped publication rules and fixture/template exceptions |
| `docs/USAGE.md` | Clarify portable installation recipes versus local snapshots |
| `docs/torch_migration/REPORT.md` | Update installation wording and record this stage |
| `docs/torch_migration/20_repository_hygiene.md` | Record intent, implementation and checks |
| `docs/torch_migration/numerical_repository_hygiene.json` | Retain ignore-rule verification evidence |

## Verification

Git's `check-ignore --no-index` checked the actual worktree using a separate
temporary Git repository: **88 representative generated paths are ignored and
222 required or intentionally retained paths are visible**. This includes every
existing source/config/example/installer/tool/test/documentation file outside
known caches and local environment snapshots. Fixture and environment-template
exceptions were checked explicitly. See
[the verification record](numerical_repository_hygiene.json).

The existing source-oracle numerical regression suite passed **11 tests in
4.431 seconds with zero failures, errors or skips**. Independent subagent review
found a compressed-molecular-input omission; compressed archives and Slurm job
logs were added and rechecked. No required-file exclusions remain. No model
implementation, checkpoint, sampling, optimizer, geometry or scientific acceptance
threshold was changed.

## Publication scope

No local files were deleted, and no Git index was changed. The standalone directory
has no Git metadata; validation did not initialize it. These rules apply when a
repository is initialized or new files are staged. If used in an already existing
repository, ignore rules alone do not remove previously tracked files.

Existing numerical limitations in the overall report remain unchanged. This is a
repository hygiene change, not additional certification of scientific performance.
