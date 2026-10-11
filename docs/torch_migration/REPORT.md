# MaSIF Torch migration and standalone integration report

## Objective and delivered architecture

Provide a usable PyTorch MaSIF implementation in an isolated modern environment,
read the original trained weights through lossless checkpoint conversion, and
retain inspectable numerical evidence for each implementation stage.

The project is now `/home/tian/git_repo/masif_torch`, physically
`/work/lpdi/users/tian/git_repo/masif_torch`. The former `masif/torch` directory was
moved in full, including its converted checkpoints and numerical artifacts.
The original TensorFlow repository remains a separate checkout at `../masif`.
All implementation code lives directly under `src`; there is no nested
`src/masif_torch` directory. The installed package name remains `masif_torch`.

| Task directory | Responsibility |
|---|---|
| `src/tasks/site` | Actual site model and prepared-surface prediction |
| `src/tasks/search` | Actual search model, descriptor orientations and loss |
| `src/tasks/ligand` | Actual ligand pocket classifier |
| `src/tasks/score` | Actual six-layer ScoreNN |
| `src/tasks/alignment` | Actual five-layer IPA classifier |
| `src/tasks/seed` | Retrieval, docking, scoring, clashes and candidate output |
| `src/tasks/neosurface` | Inherited seed search and aligned-complex scoring |
| `src/tasks/preparation` | Raw PDB/ligand geometry, chemistry, patches and prediction |
| `src/tasks/rosetta` | Original external Rosetta relaxation and interface design |
| `src/tasks/benchmark` | Full public neosurf benchmark and legacy-registration comparison |

Shared numerical layers live in `src/shared/layers.py`; shared molecular data,
feature and registration implementations live in `src/shared/design`. Task
registration is lazy, and user factories can extend the same `run()` contract.
The model split preserves checkpoint state names and the original computation
AST. Worker implementations ship with regular wheels as well as editable installs.
No new neural entities were introduced solely to represent a task name.

## Installation and usage

[README](../../README.md) and [INSTALL](../../INSTALL.md) are detailed English
standalone documents. They cover all neural and molecular tasks, conversion,
training/recovery, raw preparation, Rosetta, benchmark data and numerical reruns.

The validated target uses Python3.12.15, Torch2.14.1+cu126, NumPy2.5.3 and
SciPy1.18.1, with optional Open3D0.19.0 and modern chemical packages. Torch runs
without TensorFlow installed. TensorFlow2.21 in `masif_nn` is an independent
checkpoint/oracle environment. Portable installation recipes are under
`environments`; optional machine-specific `*.lock.txt` snapshots remain local.

Prepared neural/design data can be used entirely in the modern environment.
Raw MSMS/PyMesh/geodesic geometry is explicitly delegated to a compatible external
interpreter, preserving original numerical dependencies. The original Rosetta
binary, database and DAlphaBall remain external configured tools. Their actual
paths are recorded in examples and resolved run reports; a fresh build of every
historical dependency on another machine is not claimed.

The standalone installer now resolves the project root correctly. License and
citation files have been copied to the standalone root.

## Numerical results and their limits

| Comparison | Result and scope |
|---|---|
| Converted finite TensorFlow/Keras weights | Exact reversible tensor conversion, including layout transpositions |
| Model/task folder split | Original 11 class/helper/factory bodies unchanged by AST comparison |
| Original direct protein preparation versus bridge | All 10 vertex properties and the face array exactly equal (11 checks) |
| Original site/search precomputation versus bridge | All 18 arrays exactly equal, including polar coordinates and ragged indices |
| Prediction-colored PLY surfaces | Original PyMesh coordinates, faces and all unchanged attributes exactly equal for protein and FKA inputs |
| New raw protein/FKA surfaces, independent TF/Torch | All 8 CPU/CUDA groups and 400 reference tensors pass `atol=1e-5, rtol=1e-4` |
| Original Rosetta XML, repeated seed42 design | Exported `3466×3` coordinates and all 36 scores exactly equal |
| Full old Open3D0.5 1A7X benchmark case | All 30,810 candidates processed; correct partner rank1 |
| Original Keras versus Torch on 53 accepted old poses | CPU/CUDA strict FP32 comparisons pass |
| New versus old Open3D docking | Different accepted candidates and poses; both recover the tested partner at rank1 |
| Independent IPA training trajectories | Some pre-BN biases fail strict tolerance after updates; failure retained |

Raw neural probability differences for site are at most about `3e-7`; the tested
search descriptor differences are at most about `1.67e-6`. Forward, loss,
intermediate-layer and every parameter-gradient comparisons are retained.
These strict tolerances are distinct from bitwise neural equality. No rounding,
NaN replacement, looser hidden thresholds or TensorFlow execution in the Torch
runtime is used to fabricate equality.

The seven nonfinite original site/search checkpoint families remain rejected.
The working site backup and pinned historical search checkpoint are explicit
alternatives with their own provenance; they are not presented as identical to
those corrupt models. Keras optimizer slots are not falsely treated as native
Torch recovery state. Same-gradient optimizer replay passes, while independent
IPA bias updates retain the documented failure caused by FP32 residual gradients
and Adam amplification.

Modern TF32 defaults can also change GPU predictions. CLI validation configures
strict FP32; public Python GPU examples call `deterministic(seed)` first. A failed
TF32 benchmark comparison remains separately recorded beside the successful FP32
comparison.

## Raw PDB and ligand preparation

Both real original 1A7X inputs succeed through Reduce/MSMS/PyMesh/APBS, original
9Å/12Å patches and Torch predictions:

- Protein target A: 2,253 vertices; site `[1,2253]`, both descriptors `[2253,80]`.
- Target A with FKA_B: 2,543 vertices; site `[1,2543]`, both descriptors `[2543,80]`.

Actual failures were localized and corrected: unavailable historical APBS shared
libraries, the PDB2PQR APBS-input CLI dialect, incomplete ligand hydrogens,
OpenBabel MOL2 residue-table omission, and modern KDTree equidistant-neighbor
selection. The final bridge retains original interpolation semantics and original PyMesh prediction-surface serialization. An independently reproduced duplicate-attribute serialization failure was corrected; final protein and FKA prediction surfaces reload through the actual old PyMesh reader with exact geometry and unchanged attributes. Ligand H
addition preserves original heavy coordinates but introduces an explicit new
protonation baseline. Cross-version ligand chemistry equality is not certified.

Details and source changes: [stage12](12_raw_preparation.md),
[exact geometry](numerical_raw_geometry.json),
[exact precomputation](numerical_raw_precomputation.json),
[exact prediction-surface geometry](numerical_prediction_surface_geometry.json),
[400 neural comparisons](numerical_raw_neural.json).

## Rosetta

The original relaxation and interface-design XML files were executed unchanged
on the real original 1A7X/FKA complex. Relaxation and design finished successfully,
retained all 143 ligand atoms and produced finite scores. Same-seed design repeat
matched every exported coordinate and score exactly. Actual original inputs,
parameterization, constraints, binary/protocol hashes and output validation are
recorded. The prepared-surface-input integration also completed all six stages
successfully and retained all ligand atoms; its distinct hydrogen baseline gives
different scores, which are recorded separately. Scaffold grafting has not been executed.

Details: [stage13](13_rosetta.md), [numerical result](numerical_rosetta.json),
[prepared-input result](numerical_rosetta_prepared.json).

## Public benchmark and numerical comparison

The complete official archive was downloaded and verified:

- 9,961,569,979 bytes; official MD5 `2b45977254f45c6e229390cc62f3685d`.
- All 6,156 archive entries recovered despite wrapped ZIP offsets; 5,339 files
  extracted with per-member size/CRC verification, totaling 17,289,741,138 bytes.
- All 28 target directions and 200 decoy sources are present.
- The full configured evaluation has 112 combinations: both target conditions,
  all 28 directions, and both target-source/200-decoy databases, with no candidate cap.
- Original 12A0129 classifier, COM, site selection, random initial transforms,
  descriptor/score/clash thresholds and source-frame composition are recorded.
- Thirty-three independently reviewed COM/rotation/composition numerical checks
  are exactly equal to original-source expressions.
- Resuming requires matching checkpoint, data, original complexes, protocol,
  implementation dependencies and runtime fingerprints.

**Final scope:** the user requested numerical comparison without completing all
112 searches. The full run was stopped at **29 completed searches**: all 28
with-ligand target-source searches and one with-ligand 200-decoy search. These
processed **1,925,440 candidates** and retained **1,738 poses**. Completed results,
interrupted case artifacts, logs and execution provenance remain under
`artifacts/benchmark_full/run`. The full 112-case benchmark is **not completed**;
no full-dataset recovery-rate claim is made.

The requested numerical comparisons are complete: original-source protocol
expressions, all 284 surface-role records, all 56 target site selections, original
saved-PDB RMSD/iRMSD are exactly equal in their audited checks. Torch
eager-versus-deferred accepted-pose importance and RNG states are also exactly
equal; that optimization check is not a cross-framework importance comparison. Original Keras versus Torch on all 53 captured legacy poses passes the
strict CPU/CUDA FP32 tolerances. Modern Open3D docking remains a different pose
sampler from original Open3D0.5, as the complete 1A7X comparison demonstrates.
The correspondence-cache performance experiment is diagnostic only and was not
activated; the active benchmark retains the independently verified deferred
importance optimization. Optional complete-run and combined-ranking commands
remain available for future use.

Details: [stage14](14_benchmark.md),
[preserved run and stop proof](numerical_benchmark_partial_run.json),
[original protocol review](numerical_benchmark_protocol_review.json),
[exact surface-reader comparison](numerical_benchmark_readers.json),
[original PDB metrics](numerical_benchmark_metrics_review.json),
[exact accepted importance](numerical_benchmark_importance.json),
[53 original-pose neural scores](numerical_benchmark_legacy_scores.json).
This is the published public 200-decoy benchmark; it is distinct from the much
larger seed-design database and experimental wet-lab validation.

## Review, testing and stage records

Independent subagents continuously reviewed model math, source preservation,
package structure, geometry/protocol settings, documentation and failure handling.
The installed dependency check passes. At the earlier standalone stage, the regression suite passed all 28 tests without skips. That historical regular wheel imported all 10 task folders and packaged all four external workers; all 41 Python sources matched that stage byte for byte. Current version0.2 checks are recorded below. All implementation and tool functions have docstrings. See [installation evidence](numerical_standalone_install.json). Tests cover source-oracle numerics,
scientific edge cases, task registration, input formats, checkpoint conversion,
training/recovery, trusted geometry inputs, partial ligand hydrogens, MOL2
preservation, benchmark provenance and Rosetta failure reports. Regular wheel
and editable installation are both checked.

Stage records preserve purpose, intended work, implementation and changed files:

- Original neural migration: stages00–06.
- Seed/neosurface integration and numerical audit: stages07–10.
- [Standalone structure and English documentation](11_standalone_structure.md).
- [Original PDB/small-molecule surface preparation](12_raw_preparation.md).
- [Real Rosetta protocols and repeat](13_rosetta.md).
- [Public benchmark and old-registration comparison](14_benchmark.md).
- [Hydra YAML configuration and native training](15_hydra_configuration.md).
- [Exact engineering optimizations and assessed opportunities](16_engineering_acceleration.md).
- [Application README and installation guide](17_readme_packaging.md).
- [MaSIF distribution and hardware-aware runtime setup](18_distribution_runtime.md).
- [MaSIF method and application references](19_references.md).

Earlier stage records retain historical paths and precision failures; current
standalone commands are in README/INSTALL. Nothing has been committed.

## Current usability update: Hydra, engineering reuse and the MaSIF package

### Configuration and training

All 18 example configurations are YAML. Actual Hydra defaults/groups select the
model, task, dataset and trainer; application commands accept `key=value` overrides
and `--cfg job`. All eight migrated configurations retain their old values exactly.
Built-in presets ship in the wheel, so composition does not depend on a checkout.

Fresh native ScoreNN training and resumed training through the previous argument
frontend and the Hydra frontend produce exactly equal complete checkpoint values,
including all 56 model tensors, optimizer moments, class weights, history and Torch
RNG. Training saves the resolved YAML configuration. No optimizer/model equation or
sampling mechanism was changed. See [stage15](15_hydra_configuration.md) and
[the numerical comparison](numerical_hydra_configuration.json).

### Engineering improvements and evaluated opportunities

Existing objects reuse bounded coordinate trees and target patch preparation;
descriptor retrieval avoids sorting an already ordered intersection. The distance
and model arithmetic, cutoffs, vertex order and RNG remain unchanged. A cache-off
option and mutation/tie/bound regressions retain an explicit baseline path.

All 53 real captured poses compare exactly before/after on CPU and CUDA for
features, neural scores, importance (with equal-NaN handling), clashes and RNG.
An independent CPU review repeats that comparison. The complete Seed workflow
retains every recorded row field and RNG state for 53 candidates; 42 descriptor
queries retain exact selected indices, distances and vertex order. Isolated geometry processing
measured approximately 1.47× faster; an independent whole-scorer CPU measurement
was essentially unchanged. These are component timings, not a whole-benchmark
speed claim. See [stage16](16_engineering_acceleration.md) and
[geometry evidence](numerical_engineering_geometry.json).

Preparation and Rosetta were assessed. Their expensive external stages remain
unchanged; persistent geometry workers, protocol/thread changes and unverified
intermediate caches were not enabled. Content-verified intermediate reuse and
isolated independent-job concurrency are documented future opportunities requiring
further exact gates. Registration correspondence caching remains diagnostic only.
No full benchmark was restarted. These engineering comparisons are between the
same Torch model before/after; they do not upgrade TF/Torch tolerance to bitwise
equality or remove previously documented scientific failures.

### Installation and application documentation

The distribution, public import and primary command are named `masif`. The
canonical implementation/classes retain `masif_torch` module identities and old
state names. A regular wheel ships all task packages, external workers and Hydra
resources. Both console names and both module commands remain available.

The shorter English README follows the original MaSIF method/application structure
and explains site/search/ligand/ScoreNN/alignment/seed/neosurface/preparation/Rosetta
and training/recovery. The previous detailed guide lives in `docs/USAGE.md`.

`masif install-runtime --backend auto` selects an official Torch wheel from GPU
architecture and driver, then checks actual calculations in a fresh process. The
validated L40S retained CUDA12.6 and passed exact matrix/convolution checks. CUDA
versions above12.2 alone are insufficient for compatibility. Ordinary pip cannot
select hardware-specific dependencies automatically; runtime setup is an explicit
step. This package has not been published to PyPI. Local source and wheel installs
are supported; public `pip install masif` requires a published release. CUDA13 and
fresh pure CPU wheels are not certified against every earlier scientific reference
by the runtime smoke tests.

See [stage17](17_readme_packaging.md), [stage18](18_distribution_runtime.md),
[runtime evidence](numerical_runtime.json), and
[regular wheel evidence](numerical_distribution.json).

### Final installed acceptance

The final regular `masif-0.2.0` wheel matches **47 Python files and 30 YAML
resources** byte for byte. All 10 task folders import, 12 built-in presets compose,
the public API retains canonical object/model-state identity, and both console
commands and module commands run outside the checkout. Runtime planning also
runs with third-party site packages disabled. The installed regression suite
passes **43 tests with zero skips**, including original-source numerical references,
Hydra fresh/resume checkpoint equality and new cache/runtime contracts.

The active `masif_torch` environment also has an editable `masif0.2.0` installation.
The wheel audit used an isolated prefix with existing validated dependencies; it
does not claim a fresh complete dependency download, PyPI publication, universal
CUDA13 parity, or a completed benchmark. Continuous independent subagent reviews
found and verified fixes for CLI dispatch, console exit status and fixture-contract
issues before final acceptance. See [wheel evidence](numerical_distribution.json).

## Method references

The README and detailed usage guide now cite all three original methods behind the
integrated components plus four directly related MaSIF methods/applications. Seven
complete BibTeX records are provided in `citation.bib`. Publication metadata was
checked against original publishers, conference/workshop papers and author records.
See [stage19](19_references.md). This documentation change does not change model
weights, computation or the numerical acceptance scope.


## Repository publication hygiene

The standalone ignore rules exclude local datasets, model weights, run output,
validation artifacts, caches, build output, virtual environments and editor files.
Portable installation recipes, source, Hydra examples, tools, small intentional
fixtures, tests and all numerical documentation remain eligible for publication.
Machine-specific environment snapshots are local audit records. Actual Git ignore
checks passed for 88 generated paths and 222 retained paths, without initializing
the standalone repository, deleting files or changing an index. See
[stage20](20_repository_hygiene.md) and
[the verification record](numerical_repository_hygiene.json).

## Pretrained weight catalog

Fourteen usable weight files now live under `weights/`, grouped by application and
named for interface prediction, descriptors, pocket classification, second-stage
scoring or IPA variants. The catalog and YAML manifest describe source provenance,
patch/feature choices and known limits. README and ten task examples use these
paths. A small organizer validates task/configuration and finite tensors, copies
bytes, verifies hashes and refuses to overwrite different weights.

All fourteen source/destination files and complete checkpoint payloads are exactly
equal. Twenty-eight fixed-input CPU/CUDA inference groups have `max_abs=0`.
The existing 11 numerical and 4 Hydra regressions pass with zero skips. Binary
weights remain local; catalog, manifest and reconstruction tool are publishable.
No model equation or scientific acceptance scope changed. See
[stage21](21_weight_catalog.md), [the catalog](../../weights/README.md) and
[the verification record](numerical_weight_layout.json).

## Source repository publication

The standalone source is prepared for the user-selected GitHub repository
[TianZhuAI4S/MaSIF_torch](https://github.com/TianZhuAI4S/MaSIF_torch) on branch `main`.
The initial remote is empty. Source, portable installation/configuration files,
examples, tests, documentation, numerical evidence and the weight catalog/manifest
are included. Binary weights and local artifacts remain excluded by the clean
repository rules. Publication changes no model computation or scientific claims.
See [stage22](22_github_publication.md) for the publication checks.
