# Stage 14 — Public benchmark integration and numerical comparison

## Purpose

Run the complete published small-decoy benchmark with the standalone PyTorch package: 14 ligand-induced complexes, both binding directions, with/without ligand targets, searched against the 28 protein subunits and against 200 decoys. This defines **112 searches**. The separate optional 8,879-decoy PDBbind dataset and the paper's full seed design database are different datasets.

## Updated user scope

The user subsequently requested **numerical comparisons only** and explicitly waived completion of the full benchmark. The running pool was stopped at that request. Numerical comparisons are complete; **29 of 112 searches** were completed and preserved. The optional complete-run implementation remains available for a future explicitly requested run.

## Implementation

- `src/tasks/benchmark/` owns the benchmark orchestration and composes the existing seed workflow, shared surface records, IPA scorer, and registration backend.
- `tools/download_benchmark.py` downloads the full public archive and verifies its official byte count and MD5 before promoting the file.
- `tools/extract_benchmark.py` recovers physical offsets from the ZIP32 central directory. The public 9.96 GB archive wraps offsets at 4 GB; ordinary extraction can seek to incorrect members. Each decoded local filename, uncompressed byte count, and CRC32 is checked.
- `examples/benchmark.yaml` describes the complete run. No candidate, target, or decoy cap is applied.
- Four conditions each contain 28 cases. Immutable surface arrays are cached once and shared with 16 CPU worker processes. Every accepted pose is written as a PDB and numeric packet. Every candidate retains rejection diagnostics.
- Original complexes are downloaded from RCSB to compute the original float32 protein/ligand interface center of mass. Atom distances use the original strict 4 Å threshold; surface center selection uses the original strict 10 Å threshold and fallback. Three sites are chosen.
- Source coordinates are randomized using the original Graphics Gems formula before RANSAC. This avoids starting ICP from the crystallographic pose. The final transform is composed back into the original coordinate frame.

## Protocol

The checkpoint follows **the actual `computational_benchmark/make_param_files.py` configuration**: `models_std/weights_12A_0129.hdf5`, converted losslessly to PyTorch. The separately named `models_benchmark/weights_seed_benchmark_12A_0123` checkpoint is not substituted.

Descriptor distance <3.0; source interface score >0.0; IPA probability >0.8; three target sites; RANSAC 2,000 iterations, radius 1.5 Å; outward normal shift 0.25 Å; fitness ≥0.3; zero allowed CA clashes and five allowed heavy atom clashes at radius 2 Å. The original `compute_nn_score` hardcodes importance cutoff 0.9. The benchmark computes the same probability for every candidate, then computes original sequential leave-one-point-out importance only after clash acceptance. Accepted output values and the sampling stream are unchanged; rejected importance diagnostics are explicitly omitted. Default seed tasks retain eager importance evaluation.

## Numerical checks

`numerical_benchmark_protocol.json` records exact array equality for:

1. The initial source transform compared with functions extracted from the unchanged official source.
2. The 1A7X float32 interface COM compared with the original full distance-matrix expression. Chunking preserves arithmetic while limiting memory.

The automatically recovered offsets of all 6,156 archive entries equal the previously independently recovered inventory. Full extraction verified **5,339 files**, **17,289,741,138 uncompressed bytes**, and every member CRC32. The downloaded archive is **9,961,569,979 bytes**, MD5 **2b45977254f45c6e229390cc62f3685d**.

Existing independent TensorFlow/PyTorch IPA tests cover the actual `0129` checkpoint on CPU and CUDA; those reports remain under `numerical_design_weights.json`.

### Reader precision and accepted diagnostics

The actual old Open3D 0.5 and PyMesh 0.2.1 environment read all 256 public surfaces. `numerical_benchmark_readers.json` covers **284 task-role records**, including 56 target records. Coordinates and normals match exactly. ASCII PLY target interface values are parsed from decimal text directly into float64, matching PyMesh rather than narrowing through the PLY header's float32 declaration; binary values preserve the stored precision. All 56 selected site sets were identical before and after this correction.

`numerical_benchmark_importance.json` verifies the deferred benchmark diagnostics on all **53 real accepted original-registration poses**: probability and point importance match eager evaluation exactly (including NaNs), with exactly unchanged NumPy RNG state. A guaranteed >200-point fixture also verifies that sampled inputs are drawn only once. Six focused regression tests cover resume integrity and ASCII/binary interface values.

Metrics now use actual exported three-decimal PDBs and original Bio.PDB float32 protein-heavy atom selections, with the native target without ligand. Independent original-source AST checks cover all 28 native selections and 18 accepted native-partner pose metrics; RMSD and interface RMSD match exactly. Historical unrounded float64 metric differences remain inspectable in `numerical_benchmark_metrics_review.json`.

## Reproduction

```bash
cd /home/tian/git_repo/masif_torch
python tools/download_benchmark.py --output artifacts/benchmark_full
python tools/extract_benchmark.py --archive artifacts/benchmark_full/dataset.zip --output artifacts/benchmark_full/dataset
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 masif-torch run --config examples/benchmark.yaml
```

## Actual original Open3D 0.5 comparison

A complete `with_ligand / targets / 1A7X_A` case was also run through the actual original Open3D **0.5.0.0** binary in its legacy Python environment: all **30,810 candidates**, all three sites, all 28 database subunits. The original RANSAC stopping arguments `(2000, 500)` and original ICP were executed in a persistent external worker, followed by the converted IPA checkpoint.

| Result | Original Open3D 0.5 poses + Torch IPA | Modern Open3D 0.19 poses + Torch IPA |
|---|---:|---:|
| Descriptor candidates | 30,810 | 30,810 |
| Accepted poses | 53 | 67 |
| Known partner rank | 1 | 1 |
| Best partner probability | 0.99707115 | 0.99700695 |
| Best partner interface RMSD | 0.86459 Å | 0.38569 Å |

Ranking retains the best pose per `(site, source protein)` as the official `analysis_utils.parse_results` does. Recovery uses interface heavy atoms within 5 Å in the reference complex and requires interface RMSD <5 Å. This table directly demonstrates the registration boundary: the two Open3D versions produce different poses and accepted hit counts.

Every accepted original-registration pose was captured: **53 real feature packets**. The unchanged official Keras layers and original `0129` HDF5 tensors were then evaluated on the same prepared `[53,200,4]` batch. PyTorch comparison passed strict `atol=1e-5, rtol=1e-4` on both devices:

| Device | Maximum logits error | Maximum probability error | Bitwise equality |
|---|---:|---:|---|
| CPU | 3.4571e-6 | 7.7486e-7 | No |
| CUDA, FP32 | 1.4305e-6 | 2.3842e-7 | No |

See `numerical_benchmark_legacy_scores.json` and the captured packets under `artifacts/benchmark_full/legacy_comparison/`. The separate original-registration run uses Torch IPA for docking decisions; the independent original Keras comparison covers all 53 accepted fixed poses. It is not a full TensorFlow benchmark rerun.

### Precision for the Python API

The CLI enables the project's strict precision settings. Direct GPU calls through the public Python API must enable them before inference:

```python
from masif_torch.tasks import build_task
from masif_torch.validation import deterministic

deterministic(seed=42)  # FP32, TF32 disabled, deterministic kernels.
report = build_task(config).run()
```

The initial new validation tool omitted this helper. Default CUDA TF32 produced larger errors: maximum logits error **6.6090e-4**, probability error **6.7055e-5**, which failed the declared tolerances. That failure remains visible in `numerical_benchmark_legacy_scores_tf32.json`. The corrected validation uses the existing CLI precision policy and the original tolerances.

## Resume integrity

Resume requires a matching `provenance.json`: SHA256 of checkpoint bytes, benchmark definitions, all consumed processed-surface files, original complex PDBs, task cases, the scoring/registration source dependency closure, and the Torch/NumPy/SciPy/Open3D/Biopython/PLY-reader versions. Cached surface packets also retain source, packet, and adapter implementation hashes. Changing inputs cannot silently reuse a previous summary. Four regression tests cover changed weights, definitions, surfaces, and unverified summaries. The migration retained 28 completed cases through an explicit certificate (`artifacts/benchmark_full/migration_audit/resume_certificate.json`): unchanged scientific input hashes, all selected site IDs, exact accepted diagnostics, and recomputed file-based metrics. The historical execution manifest and source snapshots are preserved separately; their execution is not relabeled as current source.

### Registration timing experiment

`tools/profile_benchmark_registration.py` is a diagnostic experiment, **not an active backend change**. It compares the official feature-matching entry point with cached descriptor correspondences using unchanged RANSAC/ICP parameters. More than 500 distinct real randomized candidate keys, descriptor ties, direct benchmark transforms/fitness, and NumPy RNG state match exactly. The larger experiment gave a modest gain after cache construction; restarting the full run would discard accumulated work. The benchmark therefore retains the existing modern registration backend. See `numerical_benchmark_registration_profile.json`.

The delegation used in this experiment is visible in the [official Open3D 0.19 source](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/pipelines/registration/Registration.cpp).

## Combined database analysis

`tools/analyze_benchmark.py` merges the separately executed 28-subunit and 200-decoy searches into the full **228-source** ranking for each target and ligand condition. It retains each best `(site, source)` pose and reuses the independently validated saved-PDB metrics recorded in per-case summaries; the analysis tool does not itself parse PDB coordinates. The inclusive-rank summary is reported separately from the original source's `get_topk` behavior, which skips lists with length ≤K; the historical boundary is preserved explicitly rather than silently relabeled.

Independent original-source AST checks reconstruct the 28 completed native-database cases and test synthetic TopK boundaries, ties, and malformed input (`numerical_benchmark_analysis_review.json`). This validates analysis logic and summary metrics, not a completed 228-source merged ranking across the full benchmark. A full combined report requires all 112 searches. The preserved partial run must be labeled incomplete when analyzed.

## Result and limits

The run was stopped at the user's request with **29/112 completed searches**, **1,925,440 candidates**, and **1,738 accepted poses**. All 28 with-ligand target-database cases completed (1,785,407 candidates; 1,529 accepted poses; 19 native-partner recoveries under the inclusive per-case criterion). One with-ligand decoy case, `6ENG_A`, completed (140,033 candidates; 209 accepted poses). No without-ligand searches completed. These partial counts do not establish the complete benchmark's recovery or decoy-ranking rates.

`artifacts/benchmark_full/run/status.json` records the explicit `stopped_at_user_request` state, full input/code/runtime fingerprints, completed/pending cases, and numerical evidence hashes. All 16 workers and their parent were terminated and verified stopped; proof is `artifacts/benchmark_full/migration_audit/user_stop_processes.json`. Historical execution manifests, corrected summaries, and logs remain under `migration_audit/`. Accepted numeric packets, aligned PDBs, per-case `summary.json`, and candidate `results.json` are preserved. A completed full-run `benchmark.json` is not claimed.

Modern Open3D 0.19 RANSAC has different random sampling and stopping semantics from Open3D 0.5. This run therefore does **not** establish identical docked poses, benchmark ranks, or paper recovery rates. Sorting the database for reproducibility also fixes an ordering that the original script obtained from the filesystem. Neural checkpoint conversion and fixed-pose neural comparisons remain separate, inspectable numerical guarantees.
