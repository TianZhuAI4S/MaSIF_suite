# Stage 16 — Exact engineering reuse

## Purpose and scope

Reduce repeated geometry preparation and indexing in alignment scoring, Seed, and Neosurface without changing model computation, numerical precision, sampling, candidate order, thresholds, or rigid-transform arithmetic. The scope is engineering reuse in the existing object-oriented owners. The user requested numerical comparison and stopped the full benchmark; this stage does not restart it.

The baseline files were preserved byte-for-byte before editing under `artifacts/engineering_acceleration/baseline_source/`. Comparisons below use the same converted PyTorch checkpoint before and after engineering changes. They establish exact before/after equality on tested inputs. They do not change the earlier TensorFlow-versus-PyTorch tolerance boundary or the Open3D 0.5-versus-modern pose boundary.

## Step 1 — Reuse exact coordinate indexes

**Purpose:** Avoid reconstructing target patch and target heavy-atom nearest-neighbor indexes for every scored pose and clash check.

**Intention:** Retain the same SciPy tree constructor, nearest-neighbor queries, input values, tie behavior, distance clipping, and feature arithmetic.

**Implemented:** `AlignmentScorer` owns an eight-entry LRU of coordinate trees. Keys include shape, dtype, and a SHA-256 of exact contiguous coordinate bytes. Trees own their indexed data, so caller mutations cannot corrupt previous entries. Changed bytes select a different entry. `clear_geometry_cache()` releases geometry without changing model state or sampling RNG. Optional tree arguments allow feature and clash helpers to reuse the same indexes; direct helper calls retain their previous behavior. `cache_geometry=False` disables scorer reuse.

**Changed files:** `src/shared/design/features.py`; regression coverage in `tests/test_engineering_cache.py` and `tools/validate_engineering_geometry.py`.

**Numerical result:** All 53 captured original-registration poses have exactly equal reconstructed features, probabilities, descriptor scores, importance arrays (including equal NaNs), clash counts, and final NumPy RNG state before/after the changes, on CPU and CUDA. Reconstructed features also equal the original captured feature packets. An additional above-200-point input preserves sampling output and RNG exactly. Unit tests cover duplicate-point ties, float32/float64 atom inputs, caller mutation, bounded storage, and cache clearing.

## Step 2 — Reuse patch preparation and simplify screening

**Purpose:** Remove repeated target indexing/shift/flip preparation and redundant sorting during descriptor retrieval.

**Intention:** Keep source vertex order, strict interface/descriptor thresholds, descriptor distance arithmetic, and the original shift-before-transform coordinate order.

**Implemented:** `SeedWorkflow` caches target patch templates for one `search()` call and gives each pose provider fresh writable copies of points, normals, and descriptors. The cache resets at the start of each search. Built-in providers receive the same values as before; an input-mutating provider cannot damage subsequent coordinate/feature templates. `cache_geometry=False` restores fresh patch construction in Seed. `match_descriptors()` uses one Boolean mask and `flatnonzero` instead of intersecting two already ordered index arrays. Distances and thresholds remain unchanged. Seed reuses the scorer's target atom index for clash filtering. Neosurface and Benchmark inherit the same Seed search path.

**Changed files:** `src/tasks/seed/workflow.py`, `src/shared/design/features.py`, `tests/test_engineering_cache.py`, and `tools/validate_engineering_workflow.py`. `src/shared/design/registration.py` and `src/tasks/benchmark/workflow.py` were not changed in this stage.

**Numerical result:** The preserved Seed owner and optimized owner process the same 53 real recorded candidates with every output row field exactly equal, including pose, probability, raw features, importance, acceptance metadata, and clashes; final sampling RNG is exact. Forty-two real descriptor queries preserve selected indices, distance values, and order. Separate tests check strict cutoff equality and isolated writable patch copies. The recorded legacy fixture contains count-like fitness values from 14 to 132; a test-only `RecordedPoses` provider supplies those values unchanged to both owners. Production `ReplayRegistration` continues to validate fitness in [0, 1]. This replay checks the 53 captured accepted inputs; it does not claim general rejected-candidate or full-database ranking coverage.

## Measurements

The geometry tool warms both scorer paths before recording CPU/CUDA timings and uses one PyTorch CPU thread. These are local measurements, not an end-to-end workload guarantee.

| Measurement | Baseline | Optimized | Interpretation |
| --- | ---: | ---: | --- |
| Geometry feature construction, 1,060 real-pose calls | 0.3129 s | 0.2130 s | 1.47× for this component |
| CPU complete IPA scoring, 53 real poses | 3.7996 s | 3.7851 s | Essentially unchanged; neural computation dominates |
| CUDA complete IPA scoring, 53 real poses | 4.5201 s | 4.5175 s | Essentially unchanged |

Independent review repeated the real CPU geometry and Seed orchestration gates and the cache unit tests. Timings vary with warm-up and host load; the exact numerical comparisons are the acceptance criteria.

## Assessed opportunities and limits

- **Modern registration correspondences:** Prior diagnostics cover 549 distinct real candidate keys, tie cases, exact sequential transforms/fitness and RNG, with modest measured gains. The correspondence cache remains diagnostic only. No active registration backend, RANSAC/ICP configuration, descriptor casting, or legacy Open3D worker was changed.
- **Search inference:** This stage does not change model forward operations, batch sizes, attention/convolution math, or precision. Shared descriptor retrieval is accelerated by removing redundant intersection sorting. No new inference batching speedup is claimed.
- **Preparation:** Content-verified reuse of protonation, mesh, chemistry, APBS, and precomputation outputs is a feasible separate improvement when every input, tool version, and parameter is included in provenance. This stage implements no persistent PyMesh worker or preparation cache; mesh ordering and process state require a separate equivalence gate.
- **Rosetta:** Recorded relax/design durations were approximately 209.98 s / 145.54 s; worker preparation was approximately 2 s. Combining stochastic runs or changing thread/minimizer settings has limited startup benefit and can change results. Content-verified ligand parameter reuse and isolated independent-job scheduling can be assessed separately. This stage changes no Rosetta protocol, seed, worker, or threading.
- Target surfaces are immutable inputs during each search, as used by the built-in workflows. Template coordinate/feature arrays are copied for providers; neighbor indices retain the previous interface behavior.
- No complete benchmark was rerun. Historical results and their execution manifests remain historical; modified source naturally requires new provenance for future runs.

## Reproduction and evidence

```bash
python -m unittest discover -s tests -p test_engineering_cache.py -v
python -m unittest discover -s tests -p test_design.py -v
python tools/validate_engineering_geometry.py --config examples/benchmark.yaml --devices cpu cuda
python tools/validate_engineering_workflow.py --config examples/benchmark.yaml
```

The numerical tools require the downloaded real fixture/cache artifacts and original converted checkpoint; they do not download or execute a full benchmark. Use strict FP32 CUDA settings through the supplied deterministic initialization, as in the previous migration gates.

- `numerical_engineering_geometry.json`: exact CPU/CUDA geometry/scoring and sampling evidence, source/checkpoint hashes, component timings.
- `numerical_engineering_workflow.json`: exact 53-row replay and 42 real descriptor-query comparisons.
- `numerical_engineering_geometry_independent_review.json` and `numerical_engineering_workflow_independent_review.json`: independent rerun evidence.
- `tests/test_engineering_cache.py`: five mutable-input, tie, dtype, boundary, and lifecycle tests.
- Existing `tests/test_design.py`: seven original geometry/sampling/replay/PDB contract tests also pass.

## Outcome

Geometry construction avoids repeat work while retaining exact outputs on all tested real inputs and explicit boundary tests. The measured benefit is concentrated in geometry preparation; total IPA scoring remains dominated by unchanged neural and importance computation. Registration, preparation, and Rosetta opportunities are assessed honestly without unvalidated scientific changes.
