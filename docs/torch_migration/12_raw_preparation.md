# Stage 12 — Raw protein and small-molecule surfaces

## Purpose

Run from the original crystallographic PDB to a surface, neural-network patches,
site probabilities and straight/flipped search descriptors in the standalone project.
Preserve the original scientific geometry and explicitly test interpreter boundaries.

## Intended implementation

- Keep PyMesh and its original numerical dependencies isolated in the existing
  compatible geometry interpreter.
- Use modern RDKit/OpenBabel/ProDy for the original neosurf chemical functions.
- Transfer primitive arrays through NPZ; import no TensorFlow or PyMesh into Torch.
- Use the unchanged upstream geodesic precomputation and converted finite weights.
- Execute real protein-only and FKA-containing 1A7X inputs; compare original outputs
  and independent TensorFlow predictions, loss, intermediate layers and gradients.

## Code changes

`src/tasks/preparation/pipeline.py` owns checked, logged stages and PyTorch inference.
`worker_geometry.py` runs original Reduce, chain extraction, MSMS, PyMesh repair,
normal computation, interface labeling, interpolation and PLY writing.
`worker_chemistry.py` runs the original neosurf chemical feature functions in a
modern interpreter. Its raw numeric features return to the geometry worker for
interpolation. Workers ship inside the installed package and run as direct scripts,
including in Python 3.6. The registry and explicit package mapping include this task.

- `examples/preparation_protein.yaml` and `preparation_ligand.yaml` declare actual
  source paths, external tools, optional ligand template, output and checkpoints.
- `tools/pack_preparation_references.py` freezes full site indexing and 32 real
  search patches for independent TensorFlow graph validation.
- `pyproject.toml` adds the chemistry extra; the actual installed environment is
  recorded in `environments/masif_torch-chemistry.lock.txt`.
- Existing success reports are invalidated at run start. Nonzero external stages
  save a failed result and preserve the diagnostic log. Configuration rejects a
  requested checkpoint when its required preprocessing task is absent.

## Actual failures and corrections

1. The first historical APBS binary lacked shared libraries. The working rebuilt
   APBS 1.5 and multivalue executables were selected explicitly.
2. Neosurf's `--apbs-input=filename` assumes a different PDB2PQR interface. The
   available PDB2PQR 2.1.1 accepts bare `--apbs-input`. The adapter exposes this
   dialect, keeps original PARSE/noopt/whitespace settings, checks all three tool
   exit codes, and checks a finite potential for every vertex.
3. Reduce removed ligand H but did not regenerate FKA hydrogens. Before MSMS,
   RDKit assigns the supplied SDF bond orders and adds H coordinates. Original
   heavy-atom PDB lines and coordinates remain unchanged; existing partial
   hydrogens are retained and new names are unique. This is an explicit input
   repair, not a claim that historical protonation coordinates are identical.
4. OpenBabel3 omitted the MOL2 SUBSTRUCTURE table required by PDB2PQR2.1. The
   adapter restores only residue bookkeeping; atom, bond and charge records remain
   byte-for-byte unchanged. Repeated application makes no further changes.
5. Modern sklearn KDTree selected different neighbors at equal distances. The
   first protein comparison differed by 0.0130501 in hydrogen-bond features and
   0.18749 in hydrophobicity. Executing the unchanged interpolation function in
   the original KDTree environment removed both differences, without loosening
   tolerances or rounding outputs.
6. The prediction PLY writer initially narrowed original ASCII decimal coordinates
   through a float32 schema. Surface coloring now uses the original PyMesh
   load/save path, preserving geometry, chemical fields, normals and faces.

## Real runs and numerical evidence

| Input | Geometry and complete pipeline | Torch outputs |
|---|---|---|
| Original 1A7X, target A, protein only | Successful; 2,253 vertices, 4,502 faces | site `[1,2253]`; both search orientations `[2253,80]` |
| Original 1A7X, target A plus FKA_B | Successful; 2,543 vertices | site `[1,2543]`; both search orientations `[2543,80]` |

Logs, resolved configuration and native products are under
`artifacts/raw_pipeline/{protein,protein_bridge,ligand}/`.

- [Raw geometry comparison](numerical_raw_geometry.json): all 10 vertex properties
  and the face array (11 checks) match the original direct protein pipeline exactly, including
  charges, hydrogen bonds, hydrophobicity, normals and interface labels.
- [Raw precomputation comparison](numerical_raw_precomputation.json): all 18
  original site/search input arrays match exactly, including ragged neighbor
  indices and polar coordinates.
- `numerical_prediction_surface_geometry.json` checks that prediction-surface
  coloring preserves original float64 geometry and unmodified attributes.
- `numerical_raw_neural.json` summarizes strict independent TF/Torch checks for
  both raw inputs on CPU and CUDA. Full per-tensor reports remain alongside the
  products in `artifacts/raw_pipeline/numerical_*.json`.
- Regressions cover partial ligand hydrogens, exact preserved PDB lines, unique
  hydrogen names, repeated protonation, MOL2 record preservation/idempotence,
  and configuration failures before external execution.

Strict neural checks use `atol=1e-5`, `rtol=1e-4`; checkpoint weights and protein
geometry/input arrays have stronger exact-equality evidence. Float32 neural
kernels are not advertised as bitwise identical on arbitrary devices or inputs.

## Reproduction

See [the English installation guide](../../INSTALL.md) for external geometry
interpreter/tool configuration and [README](../../README.md) for commands.

```bash
masif-torch run --config examples/preparation_protein.yaml
masif-torch run --config examples/preparation_ligand.yaml
python tools/pack_preparation_references.py \
  --root artifacts/raw_pipeline/ligand --identifier 1A7X_A \
  --output artifacts/raw_pipeline/references/ligand
```

The full neural conversion/export steps are listed in README. Only freshly written,
locally trusted original neighbor indices are unpickled by preparation.

## Remaining precision boundaries

The FKA repair adds hydrogen coordinates; it cannot reproduce a historical missing
or differently protonated ligand input. Its current geometry and predictions run
successfully and are tested independently. Numerical equality to an earlier RDKit
or OpenBabel chemistry release is not certified. Seven corrupt original site/search
checkpoints remain unusable; these runs explicitly use the finite site backup and
pinned historical search checkpoint, not replacements presented as the same weights.
The raw pipeline still needs compatible external PyMesh/MSMS/APBS tools; the neural,
prepared-data and modern design environment remains separate and TensorFlow-free.
