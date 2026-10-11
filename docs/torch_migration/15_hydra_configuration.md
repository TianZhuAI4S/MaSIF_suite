# Stage 15 — Hydra YAML configuration and task commands

## Purpose

Make every task and native training run configurable through YAML, including when
MaSIF is installed from a normal wheel outside this repository. Preserve the
existing model constructors, packet sampling, optimizer, tensor arithmetic,
checkpoint resume order, and random-number streams.

## Implementation

- `src/configuration.py` uses the actual Hydra Compose API and OmegaConf. Packaged
  presets use `initialize_config_module`; external YAML uses a resolved directory
  and `initialize_config_dir`. Required `???` values are rejected before execution.
- `src/conf/` ships task presets and separate `task`, `model`, `dataset`, and
  `trainer` groups. The training task selector chooses its model group through
  Hydra defaults-list interpolation. `task=search` selects the `ppi_search` model.
- `src/cli.py` adds named task commands and YAML composition. Native training is
  extracted into one shared helper used by both Hydra and the previous argument
  frontend. The new frontend adapts configuration values to that same helper.
- All example configuration files are YAML. The eight existing JSON examples were
  removed only after their YAML counterparts composed to identical dictionaries.
  `neosurf.json` becomes `neosurface.yaml`. The original alignment model example
  remains `alignment.yaml`; its inference workflow is `inference_alignment.yaml`.
- Training writes `resolved_config.yaml` after checkpoint and dataset validation.
  Reports, benchmark journals, and checkpoint metadata keep their existing formats.
- Packaging includes the config resources; the lightweight launcher and runtime
  installer are described separately in Stage 18.

Each added Python function has a docstring. No model/trainer/optimizer equation or
training RNG mechanism is changed in this stage.

## Installed command examples

Built-in presets work without a local `examples` or `src` directory:

```bash
masif site checkpoint=/path/site.pt input=/path/site.npz output=results/site.npy
masif search checkpoint=/path/search.pt input=/path/search.npz output=results/search.npy
masif ligand checkpoint=/path/ligand.pt input=/path/ligand.npz output=results/ligand.npy
masif score checkpoint=/path/score.pt input=/path/alignments.npz output=results/scores.npy
masif alignment checkpoint=/path/alignment.pt input=/path/alignments.npz output=results/ipa.npy
masif seed --config /path/seed.yaml
masif neosurface --config /path/neosurface.yaml
masif prepare --config /path/preparation_protein.yaml
masif prepare --config-name prepare_ligand --cfg job
masif rosetta --config /path/rosetta_prepared.yaml
masif benchmark --config /path/benchmark.yaml
```

The neural task presets consume safe numeric NPZ packets and save NPY outputs.
For original per-protein precomputation directories, the existing `masif surface`
command still generates site probabilities or straight/flipped search descriptors.
The real geometry, chemistry and Rosetta presets require installed external tools
and caller-provided paths; a YAML configuration does not install these tools.

Native training and resume:

```bash
masif train task=site dataset.input=/path/packets trainer.output=results/site \
  trainer.epochs=10 trainer.seed=42 trainer.device=cpu
masif train task=search dataset=search_cache dataset.input=/path/search_cache \
  trainer.output=results/search dataset.steps=1000 dataset.batch_size=32
masif train task=score dataset.input=/path/alignments trainer.output=results/score \
  'trainer.class_weights=[0.005,1.0]'
masif train task=site dataset.input=/path/packets trainer.output=results/site_resume \
  trainer.checkpoint=/path/last.pt trainer.resume=true trainer.epochs=10
masif train --config /path/train_site.yaml trainer.epochs=2
```

Training selectors are `site`, `search`, `ligand`, `score`, and `alignment`.
Device configuration for training is `trainer.device`; inference uses `device`.
The selected task must match the checkpoint task. When a checkpoint is loaded,
its saved model configuration defines the network. To change a fresh network,
override a real constructor field such as `model.n_thetas=8` or
`'model.feat_mask=[1,1,1,1,1]'`; `n_feat` is derived from the feature mask.

## Composition and path semantics

```bash
masif train task=ligand --cfg job
masif site --cfg job --resolve
masif train --config-path /path/configs --config-name train_site trainer.epochs=2
```

`--cfg job` prints the composed YAML without executing a task or validating missing
input paths. `--resolve` resolves interpolations in that printed configuration.
Actual execution resolves interpolations and rejects missing required values.
Unknown keys require intentional Hydra `+key=value` syntax. Defaults lists,
config-group selection, list overrides and OmegaConf interpolation are supported.
This CLI uses the Compose API; Hydra multirun launchers and automatic run-directory
management are not provided.

The caller's working directory is preserved. Relative input, checkpoint and output
values refer to that directory; they are not implicitly rebased to the YAML file.
The YAML file itself is located through `--config` or `--config-path`.
Hydra configuration filenames use the `.yaml` extension. External training examples
import built-in groups with:

```yaml
hydra:
  searchpath:
    - pkg://masif_torch.conf
```

Older argument commands and explicit JSON configuration reads remain compatible.
JSON does not support Hydra overrides; convert such application configuration to
YAML first. New documentation and examples use YAML.

## Numerical verification

[Configuration evidence](numerical_hydra_configuration.json) records:

- All eight migrated example dictionaries exactly equal their old JSON values.
- All 18 YAML examples compose successfully; all five training task/model selectors
  and the original search-cache dataset group are validated.
- Actual ScoreNN training through the previous argument frontend versus Hydra
  produces identical complete native checkpoint values after one epoch. This
  includes model parameters/buffers, Adam state, history, class weights and Torch RNG.
- Both frontends resume that checkpoint for another epoch with the same explicit
  learning-rate and class-weight overrides. Complete checkpoint values are again
  identical, including the restored epoch and step.
- Missing required values, unknown overrides, task/checkpoint mismatches, dry-run
  execution avoidance and default `main(argv=None)` dispatch have regression checks.

Smallest relevant verification:

```bash
python -m unittest discover -s tests -p test_configuration.py -v
python -m masif_torch --help
python -m masif_torch train task=search --cfg job
```

These checks prove that configuration changes preserve the tested native training
path. They do not expand the earlier TensorFlow/PyTorch equivalence claim: network
outputs retain the documented FP32 tolerance, damaged source checkpoints remain
rejected, and independent IPA training and old/modern Open3D poses retain their
previously reported differences.

Hydra behavior follows its official [Compose API](https://hydra.cc/docs/1.3/advanced/compose_api/),
[defaults list](https://hydra.cc/docs/1.3/advanced/defaults_list/) and
[config search path](https://hydra.cc/docs/1.3/advanced/search_path/) documentation.
