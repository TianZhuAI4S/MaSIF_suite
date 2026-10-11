# Stage 17 — Scientific README and application installation

## Purpose

Present MaSIF as a scientific protein-surface learning toolkit, with an application-focused English README resembling the official project. Keep the detailed conversion, data, numerical, and implementation material available in a separate guide. Document the new `masif` distribution/command and Hydra applications without claiming an unpublished public release.

## Intended work

1. Inspect the local original README and current official GitHub README for the description, method, preparation, applications, and reference structure.
2. Preserve the existing detailed README as `docs/USAGE.md` and rebase its local links.
3. Write a shorter main README organized around site/search/ligand/ScoreNN/seed/neosurface, raw preparation, Rosetta, and training/recovery.
4. Explain source/wheel pip installation and the separate hardware-aware runtime installer.
5. Validate documented commands and package/bootstrap behavior against the implementation, retaining the numerical limitations.

## Changes implemented

- `README.md` now starts with the molecular-recognition problem, surface-patch method, scientific application table, and examples of the common `masif` command. Detailed tensor/optimizer/weight-mapping material is linked instead of being the main narrative.
- `docs/USAGE.md` preserves the previous detailed guide. Its local Markdown links are rebased to the `docs` directory; compatibility commands still run from the project root.
- `INSTALL.md` describes local `pip install .` or an externally supplied wheel, `masif install-runtime`, CPU/CUDA choices, optional molecular extras, packaged Hydra configuration, and the independent TensorFlow reader environment.
- The documentation explicitly distinguishes distribution/CLI naming from publication: this implementation has not been uploaded to PyPI. It does not present public `pip install masif` as an available release.
- The application README retains scientific limits: strict FP32 tolerances versus bitwise equality, rejected nonfinite weights, independent IPA training-update failures, modern-versus-historical docking differences, and the user-requested stopped benchmark.

The packaging, alias, runtime-installer, and Hydra implementation are owned by their respective implementation agents. This documentation stage changes no neural mathematics or dataset sampling.

## Scientific source and references

The documentation author inspected `../masif/README.md` and browsed the current [official MaSIF repository](https://github.com/LPDI-EPFL/masif). The new description is written for this PyTorch implementation; it does not claim ownership of the original method or repeat the official TensorFlow installation instructions as current Torch instructions. Original concept/method illustrations and paper references remain attributed to MaSIF.

## Review and tests

Independent review checked the four documentation deliverables and their corresponding behavior:

| Deliverable | Outcome and evidence |
|---|---|
| Scientific main README | 200 lines; application-first wording, original method attribution, one CLI, Hydra training/recovery, and explicit numerical scope. |
| Preserved detailed guide | Existing conversion/data/numerical recipes retained; local links rebased and current examples updated to native YAML. Compatibility commands and historical JSON reports remain documented. |
| Installation guide | Runtime architecture/driver rules match the implementation. CPU auto-selection applies when no NVIDIA GPU is accessible; incompatible architectures/allocations/drivers produce an error. Publication and fresh-environment limits are explicit. |
| Documentation and implementation review | Four independent Hydra tests and six runtime tests passed without skips. The Hydra gate compares complete fresh/resumed ScoreNN checkpoints exactly; other training selectors are configuration/model-group checks, rather than new cross-framework training audits. |

The final static review checked 110 local Markdown paths/anchors, including package evidence, and all shell code-block syntax; all 17 application examples in README/INSTALL composed with `--cfg job`, avoiding execution on illustrative paths. The independent bootstrap/API review confirmed lazy imports, preserved canonical model identity, packaged Hydra resources, and hardware verification on every visible device. After the console-return fix, an independent temporary-prefix installation of the rebuilt wheel verified both `masif` and `masif-torch` CPU dry-run consoles and `python -S -m masif` bootstrap exit 0 outside the checkout. The shipped entrypoint bytes matched current source. The final regular-wheel audit records exact bytes for 47 Python files and 30 YAML resources, plus 43 installed regressions without skips, in [distribution evidence](numerical_distribution.json); [Stage 18](18_distribution_runtime.md) owns that final package validation.

This review does not claim a public PyPI release, a fresh complete conda installation, or fresh CPU/CUDA13 hardware parity. The observed runtime used the already installed validated CUDA12.6 wheel. Existing TF/Torch and molecular evidence remains linked from [the detailed guide](../USAGE.md) and [overall report](REPORT.md).

## Changed files

- `README.md`
- `INSTALL.md`
- `docs/USAGE.md`
- `docs/torch_migration/17_readme_packaging.md`
