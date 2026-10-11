#!/usr/bin/env bash
# Install an isolated environment, then select and verify its official Torch runtime.
# Usage: bash scripts/install.sh auto|cpu|cu126|cu130 [environment-prefix]
set -euo pipefail
compute_platform="${1:-auto}"
masif_prefix="${2:-$HOME/.conda/envs/masif_torch}"
case "$compute_platform" in
  auto|cpu|cu126|cu130) ;;
  *) echo 'Choose auto, cpu, cu126 or cu130' >&2; exit 2 ;;
esac
masif_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda env create --prefix "$masif_prefix" --file "$masif_root/environments/masif_torch.yml"
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' conda run --prefix "$masif_prefix" python -m pip install \
  --index-url https://pypi.org/simple -r "$masif_root/environments/runtime-requirements.txt"
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' conda run --prefix "$masif_prefix" python -m pip install \
  --no-deps --no-build-isolation -e "$masif_root"
# This launcher imports only standard-library modules before installing Torch.
conda run --prefix "$masif_prefix" masif install-runtime --backend "$compute_platform"
conda run --prefix "$masif_prefix" masif doctor
conda run --prefix "$masif_prefix" python -m pip check
