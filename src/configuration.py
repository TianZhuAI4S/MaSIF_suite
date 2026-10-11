"""Hydra composition for packaged task presets and explicit user YAML files."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import DictConfig, OmegaConf


def compose_config(config_name: str, overrides: Sequence[str] = (),
                   config_path: str | Path | None = None) -> DictConfig:
    """Compose real Hydra defaults/groups without changing the caller's directory.

    Built-in presets come from the installed ``masif_torch.conf`` package.
    A user directory is resolved against the calling working directory; values
    such as input/output paths remain relative to that same working directory.
    """
    context = (initialize_config_module(config_module="masif_torch.conf", version_base="1.3")
               if config_path is None else
               initialize_config_dir(config_dir=str(Path(config_path).expanduser().resolve()),
                                     version_base="1.3"))
    with context:
        return compose(config_name=config_name, overrides=list(overrides))


def configuration_dict(config: DictConfig) -> dict:
    """Resolve interpolations and reject missing required values before execution."""
    result = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(result, dict):
        raise ValueError("The application configuration must be a YAML mapping")
    return result


def load_config_file(path: str | Path, overrides: Sequence[str] = ()) -> dict:
    """Read YAML through Hydra; retain old JSON mappings for existing integrations.

    YAML supports defaults lists and genuine Hydra override grammar. Legacy JSON
    is read only for backwards compatibility and accepts no composition overrides.
    Reports and saved checkpoint metadata continue using their original formats.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".json":
        if overrides:
            raise ValueError("Convert legacy JSON configuration to YAML before adding Hydra overrides")
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError("The application configuration must be a mapping")
        return result
    if path.suffix.lower() != ".yaml":
        raise ValueError("Hydra configuration files must use the .yaml extension")
    return configuration_dict(compose_config(path.name, overrides, path.parent))
