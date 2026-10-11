"""Strict conversion from exported TensorFlow tensors to portable .pt checkpoints."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .models import build_model


def parameter_mapping(model):
    """Return torch name → (TF name, transpose axes), including legacy site swaps."""
    mapping = {}
    if model.config["task"] in ("score", "alignment"):
        suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
        for i in range(len(model.convs)):
            for source, target, axes in (("kernel", "weight", (2, 1, 0)),
                                          ("bias", "bias", None)):
                mapping[f"convs.{i}.{target}"] = (f"layer-{3*i}/{source}{suffix}", axes)
            for source, target in (("gamma", "weight"), ("beta", "bias"),
                                   ("moving_mean", "running_mean"),
                                   ("moving_variance", "running_var")):
                mapping[f"norms.{i}.{target}"] = (f"layer-{3*i+1}/{source}{suffix}", None)
        for i in range(7):
            layer = 3 * len(model.convs) + 1 + i
            mapping[f"dense.{i}.weight"] = (f"layer-{layer}/kernel{suffix}", (1, 0))
            mapping[f"dense.{i}.bias"] = (f"layer-{layer}/bias{suffix}", None)
        return mapping
    for i in range(model.n_feat):
        for name in ("mu_rho", "sigma_rho", "mu_theta", "sigma_theta"):
            mapping[f"convs.{i}.{name}"] = (f"{name}_{i}", None)
        mapping[f"convs.{i}.weight"] = (f"W_conv_{i}", None)
        mapping[f"convs.{i}.bias"] = (f"b_conv_{i}", None)
    if model.config["task"] == "site":
        for i in range(model.config["n_conv_layers"] - 1):
            suffix = f"l{i+2}"
            for target, source in (("mu_rho", "mu_rho"), ("sigma_rho", "mu_theta"),
                                   ("mu_theta", "sigma_rho"), ("sigma_theta", "sigma_theta")):
                mapping[f"extra_convs.{i}.{target}"] = (f"{source}_{suffix}", None)
            mapping[f"extra_convs.{i}.weight"] = (f"W_conv_{suffix}", None)
            mapping[f"extra_convs.{i}.bias"] = (f"b_conv_{suffix}", None)
        layers = ("fc1", "fc2", "fc3", "fc4")
    elif model.config["task"] == "ligand":
        layers = ("fc1", "fc2", "fc3")
    else:
        layers = ("fc",)
    for i, layer in enumerate(layers):
        source = "fully_connected" + (f"_{i}" if i else "")
        mapping[f"{layer}.weight"] = (f"{source}/weights", (1, 0))
        mapping[f"{layer}.bias"] = (f"{source}/biases", None)
    return mapping


def convert(export_dir, output, include_optimizer=False):
    """Validate every exported parameter and write a TensorFlow-free checkpoint."""
    export_dir, output = Path(export_dir), Path(output)
    config = json.loads((export_dir / "config.json").read_text())
    model = build_model(config)
    mapping = parameter_mapping(model)
    state = model.state_dict()
    manifest = json.loads((export_dir / "manifest.json").read_text())
    with np.load(export_dir / "weights.npz", allow_pickle=False) as weights:
        used = set()
        audit = []
        for target, (source, axes) in mapping.items():
            if source not in weights:
                raise ValueError(f"Missing TensorFlow parameter: {source}")
            array = weights[source]
            if axes is not None:
                array = array.transpose(axes)
            if tuple(array.shape) != tuple(state[target].shape):
                raise ValueError(f"Shape mismatch for {source}: {array.shape}, expected {state[target].shape}")
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Parameter {source} must be finite float32")
            state[target] = torch.from_numpy(array.copy())
            # The conversion must be bitwise reversible, including transpositions.
            restored = state[target].numpy()
            if axes is not None:
                restored = restored.transpose(np.argsort(axes))
            if not np.array_equal(restored, weights[source]):
                raise AssertionError(f"Non-reversible weight conversion: {source}")
            used.add(source)
            audit.append(dict(source=source, target=target, shape=list(array.shape)))
        ignored = {name for name in weights.files if "/Adam" in name or "/.OPTIMIZER_SLOT/" in name or
                   name.startswith("optimizer/") or
                   name in ("beta1_power", "beta2_power", "_CHECKPOINTABLE_OBJECT_GRAPH") or
                   name.endswith("/.ATTRIBUTES/OBJECT_CONFIG_JSON")}
        unknown = set(weights.files) - used - ignored
        if unknown:
            raise ValueError(f"Unmapped checkpoint variables: {sorted(unknown)}")
        optimizer = None
        if include_optimizer:
            if config["task"] in ("score", "alignment"):
                raise ValueError("Keras optimizer slots require a separate audited conversion; export supports inference weights")
            from .optim import TFAdam
            hyper = manifest.get("optimizer")
            if hyper is None:
                raise ValueError("Export lacks original optimizer hyperparameters; re-export this checkpoint")
            opt = TFAdam(model.parameters(), lr=hyper["learning_rate"],
                         betas=(hyper["beta1"], hyper["beta2"]), eps=hyper["epsilon"])
            named = dict(model.named_parameters())
            for target, (source, axes) in mapping.items():
                moments = {}
                for key, suffix in (("m", "/Adam"), ("v", "/Adam_1")):
                    if source + suffix not in weights:
                        raise ValueError(f"Missing optimizer slot: {source + suffix}")
                    array = weights[source + suffix]
                    if axes is not None:
                        array = array.transpose(axes)
                    if array.shape != tuple(named[target].shape) or not np.isfinite(array).all():
                        raise ValueError(f"Invalid optimizer slot for {source}")
                    moments[key] = torch.from_numpy(array.copy())
                opt.state[named[target]] = moments
            for key in ("beta1_power", "beta2_power"):
                opt.param_groups[0][key] = torch.tensor(float(weights[key]), dtype=torch.float32)
            optimizer = opt.state_dict()
    model.load_state_dict(state, strict=True)
    payload = dict(format_version=1, config=model.config, state_dict=state,
                   provenance=manifest, optimizer=optimizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = dict(parameters=audit, ignored=sorted(ignored),
                  weights_sha256=hashlib.sha256((export_dir / "weights.npz").read_bytes()).hexdigest(),
                  exact_weight_roundtrip=True, optimizer_included=include_optimizer)
    output.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_checkpoint(path, device="cpu"):
    """Load a safe tensor/dictionary checkpoint and reconstruct its model."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported MaSIF checkpoint format")
    model = build_model(payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def save_checkpoint(model, path, optimizer=None, **metadata):
    """Save model configuration, tensors, and optional resumable training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format_version=1, config=model.config, state_dict=model.state_dict(),
                    optimizer=None if optimizer is None else optimizer.state_dict(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state_all=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                    **metadata), path)
