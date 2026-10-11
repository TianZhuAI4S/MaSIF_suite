"""Compare PyTorch against independently exported TensorFlow graphs and tensors."""

import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from .checkpoints import load_checkpoint, parameter_mapping
from .optim import TFAdam


def deterministic(seed=0):
    """Disable reduced precision and nondeterministic kernels before validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def compare(actual, expected, atol=1e-5, rtol=1e-4):
    """Compute errors and fail finite/shape checks without hiding near-zero errors."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, reason=f"Shape mismatch: {actual.shape} vs {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return dict(passed=False, reason="NaN/Inf encountered")
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(passed=bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
                max_abs=float(delta.max(initial=0)),
                max_rel=float((delta / np.maximum(np.abs(expected), 1e-12)).max(initial=0)),
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-12)),
                rmse=float(np.sqrt(np.mean(delta ** 2))), atol=atol, rtol=rtol)


def validate(export_dir, checkpoint, device="cpu", atol=1e-5, rtol=1e-4,
             grad_atol=1e-5, grad_rtol=1e-4):
    """Compare forward, probabilities, convolution layers, loss and all gradients."""
    deterministic()
    model, _ = load_checkpoint(checkpoint, device)
    model.eval()
    task = model.config["task"]
    with np.load(Path(export_dir) / "fixture.npz", allow_pickle=False) as fixture:
        keys = {"features", "rho", "theta", "mask", "indices", "labels", "pos_idx", "neg_idx"}
        data = {k: torch.from_numpy(fixture[k]).to(device) for k in fixture.files if k in keys}
    traces = {}
    hooks = []
    def capture(name):
        """Capture a module's numerical output without retaining its graph."""
        def hook(module, inputs, output):
            """Store channels-last post-ReLU for score BN, native output otherwise."""
            value = torch.relu(output).transpose(1, 2) if task in ("score", "alignment") else output
            traces["trace/" + name] = value.detach().cpu().numpy()
        return hook
    modules = list(model.named_modules())
    for name, module in modules:
        if task in ("score", "alignment") and name.startswith("norms."):
            hooks.append(module.register_forward_hook(capture("block" + name.split(".")[1])))
        elif task not in ("score", "alignment") and ((name.startswith("convs.") or
                                               name.startswith("extra_convs.")) and name.count(".") == 1):
            hooks.append(module.register_forward_hook(capture(name)))
    try:
        output = (model(data["features"]) if task in ("score", "alignment") else
                  model(data["features"], data["rho"], data["theta"], data["mask"], data.get("indices")))
    finally:
        for hook in hooks:
            hook.remove()
    probability = (output.sigmoid()[:, 0] if task == "site" else
                   output.softmax(-1) if task in ("ligand", "score", "alignment") else output)
    actual = dict(output=output.detach().cpu().numpy(),
                  probability=probability.detach().cpu().numpy(), **traces)
    if task in ("score", "alignment"):
        loss = torch.nn.functional.cross_entropy(output, data["labels"].long())
    elif task == "site":
        loss = model.loss(output, data["labels"], data["pos_idx"], data["neg_idx"])
    elif task == "ligand":
        loss = model.loss(output, data["labels"])
    else:
        loss = model.loss(output)
    loss.backward()
    actual["loss"] = loss.detach().cpu().numpy()
    named = dict(model.named_parameters())
    for target, (source, axes) in parameter_mapping(model).items():
        if target not in named:  # BN running statistics are buffers, not differentiable weights.
            continue
        grad = named[target].grad
        if grad is None:
            raise AssertionError(f"Missing PyTorch gradient: {target}")
        value = grad.detach().cpu().numpy()
        if axes is not None:
            value = value.transpose(np.argsort(axes))
        actual["grad/" + source] = value
    with np.load(Path(export_dir) / "reference.npz", allow_pickle=False) as reference:
        results = {k: compare(actual[k], reference[k],
                              grad_atol if k.startswith("grad/") else atol,
                              grad_rtol if k.startswith("grad/") else rtol) for k in reference.files}
    return dict(checkpoint=str(checkpoint), device=str(device), task=task,
                torch=str(torch.__version__), passed=all(v["passed"] for v in results.values()),
                tensors=results)


def validate_adam(reference_file):
    """Check three TF Adam updates, including epsilon-sensitive tiny gradients."""
    with np.load(reference_file, allow_pickle=False) as reference:
        parameter = torch.nn.Parameter(torch.from_numpy(reference["initial"].copy()))
        optimizer = TFAdam([parameter], eps=float(reference["epsilon"]) if "epsilon" in reference else 1e-8)
        results = []
        for grad, expected in zip(reference["gradients"], reference["updates"], strict=True):
            parameter.grad = torch.from_numpy(grad.copy())
            optimizer.step()
            results.append(compare(parameter.detach().numpy(), expected, atol=1e-7, rtol=1e-6))
    return dict(passed=all(v["passed"] for v in results), updates=results)


def write_report(report, output):
    """Save machine-readable evidence beside its checkpoint or under the docs folder."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
