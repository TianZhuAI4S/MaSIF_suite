"""Run in masif_nn: export original checkpoint tensors and independent TF references."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from legacy_weights import read_weights

tf.compat.v1.disable_eager_execution()


def infer_config(weights, overrides=None):
    """Infer repository architectures; radius/rotation defaults follow its callers."""
    if "layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE" in weights:
        first = weights["layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"]
        if first.shape[2] == 16:
            config = dict(task="alignment", selected_features=list(range(first.shape[1])),
                          max_npoints=200, n_negatives=100, n_positives=1)
        elif first.shape[1:] == (3, 8):
            config = dict(task="score")
        else:
            raise ValueError(f"Unknown Keras classifier architecture: first kernel {first.shape}")
    else:
        task = "site" if "fully_connected_3/weights" in weights else (
            "ligand" if "fully_connected_2/weights" in weights else "ppi_search")
        features = sum(name.startswith("W_conv_") and name[7:].isdigit() for name in weights)
        gaussians = weights["mu_rho_0"].shape[-1]
        theta = (overrides or {}).get("n_thetas", weights["fully_connected_2/biases"].size if task == "site" else 16)
        if gaussians % theta:
            raise ValueError("Cannot infer radial grid; provide a supported configuration")
        config = dict(task=task, max_rho=9.0 if task == "site" else 12.0,
                      n_thetas=theta, n_rhos=gaussians // theta,
                      n_rotations=4 if task == "site" else 16, feat_mask=[1] * features)
        if task == "site":
            config["n_conv_layers"] = 1 + sum(f"W_conv_l{i}" in weights for i in (2, 3, 4))
        if task == "ligand":
            config.update(n_ligands=weights["fully_connected_2/biases"].size, dropout=0.0)
    if overrides:
        config.update(overrides)
    return config


def fixture(config):
    """Make deterministic synthetic inputs, including boundaries and a zero mask."""
    rng = np.random.default_rng(20261010)
    if config["task"] in ("score", "alignment"):
        width = len(config["selected_features"]) if config["task"] == "alignment" else 3
        points = config.get("max_npoints", 200)
        return dict(features=rng.normal(size=(8, points, width)).astype(np.float32),
                    labels=np.arange(8, dtype=np.int64) % 2)
    n, v, f = (32 if config["task"] == "ligand" else 8), 11, sum(config["feat_mask"])
    features = rng.normal(size=(n, v, f)).astype(np.float32)
    rho = rng.uniform(0, config["max_rho"], (n, v)).astype(np.float32)
    theta = rng.uniform(-2 * np.pi, 4 * np.pi, (n, v)).astype(np.float32)
    theta[:, :3] = np.array([0, 2 * np.pi, -2 * np.pi], dtype=np.float32)
    rho[:, 0] = 0
    mask = (rng.uniform(size=(n, v)) > 0.2).astype(np.float32)
    mask[0] = 0
    indices = rng.integers(n, size=(n, v), dtype=np.int64)
    labels = (np.eye(config["n_ligands"], dtype=np.float32)[2] if config["task"] == "ligand"
              else np.eye(2, dtype=np.float32)[np.arange(n) % 2])
    return dict(features=features, rho=rho, theta=theta, mask=mask, indices=indices,
                labels=labels, pos_idx=np.arange(0, n, 2, dtype=np.int64),
                neg_idx=np.arange(1, n, 2, dtype=np.int64))


def score_reference(weights, data, config=None):
    """Reconstruct the saved Keras architecture with TF ops when no .meta exists."""
    graph = tf.Graph()
    with graph.as_default():
        x = tf.constant(data["features"])
        fetches = {}
        parameters = {}
        def value(name, trainable=True):
            """Track original trainable tensor identities for independent TF gradients."""
            if not trainable:
                return tf.constant(weights[name])
            parameter = tf.Variable(weights[name])
            parameters[name] = parameter
            return parameter
        count = 5 if (config or {}).get("task") == "alignment" else 6
        for i in range(count):
            prefix = f"layer-{3*i}/"
            kernel = value(prefix + "kernel/.ATTRIBUTES/VARIABLE_VALUE")
            bias = value(prefix + "bias/.ATTRIBUTES/VARIABLE_VALUE")
            x = tf.nn.conv1d(x, kernel, stride=1, padding="VALID") + bias
            prefix = f"layer-{3*i+1}/"
            values = [value(prefix + name + "/.ATTRIBUTES/VARIABLE_VALUE", name in ("beta", "gamma"))
                      for name in ("moving_mean", "moving_variance", "beta", "gamma")]
            x = tf.nn.relu(tf.nn.batch_normalization(x, *values, variance_epsilon=1e-3))
            fetches[f"trace/block{i}"] = x
        x = tf.reduce_mean(x, axis=1)
        for i in range(7):
            prefix = f"layer-{3*count+1+i}/"
            x = (tf.matmul(x, value(prefix + "kernel/.ATTRIBUTES/VARIABLE_VALUE")) +
                 value(prefix + "bias/.ATTRIBUTES/VARIABLE_VALUE"))
            if i < 6:
                x = tf.nn.relu(x)
        fetches["output"] = x
        fetches["probability"] = tf.nn.softmax(x)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=data["labels"], logits=x))
        fetches["loss"] = loss
        gradients = tf.gradients(loss, list(parameters.values()))
        fetches.update({"grad/" + name: grad for name, grad in zip(parameters, gradients, strict=True)})
        init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session(graph=graph, config=session_config()) as session:
        session.run(init)
        first = session.run(fetches)
        second = session.run(fetches)
    if not all(np.array_equal(first[k], second[k]) for k in first):
        raise AssertionError("TensorFlow reference is not repeatable")
    return first


def session_config():
    """Use one CPU thread and clear GPU placement for reproducible baseline runs."""
    return tf.compat.v1.ConfigProto(device_count={"GPU": 0}, intra_op_parallelism_threads=1,
                                    inter_op_parallelism_threads=1)


def geometry_array(data, name, channel_axis):
    """Accept both legacy geometry layouts and feed exactly the saved graph rank."""
    value = np.asarray(data[name], dtype=np.float32)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2 or value.shape != np.shape(data["features"])[:2]:
        raise ValueError(f"{name} must match patch shape [N,V] or [N,V,1]")
    return value[..., None] if channel_axis else value


def graph_reference(prefix, config, data):
    """Execute the original saved graph, its actual loss, and all parameter gradients."""
    task = config["task"]
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(str(prefix) + ".meta", clear_devices=True)
        get = graph.get_tensor_by_name
        placeholders = [op.outputs[0] for op in graph.get_operations() if op.type == "Placeholder"]
        if task == "site":
            feeds = [geometry_array(data, "rho", False), geometry_array(data, "theta", False),
                     data["features"], geometry_array(data, "mask", True),
                     data["pos_idx"].astype(np.int32), data["neg_idx"].astype(np.int32),
                     data["labels"].astype(np.int32), data["indices"].astype(np.int32), 1.0]
            output = get("fully_connected_3/BiasAdd:0")
            loss = tf.reduce_sum(get("logistic_loss:0"))
        elif task == "ppi_search":
            feeds = [1.0, geometry_array(data, "rho", True), geometry_array(data, "theta", True),
                     data["features"], geometry_array(data, "mask", True)]
            output = get("fully_connected/BiasAdd:0")
            # The last Add before optimizer gradients is the original variance loss.
            before_gradients = []
            for op in graph.get_operations():
                if op.name.startswith("gradients/"):
                    break
                if op.type in ("Add", "AddV2") and op.name.startswith("add_"):
                    before_gradients.append(op)
            loss = before_gradients[-1].outputs[0]
        else:
            feeds = [1.0, geometry_array(data, "rho", True), geometry_array(data, "theta", True),
                     data["features"], geometry_array(data, "mask", True), data["labels"]]
            output = tf.expand_dims(get("fully_connected_2/BiasAdd:0"), 0)
            cross_entropy = [op for op in graph.get_operations()
                             if op.type == "SoftmaxCrossEntropyWithLogits"]
            loss = tf.reduce_sum(cross_entropy[0].outputs[0])
        fetches = dict(output=output, loss=loss)
        fetches["probability"] = (tf.sigmoid(output)[:, 0] if task == "site" else
                                  tf.nn.softmax(output) if task == "ligand" else output)
        for i in range(sum(config["feat_mask"])):
            fetches[f"trace/convs.{i}"] = get(("Relu" + (f"_{i}" if i else "")) + ":0")
        if task == "site":
            for i in range(config["n_conv_layers"] - 1):
                fetches[f"trace/extra_convs.{i}"] = get(f"Relu_{sum(config['feat_mask'])+i}:0")
        variables = tf.compat.v1.trainable_variables()
        for variable, grad in zip(variables, tf.gradients(loss, variables)):
            if grad is None:
                raise ValueError(f"Missing original graph gradient: {variable.name}")
            fetches["grad/" + variable.name[:-2]] = tf.convert_to_tensor(grad)
        for name in ("learning_rate", "beta1", "beta2", "epsilon"):
            fetches["optimizer/" + name] = get(f"Adam/{name}:0")
    feed = dict(zip(placeholders, feeds, strict=True))
    with tf.compat.v1.Session(graph=graph, config=session_config()) as session:
        saver.restore(session, str(prefix))
        first = session.run(fetches, feed)
        second = session.run(fetches, feed)
    failures = {k: dict(finite=bool(np.isfinite(first[k]).all()),
                         max_abs=float(np.max(np.abs(first[k] - second[k]))))
                for k in first if not np.array_equal(first[k], second[k])}
    if failures:
        raise AssertionError(f"Original TensorFlow graph is not repeatable: {failures}")
    optimizer = {name: float(first.pop("optimizer/" + name))
                 for name in ("learning_rate", "beta1", "beta2", "epsilon")}
    return first, optimizer


def export(prefix, destination, overrides=None, input_file=None):
    """Write weights, config, fixed inputs, references and provenance per checkpoint."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    weights, source_format = read_weights(prefix)
    config = infer_config(weights, overrides)
    invalid = {name: int(np.count_nonzero(~np.isfinite(value))) for name, value in weights.items()
               if np.asarray(value).dtype.kind in "fc" and not np.isfinite(value).all()}
    if invalid:
        report = dict(source=str(prefix), rejected=True, reason="Non-finite checkpoint tensors",
                      nonfinite_counts=invalid)
        (destination / "rejected.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"REJECTED {prefix}: {len(invalid)} tensors contain NaN/Inf", flush=True)
        return False
    if input_file is None:
        data = fixture(config)
    else:
        with np.load(input_file, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        data["features"] = np.asarray(data["features"], dtype=np.float32)
        if config["task"] in ("score", "alignment"):
            data["labels"] = np.asarray(data["labels"], dtype=np.int64)
        else:
            for name in ("rho", "theta", "mask"):
                data[name] = geometry_array(data, name, False)
            if config["task"] == "site":
                data.setdefault("pos_idx", np.flatnonzero(data["labels"][:, 0] == 1))
                data.setdefault("neg_idx", np.flatnonzero(data["labels"][:, 1] == 1))
                data["indices"] = np.asarray(data["indices"], dtype=np.int64)
            elif config["task"] == "ligand":
                if "labels" not in data or data["labels"].size != config["n_ligands"]:
                    raise ValueError("Ligand reference requires a fixed sampled pocket and one-hot labels [C]")
        for name in ("pos_idx", "neg_idx"):
            if name in data:
                data[name] = np.asarray(data[name], dtype=np.int64)
    if config["task"] in ("score", "alignment"):
        reference, optimizer = score_reference(weights, data, config), None
    else:
        reference, optimizer = graph_reference(prefix, config, data)
    np.savez_compressed(destination / "weights.npz", **weights)
    np.savez_compressed(destination / "fixture.npz", **data)
    np.savez_compressed(destination / "reference.npz", **reference)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    files = [Path(prefix)] if Path(prefix).is_file() else list(Path(prefix).parent.glob(Path(prefix).name + ".*"))
    manifest = dict(source=str(prefix), tensorflow=tf.__version__, numpy=np.__version__,
                    device="cpu", synthetic_input=input_file is None, repeat_exact=True,
                    source_format=source_format,
                    optimizer_slots_present=any("/.OPTIMIZER_SLOT/" in name for name in weights),
                    oracle="tf_ops_reconstruction" if config["task"] in ("score", "alignment") else "saved_meta_graph",
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    optimizer=optimizer, shapes={k: list(np.shape(v)) for k, v in weights.items()})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {prefix} -> {destination} ({config['task']})", flush=True)
    return True


def export_adam(destination, epsilon=1e-8, filename="adam_reference.npz"):
    """Save a small three-step TF Adam oracle with tiny and normal gradients."""
    graph = tf.Graph()
    initial = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    grads = np.array([[1e-9, -0.3, 2.0], [3e-8, 0.8, -1.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    with graph.as_default():
        parameter = tf.Variable(initial)
        gradient = tf.compat.v1.placeholder(tf.float32, [3])
        update = tf.compat.v1.train.AdamOptimizer(1e-3, epsilon=epsilon).apply_gradients([(gradient, parameter)])
        init = tf.compat.v1.global_variables_initializer()
    results = []
    with tf.compat.v1.Session(graph=graph, config=session_config()) as session:
        session.run(init)
        for grad in grads:
            session.run(update, {gradient: grad})
            results.append(session.run(parameter))
    np.savez(Path(destination) / filename, initial=initial, gradients=grads,
             updates=np.array(results), epsilon=np.asarray(epsilon))


def main():
    """Export one checkpoint or every repository checkpoint with its own reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON overrides for a custom checkpoint")
    parser.add_argument("--input", type=Path, help="Prepared numeric NPZ fixture for real data")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.repo):
        parser.error("Supply exactly one of --checkpoint or --repo")
    if args.repo:
        if args.config or args.input:
            parser.error("Custom config/input apply to a single checkpoint")
        prefixes = sorted(args.repo.glob("data/**/*.index"))
        results = []
        for index in prefixes:
            prefix = Path(str(index)[:-6])
            success = export(prefix, args.output / prefix.relative_to(args.repo))
            results.append(dict(source=str(prefix), exported=success))
        (args.output / "export_report.json").write_text(json.dumps(results, indent=2) + "\n")
        if not any(row["exported"] for row in results):
            raise SystemExit("No valid checkpoints exported")
    else:
        overrides = json.loads(args.config.read_text()) if args.config else None
        if not export(args.checkpoint, args.output, overrides, args.input):
            raise SystemExit("Checkpoint rejected; see rejected.json")
    export_adam(args.output)


if __name__ == "__main__":
    main()
