"""Execute official IPA Keras source with builder API compatibility adjustments only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from legacy_weights import read_weights


def source_model(source, config, weights):
    """Build unchanged original layer sequence and assign original checkpoint tensors."""
    text = Path(source).read_text()
    for line in ("import tensorflow as tf", "from scipy.spatial import cKDTree",
                 "from sklearn.model_selection import train_test_split", "from sklearn.metrics import roc_auc_score"):
        text = text.replace(line, "", 1)
    # Modern Keras renamed these keyword arguments; the numerical values stay identical.
    text = text.replace("keras.regularizers.l2(l=", "keras.regularizers.l2(l2=")
    text = text.replace("keras.optimizers.Adam(lr=", "keras.optimizers.Adam(learning_rate=")
    proxy = SimpleNamespace(ConfigProto=tf.compat.v1.ConfigProto, Session=tf.compat.v1.Session,
                            random=SimpleNamespace(set_random_seed=tf.random.set_seed), nn=tf.nn)
    namespace = dict(tf=proxy)
    exec(compile(text, str(source), "exec"), namespace)
    model = namespace["AlignmentEvaluationNN"]("unused", config["selected_features"], config["max_npoints"],
                                               config["n_negatives"], config["n_positives"])
    model.model.build((None, config["max_npoints"], len(config["selected_features"])))
    for index, layer in enumerate(model.model.layers):
        for variable in layer.weights:
            variable.assign(weights[f"layer-{index}/{variable.name}/.ATTRIBUTES/VARIABLE_VALUE"])
    # Direct eager dispatch avoids TF Dataset threads while executing the same original layers.
    model.model.predict = lambda value: model.model(value, training=False).numpy()
    return model


def export_cases(source, checkpoint, config, output, real_features=None):
    """Capture original eval_model RNG/padding/importance and original Keras training gradients."""
    output.mkdir(parents=True, exist_ok=True)
    weights, _ = read_weights(checkpoint)
    model = source_model(source, config, weights)
    rng = np.random.RandomState(7)
    # Evaluate two consecutive oversized packets against one global legacy sampling stream.
    cases = [("padding", 7, -1.0), ("exact", config["max_npoints"], 1.0),
             ("trim_1", config["max_npoints"] + 13, 1.0),
             ("trim_2", config["max_npoints"] + 13, 1.0), ("degenerate", 5, -1.0)]
    np.random.seed(42)
    for name, count, cutoff in cases:
        raw = rng.uniform(0.1, 2.0, (count, 4)).astype(np.float64)
        if name == "degenerate":
            raw.fill(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            probability, importance = model.eval_model(raw[:, config["selected_features"]], cutoff)
        np.savez_compressed(output / f"{name}.npz", raw=raw, probability=probability,
                            importance=importance, cutoff=cutoff)
    if real_features is not None:
        with np.load(real_features, allow_pickle=False) as data:
            raw = data["features"]
        probability, importance = model.eval_model(raw[:, config["selected_features"]], 0.9)
        np.savez_compressed(output / "real.npz", raw=raw, probability=probability,
                            importance=importance, cutoff=0.9)
    batch = rng.normal(size=(8, config["max_npoints"], len(config["selected_features"]))).astype(np.float32)
    labels = np.arange(8) % 2
    with tf.GradientTape() as tape:
        probability = model.model(batch, training=True)
        per_sample = tf.keras.losses.sparse_categorical_crossentropy(labels, probability)
        class_weights = tf.constant([1/config["n_negatives"], 1/config["n_positives"]], tf.float32)
        loss = tf.reduce_mean(per_sample * tf.gather(class_weights, labels))
    gradients = tape.gradient(loss, model.model.trainable_variables)
    result = dict(features=batch, labels=labels, probability=probability.numpy(), loss=loss.numpy())
    iterator = iter(gradients)
    for index, layer in enumerate(model.model.layers):
        for variable in layer.weights:
            key = f"layer-{index}/{variable.name}/.ATTRIBUTES/VARIABLE_VALUE"
            if variable.trainable:
                result["grad/" + key] = next(iterator).numpy()
            elif "moving_" in variable.name:
                result["moving/" + key] = variable.numpy()
    np.savez_compressed(output / "training.npz", **result)
    # A fresh original model isolates the optimizer audit from previous BN updates.
    update_model = source_model(source, config, weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-7)
    updates = dict(features=batch, labels=labels)
    for step in range(3):
        with tf.GradientTape() as tape:
            probability = update_model.model(batch, training=True)
            losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probability)
            loss = tf.reduce_mean(losses * tf.gather(class_weights, labels))
        gradients = tape.gradient(loss, update_model.model.trainable_variables)
        gradient_iterator = iter(gradients)
        for index, layer in enumerate(update_model.model.layers):
            for variable in layer.trainable_weights:
                updates[f"gradient{step+1}/layer-{index}/{variable.name}/.ATTRIBUTES/VARIABLE_VALUE"] = next(gradient_iterator).numpy()
        optimizer.apply_gradients(zip(gradients, update_model.model.trainable_variables))
        for index, layer in enumerate(update_model.model.layers):
            for variable in layer.weights:
                updates[f"step{step+1}/layer-{index}/{variable.name}/.ATTRIBUTES/VARIABLE_VALUE"] = variable.numpy()
        updates[f"probability{step+1}"] = update_model.model(batch, training=False).numpy()
    np.savez_compressed(output / "adam.npz", **updates)
    update_model.session.close()
    checkpoint = Path(checkpoint)
    files = [checkpoint] if checkpoint.is_file() else sorted(checkpoint.parent.glob(checkpoint.name + ".*"))
    (output / "manifest.json").write_text(json.dumps(dict(source=str(source), checkpoint=str(checkpoint),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        checkpoint_sha256={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files},
        oracle="official_keras_source", tensorflow=tf.__version__, config=config,
        compatibility_changes=["TF session/seed names", "Keras l2/lr keyword names", "eager predict dispatch"]), indent=2) + "\n")
    model.session.close()


def main():
    """Use exported config with the original source and original weight prefix or HDF5."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-features", type=Path, help="Fixed-pose real IPA raw features [points,4]")
    args = parser.parse_args()
    export_cases(args.source, args.checkpoint, json.loads(args.config.read_text()), args.output, args.real_features)


if __name__ == "__main__":
    main()
