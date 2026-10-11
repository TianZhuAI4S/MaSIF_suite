"""Run original TF1 model source under TF2 compat.v1 using builder-only shims."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tensorflow
from export_tf import export, export_adam, fixture, session_config

tf = tensorflow.compat.v1


def fully_connected(inputs, outputs, activation_fn):
    """Replace removed contrib's builder with the same variables and primitive ops."""
    with tf.variable_scope(None, default_name="fully_connected"):
        width = inputs.shape.as_list()[-1]
        weight = tf.get_variable("weights", [width, outputs], initializer=tf.glorot_uniform_initializer())
        bias = tf.get_variable("biases", [outputs], initializer=tf.zeros_initializer())
        return activation_fn(tf.nn.bias_add(tf.matmul(inputs, weight), bias))


def skip_parameter_print(self):
    """Avoid obsolete Dimension.value printing, which does not change computation."""
    return sum(np.prod(v.shape.as_list()) for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))


def source_class(source, task):
    """Execute the original source with only its TensorFlow import redirected."""
    name = "MaSIF_" + task
    text = (Path(source) / "masif_modules" / (name + ".py")).read_text()
    text = text.replace("import tensorflow as tf", "", 1)
    namespace = {"tf": tf}
    exec(compile(text, name + ".py", "exec"), namespace)
    cls = namespace[name]
    cls.count_number_parameters = skip_parameter_print
    return cls


def make_bn_oracle(output):
    """Record Keras non-fused BN training, moving statistics, gradients and eval."""
    graph = tf.Graph()
    value = np.array([[[0, 2, 3, 5], [4, 4, 4, 4]],
                      [[1, 4, 6, 8], [-1, 0, 1, 2]]], dtype=np.float32)
    with graph.as_default():
        x = tf.Variable(value)
        mean, variance = tf.nn.moments(x, axes=[0, 2])
        y = tf.nn.batch_normalization(x, mean[None, :, None], variance[None, :, None],
                                      tf.zeros([1, 2, 1]), tf.ones([1, 2, 1]), 1e-3)
        new_mean = mean * 0.01
        new_variance = variance * 0.01 + 0.99
        evaluation = tf.nn.batch_normalization(x, new_mean[None, :, None],
                                               new_variance[None, :, None],
                                               tf.zeros([1, 2, 1]), tf.ones([1, 2, 1]), 1e-3)
        gradient = tf.gradients(tf.reduce_sum(y * y), x)[0]
        init = tf.global_variables_initializer()
    with tf.Session(graph=graph, config=session_config()) as session:
        session.run(init)
        results = session.run(dict(train=y, moving_mean=new_mean, moving_variance=new_variance,
                                   evaluation=evaluation, gradient=gradient))
    np.savez(Path(output) / "bn_reference.npz", input=value, **results)


def make_records_oracle(output):
    """Create one synthetic legacy TFRecord and exact arrays for conversion tests."""
    config = dict(task="ligand", max_rho=12.0, feat_mask=[1]*5, n_ligands=7)
    data = fixture(config)
    record = {key: data[key] for key in ("features", "rho", "theta", "mask")}
    record["mask"] = record["mask"][..., None]
    record["pocket_labels"] = np.full((32, 1), 3, dtype=np.int64)
    record["pdb"] = np.asarray("SYNTHETIC_A")
    fields = {}
    for source, target in (("input_feat", "features"), ("rho_wrt_center", "rho"),
                           ("theta_wrt_center", "theta"), ("mask", "mask"),
                           ("pocket_labels", "pocket_labels")):
        value = record[target]
        fields[source + "_shape"] = tf.train.Feature(int64_list=tf.train.Int64List(value=value.shape))
        fields[source] = (tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))
                          if target == "pocket_labels" else
                          tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1))))
    fields["pdb"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"SYNTHETIC_A"]))
    example = tf.train.Example(features=tf.train.Features(feature=fields))
    with tf.io.TFRecordWriter(str(Path(output) / "synthetic.tfrecord")) as writer:
        writer.write(example.SerializeToString())
    np.savez(Path(output) / "record_expected.npz", **record)


def main():
    """Build small 1/2/3-layer site, search, and ligand oracles from original source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tf.contrib = SimpleNamespace(layers=SimpleNamespace(xavier_initializer=tf.glorot_uniform_initializer,
                                                       fully_connected=fully_connected))
    for task, layers in (("site", 1), ("site", 2), ("site", 3), ("ppi_search", 1), ("ligand", 1)):
        config = dict(task=task, max_rho=9.0 if task == "site" else 12.0, n_thetas=2,
                      n_rhos=2, n_rotations=3, feat_mask=[1, 1, 1])
        kwargs = {k: v for k, v in config.items() if k != "task"}
        kwargs["idx_gpu"] = "/cpu:0"
        if task == "site":
            config["n_conv_layers"] = kwargs["n_conv_layers"] = layers
        if task == "ligand":
            config["n_ligands"] = kwargs["n_ligands"] = 3
            config["dropout"] = 0.0
        destination = args.output / f"{task}_{layers}"
        destination.mkdir(parents=True, exist_ok=True)
        cls = source_class(args.source, task)
        if task == "ligand":
            with tf.Graph().as_default():
                with tf.Session(config=session_config()) as session:
                    model = cls(session=session, **kwargs)
                    model.saver.save(model.session, str(destination / "original"))
        else:
            model = cls(**kwargs)
            model.saver.save(model.session, str(destination / "original"))
            model.session.close()
        export(destination / "original", destination, config)
    export_adam(args.output)
    export_adam(args.output, epsilon=1e-7, filename="score_adam_reference.npz")
    make_bn_oracle(args.output)
    make_records_oracle(args.output)


if __name__ == "__main__":
    main()
