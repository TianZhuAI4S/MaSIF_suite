"""Read TF object checkpoints and actual legacy Keras HDF5 files without guessing."""

from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


def read_weights(prefix):
    """Normalize HDF5 layer ordering to TF object names; retain all checkpoint tensors."""
    prefix = Path(prefix)
    if prefix.is_file() and h5py.is_hdf5(prefix):
        weights = {}
        with h5py.File(prefix, "r") as archive:
            root = archive["model_weights"] if "model_weights" in archive else archive
            for index, raw in enumerate(root.attrs["layer_names"]):
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
                def collect(path, value):
                    """Translate nested Keras weight datasets using their final variable names."""
                    if isinstance(value, h5py.Dataset):
                        field = path.rsplit("/", 1)[-1].removesuffix(":0")
                        key = f"layer-{index}/{field}/.ATTRIBUTES/VARIABLE_VALUE"
                        if key in weights:
                            raise ValueError(f"Duplicate HDF5 variable: {key}")
                        weights[key] = np.asarray(value)
                root[name].visititems(collect)
        return weights, "keras_hdf5"
    reader = tf.train.load_checkpoint(str(prefix))
    return {name: reader.get_tensor(name) for name in reader.get_variable_to_shape_map()}, "tensorflow_checkpoint"
