"""Evaluate captured original-registration poses with the unchanged official Keras layers."""

import argparse
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import numpy as np
import tensorflow as tf
from export_ipa_source import source_model
from legacy_weights import read_weights


def main():
    """Export probabilities and pre-softmax logits for an explicit prepared pose batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    weights, _ = read_weights(args.checkpoint)
    model = source_model(args.source, config, weights)
    with np.load(args.input, allow_pickle=False) as packet:
        features = packet['features']
    probability = model.model(features, training=False).numpy()
    hidden_model = tf.keras.Model(inputs=model.model.inputs, outputs=model.model.layers[-2].output)
    hidden = hidden_model([features], training=False)
    layer = model.model.layers[-1]
    logits = (tf.matmul(hidden, layer.kernel) + layer.bias).numpy()
    if not np.isfinite(probability).all() or not np.isfinite(logits).all():
        raise ValueError('Original Keras produced nonfinite scores')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, probability=probability, logits=logits)
    provenance = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      original_source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      tensorflow=tf.__version__, numpy=np.__version__, poses=len(features))
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Exported official Keras scores for {len(features)} original-registration poses')


if __name__ == '__main__':
    main()
