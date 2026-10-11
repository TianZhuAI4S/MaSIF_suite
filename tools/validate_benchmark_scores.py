"""Compare actual official Keras scores on captured legacy poses against PyTorch CPU/CUDA."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from masif_torch.checkpoints import load_checkpoint
from masif_torch.validation import deterministic


def comparison(actual, expected):
    """Record strict tolerance, bitwise equality, absolute error and finiteness separately."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError('Benchmark score shapes differ')
    return dict(shape=list(actual.shape), passed=bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-4)),
                array_equal=bool(np.array_equal(actual, expected)),
                max_absolute_error=float(np.max(np.abs(actual - expected))),
                all_finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all()))


def main():
    """Require both CPU and available CUDA inference to meet the declared tolerances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    deterministic()
    with np.load(args.reference, allow_pickle=False) as archive:
        features, expected_probability, expected_logits = archive['features'], archive['probability'], archive['logits']
    groups = []
    for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        model, _ = load_checkpoint(args.checkpoint, device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(features).to(device))
            probability = logits.softmax(-1)
        groups.append(dict(device=device, logits=comparison(logits.cpu().numpy(), expected_logits),
                           probability=comparison(probability.cpu().numpy(), expected_probability)))
    report = dict(poses=len(features), features_shape=list(features.shape), atol=1e-5, rtol=1e-4,
                  precision='FP32; TF32 disabled; deterministic algorithms enabled',
                  original_registration='Open3D0.5 RANSAC2000/max_validation500 + ICP1Å', groups=groups)
    report['passed'] = all(group[key]['passed'] and group[key]['all_finite'] for group in groups for key in ('logits', 'probability'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Fixed legacy pose scores failed strict TensorFlow/PyTorch comparison')


if __name__ == '__main__':
    main()
