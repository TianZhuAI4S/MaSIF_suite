"""Alignment task neural-network implementation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...shared.layers import linear, initialize_layer, GaussianConv, PatchEncoder, KerasBatchNorm1d
from ..score.model import ScoreNN


class AlignmentEvaluationNN(ScoreNN):
    """Seed/neosurf IPA: five Conv1D blocks and explicit point sampling metadata."""

    def __init__(self, selected_features=(0, 1, 2, 3), max_npoints=200, n_negatives=100,
                 n_positives=1):
        """Reuse the BN/dense implementation while retaining original IPA architecture."""
        if (not selected_features or len(set(selected_features)) != len(selected_features) or
                any(i not in range(4) for i in selected_features)):
            raise ValueError("IPA selected_features must select unique columns from the four raw features")
        if max_npoints not in (100, 200) or min(n_negatives, n_positives) <= 0:
            raise ValueError("IPA requires 100/200 points and positive class frequencies")
        super().__init__((len(selected_features), 16, 32, 64, 128, 256))
        self.config = dict(task="alignment", selected_features=list(selected_features),
                           max_npoints=max_npoints, n_negatives=n_negatives, n_positives=n_positives)
        self._sampling_rng = np.random.RandomState(42)

    def prepare(self, features, rng):
        """Select raw columns, pad with zeros or use legacy RandomState.choice rows."""
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != 4 or not len(features):
            raise ValueError("Raw IPA features must have shape [points,4]")
        if not np.isfinite(features).all():
            raise ValueError("IPA features must be finite; identical descriptors have undefined inverse distance")
        count = self.config["max_npoints"]
        prepared = np.zeros((1, count, self.in_features), dtype=np.float32)
        selected = (np.arange(len(features)) if len(features) <= count else
                    rng.choice(len(features), count, replace=False))
        prepared[0, :len(selected)] = features[selected][:, self.config["selected_features"]]
        return prepared, selected

    @torch.no_grad()
    def eval_model(self, features, nn_score_cutoff=0.9, rng=None):
        """Return positive probability and legacy signed leave-one-point-out importance."""
        self.eval()
        rng = self._sampling_rng if rng is None else rng
        prepared, _ = self.prepare(features, rng)
        device = next(self.parameters()).device
        values = torch.from_numpy(prepared).to(device)
        probability = self(values).softmax(-1)[:, 1:2].cpu().numpy()
        importance = np.zeros(len(features))
        if len(features) <= self.config["max_npoints"] and probability[0, 0] > nn_score_cutoff:
            for index in range(len(features)):
                ablated = values.clone()
                ablated[0, index] = 0
                importance[index] = (self(ablated).softmax(-1)[0, 1].cpu().item() - probability[0, 0])
            # Preserve the original std/max normalization, including NaN in degenerate cases.
            with np.errstate(divide="ignore", invalid="ignore"):
                const = np.max(np.abs(importance)) / np.std(importance)
                importance = importance / (const * np.std(importance))
        return probability, importance
