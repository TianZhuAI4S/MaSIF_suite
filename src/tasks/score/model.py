"""Score task neural-network implementation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...shared.layers import linear, initialize_layer, GaussianConv, PatchEncoder, KerasBatchNorm1d


class ScoreNN(nn.Module):
    """Second-stage alignment classifier, with Keras-compatible BN epsilon."""

    def __init__(self, channels=(3, 8, 16, 32, 64, 128, 256)):
        """Build the original classifier, sharing its math with the IPA variant."""
        super().__init__()
        self.config = dict(task="score")
        self.in_features = channels[0]
        self.convs = nn.ModuleList([initialize_layer(nn.Conv1d(a, b, 1)) for a, b in zip(channels, channels[1:])])
        self.norms = nn.ModuleList([KerasBatchNorm1d(b, eps=1e-3, momentum=0.01)
                                   for b in channels[1:]])
        widths = (256, 128, 64, 32, 16, 8, 4, 2)
        self.dense = nn.ModuleList([initialize_layer(nn.Linear(a, b)) for a, b in zip(widths, widths[1:])])

    def forward(self, features):
        """Return [N,2] logits from alignment features [N,points,3]."""
        if features.ndim != 3 or features.shape[-1] != self.in_features or min(features.shape[:2]) == 0:
            raise ValueError(f"Alignment features must have shape [N,points,{self.in_features}]")
        if not torch.isfinite(features).all():
            raise ValueError("Alignment features must be finite")
        x = features.transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)))
        x = x.mean(2)
        for layer in self.dense[:-1]:
            x = F.relu(linear(layer, x))
        return linear(self.dense[-1], x)
