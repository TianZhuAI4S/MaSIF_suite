"""Site task neural-network implementation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...shared.layers import linear, initialize_layer, GaussianConv, PatchEncoder, KerasBatchNorm1d


class MaSIF_site(PatchEncoder):
    """MaSIF-site logits and sigmoid interface scores."""

    def __init__(self, max_rho=9.0, n_thetas=4, n_rhos=3, n_rotations=4,
                 feat_mask=(1, 1, 1, 1, 1), n_conv_layers=3):
        """Initialize supported 1–3 convolution layers, preserving legacy l2/l3 use."""
        if n_conv_layers not in (1, 2, 3):
            raise ValueError("Only site layers 1–3 are supported; legacy layer 4 has incompatible shapes")
        super().__init__("site", max_rho, n_thetas, n_rhos, n_rotations, feat_mask)
        self.config["n_conv_layers"] = n_conv_layers
        self.fc1 = initialize_layer(nn.Linear(self.n_feat * self.n_gauss, self.n_gauss))
        self.fc2 = initialize_layer(nn.Linear(self.n_gauss, self.n_feat))
        self.extra_convs = nn.ModuleList([
            GaussianConv(max_rho, n_thetas, n_rhos, n_rotations, self.n_feat,
                         self.n_feat * self.n_gauss, legacy_site_layer=True)
            for _ in range(n_conv_layers - 1)])
        self.fc3 = initialize_layer(nn.Linear(self.n_feat, n_thetas))
        self.fc4 = initialize_layer(nn.Linear(n_thetas, 2))

    def forward(self, features, rho, theta, mask, indices=None):
        """Gather complete-surface neighbors between layers and return [N,2] logits."""
        features, rho, theta, mask = self.validate(features, rho, theta, mask)
        if self.extra_convs:
            if indices is None or indices.shape != features.shape[:2]:
                raise ValueError("Multilayer site requires indices [N,V] for the full surface")
            if indices.dtype != torch.int64 or indices.device != features.device:
                raise ValueError("indices must be int64 on the model device")
            if (indices < 0).any() or (indices >= features.shape[0]).any():
                raise ValueError("Neighbor indices must refer to rows of this surface")
        encoded = self.encode(features, rho, theta, mask)
        desc = F.relu(linear(self.fc2, F.relu(linear(self.fc1, encoded))))
        for conv in self.extra_convs:
            desc = conv(desc[indices], rho, theta, mask)
            desc = desc.reshape(features.shape[0], self.n_feat, self.n_gauss).mean(2)
        return linear(self.fc4, F.relu(linear(self.fc3, desc)))

    def loss(self, logits, labels, pos_idx=None, neg_idx=None):
        """Sum two-output sigmoid BCE, matching TF minimize's implicit sum."""
        if labels.shape != logits.shape:
            raise ValueError("Site labels must be one-hot [N,2], positive in column 0")
        if pos_idx is not None or neg_idx is not None:
            if pos_idx is None or neg_idx is None:
                raise ValueError("Supply both positive and negative indices")
            for index in (pos_idx, neg_idx):
                if index.ndim != 1 or index.dtype != torch.int64 or index.device != logits.device:
                    raise ValueError("Loss indices must be one-dimensional int64 on the model device")
                if (index < 0).any() or (index >= logits.shape[0]).any():
                    raise ValueError("Loss index outside the surface")
            selected = torch.cat((pos_idx, neg_idx))
            logits, labels = logits[selected], labels[selected]
        return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="sum")
