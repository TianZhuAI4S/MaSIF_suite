"""Ligand task neural-network implementation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...shared.layers import linear, initialize_layer, GaussianConv, PatchEncoder, KerasBatchNorm1d


class MaSIF_ligand(PatchEncoder):
    """MaSIF-ligand pocket classification using a noncentered Gram matrix."""

    def __init__(self, max_rho=12.0, n_ligands=7, n_thetas=16, n_rhos=5,
                 n_rotations=16, feat_mask=(1, 1, 1, 1, 1), dropout=0.0):
        """Initialize pocket layers; legacy training uses dropout probability zero."""
        super().__init__("ligand", max_rho, n_thetas, n_rhos, n_rotations, feat_mask)
        if n_ligands < 2 or not 0 <= dropout < 1:
            raise ValueError("n_ligands must be >=2 and dropout must lie in [0,1)")
        self.config.update(n_ligands=n_ligands, dropout=dropout)
        self.fc1 = initialize_layer(nn.Linear(self.n_feat * self.n_gauss, self.n_gauss))
        self.fc2 = initialize_layer(nn.Linear(self.n_gauss ** 2, 64))
        self.fc3 = initialize_layer(nn.Linear(64, n_ligands))
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, rho, theta, mask, indices=None):
        """Aggregate one pocket's N patches and return legacy [1,1,C] logits."""
        features, rho, theta, mask = self.validate(features, rho, theta, mask)
        desc = F.relu(linear(self.fc1, self.encode(features, rho, theta, mask)))
        gram = (desc.T @ desc / desc.shape[0]).reshape(1, -1)
        hidden = F.relu(linear(self.fc2, self.dropout(gram)))
        return linear(self.fc3, hidden).unsqueeze(0)

    def loss(self, logits, labels):
        """Compute summed one-hot softmax cross entropy for one pocket."""
        labels = labels.reshape_as(logits).to(logits.dtype)
        return -(labels * F.log_softmax(logits, dim=-1)).sum()
