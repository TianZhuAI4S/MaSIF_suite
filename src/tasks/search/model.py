"""Search task neural-network implementation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...shared.layers import linear, initialize_layer, GaussianConv, PatchEncoder, KerasBatchNorm1d


class MaSIF_ppi_search(PatchEncoder):
    """MaSIF-search descriptors without L2 normalization."""

    def __init__(self, max_rho=12.0, n_thetas=16, n_rhos=5, n_rotations=16,
                 feat_mask=(1, 1, 1, 1, 1)):
        """Initialize the legacy search grid and final identity projection."""
        super().__init__("ppi_search", max_rho, n_thetas, n_rhos, n_rotations, feat_mask)
        self.fc = initialize_layer(nn.Linear(self.n_feat * self.n_gauss, self.n_gauss))

    def forward(self, features, rho, theta, mask, indices=None):
        """Return [N,G] unnormalized patch descriptors."""
        features, rho, theta, mask = self.validate(features, rho, theta, mask)
        return linear(self.fc, self.encode(features, rho, theta, mask))

    def loss(self, descriptors, pos_thresh=0.0, neg_thresh=10.0):
        """Use mean plus population variance of thresholded four-block distances."""
        if descriptors.shape[0] == 0 or descriptors.shape[0] % 4:
            raise ValueError("Search training requires four equally sized patch blocks")
        pos, binder, neg, neg2 = descriptors.chunk(4)
        positive = F.relu((binder - pos).square().sum(1) - pos_thresh)
        negative = F.relu(neg_thresh - (neg - neg2).square().sum(1))
        return (positive.mean() + positive.var(correction=0) +
                negative.mean() + negative.var(correction=0))
