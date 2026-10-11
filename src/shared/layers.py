"""Shared numerical primitives; Å distances and radian angles retain legacy meanings."""

import math
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

def linear(layer: nn.Linear, x: Tensor) -> Tensor:
    """Keep the legacy MatMul then bias-add ordering instead of fused addmm."""
    return x @ layer.weight.T + layer.bias


def initialize_layer(layer):
    """Use legacy Xavier kernels and zero biases; framework RNG streams differ."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class GaussianConv(nn.Module):
    """Rotation-pooled Gaussian convolution of [patch, vertex, feature] inputs."""

    def __init__(self, max_rho, n_thetas, n_rhos, n_rotations, in_features=1,
                 out_features=None, legacy_site_layer=False):
        """Initialize the original radial-major grid and optional site quirks."""
        super().__init__()
        self.n_rotations = n_rotations
        self.n_gauss = n_thetas * n_rhos
        self.in_features = in_features
        rho, theta = np.meshgrid(np.linspace(0, max_rho, n_rhos + 1)[1:],
                                 np.linspace(0, 2 * np.pi, n_thetas + 1)[:-1])
        rho = torch.from_numpy(rho.T.reshape(1, -1).astype(np.float32))
        theta = torch.from_numpy(theta.T.reshape(1, -1).astype(np.float32))
        self.mu_rho = nn.Parameter(rho)
        # Site's l2/l3 attribute names were swapped in the TF graph. Preserve use.
        self.sigma_rho = nn.Parameter(theta.clone() if legacy_site_layer else
                                      torch.full_like(rho, max_rho / 8))
        self.mu_theta = nn.Parameter(torch.full_like(rho, max_rho / 8)
                                     if legacy_site_layer else theta)
        self.sigma_theta = nn.Parameter(torch.ones_like(rho))
        out_features = out_features or self.n_gauss
        self.weight = nn.Parameter(torch.empty(in_features * self.n_gauss, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: Tensor, rho: Tensor, theta: Tensor, mask: Tensor) -> Tensor:
        """Normalize masked weights over vertices, pool rotations, then ReLU."""
        batch, vertices, _ = features.shape
        rho = rho.reshape(-1, 1)
        theta = theta.reshape(-1, 1)
        outputs = []
        for rotation in range(self.n_rotations):
            angle = torch.remainder(theta + rotation * 2 * math.pi / self.n_rotations,
                                    2 * math.pi)
            radial = torch.exp(-(rho - self.mu_rho).square() /
                               (self.sigma_rho.square() + 1e-5))
            angular = torch.exp(-(angle - self.mu_theta).square() /
                                (self.sigma_theta.square() + 1e-5))
            activation = (radial * angular).reshape(batch, vertices, self.n_gauss) * mask
            activation = activation / (activation.sum(dim=1, keepdim=True) + 1e-5)
            # Flatten feature then Gaussian, matching TensorFlow reshape ordering.
            desc = (activation.unsqueeze(2) * features.unsqueeze(3)).sum(dim=1)
            outputs.append(desc.reshape(batch, -1) @ self.weight + self.bias)
        # amax splits gradients across ties, as TF reduce_max does.
        return F.relu(torch.stack(outputs).amax(dim=0))


class PatchEncoder(nn.Module):
    """Shared per-feature convolutions and strict geometry input validation."""

    def __init__(self, task, max_rho, n_thetas, n_rhos, n_rotations, feat_mask):
        """Store serializable configuration and initialize one filter per feature."""
        super().__init__()
        if max_rho <= 0 or min(n_thetas, n_rhos, n_rotations) < 1:
            raise ValueError("Radius and grid dimensions must be positive")
        if not feat_mask or any(v not in (0, 1) for v in feat_mask) or not sum(feat_mask):
            raise ValueError("feat_mask must select at least one feature using 0/1 values")
        self.config = dict(task=task, max_rho=max_rho, n_thetas=n_thetas,
                           n_rhos=n_rhos, n_rotations=n_rotations, feat_mask=list(feat_mask))
        self.n_feat = int(sum(feat_mask))
        self.n_gauss = n_thetas * n_rhos
        self.convs = nn.ModuleList([GaussianConv(max_rho, n_thetas, n_rhos, n_rotations)
                                    for _ in range(self.n_feat)])

    def validate(self, features, rho, theta, mask):
        """Accept [N,V] coordinates or [N,V,1], with matching device and dtype."""
        if features.ndim != 3 or features.shape[-1] != self.n_feat:
            raise ValueError(f"features must have shape [N,V,{self.n_feat}]")
        if not features.shape[0] or not features.shape[1]:
            raise ValueError("Empty patches or vertex dimensions are invalid")
        tensors = [features]
        for name, value in (("rho", rho), ("theta", theta), ("mask", mask)):
            value = value.unsqueeze(-1) if value.ndim == 2 else value
            if value.shape != features.shape[:2] + (1,):
                raise ValueError(f"{name} must have shape [N,V] or [N,V,1]")
            tensors.append(value)
        parameter = self.convs[0].weight
        for value in tensors:
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ValueError("Inputs and model must use the same device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError("Inputs must be finite")
        if (tensors[1] < 0).any() or (tensors[3] < 0).any() or (tensors[3] > 1).any():
            raise ValueError("rho must be nonnegative and mask must lie in [0,1]")
        return tensors

    def encode(self, features, rho, theta, mask):
        """Concatenate feature-specific descriptors in the legacy feature order."""
        return torch.cat([conv(features[..., i:i + 1], rho, theta, mask)
                          for i, conv in enumerate(self.convs)], dim=1)


class KerasBatchNorm1d(nn.BatchNorm1d):
    """Use population variance for both normalization and Keras moving statistics."""

    def forward(self, x):
        """Match Keras momentum=.99 and epsilon=.001 in train and eval modes."""
        if not self.training:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight,
                                self.bias, training=False, eps=self.eps)
        mean = x.mean(dim=(0, 2))
        variance = x.var(dim=(0, 2), correction=0)
        with torch.no_grad():
            self.num_batches_tracked.add_(1)
            self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(variance, alpha=self.momentum)
        return ((x - mean[None, :, None]) / (variance[None, :, None] + self.eps).sqrt() *
                self.weight[None, :, None] + self.bias[None, :, None])
