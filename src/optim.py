"""Adam with the original TensorFlow v1 epsilon and beta-power conventions."""

import torch


class TFAdam(torch.optim.Optimizer):
    """Use TF v1 Adam's uncorrected second moment denominator."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """Initialize global beta powers separately from per-parameter moments."""
        if lr <= 0 or eps <= 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("Invalid Adam learning rate, epsilon, or beta")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    def add_param_group(self, param_group):
        """Validate group hyperparameters and initialize that group's own beta powers."""
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group["lr"] <= 0 or group["eps"] <= 0 or any(not 0 <= b < 1 for b in group["betas"]):
            raise ValueError("Invalid Adam group hyperparameters")
        group["beta1_power"] = torch.tensor(group["betas"][0], dtype=torch.float32)
        group["beta2_power"] = torch.tensor(group["betas"][1], dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one update; persist beta powers in optimizer state checkpoints."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            step_lr = (group["lr"] * (1 - group["beta2_power"]).sqrt() /
                       (1 - group["beta1_power"]))
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise ValueError("TFAdam supports dense MaSIF gradients only")
                state = self.state[parameter]
                if not state:
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)
                # TF computes (1-beta) using float32 beta tensors, not Python doubles.
                b1 = parameter.new_tensor(beta1)
                b2 = parameter.new_tensor(beta2)
                state["m"].add_((grad - state["m"]) * (1 - b1))
                state["v"].add_((grad.square() - state["v"]) * (1 - b2))
                parameter.sub_(step_lr.to(parameter.device) * state["m"] /
                               (state["v"].sqrt() + group["eps"]))
            group["beta1_power"] *= beta1
            group["beta2_power"] *= beta2
        return loss
