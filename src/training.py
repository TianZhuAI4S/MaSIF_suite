"""Explicit native PyTorch training, validation and checkpoint resumption."""

import json
from pathlib import Path
import numpy as np
import torch
from .checkpoints import save_checkpoint
from .data import tensors
from .optim import TFAdam


class Trainer:
    """Run single-structure updates with original MaSIF loss and Adam semantics."""

    def __init__(self, model, device="cpu", learning_rate=None, class_weights=None):
        """Move the model once and select original per-task default learning rates."""
        self.model, self.device = model.to(device), device
        rate = learning_rate or (1e-4 if model.config["task"] in ("ligand", "score", "alignment") else 1e-3)
        self.optimizer = TFAdam(model.parameters(), lr=rate,
                                eps=1e-7 if model.config["task"] in ("score", "alignment") else 1e-8)
        # Original second-stage training uses 200 negatives per positive alignment.
        default = ((1 / model.config["n_negatives"], 1 / model.config["n_positives"])
                   if model.config["task"] == "alignment" else (0.005, 1.0))
        weights = default if class_weights is None else class_weights
        self.class_weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.step = 0
        self.history, self.best_metric = [], float("inf")

    def restore(self, payload):
        """Restore moments, class weights and random state for an exact native resume."""
        if payload.get("optimizer") is None:
            raise ValueError("Checkpoint has no optimizer state")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = payload.get("step", 0)
        self.history = list(payload.get("history", []))
        self.best_metric = payload.get("best_metric", float("inf"))
        if "class_weights" in payload:
            self.class_weights = torch.tensor(payload["class_weights"], device=self.device)
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if payload.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng_state_all"]])
        return payload.get("epoch", -1) + 1

    def compute(self, packet):
        """Compute a packet's logits and scalar loss using the task's label rules."""
        data = tensors(packet, self.device)
        task = self.model.config["task"]
        if task in ("score", "alignment"):
            output = self.model(data["features"])
            labels = data["labels"].reshape(-1).long()
            per_sample = torch.nn.functional.cross_entropy(output, labels, reduction="none")
            # Keras divides weighted losses by sample count, not sum of class weights.
            loss = ((per_sample * self.class_weights[labels]).mean() if self.model.training
                    else per_sample.mean())
        else:
            output = self.model(data["features"], data["rho"], data["theta"], data["mask"],
                                data.get("indices"))
            loss = (self.model.loss(output, data["labels"], data.get("pos_idx"), data.get("neg_idx"))
                    if task == "site" else self.model.loss(output, data["labels"])
                    if task == "ligand" else self.model.loss(output))
        if not torch.isfinite(output).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite model output or loss")
        return output, loss

    def train_step(self, packet):
        """Backpropagate once, reject non-finite gradients, and apply TF-style Adam."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.compute(packet)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Missing or non-finite gradient: {name}")
        self.optimizer.step()
        self.step += 1
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, dataset):
        """Evaluate held-out packets without parameter or BN state updates."""
        self.model.eval()
        losses, counts = [], []
        for index in range(len(dataset)):
            packet = dataset[index]
            losses.append(float(self.compute(packet)[1]))
            counts.append(len(packet["features"]) if self.model.config["task"] in ("score", "alignment") else 1)
        return float(np.average(losses, weights=counts))

    def fit(self, dataset, output, epochs=100, validation=None, seed=0, start_epoch=0):
        """Save last/best checkpoints and JSON epoch logs using explicit split data."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        history, best = self.history, self.best_metric
        for epoch in range(start_epoch, start_epoch + epochs):
            dataset.epoch = epoch
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
            losses, counts = [], []
            for index in rng.permutation(len(dataset)):
                packet = dataset[int(index)]
                losses.append(self.train_step(packet))
                counts.append(len(packet["features"]) if self.model.config["task"] in ("score", "alignment") else 1)
            row = dict(epoch=epoch, step=self.step, training_loss=float(np.average(losses, weights=counts)))
            metric = row["training_loss"]
            if validation is not None:
                row["validation_loss"] = self.evaluate(validation)
                metric = row["validation_loss"]
            history.append(row)
            improved = metric < best
            best = min(best, metric)
            self.best_metric = best
            metadata = dict(epoch=epoch, step=self.step, seed=seed, class_weights=self.class_weights.tolist(),
                            history=history, best_metric=best)
            save_checkpoint(self.model, output / "last.pt", self.optimizer, **metadata)
            if improved:
                save_checkpoint(self.model, output / "best.pt", self.optimizer, **metadata)
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        return history
