"""Shared prepared-packet application contract."""

from pathlib import Path
import numpy as np

class NeuralTask:
    """Adapt existing prepared-packet inference to the common application run contract."""

    def __init__(self, config: dict):
        """Keep a copy of configuration so callers can reuse their input dictionary."""
        self.config = dict(config)

    def run(self) -> dict:
        """Load the declared model, verify task ownership and save native output shape."""
        from ..checkpoints import load_checkpoint
        from ..cli import predict
        from ..data import load_npz
        model, _ = load_checkpoint(self.config["checkpoint"], self.config.get("device", "cpu"))
        if model.config["task"] != self.config["task"]:
            raise ValueError("Task configuration and checkpoint model must agree")
        output = Path(self.config["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        prediction = predict(model, load_npz(self.config["input"]), self.config.get("device", "cpu"),
                             self.config.get("batch_size", 100))
        np.save(output, prediction)
        return dict(task=self.config["task"], output=str(output), shape=list(prediction.shape))
