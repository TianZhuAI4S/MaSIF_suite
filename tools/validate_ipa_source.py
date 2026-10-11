"""Compare IPA evaluation sampling/importance and training against official Keras source."""

import argparse
from pathlib import Path

import numpy as np
import torch

from masif_torch.checkpoints import load_checkpoint, parameter_mapping
from masif_torch.training import Trainer
from masif_torch.validation import compare, deterministic, write_report


def validate_source(reference, checkpoint, device="cpu"):
    """Compare independent Keras inference cases, all gradients and moving BN buffers."""
    deterministic()
    model, _ = load_checkpoint(checkpoint, device)
    results = {}
    names = ["padding", "exact", "trim_1", "trim_2", "degenerate"]
    if (Path(reference) / "real.npz").exists():
        names.append("real")
    for name in names:
        with np.load(Path(reference) / f"{name}.npz", allow_pickle=False) as data:
            probability, importance = model.eval_model(data["raw"], float(data["cutoff"]))
            results[name + "/probability"] = compare(probability, data["probability"])
            expected = data["importance"]
            same_nan = np.array_equal(np.isnan(importance), np.isnan(expected))
            finite = np.isfinite(expected)
            result = compare(importance[finite], expected[finite]) if finite.any() else dict(passed=True)
            result["passed"] &= same_nan
            result["degenerate_nan_mask_equal"] = same_nan
            results[name + "/importance"] = result
    trainer = Trainer(model, device)
    with np.load(Path(reference) / "training.npz", allow_pickle=False) as data:
        model.train()
        output, loss = trainer.compute(dict(features=data["features"], labels=data["labels"]))
        loss.backward()
        results["training/probability"] = compare(output.softmax(-1).detach().cpu().numpy(), data["probability"])
        results["training/loss"] = compare(loss.detach().cpu().numpy(), data["loss"])
        named, state = dict(model.named_parameters()), model.state_dict()
        for target, (source, axes) in parameter_mapping(model).items():
            key = ("grad/" if target in named else "moving/") + source
            actual = named[target].grad if target in named else state[target]
            actual = actual.detach().cpu().numpy()
            if axes is not None:
                actual = actual.transpose(np.argsort(axes))
            results[key] = compare(actual, data[key])
    adam_path = Path(reference) / "adam.npz"
    if adam_path.exists():
        updated, _ = load_checkpoint(checkpoint, device)
        trainer = Trainer(updated, device)
        with np.load(adam_path, allow_pickle=False) as data:
            for step in range(3):
                trainer.train_step(dict(features=data["features"], labels=data["labels"]))
                for target, (source, axes) in parameter_mapping(updated).items():
                    actual = updated.state_dict()[target].detach().cpu().numpy()
                    if axes is not None:
                        actual = actual.transpose(np.argsort(axes))
                    key = f"step{step+1}/" + source
                    results["keras_adam/" + key] = compare(actual, data[key])
                updated.eval()
                with torch.no_grad():
                    probability = updated(torch.as_tensor(data["features"], device=device)).softmax(-1).cpu().numpy()
                results[f"updated_probability/step{step+1}"] = compare(probability, data[f"probability{step+1}"])
            replay, _ = load_checkpoint(checkpoint, device)
            replay_trainer = Trainer(replay, device)
            named = dict(replay.named_parameters())
            for step in range(3):
                replay_trainer.optimizer.zero_grad(set_to_none=True)
                for target, (source, axes) in parameter_mapping(replay).items():
                    if target in named:
                        gradient = data[f"gradient{step+1}/" + source]
                        if axes is not None:
                            gradient = gradient.transpose(axes)
                        named[target].grad = torch.as_tensor(gradient.copy(), device=device)
                replay_trainer.optimizer.step()
                for target, (source, axes) in parameter_mapping(replay).items():
                    if target in named:
                        actual = named[target].detach().cpu().numpy()
                        if axes is not None:
                            actual = actual.transpose(np.argsort(axes))
                        key = f"step{step+1}/" + source
                        results["same_gradient_adam/" + key] = compare(actual, data[key])
    groups = {name: all(value["passed"] for key, value in results.items() if predicate(key))
              for name, predicate in {
                  "source_inference_loss_gradients_bn": lambda key: not key.startswith(("keras_adam/", "same_gradient_adam/", "updated_probability/")),
                  "independent_training_parameter_updates": lambda key: key.startswith("keras_adam/"),
                  "same_gradient_optimizer_updates": lambda key: key.startswith("same_gradient_adam/"),
                  "updated_eval_probabilities": lambda key: key.startswith("updated_probability/")}.items()}
    return dict(device=device, passed=all(r["passed"] for r in results.values()), groups=groups, tensors=results)


def main():
    """Run the requested CPU/GPU source comparison and retain every numerical failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    reports = [validate_source(args.reference, args.checkpoint, device)
               for device in (["cpu", "cuda"] if args.cuda else ["cpu"])]
    write_report(dict(models=reports, passed=all(r["passed"] for r in reports)), args.report)
    for report in reports:
        print(report["device"], report["passed"], {k: v for k, v in report["tensors"].items() if not v["passed"]})
    if not all(r["passed"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
