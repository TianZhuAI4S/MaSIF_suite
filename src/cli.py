"""Command line installation checks, conversion, validation, inference and training."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from .checkpoints import convert, load_checkpoint
from .configuration import compose_config, configuration_dict, load_config_file
from omegaconf import OmegaConf
from .data import PacketDataset, SearchCacheDataset, flip_binder, load_npz, load_surface, tensors
from .models import build_model
from .training import Trainer
from .validation import deterministic, validate, validate_adam, write_report


def predict(model, data, device="cpu", batch_size=100):
    """Predict complete surfaces or one pocket, preserving task-specific output shapes."""
    model.eval()
    task = model.config["task"]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with torch.no_grad():
        if task in ("score", "alignment"):
            if not len(data["features"]):
                raise ValueError("Cannot score an empty alignment set")
            return np.concatenate([model(tensors(dict(features=data["features"][start:start + batch_size]),
                                                 device)["features"]).softmax(-1).cpu().numpy()
                                   for start in range(0, len(data["features"]), batch_size)])
        if task == "ppi_search":
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            results = []
            for start in range(0, len(data["features"]), batch_size):
                batch = {k: v[start:start + batch_size] for k, v in data.items()
                         if k in ("features", "rho", "theta", "mask")}
                values = tensors(batch, device)
                results.append(model(values["features"], values["rho"], values["theta"],
                                     values["mask"]).cpu().numpy())
            if not results:
                raise ValueError("Cannot describe an empty surface")
            return np.concatenate(results)
        values = tensors(data, device)
        logits = model(values["features"], values["rho"], values["theta"], values["mask"],
                       values.get("indices"))
        return (logits.sigmoid()[:, 0][None] if task == "site" else logits.softmax(-1)).cpu().numpy()


def evaluate_ligand(model, data, device, samples=100, seed=0):
    """Reproduce evaluation's 100 draws of 32 patches for every eligible pocket."""
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    labels = data["pocket_labels"]
    predictions, classes = [], []
    for pocket in range(labels.shape[1]):
        points = np.flatnonzero(labels[:, pocket])
        if len(points) < 32:
            continue
        classes.append(int(labels[:, pocket].max()) - 1)
        draws = []
        for _ in range(samples):
            selected = rng.choice(points, 32, replace=False)
            batch = {k: v[selected] for k, v in data.items()
                     if k in ("features", "rho", "theta", "mask")}
            draws.append(predict(model, batch, device))
        predictions.append(draws)
    return np.array(classes), np.asarray(predictions, dtype=np.float32)


def _train(args, model_config=None, expected_task=None, resolved_config=None):
    """Run the original training path with unchanged initialization/resume order."""
    if bool(args.config or model_config is not None) == bool(args.checkpoint):
        raise ValueError("Training requires exactly one model configuration or checkpoint")
    if args.resume and not args.checkpoint:
        raise ValueError("resume requires a checkpoint")
    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, args.device)
    else:
        model, payload = build_model(model_config if model_config is not None else load_config_file(args.config)), {}
    if expected_task is not None and model.config["task"] != expected_task:
        raise ValueError("The task selector must match the checkpoint task")
    args.batch_size = args.batch_size or (100 if model.config["task"] == "site" else 32)
    trainer = Trainer(model, args.device, args.learning_rate, args.class_weights)
    args.seed = args.seed if args.seed is not None else payload.get("seed", 0)
    start_epoch = 0
    if args.resume:
        if payload.get("optimizer") is None:
            raise ValueError("Checkpoint has no optimizer state; use converted --include-optimizer")
        start_epoch = trainer.restore(payload)
        if args.learning_rate is not None:
            for group in trainer.optimizer.param_groups:
                group["lr"] = args.learning_rate
        if args.class_weights is not None:
            trainer.class_weights = torch.tensor(args.class_weights, device=args.device)
    if args.search_cache:
        if model.config["task"] != "ppi_search":
            raise ValueError("The search_cache dataset requires the search model")
        dataset = SearchCacheDataset(args.input, model.config, args.steps, args.batch_size, args.seed)
    else:
        dataset = PacketDataset(args.input, model.config, args.seed, batch_size=args.batch_size)
    validation = (PacketDataset(args.validation, model.config, args.seed, training=False,
                                batch_size=args.batch_size) if args.validation else None)
    excluded = dict(training=getattr(dataset, "excluded", []),
                    validation=getattr(validation, "excluded", []))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    if any(excluded.values()):
        print(json.dumps(dict(excluded=excluded)), flush=True)
    if resolved_config is not None:
        # Record the executed configuration after dataset/checkpoint validation.
        (args.output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(resolved_config))
    trainer.fit(dataset, args.output, args.epochs, validation, args.seed, start_epoch)


def _train_configuration(config):
    """Adapt composed model/dataset/trainer groups to the existing native trainer."""
    dataset, trainer = config["dataset"], config["trainer"]
    if dataset["kind"] not in ("packets", "search_cache"):
        raise ValueError("dataset.kind must be packets or search_cache")
    checkpoint = trainer.get("checkpoint")
    model_config = None if checkpoint else config["model"]
    if model_config is not None and model_config["task"] != config["task"]:
        raise ValueError("The task selector and model.task must agree")
    args = argparse.Namespace(
        config=None, checkpoint=checkpoint, resume=trainer["resume"],
        input=Path(dataset["input"]), output=Path(trainer["output"]),
        validation=Path(dataset["validation"]) if dataset.get("validation") else None,
        epochs=trainer["epochs"], learning_rate=trainer.get("learning_rate"),
        seed=trainer.get("seed"), class_weights=trainer.get("class_weights"),
        device=trainer["device"], search_cache=dataset["kind"] == "search_cache",
        steps=dataset["steps"], batch_size=dataset.get("batch_size"))
    # Initialize before constructing/loading the model; restore subsequently restores
    # checkpoint RNG state exactly as in the original command-line training path.
    deterministic(args.seed or 0)
    _train(args, model_config, config["task"], config)


_TASK_PRESETS = {"site": "site", "search": "search", "ligand": "ligand",
                 "score": "score", "alignment": "alignment", "seed": "seed",
                 "neosurface": "neosurface", "neosurf": "neosurface",
                 "prepare": "prepare_protein", "preparation": "prepare_protein",
                 "rosetta": "rosetta", "benchmark": "benchmark", "train": "train"}


def _configured_command(argv):
    """Compose task presets or a user YAML and apply actual Hydra overrides."""
    parser = argparse.ArgumentParser(prog="masif " + argv[0],
                                     description="Compose Hydra YAML; paths use the calling directory.")
    parser.add_argument("--config", type=Path, help="Explicit YAML configuration file")
    parser.add_argument("--config-path", type=Path, help="Directory containing user YAML presets")
    parser.add_argument("--config-name", help="Built-in or user preset name")
    parser.add_argument("--cfg", choices=("job",), help="Print configuration without running a task")
    parser.add_argument("--resolve", action="store_true", help="Resolve interpolations in printed YAML")
    args, overrides = parser.parse_known_args(argv[1:])
    if any(item.startswith("--") for item in overrides):
        parser.error("Task settings use Hydra key=value overrides")
    if args.config and (args.config_path or args.config_name):
        parser.error("Use --config or --config-path/--config-name")
    if args.config:
        if args.config.suffix.lower() == ".json":
            config = OmegaConf.create(load_config_file(args.config, overrides))
        else:
            if args.config.suffix.lower() != ".yaml":
                parser.error("Hydra configuration files must use the .yaml extension")
            config = compose_config(args.config.name, overrides, args.config.parent)
    else:
        name = args.config_name or _TASK_PRESETS.get(argv[0])
        if not name:
            parser.error("run requires --config or --config-name")
        config = compose_config(name, overrides, args.config_path)
    if args.cfg:
        print(OmegaConf.to_yaml(config, resolve=args.resolve), end="")
        return
    values = configuration_dict(config)
    if argv[0] == "train" and values.get("operation") != "train":
        raise ValueError("The train command requires a training configuration")
    if argv[0] not in ("run", "train"):
        expected = {"search": "ppi_search", "neosurface": "neosurf", "neosurf": "neosurf",
                    "prepare": "preparation"}.get(argv[0], argv[0])
        actual = {"search": "ppi_search", "neosurface": "neosurf"}.get(values.get("task"), values.get("task"))
        if actual != expected or values.get("operation") == "train":
            raise ValueError("The command and configured task must agree; use run for generic dispatch")
    if values.get("operation") == "train":
        _train_configuration(values)
    else:
        from .tasks import build_task
        deterministic(values.get("seed", 42))
        print(json.dumps(build_task(values).run(), indent=2))


def main(argv=None):
    """Dispatch documented task commands with explicit paths and configuration."""
    argv = sys.argv[1:] if argv is None else list(argv)
    legacy_training = (argv and argv[0] == "train" and any(
        option in argv for option in ("--input", "--output", "--checkpoint", "--resume")))
    if argv and (argv[0] == "run" or argv[0] in _TASK_PRESETS) and not legacy_training:
        _configured_command(argv)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in _TASK_PRESETS:
        if name != "train":
            sub.add_parser(name, help="Run a task using Hydra YAML and key=value overrides")
    sub.add_parser("install-runtime", help="Install a compatible CPU/CUDA runtime (lightweight launcher)")
    sub.add_parser("doctor", help="Check Python, Torch, CUDA and TensorFlow-free imports")
    application = sub.add_parser("run", help="Run any registered task from Hydra YAML")
    application.add_argument("--config", type=Path, required=True)
    adapter = sub.add_parser("pack-design", help="Convert processed seed/neosurf geometry to safe NPZ")
    adapter.add_argument("--root", type=Path, required=True)
    adapter.add_argument("--identifier", required=True)
    adapter.add_argument("--pid", choices=("p1", "p2"), default="p1")
    adapter.add_argument("--trusted-indices", action="store_true")
    adapter.add_argument("--role", choices=("target", "source"), default="target")
    adapter.add_argument("--output", type=Path, required=True)
    conversion = sub.add_parser("convert", help="Convert an export made by tools/export_tf.py")
    conversion.add_argument("--export", type=Path, required=True)
    conversion.add_argument("--output", type=Path, required=True)
    conversion.add_argument("--include-optimizer", action="store_true")
    check = sub.add_parser("validate", help="Compare one checkpoint against its original graph")
    check.add_argument("--export", type=Path, required=True)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--device", default="cpu")
    check.add_argument("--grad-atol", type=float, default=1e-5)
    check.add_argument("--grad-rtol", type=float, default=1e-4)
    adam = sub.add_parser("validate-adam")
    adam.add_argument("--reference", type=Path, required=True)
    inference = sub.add_parser("predict", help="Predict one prepared numeric NPZ packet")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    surface = sub.add_parser("surface", help="Predict site or save straight/flipped search descriptors")
    surface.add_argument("--input", type=Path, required=True, help="Original per-protein precomputation directory")
    surface.add_argument("--output", type=Path, required=True, help="Site .npy file or search descriptor directory")
    surface.add_argument("--pid", choices=("p1", "p2"), default="p1")
    surface.add_argument("--trusted-indices", action="store_true")
    pack = sub.add_parser("pack-surface", help="Convert one original surface to a numeric training NPZ")
    pack.add_argument("--input", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--pid", choices=("p1", "p2"), default="p1")
    pack.add_argument("--trusted-indices", action="store_true")
    pack.add_argument("--feat-mask", nargs="+", type=int, default=[1, 1, 1, 1, 1])
    ligand = sub.add_parser("evaluate-ligand", help="Evaluate converted TFRecord structure packets")
    ligand.add_argument("--input", type=Path, required=True)
    ligand.add_argument("--output", type=Path, required=True)
    ligand.add_argument("--samples", type=int, default=100)
    ligand.add_argument("--seed", type=int, default=0)
    train = sub.add_parser("train", help="Train with Hydra model/dataset/trainer groups (legacy flags also supported)")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--validation", type=Path)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--search-cache", action="store_true")
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--class-weights", type=float, nargs=2, help="ScoreNN weights for classes 0,1")
    for command in (inference, surface, ligand):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=100)
    for command in (inference, surface, ligand, train):
        command.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    deterministic(getattr(args, "seed", 0) or 0)
    if args.command == "doctor":
        import importlib.util
        print(json.dumps(dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                              cuda=torch.version.cuda, gpu_available=torch.cuda.is_available(),
                              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                              tensorflow_installed=importlib.util.find_spec("tensorflow") is not None), indent=2))
    elif args.command == "run":
        from .tasks import build_task
        config = load_config_file(args.config)
        deterministic(config.get("seed", 42))
        print(json.dumps(build_task(config).run(), indent=2))
    elif args.command == "pack-design":
        from .shared.design.data import SurfaceRecord
        SurfaceRecord.from_processed(args.root, args.identifier, args.pid, args.trusted_indices, args.role).save(args.output)
    elif args.command == "convert":
        report = convert(args.export, args.output, args.include_optimizer)
        print(f"Converted {len(report['parameters'])} tensors; exact roundtrip={report['exact_weight_roundtrip']}")
    elif args.command == "validate":
        report = validate(args.export, args.checkpoint, args.device,
                          grad_atol=args.grad_atol, grad_rtol=args.grad_rtol)
        write_report(report, args.report)
        print(json.dumps(dict(passed=report["passed"], report=str(args.report))))
        if not report["passed"]:
            raise SystemExit(1)
    elif args.command == "validate-adam":
        report = validate_adam(args.reference)
        print(json.dumps(report))
        if not report["passed"]:
            raise SystemExit(1)
    elif args.command == "pack-surface":
        data = load_surface(args.input, args.pid, args.feat_mask, args.trusted_indices)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **data)
    elif args.command == "train":
        _train(args)
    else:
        model, _ = load_checkpoint(args.checkpoint, args.device)
        if args.command == "surface":
            task = model.config["task"]
            if task not in ("site", "ppi_search"):
                parser.error("surface requires a site or search checkpoint")
            data = load_surface(args.input, args.pid, model.config["feat_mask"],
                                args.trusted_indices, require_indices=task == "site")
            if task == "site":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.output, predict(model, data, args.device, args.batch_size))
            else:
                args.output.mkdir(parents=True, exist_ok=True)
                np.save(args.output / f"{args.pid}_desc_straight.npy",
                        predict(model, data, args.device, args.batch_size))
                np.save(args.output / f"{args.pid}_desc_flipped.npy",
                        predict(model, flip_binder(data, model.config["feat_mask"]), args.device, args.batch_size))
        elif args.command == "evaluate-ligand":
            if model.config["task"] != "ligand":
                parser.error("evaluate-ligand requires a ligand checkpoint")
            paths = [args.input] if args.input.is_file() else sorted(args.input.glob("*.npz"))
            if not paths:
                raise ValueError("No ligand packets found")
            args.output.mkdir(parents=True, exist_ok=True)
            for path in paths:
                data = load_npz(path)
                labels, probability = evaluate_ligand(model, data, args.device, args.samples, args.seed)
                np.save(args.output / f"{path.stem}_labels.npy", labels)
                np.save(args.output / f"{path.stem}_logits.npy", probability)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output, predict(model, load_npz(args.input), args.device, args.batch_size))


if __name__ == "__main__":
    main()
