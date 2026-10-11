"""Collect validated pretrained weights under task and purpose names without changing bytes."""

import argparse
import hashlib
from pathlib import Path
import shutil

from omegaconf import OmegaConf


BASE_MODELS = {
    'site': ('site_backup.pt', 'interface_prediction_backup.pt',
             'Protein interface prediction; finite three-layer peptides backup.'),
    'search': ('ppi_search_historical.pt', 'surface_descriptors_historical.pt',
               'Surface descriptors; pinned historical upstream revision 51e5422.'),
    'ligand': ('ligand.pt', 'pocket_classification.pt',
               'Seven-class ligand pocket classification with all surface features.'),
    'score': ('score.pt', 'second_stage_interface_scoring.pt',
              'Original six-layer ScoreNN for second-stage interface scoring.'),
}
IPA_MODELS = {
    'models_std/weights_12A_012.hdf5': ('ipa_12A_surface.pt',
        'IPA interface scoring with three surface features and 200 points.'),
    'models_std/weights_12A_0129.hdf5': ('ipa_12A_surface_atom.pt',
        'IPA interface scoring with surface and atom-distance features; official workflow default, 200 points.'),
    'models_std/weights_9A_0123.hdf5': ('ipa_9A_surface_atom.pt',
        'IPA interface scoring for 9 Angstrom patches with four features and 100 points.'),
    'models_benchmark/weights_seed_benchmark_12A_0123': ('ipa_12A_benchmark.pt',
        'IPA benchmark-training snapshot from the upstream models_benchmark directory; four features, 200 points.'),
    'models_std/weights_seed_benchmark_12A_0123.hdf5': ('ipa_12A_benchmark_alternative.pt',
        'Distinct IPA benchmark-training snapshot from upstream standard HDF5 weights; four features, 200 points.'),
}


def catalog(root: Path, tasks: list[str]) -> list[dict]:
    """Map each selected upstream snapshot to an explicit task and purpose filename."""
    entries = []
    for task in tasks:
        if task in BASE_MODELS:
            source, name, purpose = BASE_MODELS[task]
            entries.append(dict(task=task, source=f'artifacts/checkpoints/{source}',
                                checkpoint=f'weights/{task}/{name}', purpose=purpose))
        else:
            repository = 'masif_seed' if task == 'seed' else 'masif-neosurf'
            for source, (name, purpose) in IPA_MODELS.items():
                features = [0, 1, 2] if source == 'models_std/weights_12A_012.hdf5' else [0, 1, 2, 3]
                points = 100 if source == 'models_std/weights_9A_0123.hdf5' else 200
                entries.append(dict(task=task,
                    source=f'artifacts/design_checkpoints/{repository}/{source}/converted.pt',
                    checkpoint=f'weights/{task}/{name}', purpose=purpose,
                    expected_config=dict(selected_features=features, max_npoints=points)))
    for entry in entries:
        if not (root / entry['source']).is_file():
            raise FileNotFoundError(f"Generate the validated source weight first: {entry['source']}")
    return entries


def organize(root: Path, tasks: list[str]) -> None:
    """Copy checkpoint bytes, verify safe strict loading, and retain a YAML provenance manifest."""
    import torch
    from masif_torch.checkpoints import load_checkpoint

    entries = catalog(root, tasks)
    # Validate all selected sources and destinations before copying any files.
    for entry in entries:
        source = root / entry['source']
        model, payload = load_checkpoint(source)
        expected_task = 'alignment' if entry['task'] in ('seed', 'neosurface') else entry['task']
        if expected_task == 'search':
            expected_task = 'ppi_search'
        if payload['config']['task'] != expected_task:
            raise ValueError(f"Checkpoint task does not match destination: {source}")
        for key, expected in entry.get('expected_config', {}).items():
            if payload['config'].get(key) != expected:
                raise ValueError(f"Checkpoint configuration does not match filename: {source}: {key}")
        for name, value in model.state_dict().items():
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite source weight: {source}: {name}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = root / entry['checkpoint']
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Different weight already exists; select a new destination: {destination}")
        entry.update(sha256=digest, model_config=payload['config'])
    for entry in entries:
        destination = root / entry['checkpoint']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(root / entry['source'], destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f"Checkpoint copy verification failed: {destination}")
        print(entry['checkpoint'])
    manifest = root / 'weights/manifest.yaml'
    previous = OmegaConf.to_container(OmegaConf.load(manifest), resolve=True) if manifest.exists() else {'models': []}
    models = [entry for entry in previous['models'] if entry['task'] not in tasks] + entries
    OmegaConf.save(OmegaConf.create(dict(format_version=1, models=models)), manifest)


def main() -> None:
    """Select task weights from an existing checkout and organize their validated copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    choices = list(BASE_MODELS) + ['seed', 'neosurface']
    parser.add_argument('--task', choices=choices, action='append', help='Repeat to select tasks; default: all.')
    args = parser.parse_args()
    organize(args.root.resolve(), list(dict.fromkeys(args.task or choices)))


if __name__ == '__main__':
    main()
