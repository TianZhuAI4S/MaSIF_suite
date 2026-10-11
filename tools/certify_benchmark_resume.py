"""Certify this migration's preserved cases with explicit reader and importance evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from masif_torch.configuration import load_config_file
from masif_torch.tasks.benchmark.workflow import BenchmarkTask, file_sha256, surface_fingerprint, summarize_case


def certify(config: dict) -> dict:
    """Verify unchanged scientific inputs and publish an auditable corrected resume manifest."""
    output = Path(config['output'])
    audit = output.parent / 'migration_audit'
    audit.mkdir(exist_ok=True)
    old = json.loads((output / 'provenance.json').read_text())
    cases = json.loads((output / 'cases.json').read_text())
    readers_path = Path('docs/torch_migration/numerical_benchmark_readers.json')
    importance_path = Path('docs/torch_migration/numerical_benchmark_importance.json')
    readers, importance = json.loads(readers_path.read_text()), json.loads(importance_path.read_text())
    if not all(readers[key] for key in ['geometry_array_equal', 'all_original_site_ids_equal_before',
                                       'all_original_site_ids_equal_after', 'all_target_interface_array_equal_after']):
        raise ValueError('Reader correction lacks unchanged-site proof')
    if readers['task_role_records'] != 284 or readers['target_records'] != 56:
        raise ValueError('Reader correction did not cover the full dataset')
    if not importance['passed'] or importance['accepted_real_poses'] != 53:
        raise ValueError('Deferred importance lacks its full real-pose regression')
    task = BenchmarkTask(config)
    task.input_fingerprints = {}
    cache = output / 'surface_cache'
    for condition in ('with_ligand', 'without_ligand', 'decoys'):
        identifiers = sorted({path.name.rsplit('_', 1)[0] for path in (cache / condition).glob('*.npz')})
        for identifier in identifiers:
            key = condition + '/' + identifier
            task.input_fingerprints[key] = surface_fingerprint(Path(config['dataset']) / condition, identifier)
        print('Verified source files:', condition, len(identifiers), flush=True)
    current = task.provenance(cases)
    for key in ['checkpoint_sha256', 'definitions_sha256', 'dataset_sources', 'original_complexes', 'cases']:
        if old['inputs'][key] != current['inputs'][key]:
            raise ValueError(f'Scientific input changed: {key}')
    preserved = []
    native = Path(config['dataset']) / 'without_ligand/data_preparation/01-benchmark_pdbs'
    for case in cases:
        path = Path(case['output'])
        if not (path / 'summary.json').exists():
            continue
        relative = path.relative_to(output)
        previous = audit / 'summaries_before_file_metric_fix' / relative
        previous.mkdir(parents=True, exist_ok=True)
        if not (previous / 'summary.json').exists():
            shutil.copyfile(path / 'summary.json', previous / 'summary.json')
        report = json.loads((path / 'results.json').read_text())
        # The same corrected site IDs are required for every retained output.
        target = next(row for row in readers['records'] if row['condition'] == case['target_condition']
                      and row['identifier'] == case['target'] and row['role'] == 'target')
        if report['sites'] != target['original_sites']:
            raise ValueError(f'Preserved site IDs differ: {case["target"]}')
        partner = SimpleNamespace(pdb_path=native / (case['partner'] + '.pdb'))
        target_record = SimpleNamespace(pdb_path=native / (case['target'] + '.pdb'))
        summarize_case(case, report, partner, target_record)
        preserved.append(dict(case=str(relative), results_sha256=file_sha256(path / 'results.json'),
                              summary_sha256=file_sha256(path / 'summary.json')))
    certificate = dict(previous_execution_fingerprint=old['fingerprint'],
                       certified_resume_fingerprint=current['fingerprint'], preserved_cases=preserved,
                       evidence_sha256={str(path): file_sha256(path) for path in [readers_path, importance_path]},
                       corrections=['ASCII target iface widened from original decimal text; all 56 selected site sets unchanged.',
                                    'Metrics recomputed from actual exported PDB float32 protein-heavy coordinates, matching original analysis.',
                                    'Benchmark importance deferred past clash acceptance; all 53 real accepted outputs and sampling RNG exact.',
                                    'Packaged optional legacy registration worker and stricter resume dependency/version validation.'],
                       historical_execution_note='Previous execution used preserved workflow/data snapshots and its historical manifest; this certificate does not relabel that execution as the new source version.')
    (audit / 'resume_certificate.json').write_text(json.dumps(certificate, indent=2)+'\n')
    current['migration_certificate'] = str(audit / 'resume_certificate.json')
    (output / 'provenance.json').write_text(json.dumps(current, indent=2)+'\n')
    print('Certified', len(preserved), 'completed cases', flush=True)
    return certificate


def main() -> None:
    """Apply the explicit migration certificate to the configured historical run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples/benchmark.yaml')
    args = parser.parse_args()
    certify(load_config_file(args.config))


if __name__ == '__main__':
    main()
