#!/usr/bin/env python3
"""Combine completed target/decoy searches and audit original versus inclusive Top-K."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path


def read_json(path: Path) -> tuple[dict, str]:
    """Read one immutable completed artifact and retain its exact byte fingerprint."""
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def ranked_case(directory: Path) -> tuple[list[dict], dict]:
    """Recover best poses per site/source and validated native metrics from a case."""
    summary, summary_sha = read_json(directory / 'summary.json')
    result, result_sha = read_json(directory / 'results.json')
    accepted = [row for row in result['candidates'] if row.get('accepted')]
    if len(accepted) != summary['accepted'] or len(result['candidates']) != summary['candidates']:
        raise ValueError(f'Candidate counts differ from the completed summary: {directory}')
    best = {}
    # Stable descending ordering is the same ordering used by summarize_case.
    for row in sorted(accepted, key=lambda item: item['score'], reverse=True):
        if not math.isfinite(row['score']):
            raise ValueError(f'Nonfinite accepted probability: {directory}')
        best.setdefault((row['site'], row['source']), row)
    ranked = sorted(best.values(), key=lambda item: item['score'], reverse=True)
    if len(ranked) != summary['ranked_groups']:
        raise ValueError(f'Group counts differ from the completed summary: {directory}')
    metrics = {row['rank']: row for row in summary['partner_hits']}
    if (len(metrics) != len(summary['partner_hits']) or
            len(metrics) != sum(row['source'] == summary['partner'] for row in ranked)):
        raise ValueError(f'Duplicate or extra native metric ranks: {directory}')
    if any(not math.isfinite(row[field]) or row[field] < 0
           for row in metrics.values() for field in ('rmsd', 'interface_rmsd')):
        raise ValueError(f'Native RMSD/iRMSD must be finite and nonnegative: {directory}')
    records = []
    for rank, row in enumerate(ranked, 1):
        record = dict(site=row['site'], source=row['source'], score=row['score'])
        if row['source'] == summary['partner']:
            if rank not in metrics or metrics[rank]['score'] != row['score']:
                raise ValueError(f'Native group metrics do not match its score/rank: {directory}')
            record.update(RMSD=metrics[rank]['rmsd'], iRMSD=metrics[rank]['interface_rmsd'])
        elif rank in metrics:
            raise ValueError(f'Non-native group has native metrics: {directory}')
        records.append(record)
    proof = dict(directory=str(directory), summary_sha256=summary_sha,
                 results_sha256=result_sha, condition=summary['condition'], target=summary['target'])
    return records, proof


def original_topk(source: Path, results: dict, k_values: list[int]) -> list[int]:
    """Execute unchanged upstream get_topk, retaining its len(result_list)<=k skip."""
    module = ast.parse(source.read_text())
    definitions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'get_topk']
    if len(definitions) != 1:
        raise ValueError('Expected exactly one original get_topk function')
    namespace = {}
    code = ast.Module(body=definitions, type_ignores=[])
    exec(compile(code, str(source), 'exec'), namespace)
    with redirect_stdout(io.StringIO()):
        return namespace['get_topk'](results, k_values)


def inclusive_topk(results: dict, k_values: list[int]) -> list[int]:
    """Count native poses below 5 Å among all available groups at ranks <=K."""
    return [sum(any('iRMSD' in row and row['iRMSD'] < 5.0 for row in rows[:k])
                for rows in results.values()) for k in k_values]


def analyze(root: Path, source: Path) -> dict:
    """Merge all 112 completed searches without rerunning docking or neural scoring."""
    aggregate, aggregate_sha = read_json(root / 'benchmark.json')
    if aggregate['completed'] != 112 or not aggregate['full_public_small_decoy_benchmark']:
        raise ValueError('Combined analysis requires the complete uncapped 112-case run')
    lookup = {(row['condition'], row['target']): row for row in aggregate['cases']}
    if len(lookup) != 112:
        raise ValueError('Expected 112 distinct condition/target summaries')
    k_values = list(range(101))
    conditions, proofs = {}, []
    for condition in ('with_ligand', 'without_ligand'):
        targets = sorted(target for name, target in lookup if name == condition + '_targets')
        if len(targets) != 28:
            raise ValueError('Expected all 28 native target directions')
        results, target_reports = {}, []
        for target in targets:
            grouped = {}
            for database in ('targets', 'decoys'):
                records, proof = ranked_case(root / condition / database / target)
                if proof['condition'] != condition + '_' + database or proof['target'] != target:
                    raise ValueError('Completed case identity differs from its directory')
                proofs.append(proof)
                for row in records:
                    key = (row['site'], row['source'])
                    if key in grouped:
                        raise ValueError('Native and decoy source identifiers overlap')
                    grouped[key] = row
            ranked = sorted(grouped.values(), key=lambda row: row['score'], reverse=True)
            results[target] = ranked
            hits = [dict(rank=rank, **row) for rank, row in enumerate(ranked, 1) if 'iRMSD' in row]
            target_reports.append(dict(target=target, ranked_groups=len(ranked), native_hits=hits,
                best_recovery_rank=min((row['rank'] for row in hits if row['iRMSD'] < 5.0), default=None)))
        conditions[condition] = dict(targets=28, searched_source_proteins=228,
            recovered_at_any_rank=sum(row['best_recovery_rank'] is not None for row in target_reports),
            k_values=k_values, inclusive_topk=inclusive_topk(results, k_values),
            original_source_topk=original_topk(source, results, k_values), cases=target_reports)
    return dict(completed_searches=112, conditions=conditions, case_inputs=proofs,
        aggregate_sha256=aggregate_sha, original_analysis_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric='Original float32 heavy-atom RMSD/iRMSD from exported PDBs; recovery iRMSD<5 Å',
        ranking='Best pose per (site,source), then descending probability; native groups precede decoys on exact ties',
        original_source_boundary='Upstream get_topk skips a target whenever len(result_list)<=k; inclusive_topk includes shorter lists.',
        neural_scores_recomputed=False, bitwise_historical_pose_equivalence=False)


def main() -> None:
    """Write a standalone, source-backed combined-ranking artifact after completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--original-analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    report = analyze(options.run, options.original_analysis)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {key: value[key] for key in ('targets', 'searched_source_proteins', 'recovered_at_any_rank')}
                      for name, value in report['conditions'].items()}))


if __name__ == '__main__':
    main()
