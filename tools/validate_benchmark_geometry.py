"""Audit all public benchmark surfaces against actual old readers and correct target caches."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from masif_torch.shared.design.data import SurfaceRecord
from masif_torch.shared.design.features import target_sites
from masif_torch.tasks.benchmark.workflow import file_sha256


def main():
    """Check original geometry, original target site IDs, and decimal precision for all records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair-caches', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = project / 'artifacts/upstream_repos/masif-neosurf/masif_seed_search/source/alignment_utils.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_target_vix']
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    cases = json.loads((args.run / 'cases.json').read_text())
    points = {(case['target_condition'], case['target']): np.asarray(case['target_point']) for case in cases}
    records = []
    adapter = file_sha256(project / 'src/shared/design/data.py')
    for legacy in sorted(args.legacy.glob('*/*.npz')):
        condition, identifier = legacy.parent.name, legacy.stem
        with np.load(legacy, allow_pickle=False) as archive:
            old = {key: archive[key] for key in archive.files}
        raw_colored_equal = all(np.array_equal(old[key], old['colored_' + key]) for key in ('points', 'normals'))
        pymesh_vertices_equal = np.array_equal(old['points'], old['pymesh_points'])
        roles = ('target', 'source') if condition == 'without_ligand' else (('source',) if condition == 'decoys' else ('target',))
        for role in roles:
            path = args.run / 'surface_cache' / condition / (identifier + '_' + role + '.npz')
            with np.load(path, allow_pickle=False) as archive:
                new = {key: archive[key] for key in archive.files}
            geometry_equal = all(np.array_equal(old[key], new[key]) for key in ('points', 'normals'))
            row = dict(condition=condition, identifier=identifier, role=role,
                       original_raw_colored_geometry_equal=raw_colored_equal,
                       original_pymesh_open3d_vertices_equal=pymesh_vertices_equal,
                       geometry_array_equal=geometry_equal)
            if role == 'target':
                offsets, indices = new['neighbor_offsets'], new['neighbor_indices']
                neighbors = [indices[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]
                coordinate = points[(condition, identifier)]
                distances = np.sqrt(np.sum((old['pymesh_points'] - coordinate) ** 2, axis=1))
                if np.sum(distances < 10.0) < 3:
                    coordinate = old['pymesh_points'][np.argmin(distances)]
                    distances = np.sqrt(np.sum((old['pymesh_points'] - coordinate) ** 2, axis=1))
                selected = np.where(distances < 10.0)[0]
                original = namespace['get_target_vix'](neighbors, old['interface'].copy(), num_sites=3, selected_vertices=selected)
                record = SurfaceRecord.from_npz(path)
                before = target_sites(record, 3, selected)
                row.update(interface_array_equal_before=bool(np.array_equal(old['interface'], record.interface)),
                           interface_max_error_before=float(np.max(np.abs(old['interface'] - record.interface))),
                           original_sites=[int(value) for value in original], before_sites=before,
                           selected_site_ids_equal_before=original == before)
                if args.repair_caches:
                    if not geometry_equal or not pymesh_vertices_equal or not raw_colored_equal:
                        raise ValueError('Geometry differs; target attribute repair cannot certify unchanged poses')
                    record.interface = old['interface'].copy()
                    record.save(path)
                    metadata = json.loads(path.with_suffix('.json').read_text())
                    metadata.update(packet_sha256=file_sha256(path), adapter_sha256=adapter)
                    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
                after = target_sites(record, 3, selected)
                row.update(interface_array_equal_after=bool(np.array_equal(old['interface'], record.interface)),
                           selected_site_ids_equal_after=original == after)
            elif args.repair_caches:
                metadata = json.loads(path.with_suffix('.json').read_text())
                metadata['adapter_sha256'] = adapter
                path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
            records.append(row)
    geometry_ok = all(row['geometry_array_equal'] and row['original_pymesh_open3d_vertices_equal'] and
                      row['original_raw_colored_geometry_equal'] for row in records)
    targets = [row for row in records if row['role'] == 'target']
    report = dict(surfaces=256, task_role_records=len(records), target_records=len(targets),
                  original_source_sha256=file_sha256(source), geometry_array_equal=geometry_ok,
                  all_original_site_ids_equal_before=all(row['selected_site_ids_equal_before'] for row in targets),
                  all_original_site_ids_equal_after=all(row['selected_site_ids_equal_after'] for row in targets),
                  all_target_interface_array_equal_after=all(row['interface_array_equal_after'] for row in targets),
                  cache_correction_performed=args.repair_caches, records=records)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))
    if not geometry_ok or not report['all_original_site_ids_equal_after']:
        raise SystemExit('Original geometry/site selection audit failed')


if __name__ == '__main__':
    main()
