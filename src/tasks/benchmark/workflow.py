"""Complete published small-decoy benchmark with cached surfaces and explicit protocol."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
from importlib.metadata import version
import shutil
import multiprocessing
import subprocess
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from ...checkpoints import load_checkpoint
from ...shared.design.data import SurfaceRecord, transform_points
from ...shared.design.features import AlignmentScorer, target_sites
from ...shared.design.registration import Open3DRegistration, RegistrationBackend
from ..seed.workflow import SeedWorkflow

_CACHE = {}


def file_sha256(path: Path) -> str:
    """Hash every file byte without loading molecular arrays into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def surface_fingerprint(root: Path, identifier: str) -> str:
    """Hash all files consumed by either target/source processed-surface adapters."""
    paths = [root / 'output/all_feat_3l/pred_surfaces' / (identifier + '.ply'),
             root / 'output/all_feat_3l/pred_data' / ('pred_' + identifier + '.npy'),
             root / 'data_preparation/01-benchmark_pdbs' / (identifier + '.pdb'),
             root / 'data_preparation/04b-precomputation_12A/precomputation' / identifier / 'p1_list_indices.npy']
    paths += [path for path in (root / 'descriptors/sc05/all_feat' / identifier).glob('p1_desc_*.npy')]
    records = [(str(path.relative_to(root)), file_sha256(path)) for path in sorted(paths)]
    return hashlib.sha256(json.dumps(records, separators=(',', ':')).encode()).hexdigest()


def interface_center(pdb: Path, target_chains: str, binder_chains: str, ligand: str) -> np.ndarray:
    """Compute official float32 interface COM, including the ligand, with distances in Å."""
    model = PDBParser(QUIET=True).get_structure(pdb.stem, pdb)[0]
    name, chain = ligand.split('_')
    residues = [residue for residue in model[chain] if residue.resname == name]
    if len(residues) != 1:
        raise ValueError(f'Expected one {ligand} residue in {pdb}')
    target = [atom.coord for c in target_chains for atom in model[c].get_atoms() if is_aa(atom.parent)]
    target += [atom.coord for atom in residues[0].get_atoms()]
    binder = [atom.coord for c in binder_chains for atom in model[c].get_atoms() if is_aa(atom.parent)]
    target, binder = np.stack(target), np.stack(binder)
    selected = np.zeros(len(target), dtype=bool)
    # Chunking changes storage only: retain original float32 arithmetic and strict <4 Å test.
    for start in range(0, len(target), 128):
        distance = np.sqrt(np.sum((target[start:start + 128, None] - binder[None]) ** 2, axis=-1))
        selected[start:start + 128] = np.any(distance < 4.0, axis=1)
    if not selected.any():
        raise ValueError(f'No interface atoms for {pdb.stem}_{target_chains}')
    return np.mean(target[selected], axis=0)


class LegacyOpen3DRegistration(RegistrationBackend):
    """Bridge a persistent original Open3D0.5 process without importing it into PyTorch."""

    def __init__(self, python: str, output: str):
        """Start one isolated legacy worker and retain stderr as reviewable evidence."""
        self.directory = Path(output) / 'legacy_registration'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.packet = self.directory / 'patch.npz'
        self.log = (self.directory / 'worker.log').open('w')
        worker = Path(__file__).parent / 'worker_registration.py'
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.process = subprocess.Popen([python, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, env=environment)

    def align(self, key: str, source: dict, target: dict):
        """Send finite numeric patches and require an explicit successful worker response."""
        np.savez(self.packet, **{role + '_' + field: data[field] for role, data in [('source', source), ('target', target)]
                                for field in ('points', 'normals', 'descriptors')})
        self.process.stdin.write(json.dumps(dict(packet=str(self.packet.resolve()))) + '\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f'Legacy registration worker exited: see {self.directory / "worker.log"}')
            if line.startswith('MASIF_RESULT '):
                result = json.loads(line[len('MASIF_RESULT '):])
                matrix, fitness = np.asarray(result['transformation'], dtype=np.float64), result['fitness']
                transform_points(np.zeros((1, 3)), matrix)
                if not np.isfinite(fitness) or not 0 <= fitness <= 1:
                    raise ValueError('Legacy registration fitness must be finite and lie in [0,1]')
                return matrix, fitness
            self.log.write(line)
            self.log.flush()

    def close(self):
        """Close the request stream and require a clean worker exit."""
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.log.close()
        if code:
            raise RuntimeError(f'Legacy registration worker failed with exit code{code}')


class BenchmarkRegistration(Open3DRegistration):
    """Randomize native source coordinates before modern RANSAC to prevent pose leakage."""

    def __init__(self, surfaces, rng, legacy=None, **options):
        """Own original Graphics Gems random rotations per site and matched source."""
        super().__init__(**options)
        self.legacy = legacy
        self.surfaces, self.rng, self.initial = surfaces, rng, {}

    def align(self, key, source, target):
        """Return the composed original-frame pose after randomized docking and ICP."""
        site, name, _ = key.split(':')
        group = site + ':' + name
        if group not in self.initial:
            theta, phi, z = self.rng.uniform(size=3)
            theta, phi, z = theta * 2 * np.pi, phi * 2 * np.pi, z * 2
            vector = np.array([np.sin(phi) * np.sqrt(z), np.cos(phi) * np.sqrt(z), np.sqrt(2 - z)])
            st, ct = np.sin(theta), np.cos(theta)
            rotation = (np.outer(vector, vector) - np.eye(3)).dot(np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]]))
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = -np.mean(self.surfaces[name].points, axis=0)
            self.initial[group] = matrix
        initial = self.initial[group]
        rotated = dict(source, points=transform_points(source['points'], initial),
                       normals=source['normals'] @ initial[:3, :3].T)
        final, fitness = (self.legacy.align(key, rotated, target) if self.legacy else super().align(key, rotated, target))
        return final @ initial, fitness


class BenchmarkWorkflow(SeedWorkflow):
    """Reuse seed docking while selecting the published COM and preserving benchmark source randomization."""

    def __init__(self, config: dict, target: SurfaceRecord, database: list[SurfaceRecord], scorer):
        """Reuse immutable cached input arrays and choose all three official target sites."""
        self.config, self.target, self.database = dict(config), target, database
        coordinate = np.asarray(config['target_point'], dtype=np.float64)
        distances = np.sqrt(np.sum((target.points - coordinate) ** 2, axis=1))
        if np.sum(distances < 10.0) < 3:
            coordinate = target.points[np.argmin(distances)]
            distances = np.sqrt(np.sum((target.points - coordinate) ** 2, axis=1))
        selected = np.where(distances < 10.0)[0]
        self.centers = target_sites(target, 3, selected)
        self.shift, self.scorer = 0.25, scorer
        legacy = (LegacyOpen3DRegistration(config['legacy_python'], config['output']) if config.get('legacy_python') else None)
        if legacy:
            self.config['registration'] = dict(mode='legacy_open3d', version='0.5.0.0', iterations=2000,
                                               radius=1.5, max_validation=500)
        self.registration = BenchmarkRegistration({surface.name: surface for surface in database}, scorer.rng, legacy=legacy,
                                                   iterations=2000, radius=1.5, seed=42)

    def _score(self, target: dict, source: dict) -> dict:
        """Score every candidate, deferring expensive diagnostics until clash acceptance."""
        result = self.scorer.score(target, source, self.target.atoms, importance_cutoff=1.0)
        result['importance_status'] = 'omitted_unless_accepted'
        return result

    def _finalize_score(self, row: dict) -> None:
        """Compute identical original importance for accepted poses without advancing RNG."""
        features = row['features']
        if len(features) <= self.scorer.model.config['max_npoints'] and row['score'] > 0.9:
            # prepare() uses all rows here, so neither this call nor leave-one-out consumes RNG.
            probability, importance = self.scorer.model.eval_model(features, 0.9, self.scorer.rng)
            if float(probability[0, 0]) != row['score']:
                raise ValueError('Deferred importance changed the accepted pose probability')
            row['point_importance'] = importance
            row['importance_finite'] = bool(np.isfinite(importance).all())
        row['importance_status'] = 'computed_original_cutoff_0.9'


def _initialize_worker(checkpoint: str) -> None:
    """Load one inference model per forked worker and constrain CPU thread pools."""
    import torch
    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint, 'cpu')
    _CACHE['scorer'] = AlignmentScorer(model, 'cpu', 42)


def _run_case(case: dict) -> dict:
    """Run every descriptor candidate, write poses, and measure native partner recovery."""
    output = Path(case['output'])
    if output.exists() and not (output / 'summary.json').exists():
        # Remove only this task-owned, interrupted case directory before regenerating poses.
        shutil.rmtree(output)
    database = _CACHE[case['database']]
    target = _CACHE[case['target_condition']][case['target']]
    # Independent official jobs each reset numpy seed42; preserve that job-level boundary.
    scorer = _CACHE['scorer']
    scorer.rng = np.random.RandomState(42)
    workflow = BenchmarkWorkflow(case, target, database, scorer)
    try:
        report = workflow.run()
    finally:
        if workflow.registration.legacy:
            workflow.registration.legacy.close()
    summary = summarize_case(case, report, _CACHE['without_ligand'][case['partner']],
                             _CACHE['without_ligand'][case['target']])
    print(json.dumps(summary), flush=True)
    return summary


def protein_coordinates(path: Path) -> np.ndarray:
    """Read original analysis protein-heavy atoms with Bio.PDB's float32 coordinates."""
    model = PDBParser(QUIET=True).get_structure(Path(path).stem, path)[0]
    atoms = [atom for atom in model.get_atoms() if atom.element != 'H' and is_aa(atom.parent)]
    if not atoms:
        raise ValueError(f'No protein heavy atoms in {path}')
    return np.stack([atom.coord for atom in atoms])


def coordinate_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    """Preserve original float32 subtract/square/sum/mean/sqrt ordering exactly."""
    if first.shape != second.shape:
        raise ValueError('Original and exported protein atom coordinates differ in shape')
    difference = first - second
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=-1))))


def summarize_case(case: dict, report: dict, partner: SurfaceRecord, target: SurfaceRecord) -> dict:
    """Rank best poses per site/source as upstream analysis_utils.parse_results does."""
    accepted = [row for row in report['candidates'] if row.get('accepted')]
    accepted.sort(key=lambda row: row['score'], reverse=True)
    reference = protein_coordinates(partner.pdb_path)
    target_atoms = protein_coordinates(target.pdb_path)
    protein_interface = np.zeros(len(reference), dtype=bool)
    for start in range(0, len(reference), 128):
        distances = np.sqrt(np.sum((reference[start:start + 128, None] - target_atoms[None]) ** 2, axis=-1))
        protein_interface[start:start + 128] = np.any(distances < 5.0, axis=1)
    best_groups = {}
    for row in accepted:
        best_groups.setdefault((row['site'], row['source']), row)
    ranked = sorted(best_groups.values(), key=lambda row: row['score'], reverse=True)
    recoveries = []
    for rank, row in enumerate(ranked, 1):
        if row['source'] != case['partner']:
            continue
        moved = protein_coordinates(Path(case['output']) / Path(row['packet']).with_suffix('.pdb'))
        rmsd = coordinate_rmsd(reference, moved)
        irmsd = coordinate_rmsd(reference[protein_interface], moved[protein_interface])
        recoveries.append(dict(rank=rank, score=row['score'], rmsd=rmsd, interface_rmsd=irmsd))
    summary = dict(condition=case['condition'], target=case['target'], partner=case['partner'],
                   candidates=len(report['candidates']), accepted=len(accepted), sites=report['sites'],
                   ranked_groups=len(ranked), ranking_unit='best_pose_per_site_and_source',
                   metric_precision='original float32 coordinates parsed from exported 3-decimal PDBs',
                   partner_hits=recoveries, recovered=any(row['interface_rmsd'] < 5.0 for row in recoveries),
                   best_recovery_rank=min((row['rank'] for row in recoveries if row['interface_rmsd'] < 5.0), default=None))
    Path(case['output'], 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


class BenchmarkTask:
    """Run all 112 cases in the public 28-subunit, 200-decoy benchmark."""

    def __init__(self, config: dict):
        """Store explicit dataset, definition, checkpoint and output locations."""
        self.config = dict(config)

    def prepare(self) -> list[dict]:
        """Fetch original complexes, cache full processed surfaces, and construct all cases."""
        global _CACHE
        root, output = Path(self.config['dataset']), Path(self.config['output'])
        output.mkdir(parents=True, exist_ok=True)
        raw = output / 'original_pdbs'
        raw.mkdir(exist_ok=True)
        definitions = []
        for line in Path(self.config['definition']).read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                definitions.append(line.strip().split(','))
        if len(definitions) != 14:
            raise ValueError('Full benchmark requires all 14 complex definitions')
        centers, partners = {}, {}
        for code, left, right, ligand, _ in definitions:
            pdb = raw / (code.lower() + '.pdb')
            if not pdb.exists():
                with urlopen(f'https://files.rcsb.org/download/{code}.pdb', timeout=60) as response:
                    pdb.write_bytes(response.read())
            for target, binder in ((left, right), (right, left)):
                identifier = code + '_' + target
                centers[identifier] = interface_center(pdb, target, binder, ligand).astype(np.float64)
                partners[identifier] = code + '_' + binder
        _CACHE = {'with_ligand': {}, 'without_ligand': {}, 'targets': [], 'decoys': []}
        cache = output / 'surface_cache'
        self.input_fingerprints = {}
        for condition, identifiers in [('with_ligand', sorted(partners)), ('without_ligand', sorted(partners)),
                                       ('decoys', sorted(path.stem[5:] for path in (root / 'decoys/output/all_feat_3l/pred_data').glob('pred_*.npy')))]:
            for identifier in identifiers:
                fingerprint = surface_fingerprint(root / condition, identifier)
                self.input_fingerprints[condition + '/' + identifier] = fingerprint
                roles = ('target', 'source') if condition == 'without_ligand' else (('source',) if condition == 'decoys' else ('target',))
                for role in roles:
                    path = cache / condition / (identifier + '_' + role + '.npz')
                    metadata = path.with_suffix('.json')
                    valid = metadata.exists() and path.exists()
                    if valid:
                        record = json.loads(metadata.read_text())
                        valid = (record['source_sha256'] == fingerprint and record['packet_sha256'] == file_sha256(path) and
                                 record.get('adapter_sha256') == file_sha256(Path(__file__).resolve().parents[2] / 'shared/design/data.py'))
                    if valid:
                        surface = SurfaceRecord.from_npz(path)
                    else:
                        surface = SurfaceRecord.from_processed(root / condition, identifier, trusted_indices=True, role=role)
                        surface.save(path)
                        metadata.write_text(json.dumps(dict(source_sha256=fingerprint, packet_sha256=file_sha256(path),
                                                            adapter_sha256=file_sha256(Path(__file__).resolve().parents[2] / 'shared/design/data.py')), indent=2) + '\n')
                    if role == 'target':
                        _CACHE[condition][identifier] = surface
                    else:
                        _CACHE['targets' if condition == 'without_ligand' else 'decoys'].append(surface)
            print(f'Cached {condition}: {len(identifiers)} surfaces', flush=True)
        if len(_CACHE['targets']) != 28 or len(_CACHE['decoys']) != 200:
            raise ValueError('Full benchmark requires all 28 target sources and 200 decoys')
        cases = []
        for condition in ('with_ligand', 'without_ligand'):
            for database in ('targets', 'decoys'):
                for identifier in sorted(partners):
                    cases.append(dict(task='benchmark', action='search', target=identifier, target_condition=condition,
                                      database=database, condition=f'{condition}_{database}', partner=partners[identifier],
                                      target_point=centers[identifier].tolist(), descriptor_cutoff=3.0, iface_cutoff=0.0,
                                      nn_score_cutoff=0.8, fitness_cutoff=0.3, allowed_CA_clashes=0,
                                      allowed_heavy_atom_clashes=5, clash_radius=2.0,
                                      registration=dict(mode='open3d', iterations=2000, radius=1.5, seed=42),
                                      output=str(output / condition / database / identifier)))
        (output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
        return cases

    def provenance(self, cases: list[dict]) -> dict:
        """Bind resumed results to checkpoint, definitions, cases, source bytes and code."""
        package = Path(__file__).resolve().parents[2]
        folders = ['shared/design', 'tasks/seed', 'tasks/benchmark', 'tasks/alignment', 'tasks/score',
                   'tasks/site', 'tasks/search', 'tasks/ligand']
        dependencies = [package / name for name in ['__init__.py', 'models.py', 'checkpoints.py', 'shared/layers.py']]
        dependencies += [path for folder in folders for path in sorted((package / folder).glob('*.py'))]
        data = dict(dependency_sha256={str(path.relative_to(package)): file_sha256(path) for path in dependencies},
                    runtime_versions={name: version(name) for name in ['torch', 'numpy', 'scipy', 'open3d', 'biopython', 'plyfile']},
                    checkpoint_sha256=file_sha256(Path(self.config['checkpoint'])),
                    definitions_sha256=file_sha256(Path(self.config['definition'])),
                    dataset_sources=self.input_fingerprints,
                    original_complexes={path.name: file_sha256(path) for path in sorted(Path(self.config['output'], 'original_pdbs').glob('*.pdb'))},
                    workflow_sha256=file_sha256(Path(__file__)), cases=cases)
        fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return dict(fingerprint=fingerprint, inputs=data)

    def run(self) -> dict:
        """Execute or resume all cases and require 112 successfully completed results."""
        cases = self.prepare()
        provenance = self.provenance(cases)
        manifest = Path(self.config['output'], 'provenance.json')
        if not self.config.get('resume', True) and any(Path(case['output'], 'summary.json').exists() for case in cases):
            raise ValueError('Completed results exist; enable verified resume or select a fresh output directory')
        if self.config.get('resume', True):
            if manifest.exists():
                previous = json.loads(manifest.read_text())
                if previous['fingerprint'] != provenance['fingerprint']:
                    raise ValueError('Benchmark resume inputs or protocol changed; select a fresh output directory')
            elif any(Path(case['output'], 'summary.json').exists() for case in cases):
                raise ValueError('Existing results lack verified provenance; select a fresh output directory')
        manifest.write_text(json.dumps(provenance, indent=2) + '\n')
        results, pending = [], []
        for case in cases:
            summary = Path(case['output'], 'summary.json')
            if self.config.get('resume', True) and summary.exists():
                results.append(json.loads(summary.read_text()))
            else:
                pending.append(case)
        # Fork shares immutable full-database arrays without repeated per-job disk reads.
        context = multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=self.config.get('workers', 8), mp_context=context,
                                 initializer=_initialize_worker, initargs=(self.config['checkpoint'],)) as pool:
            for result in pool.map(_run_case, pending, chunksize=1):
                results.append(result)
                print(f'Completed {len(results)} / {len(cases)} searches', flush=True)
        groups = {}
        for condition in ('with_ligand_targets', 'with_ligand_decoys', 'without_ligand_targets', 'without_ligand_decoys'):
            rows = [row for row in results if row['condition'] == condition]
            groups[condition] = dict(cases=len(rows), candidates=sum(row['candidates'] for row in rows),
                                    accepted=sum(row['accepted'] for row in rows), cases_with_hits=sum(row['accepted'] > 0 for row in rows),
                                    recovered=sum(row['recovered'] for row in rows))
        report = dict(task='benchmark', completed=len(results), expected=112, full_public_small_decoy_benchmark=len(results) == 112,
                      groups=groups, cases=results, bitwise_pose_equivalence=False,
                      provenance_fingerprint=provenance['fingerprint'],
                      pose_boundary='Modern Open3D0.19 RANSAC differs from original Open3D0.5 sampling/stopping semantics.',
                      protocol_checkpoint=self.config['checkpoint'], descriptor_cutoff=3.0, nn_score_cutoff=0.8)
        Path(self.config['output'], 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return report
