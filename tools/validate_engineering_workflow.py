"""Compare real replayed SeedWorkflow rows against the preserved engineering baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from masif_torch.configuration import load_config_file
from masif_torch.shared.design.features import match_descriptors
from masif_torch.shared.design.registration import RegistrationBackend
from masif_torch.tasks.seed.workflow import SeedWorkflow
from masif_torch.validation import deterministic
from validate_engineering_geometry import baseline_namespace, real_poses, same_rng


def baseline_workflow(features: dict) -> type:
    """Load the preserved Seed owner with its preserved geometry and scoring dependencies."""
    path = Path('artifacts/engineering_acceleration/baseline_source/workflow.py')
    namespace = dict(__name__='masif_torch.tasks.seed._engineering_baseline',__package__='masif_torch.tasks.seed')
    exec(compile(path.read_text(),str(path),'exec'),namespace)
    for name in ['AlignmentScorer','count_clashes','match_descriptors','target_sites']:
        namespace[name] = features[name]
    return namespace['SeedWorkflow']


class RecordedPoses(RegistrationBackend):
    """Preserve actual legacy poses, including historical Open3D fitness values above one."""

    def __init__(self, rows: list[dict]):
        """Keep finite captured transforms/fitness for the test's exact candidate keys."""
        self.rows = {row['key']:row for row in rows}

    def align(self, key: str, source: dict, target: dict) -> tuple[np.ndarray, float]:
        """Return each original pose without applying modern replay fitness validation."""
        row = self.rows[key]
        return np.asarray(row['transformation']).copy(), float(row['fitness'])


def recorded_owner(base: type, config: dict, provider: RegistrationBackend, candidates: list[dict]):
    """Replay the same real candidate order through either unchanged or optimized Seed owner."""
    class RecordedCandidates(base):
        """Use already captured matches to isolate geometry/model orchestration changes."""

        def matches(self) -> list[dict]:
            """Return fresh metadata for all original accepted candidates in original order."""
            return [dict(row) for row in candidates]

    return RecordedCandidates(config,provider)


def validate(config: dict) -> dict:
    """Check replayed row values, real descriptor queries and sampling state exactly."""
    deterministic(42)
    torch.set_num_threads(1)
    features = baseline_namespace(Path('artifacts/engineering_acceleration/baseline_source/features.py'))
    original_owner = baseline_workflow(features)
    root=Path('artifacts/benchmark_full/run/surface_cache')
    fixtures=Path('artifacts/benchmark_full/legacy_comparison/with_ligand/targets/1A7X_A')
    report=json.loads((fixtures/'results.json').read_text())
    accepted=[row for row in report['candidates'] if row.get('accepted')]
    target,poses=real_poses(root,fixtures)
    sources={pose['surface'].name:pose['surface'] for pose in poses}
    descriptor_checks=0
    for source in sources.values():
        for center in [104,475,136]:
            for cutoff in [0.,.75]:
                before=features['match_descriptors'](source,target.flipped_descriptors[center],cutoff,3.)
                after=match_descriptors(source,target.flipped_descriptors[center],cutoff,3.)
                if not all(np.array_equal(first,second) for first,second in zip(before,after,strict=True)):
                    raise AssertionError('Descriptor screening indices/order/distances changed')
                descriptor_checks+=1
    with tempfile.TemporaryDirectory() as temporary:
        candidates=[{name:row[name] for name in ['key','site','source','source_vertex','target_vertex','center_desc_dist']}
                    for row in accepted]
        workflow_config=dict(task='seed',target=str(root/'with_ligand/1A7X_A_target.npz'),
             database=[str(root/'without_ligand'/(name+'_source.npz')) for name in sources],
             checkpoint=config['checkpoint'],output=str(Path(temporary)/'result'),device='cpu',seed=42,
             selected_vertices=[104,475,136],num_sites=3,nn_score_cutoff=.8,allowed_CA_clashes=0,
             allowed_heavy_atom_clashes=5,fitness_cutoff=.3,surface_outward_shift=.25)
        outputs,owners,durations={},{},{}
        for name,base in [('baseline',original_owner),('optimized',SeedWorkflow)]:
            owner=recorded_owner(base,workflow_config,RecordedPoses(accepted),candidates)
            start=time.perf_counter();outputs[name]=owner.search();durations[name]=time.perf_counter()-start
            owners[name]=owner
        for first,second in zip(outputs['baseline'],outputs['optimized'],strict=True):
            if set(first)!=set(second):
                raise AssertionError('Workflow row fields changed')
            for name in first:
                equal=(np.array_equal(first[name],second[name],equal_nan=True) if isinstance(first[name],np.ndarray)
                       else first[name]==second[name])
                if not equal:
                    raise AssertionError(f'Workflow {name} changed for {first["key"]}')
        if not same_rng(owners['baseline'].scorer.rng.get_state(),owners['optimized'].scorer.rng.get_state()):
            raise AssertionError('Workflow patch caching changed sampling order')
    return dict(passed=True,replayed_real_candidates=len(accepted),all_row_fields_exact=True,
                exact_probability_importance_features_pose_clashes=True,exact_sampling_rng=True,
                descriptor_checks=descriptor_checks,exact_descriptor_indices_distances_order=True,
                captured_legacy_fitness_range=[min(row['fitness'] for row in accepted),max(row['fitness'] for row in accepted)],
                pose_provider='Test-only RecordedPoses preserves legacy count-like fitness; production Replay validation unchanged.',
                elapsed_seconds=durations,baseline_source_sha256=hashlib.sha256(
                    Path('artifacts/engineering_acceleration/baseline_source/workflow.py').read_bytes()).hexdigest(),
                optimized_source_sha256=hashlib.sha256(Path('src/tasks/seed/workflow.py').read_bytes()).hexdigest(),
                scope='53 real recorded candidates through the complete Seed search/filter path; no full benchmark.')


def main() -> None:
    """Run and save the exact real workflow replay regression."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='examples/benchmark.yaml')
    parser.add_argument('--report',default='docs/torch_migration/numerical_engineering_workflow.json')
    args=parser.parse_args()
    report=validate(load_config_file(args.config))
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
