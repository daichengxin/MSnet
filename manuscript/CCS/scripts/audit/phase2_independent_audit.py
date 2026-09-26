import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch

experiment_root = Path(os.environ.get('CCS_SCALING_ROOT', Path.cwd() / 'ccs_scaling_runs')).resolve()
root = experiment_root / 'phase2_compute_scaling'
phase1 = experiment_root / 'phase1_dataset_scaling'
expected_n = [500, 2500, 10000, 50000, 100000, 280000, 465356]
expected_seeds = [42, 123, 2026]
expected_steps = [10, 30, 50, 100, 300, 1000, 3000, 10000, 30000, 50000]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

cfg = json.loads((root / 'compute_scaling_config.json').read_text())
runs = pd.read_csv(root / 'compute_scaling_runs.csv')
agg = pd.read_csv(root / 'compute_scaling_aggregate.csv')
expected_triplets = {(n, s, k) for n in expected_n for s in expected_seeds for k in expected_steps}
actual_triplets = set(map(tuple, runs[['N', 'seed', 'optimizer_step']].itertuples(index=False, name=None)))
assert len(runs) == 210 and actual_triplets == expected_triplets and not runs.duplicated(['N', 'seed', 'optimizer_step']).any()
assert len(agg) == 70 and set(agg.n_seeds) == {3}
assert runs['test_evaluations'].eq(1).all()
for col in ['train_L1', 'validation_L1', 'test_L1', 'median_relative_error', 'Pearson_r', 'R2']:
    assert runs[col].map(math.isfinite).all(), col
assert runs['test_L1'].min() >= 0

checkpoint_errors = []
for row in runs.itertuples(index=False):
    path = Path(row.checkpoint)
    if not path.is_file() or sha256(path) != row.checkpoint_sha256:
        checkpoint_errors.append((row.N, row.seed, row.optimizer_step, 'missing-or-hash'))
        continue
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('optimizer_step') != row.optimizer_step:
        checkpoint_errors.append((row.N, row.seed, row.optimizer_step, state.get('optimizer_step')))
assert not checkpoint_errors, checkpoint_errors[:5]

index_sha = {}
for seed in expected_seeds:
    path = phase1 / 'nested_indices' / f'seed_{seed}_permutation.npy'
    index_sha[str(seed)] = sha256(path)
assert index_sha == cfg['nested_index_sha256']

run_jsons = sorted((root / 'runs').glob('N*_seed*.json'))
assert len(run_jsons) == 21
for path in run_jsons:
    data = json.loads(path.read_text())
    assert data['final_optimizer_step'] == 50000
    assert data['checkpoint_steps'] == expected_steps
    assert [item['optimizer_step'] for item in data['evaluations']] == expected_steps
    assert data['test_used_during_training'] is False
    assert data['validation_used_for_training_decisions'] is False
    assert data['trainable_params'] == 712428

bad_text = []
for path in list((root / 'scripts').glob('*')) + [root / 'compute_scaling_config.json']:
    if path.is_file():
        text = path.read_text(errors='ignore').lower()
        if 'pxd017703' in text and 'excluded' not in text:
            bad_text.append(str(path))
assert not bad_text, bad_text

out = {
    'status': 'PASS',
    'completed_runs': 21,
    'checkpoint_rows': len(runs),
    'aggregate_rows': len(agg),
    'checkpoint_files_verified_with_sha256_and_metadata': 210,
    'true_step_audit': 'PASS',
    'all_checkpoints_present': 'PASS',
    'three_seeds_complete': 'PASS',
    'nested_indices_reused': 'PASS',
    'pxd017703_excluded': 'PASS',
    'test_not_used_for_training_decisions': 'PASS',
    'test_L1_min': float(runs.test_L1.min()),
    'test_L1_max': float(runs.test_L1.max()),
    'artifact_sha256': {p.name: sha256(p) for p in [
        root / 'compute_scaling_runs.csv', root / 'compute_scaling_aggregate.csv',
        root / 'compute_scaling_config.json', root / 'compute_scaling_report.md',
        root / 'compute_scaling_preview.png', root / 'compute_scaling_preview.svg']},
}
print(json.dumps(out, indent=2))
