import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch


experiment_root = Path(os.environ.get('CCS_SCALING_ROOT', Path.cwd() / 'ccs_scaling_runs')).resolve()
root = experiment_root / 'phase3_parameter_scaling'
p0 = experiment_root / 'phase0d' / 'artifacts'
hidden_dims = [32, 64, 128, 256, 512, 1024]
seeds = [42, 123, 2026]
params = {32: 49164, 64: 94764, 128: 235116, 256: 712428, 512: 2453484, 1024: 9081324}
frozen = {
    'training_pool.parquet': '2618f499d23429a42c282e1dae5ce17e22dbe95a8efa7ec2ade2bab62f2cb73a',
    'validation_set.parquet': 'bc997242ab689c002fac31b2378ba826cb9880069ce87b04c49e24c209b9ff0b',
    'proteometools_test.parquet': '8e771af8c1a196a2cce9676e7c172795264f0d1dfe4aef560ce26aba0f7b497f',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


for name, expected in frozen.items():
    assert sha256(p0 / name) == expected, name

manifest = json.loads((root / 'parameter_scaling_manifest.json').read_text())
runs = pd.read_csv(root / 'parameter_scaling_runs.csv')
agg = pd.read_csv(root / 'parameter_scaling_aggregate.csv')
curves = pd.read_csv(root / 'parameter_scaling_learning_curves.csv')
fits = json.loads((root / 'parameter_scaling_fit.json').read_text())

expected_pairs = {(h, s) for h in hidden_dims for s in seeds}
actual_pairs = set(map(tuple, runs[['hidden_dim', 'seed']].itertuples(index=False, name=None)))
assert len(runs) == 18 and actual_pairs == expected_pairs
assert not runs.duplicated(['hidden_dim', 'seed']).any()
assert len(agg) == 6 and set(agg.n_seeds) == {3}
assert dict(zip(agg.hidden_dim, agg.actual_parameter_count)) == params
assert runs.apply(lambda r: r.actual_parameter_count == params[r.hidden_dim], axis=1).all()

for col in ['best_validation_L1', 'test_L1', 'median_relative_error', 'Pearson_r', 'R2', 'wall_time_seconds', 'GPU_time_seconds']:
    assert runs[col].map(math.isfinite).all(), col
assert runs.status.eq('complete').all()
assert runs.fresh_random_initialization.eq(True).all()
assert runs.pretrained.eq(False).all() and runs.fine_tuning.eq(False).all() and runs.warm_start.eq(False).all()
assert runs.test_evaluations.eq(1).all()
assert runs.test_used_for_training_decisions.eq(False).all()
assert runs.validation_only_checkpoint_selection.eq(True).all()
assert runs.training_sha256.eq(frozen['training_pool.parquet']).all()
assert runs.validation_sha256.eq(frozen['validation_set.parquet']).all()
assert runs.test_sha256.eq(frozen['proteometools_test.parquet']).all()

for hidden, group in runs.groupby('hidden_dim'):
    assert group.initial_state_sha256.nunique() == 3, hidden

checkpoint_verified = 0
for row in runs.itertuples(index=False):
    path = Path(row.checkpoint)
    assert path.is_file() and sha256(path) == row.checkpoint_sha256
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['selection'] == 'minimum validation L1 only'
    assert state['hidden_dim'] == row.hidden_dim and state['actual_parameter_count'] == row.actual_parameter_count
    assert state['seed'] == row.seed and state['epoch'] == row.best_epoch
    assert math.isclose(state['validation_L1'], row.best_validation_L1, rel_tol=0, abs_tol=1e-12)
    assert state['global_optimizer_step'] == row.optimizer_steps_at_best
    assert state['examples_seen'] == row.examples_seen_at_best
    assert state['initial_state_sha256'] == row.initial_state_sha256
    checkpoint_verified += 1

for (hidden, seed), group in curves.groupby(['hidden_dim', 'seed']):
    group = group.sort_values('epoch')
    row = runs[(runs.hidden_dim == hidden) & (runs.seed == seed)].iloc[0]
    assert list(group.epoch) == list(range(1, int(row.epochs_trained) + 1))
    assert int(group.optimizer_steps.iloc[-1]) == int(row.true_optimizer_steps)
    assert int(group.examples_seen.iloc[-1]) == int(row.examples_seen)
    assert int(row.true_optimizer_steps) == int(row.epochs_trained) * 485
    min_row = group.loc[group.validation_L1.idxmin()]
    assert int(min_row.epoch) == int(row.best_epoch)
    assert math.isclose(float(min_row.validation_L1), float(row.best_validation_L1), rel_tol=0, abs_tol=1e-12)
    assert int(row.epochs_trained) <= 200

assert manifest['pretrained'] is False and manifest['fine_tuning'] is False and manifest['warm_start'] is False
assert manifest['architecture_audit']['baseline_equivalence_check'] == 'PASS'
assert manifest['actual_parameter_counts'] == {str(k): v for k, v in params.items()}
assert json.loads((root / 'oom_preflight.json').read_text())['status'] == 'PASS'
assert all(math.isfinite(fits[m]['R2']) for m in ['pure_power_law', 'floor_power_law'])

artifacts = [
    'parameter_scaling_runs.csv', 'parameter_scaling_aggregate.csv',
    'parameter_scaling_learning_curves.csv', 'parameter_scaling_fit.json',
    'parameter_scaling_report.md', 'parameter_scaling_preview.png',
    'parameter_scaling_preview.svg', 'parameter_scaling_floor_diagnostic.png',
    'parameter_scaling_floor_diagnostic.svg',
]
for name in artifacts:
    assert (root / name).is_file() and (root / name).stat().st_size > 0

best_mean = agg.loc[agg.mean_test_L1.idxmin()]
audit = {
    'status': 'PASS',
    'completed_runs': 18,
    'expected_runs': 18,
    'checkpoint_files_verified_with_sha256_and_metadata': checkpoint_verified,
    'actual_parameter_counts_verified': 'PASS',
    'random_initialization_unique_across_seeds': 'PASS',
    'same_frozen_training_validation_test': 'PASS',
    'validation_only_checkpoint_selection': 'PASS',
    'test_evaluated_once_after_selection': 'PASS',
    'true_optimizer_steps': 'PASS',
    'pxd017703_excluded': 'PASS',
    'best_mean_model': {
        'hidden_dim': int(best_mean.hidden_dim),
        'actual_parameter_count': int(best_mean.actual_parameter_count),
        'mean_test_L1': float(best_mean.mean_test_L1),
        'SD_test_L1': float(best_mean.SD_test_L1),
    },
    'artifact_sha256': {name: sha256(root / name) for name in artifacts},
}
(root / 'phase3_independent_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
