"""Structural resume guard: reject incomplete rank/stream states before launch."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/pose_b19618_preflight.py'
spec = importlib.util.spec_from_file_location('pose_b19618_preflight', SCRIPT)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def state_fixture():
    cfg = {'seed': 0, 'batch_size': 32}
    rng = torch.arange(10, dtype=torch.uint8)
    state = {'student': {'weight': torch.ones(2)}, 'teacher': {'weight': torch.ones(2)},
             'step': 5000, 'config': cfg, 'world_size': 4,
             'stream': {'num_workers': 6, 'segment': 1200, 'world_size': 4},
             'data_signature': {'same': True}, 'rank_rng': [{'torch_rng': rng, 'cuda_rng': rng} for _ in range(4)],
             'torch_rng': rng, 'cuda_rng': [rng], 'alarm_counts': (0, 0),
             'optimizer': {'state': {0: {'step': torch.tensor(5000.), 'exp_avg': torch.zeros(2), 'exp_avg_sq': torch.ones(2)}},
                           'param_groups': [{'params': [0], 'lr': 1e-4}]}}
    return state, cfg


def test_complete_state_exports_metadata_without_tensor_values():
    state, cfg = state_fixture()
    summary = P.summarize_state(state, 5000, cfg, {'same': True})
    json.dumps(summary, allow_nan=False)
    assert summary['rank_rng'][0]['torch_rng']['shape'] == [10]
    assert len(summary['rank_rng'][0]['torch_rng']['sha256']) == 64
    assert summary['optimizer']['step_counts'] == {'5000.0': 1}
    assert 'weight' not in summary['model_states']['student']


@pytest.mark.parametrize('kind', ['rank_missing', 'cuda_missing', 'stream_changed', 'signature_changed'])
def test_incomplete_or_changed_resume_identity_rejected(kind):
    state, cfg = state_fixture()
    state = copy.deepcopy(state)
    if kind == 'rank_missing':
        state['rank_rng'].pop()
    elif kind == 'cuda_missing':
        state['rank_rng'][0]['cuda_rng'] = None
    elif kind == 'stream_changed':
        state['stream']['num_workers'] = 2
    else:
        state['data_signature'] = {'different': True}
    with pytest.raises(ValueError):
        P.summarize_state(state, 5000, cfg, {'same': True})


@pytest.mark.parametrize('pose_on', [False, True])
def test_signature_matches_frozen_trainer_definition(tmp_path, pose_on):
    from wearusfm.training import run as R
    from wearusfm.training.spectral_calibration import calibration_signature
    cfg = R.sanity_config()
    cfg = replace(cfg, loader=replace(cfg.loader, pose_root='/test/pose' if pose_on else None))
    manifest = tmp_path / 'manifest'
    manifest.write_bytes(b'fixed test manifest')
    scales, windows = {'a': 2.0}, {'a': 4.0}
    actual = P.data_signature(R.config_to_dict(cfg), P.sha(manifest), scales, windows)
    assert actual == calibration_signature(cfg, manifest, scales, windows)
