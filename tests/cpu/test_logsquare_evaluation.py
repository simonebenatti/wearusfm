"""Strict real checkpoint load, legacy defaults and production-path LQ sensitivity."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import logsquare_evaluation as V
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from test_tokenizer_activation import CFG
from test_pose_eval import _setup

CODE, HASH = 'a' * 40, 'b' * 64


def pin(path):
    return {k: V.P.file_identity(path)[k] for k in ('path', 'sha256', 'bytes')}


def save_scene(tmp_path, *, log=True):
    torch.manual_seed(9)
    cfg = replace(R.sanity_config(), max_steps=5000, model=replace(CFG, pose_head=True, tokenizer_logsquare=log))
    model = WearUsFM(cfg.model).eval()
    with torch.no_grad():
        model.tokenizer.fe_gain.fill_(2.7)
        model.pose_head.out.weight.normal_(std=.05)
        if log:
            model.tokenizer.logsquare_weight.normal_(std=.03)
    config = R.config_to_dict(cfg)
    if not log:
        for key in V.DEFAULTS:
            config['model'].pop(key)
    signature = {'manifest_sha256': 'c' * 64}
    state = {'step': 5000, 'config': config, 'teacher': model.state_dict(), 'student': copy.deepcopy(model.state_dict()),
             'pose_stats_sha256': HASH, 'data_signature': signature}
    cp, cf, rp = (tmp_path / x for x in ('checkpoint_step0005000.pt', 'config.json', 'receipt.json'))
    torch.save(state, cp)
    cf.write_text(json.dumps(config))
    receipt = {'protocol': V.TRAINING_PROTOCOL if log else 'tokenizer-training-20261103-v1', 'complete': True,
        'arm': V.ARM if log else 'T100-L', 'checkpoint_step': 5000, 'checkpoint': str(cp),
        'checkpoint_sha256': pin(cp)['sha256'], 'config_path': str(cf), 'config_file_sha256': pin(cf)['sha256'],
        'code_sha': CODE, 'plan_sha256': HASH, 'release_sha256': HASH}
    rp.write_text(json.dumps(receipt))
    p = {**pin(cp), 'config_path': str(cf), 'config_sha256': pin(cf)['sha256'], 'receipt_path': str(rp), 'receipt_sha256': pin(rp)['sha256'],
         'training_code_sha': CODE, 'training_plan_sha256': HASH, 'release_sha256': HASH}
    return p, R.config_to_dict(cfg), signature, state, receipt


def guard(scene, arm=V.ARM):
    p, cfg, sig, _, _ = scene
    return V.validate_checkpoint(p, cfg, 5000, HASH, sig, arm=arm)


def test_loaded_log_matrix_gain_and_all64_columns_match(tmp_path):
    scene = save_scene(tmp_path)
    before = pin(scene[0]['path'])
    got = guard(scene)
    assert got['postload_recipe_passed'] and got['fe_gain'] == pytest.approx(2.7)
    for kind in ('student', 'teacher'):
        expected = torch.linalg.vector_norm(scene[3][kind][V.NEW_WEIGHT].double(), dim=0).numpy()
        assert np.array_equal(got['log_weights'][kind]['column_l2_norms'], expected)
        assert got['log_weights'][kind]['families']['mlp']['n'] == 16
        assert got['log_weights'][kind]['descriptive_only']
    assert pin(scene[0]['path']) == before


@pytest.mark.parametrize('bad', ['step', 'flag', 'epsilon', 'missing_weight', 'wrong_shape', 'nonfinite', 'gain', 'signature', 'stats', 'receipt', 'code', 'release', 'plan', 'STOP', 'missing_other_field'])
def test_guard_fails_closed_with_repinned_corruption(tmp_path, bad):
    scene = save_scene(tmp_path)
    p, cfg, sig, state, rec = scene
    if bad == 'step': state['step'] = 19618
    elif bad == 'flag': state['config']['model']['tokenizer_logsquare'] = False
    elif bad == 'epsilon': state['config']['model']['tokenizer_logsquare_eps'] = .01
    elif bad == 'missing_weight': state['teacher'].pop(V.NEW_WEIGHT)
    elif bad == 'wrong_shape': state['teacher'][V.NEW_WEIGHT] = torch.zeros(16, 63)
    elif bad == 'nonfinite': state['teacher'][V.NEW_WEIGHT][0, 0] = float('nan')
    elif bad == 'gain': state['teacher']['tokenizer.fe_gain'].fill_(0)
    elif bad == 'signature': state['data_signature'] = {}
    elif bad == 'stats': state['pose_stats_sha256'] = 'd' * 64
    elif bad == 'receipt': rec['complete'] = False
    elif bad == 'code': rec['code_sha'] = 'e' * 40
    elif bad == 'release': rec['release_sha256'] = 'f' * 64
    elif bad == 'plan': rec['plan_sha256'] = 'e' * 64
    elif bad == 'STOP': (tmp_path / 'STOP').touch()
    elif bad == 'missing_other_field': state['config']['model'].pop('pose_head')
    torch.save(state, p['path']); p.update(pin(p['path'])); rec['checkpoint_sha256'] = p['sha256']
    Path(p['config_path']).write_text(json.dumps(state['config'])); p['config_sha256'] = pin(p['config_path'])['sha256']; rec['config_file_sha256'] = p['config_sha256']
    Path(p['receipt_path']).write_text(json.dumps(rec)); p['receipt_sha256'] = pin(p['receipt_path'])['sha256']
    with pytest.raises((ValueError, RuntimeError)):
        guard(scene)


def test_legacy_accepts_only_two_missing_defaults_and_preserves_bytes(tmp_path):
    scene = save_scene(tmp_path, log=False)
    before = Path(scene[0]['config_path']).read_bytes()
    got = guard(scene, arm='T100-L')
    assert got['model_config']['tokenizer_logsquare'] is False
    assert got['log_weights'] is None and Path(scene[0]['config_path']).read_bytes() == before
    for bad in ('window_ms', 'tokenizer_activation', 'pose_head', 'grad_checkpoint'):
        cfg = copy.deepcopy(scene[1]); cfg['model'].pop(bad)
        with pytest.raises(ValueError, match='two legacy'):
            V.canonical_config(cfg)


def loaded_pair(tmp_path, nonzero):
    torch.manual_seed(123)
    base = WearUsFM(replace(CFG, pose_head=True)).eval()
    torch.manual_seed(123)
    branch = WearUsFM(replace(CFG, pose_head=True, tokenizer_logsquare=True)).eval()
    with torch.no_grad():
        base.pose_head.out.weight.normal_(std=.03)
        branch.pose_head.out.weight.copy_(base.pose_head.out.weight)
        if nonzero:
            branch.tokenizer.logsquare_weight.normal_(std=.02)
    result = []
    for i, model in enumerate((base, branch)):
        path = tmp_path / f'model{i}.pt'
        cfg = R.config_to_dict(replace(R.sanity_config(), model=model.cfg))
        if i == 0:
            for key in V.DEFAULTS: cfg['model'].pop(key)
        torch.save({'config': cfg, 'teacher': model.state_dict()}, path)
        result.append(V.E.FF.load_model(str(path), 'cpu', 'teacher'))
    return result


@pytest.mark.parametrize('nonzero', [False, True])
def test_d3_real_pose_extract_zero_matches_active_changes(tmp_path, nonzero):
    root, pose_root, csv = _setup(tmp_path)
    recs = V.E.PE.recordings_for(csv, root, 'train')[:1]
    store = V.E.PT.PoseStore(pose_root)
    a, b = loaded_pair(tmp_path, nonzero)
    ev = V.E.PE.EvalConfig(n_windows=1, window_s=1.)
    x, y = [V.E.PE.extract(model, recs, store, ev, log=lambda _: None) for model in (a, b)]
    assert np.array_equal(x['target'], y['target']) and np.array_equal(x['mask'], y['mask']) and x['meta'] == y['meta']
    assert np.array_equal(x['lat'], y['lat']) is (not nonzero)
    assert np.array_equal(x['pred'], y['pred']) is (not nonzero)


@pytest.mark.parametrize('nonzero', [False, True])
@pytest.mark.parametrize('fs', [200., 2000.])
def test_d3_real_classification_extraction_includes_decoder(tmp_path, nonzero, fs):
    a, b = loaded_pair(tmp_path, nonzero)
    windows = np.random.default_rng(6).normal(size=(2, int(fs), 8))
    subjects = np.array(['s', 's'])
    montage = V.E.FF.myo8_montage('epn612', fs)
    outputs = [V.E.C.CR.extract_channel_readouts(m, windows, fs, montage, subjects, batch=2) for m in (a, b)]
    for layer in V.E.C.CR.LAYERS:
        assert np.array_equal(outputs[0][layer], outputs[1][layer]) is (not nonzero), layer


def test_extraction_commands_use_production_paths_and_teacher():
    p = json.loads((V.REPO / 'results/infra/logsquare_evaluation_20261104/extraction_plan.json').read_text())
    cmds = V.E.extraction_commands('extract-pose', p, '/candidate.pt', Path('/new'), None, 'cuda')
    assert set(cmds) == {'train', 'val'}
    assert all('teacher' in c and 'scripts/pose_eval.py' in ' '.join(c) for c in cmds.values())
    c = V.E.extraction_commands('extract-classification', p, '/candidate.pt', Path('/new'), 'p2', 'cuda')['p2']
    assert c[c.index('--batch') + 1] == '64' and 'scripts/channel_readout.py' in ' '.join(c)


def test_unreviewed_plan_fails_before_inputs_open(monkeypatch):
    path = V.REPO / 'results/infra/logsquare_evaluation_20261104/extraction_plan.json'
    monkeypatch.setattr(V.P, 'code_identity', lambda _: CODE)
    with pytest.raises(ValueError, match='review/pins'):
        V.load_context(path, 5000, CODE, pin(path)['sha256'])


def test_benchmark_context_survives_late_pin_but_rejects_data_or_recipe_changes():
    plan = json.loads((V.REPO / 'results/infra/logsquare_evaluation_20261104/extraction_plan.json').read_text())
    first = V.context_sha256(plan)
    later = copy.deepcopy(plan)
    later.update(status='APPROVED_FOR_EXTRACTION', inference_receipt={'path': '/receipt', 'sha256': HASH}, allocations={'pose_5000': {}})
    later['checkpoints']['19618'] = {'path': '/future', 'sha256': HASH}
    assert V.context_sha256(later) == first
    for key in ('inputs', 'paths', 'candidate_config', 'training_plan', 'source_sha256'):
        changed = copy.deepcopy(later); changed[key]['changed'] = 'unapproved'
        assert V.context_sha256(changed) != first
    changed = copy.deepcopy(later); changed['checkpoints']['5000'] = {'sha256': HASH}
    assert V.context_sha256(changed) != first
