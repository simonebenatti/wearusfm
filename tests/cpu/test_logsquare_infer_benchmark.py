"""Independent timing/aggregation oracles and train-only sampling gates."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import logsquare_infer_benchmark as B
from test_tokenizer_activation import inputs


def test_fixed_order_sync_warmup_excluded_and_input_unchanged():
    order, syncs = [], []
    times = iter([float(i) for i in range(32)])
    def forward(model, inp):
        order.append(model)
        return {'pool': torch.ones(1, 2)}
    result, _ = B.paired_measure(dict(zip(B.ARMS, B.ARMS)), inputs(200), forward, warmup=2, repeats=6,
                                sync=lambda: syncs.append(1), clock=lambda: next(times))
    assert order == list(B.ARMS + B.ARMS[::-1]) * 4
    assert len(syncs) == 32 and len(result['calls']) == 16
    assert sum(not x['warmup'] for x in result['calls']) == 12 and result['rho_infer'] == 1


def test_nonfinite_candidate_rejected_not_hidden_by_baseline_result():
    def forward(model, inp): return {'pool': torch.tensor([float('nan') if model == 'bad' else 1.])}
    with pytest.raises(ValueError, match='nonfinite'):
        B.paired_measure({'T100-L': 'ok', 'T100-LQ': 'bad'}, inputs(200), forward, warmup=2, repeats=6, sync=lambda: None)


def test_mutated_support_rejected():
    def forward(model, inp):
        inp.qc_valid[0] = ~inp.qc_valid[0]
        return {'pool': torch.tensor([1.])}
    # An odd mutation is guaranteed to remain after16calls by changing once.
    called = []
    def once(model, inp):
        if not called: inp.qc_valid[0] = ~inp.qc_valid[0]
        called.append(1); return {'pool': torch.ones(1)}
    with pytest.raises(ValueError, match='mutated'):
        B.paired_measure({'T100-L': None, 'T100-LQ': None}, inputs(200), once, warmup=2, repeats=6, sync=lambda: None)


def test_scale_completion_preserves_original_first10_subject_windows():
    subjects = np.array(['a'] * 60 + ['b'] * 30)
    selected = B.first64_with_scale_completion(np.zeros((90, 2, 1)), subjects)
    assert selected.tolist() == list(range(70))
    for s in ('a', 'b'):
        assert np.array_equal(np.flatnonzero(subjects[selected] == s)[:10], np.flatnonzero(subjects == s)[:10])
    with pytest.raises(ValueError, match='fewer'):
        B.first64_with_scale_completion(np.zeros((63, 2, 1)), subjects[:63])


def test_aggregation_uses_max_cap_historical_window_weight_point_and_rounding():
    groups = {'a': {'rho_infer': .8}, 'b': {'rho_infer': 1.2}}
    h = {'p1_19618': {'historical_window_counts': {'a': 9, 'b': 1}, 'historical_elapsed_seconds': 1000, 'historical_walltime_seconds': 1800}}
    p = B.resource_proposal(groups, h)['jobs']['p1_19618']
    assert p['rho_infer_cap'] == 1.2 and p['rho_infer_point'] == pytest.approx(.84)
    assert p['walltime_seconds'] == 2400 and p['expected_seconds_proxy'] == pytest.approx(840)
    assert p['gpu_hours_cap'] == pytest.approx(2 / 3)
    h['p1_19618']['historical_window_counts'] = {'a': 1}
    assert B.resource_proposal(groups, h)['jobs']['p1_19618']['walltime_seconds'] == 1800


@pytest.mark.parametrize('fault', ['missing_group', 'missing_count', 'negative', 'nan', 'over_cap'])
def test_budget_fails_for_incomplete_or_invalid_evidence(fault):
    groups = {'a': {'rho_infer': 1.}}
    h = {'p1_19618': {'historical_window_counts': {'a': 10}, 'historical_elapsed_seconds': 1000, 'historical_walltime_seconds': 1800}}
    if fault == 'missing_group': groups.clear()
    if fault == 'missing_count': h['p1_19618']['historical_window_counts'] = {}
    if fault == 'negative': h['p1_19618']['historical_window_counts']['a'] = -1
    if fault == 'nan': groups['a']['rho_infer'] = float('nan')
    if fault == 'over_cap': groups['a']['rho_infer'] = 100
    with pytest.raises(ValueError): B.resource_proposal(groups, h)


def test_shape_coverage_failure_does_not_invent_weights():
    rows = []
    for group, fs in B.GROUPS.items():
        n = 10 if group == 'pose' else 64
        calls = [{'round': k, 'arm': arm, 'warmup': k < 2, 'seconds': 1. if arm == 'T100-L' else 2.}
                 for k in range(8) for arm in (B.ARMS if k % 2 == 0 else B.ARMS[::-1])]
        rows.append({'group': group, 'input': {'counts': [8] * n, 'shapes': [[8, int(fs)]] * n, 'fs': [fs] * n, 'n_windows': n},
            'mean_seconds': {'T100-L': 1., 'T100-LQ': 2.}, 'rho_infer': 2., 'calls': calls,
            'precision': 'fp32_no_autocast' if group == 'pose' else 'bf16_cuda_with_decoder'})
    assert set(B.group_ratios(rows)) == set(B.GROUPS)
    with pytest.raises(ValueError, match='coverage'): B.group_ratios(rows[:-1])
    wrong = copy.deepcopy(rows[0]); wrong['input']['shapes'][0][1] = 100
    with pytest.raises(ValueError, match='multiple shapes'): B.group_ratios(rows + [wrong])


def test_p2_train_only_file_reader_does_not_open_excluded_subjects(tmp_path, monkeypatch):
    root = tmp_path / 'trainingJSON'
    for s in ('user1', 'user2', 'user3'):
        p = root / s; p.mkdir(parents=True); (p / (s + '.json')).write_text('{}')
    roles = [{'train': ['trainingJSON/user2'], 'val': ['trainingJSON/user1'], 'held': ['trainingJSON/user3']}]
    monkeypatch.setattr(B.C.CP, 'p2_internal_subjects', lambda s: s)
    monkeypatch.setattr(B.C.CP, 'fold_roles', lambda s: roles)
    opened = []
    def read(path):
        opened.append(path.parent.name)
        assert path.parent.name == 'user2'
        return [np.ones((1000, 8)) for _ in range(80)], ['label'] * 80, 200.
    monkeypatch.setattr(B.B, 'load_epn_user', read)
    w, s, fs, _, selection = B.load_train_p2({'paths': {'epn_roots': [str(root)]}}, 'epn612')
    assert opened == ['user2'] and len(w) == 64 and set(s) == {'trainingJSON/user2'} and fs == 200
    assert selection['fold_train'] == roles[0]['train']


def test_review_plan_freezes_selection_aggregation_and_null_candidate_pins():
    root = B.V.REPO / 'results/infra/logsquare_evaluation_20261104'
    plan = json.loads((root / 'inference_plan.json').read_text())
    assert plan['protocol'] == B.PROTOCOL
    assert plan['status'] in ('REVIEW_REQUIRED', 'APPROVED_FOR_BENCHMARK')
    assert plan['selection'] == B.SELECTION and plan['aggregation'] == B.POLICY
    assert plan['allocation'] == {'nodes': 1, 'gpus': 1, 'cpus': 8, 'memory': '100G',
        'walltime_seconds': 1800, 'gpu_hours_cap': .5, 'allocated_core_hours_cap': 4.}

    def pinned_local_json(pin):
        path = B.V.REPO / pin['path']
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin['sha256']
        if 'bytes' in pin:
            assert type(pin['bytes']) is int and len(raw) == pin['bytes']
        return json.loads(raw)

    assert B.V.REPO / plan['extraction_plan']['path'] == root / 'extraction_plan.json'
    e = pinned_local_json(plan['extraction_plan'])
    assert e['protocol'] == B.V.PROTOCOL and e['arm'] == 'T100-LQ'
    assert e['status'] == plan['status']
    assert set(e['checkpoints']) == {'5000', '19618'}
    assert e['checkpoints']['19618'] is None
    assert e['inference_receipt'] is None and e['allocations'] == {}
    assert e['source_sha256'] == plan['source_sha256']
    if plan['status'] == 'REVIEW_REQUIRED':
        assert e['checkpoints']['5000'] is None
    else:
        # Approval requires the immutable 5k endpoint and its full provenance;
        # accepting the new state must not also accept the old placeholders.
        cp = e['checkpoints']['5000']
        assert isinstance(cp, dict)
        assert set(cp) == {'path', 'sha256', 'bytes', 'config_path', 'config_sha256',
            'receipt_path', 'receipt_sha256', 'training_code_sha',
            'training_plan_sha256', 'release_sha256'}
        for key in ('sha256', 'config_sha256', 'receipt_sha256',
                    'training_plan_sha256', 'release_sha256', 'training_code_sha'):
            digest = cp[key]
            assert isinstance(digest, str) and len(digest) == (40 if key == 'training_code_sha' else 64)
            assert set(digest) <= set('0123456789abcdef')
        assert type(cp['bytes']) is int and cp['bytes'] > 0
        assert all(Path(cp[key]).is_absolute() for key in ('path', 'config_path', 'receipt_path'))
        training = pinned_local_json(e['training_plan'])
        assert training['status'] == 'APPROVED_FOR_TRAINING'
        assert cp['path'] == str(Path(training['run_dir']) / 'checkpoint_step0005000.pt')
        assert cp['receipt_path'] == training['phase_receipts']['5000']
        assert cp['config_path'] == str(Path(cp['receipt_path']).with_name('config.json'))
        assert cp['training_plan_sha256'] == e['training_plan']['sha256']
        assert cp['release_sha256'] == training['training_release']['sha256']
        pinned_local_json(training['training_release'])
        assert cp['config_sha256'] == e['candidate_config']['sha256']
        pinned_local_json(e['candidate_config'])


@pytest.mark.parametrize('name', ['logsquare_infer_benchmark', 'logsquare_evaluation'])
def test_launchers_syntax_and_fail_without_required_arguments(name):
    path = B.V.REPO / 'scripts/slurm' / (name + '.sbatch')
    subprocess.run(['bash', '-n', str(path)], check=True)
    completed = subprocess.run(['bash', str(path)], capture_output=True)
    assert completed.returncode == 2 and b'arguments' in completed.stderr
    text = path.read_text()
    assert 'submit_job' not in text and 'sbatch ' not in text and '--expected-plan-sha256' in text and 'code_dir_setup' in text


def file_pin(path):
    return {k: B.P.file_identity(path)[k] for k in ('path', 'sha256', 'bytes')}


def write_pin(path, obj):
    path.write_text(json.dumps(obj)); return file_pin(path)


def all_group_records():
    result = []
    for g, fs in B.GROUPS.items():
        n = 10 if g == 'pose' else 64
        calls = [{'round': k, 'arm': arm, 'warmup': k < 2, 'seconds': 1. if arm == 'T100-L' else 2.}
            for k in range(8) for arm in (B.ARMS if k % 2 == 0 else B.ARMS[::-1])]
        result.append({'group': g, 'input': {'n_windows': n, 'fs': [fs] * n, 'counts': [8] * n, 'shapes': [[8, int(fs)]] * n},
            'precision': 'fp32_no_autocast' if g == 'pose' else 'bf16_cuda_with_decoder',
            'calls': calls, 'mean_seconds': {'T100-L': 1., 'T100-LQ': 2.}, 'rho_infer': 2.})
    return result


def inference_scene(tmp_path):
    specs = {}
    for key in ('pose_5000', 'pose_19618', 'p1_19618', 'p2_19618'):
        step = 5000 if key.endswith('5000') else 19618
        names = ['train', 'val'] if key.startswith('pose') else B.E.DATASETS[key.split('_')[0]]
        outputs = {}
        for name in names:
            prov = {'arm': 'T100-L', 'which': 'teacher', 'checkpoint_step': step, 'cache_sha256': 'a' * 64,
                    'code_sha': 'b' * 40, 'checkpoint_sha256': 'c' * 64, 'n_windows': 100}
            pp = tmp_path / (key + '_' + name + '.json'); identity = write_pin(pp, prov)
            outputs[name] = {'sha256': prov['cache_sha256'], 'provenance_path': str(pp), 'provenance_sha256': identity['sha256']}
        rec = {'protocol': B.E.PROTOCOL, 'complete': True, 'which': 'teacher', 'arm': 'T100-L', 'checkpoint_step': step,
               'outputs': outputs, 'code_sha': 'b' * 40, 'checkpoint_sha256': 'c' * 64, 'elapsed_seconds': 1000}
        specs[key] = {'receipt': write_pin(tmp_path / (key + '.json'), rec), 'walltime_seconds': 1800 if key == 'p1_19618' else 2700}
    history, _ = B.historical_jobs(specs)
    records = all_group_records()
    groups = B.group_ratios(records)
    plan = json.loads((B.V.REPO / 'results/infra/logsquare_evaluation_20261104/extraction_plan.json').read_text())
    plan['checkpoints']['5000'] = {'sha256': 'd' * 64}
    plan['_benchmark_context_sha256'] = B.V.context_sha256(plan)
    infer = {'protocol': B.PROTOCOL, 'status': 'APPROVED_FOR_BENCHMARK', 'historical_jobs': specs, 'selection': B.SELECTION,
             'aggregation': B.POLICY, 'source_sha256': plan['source_sha256'], 'control_checkpoint': {'sha256': 'c' * 64}}
    ip = write_pin(tmp_path / 'inferplan.json', infer)
    report = {'protocol': B.PROTOCOL, 'complete': True, 'code_sha': 'e' * 40, 'plan_sha256': ip['sha256'],
              'aggregation': B.POLICY, 'source_sha256': plan['source_sha256'], 'groups': groups, 'records': records,
              'historical_jobs': history, 'resource_proposal': B.resource_proposal(groups, history),
              'benchmark_context_sha256': plan['_benchmark_context_sha256'], 'checkpoint_step': 5000,
              'control_checkpoint_sha256': 'c' * 64, 'candidate_checkpoint_sha256': 'd' * 64}
    plan['inference_plan'] = ip
    plan['inference_code_sha'] = 'e' * 40
    plan['inference_receipt'] = write_pin(tmp_path / 'inference_receipt.json', report)
    plan['allocations'] = report['resource_proposal']['jobs']
    return plan, report


def test_pinned_inference_budget_reconstructed_before_extraction(tmp_path):
    plan, report = inference_scene(tmp_path)
    key, states = B.V.allocation_for(plan, 'extract-pose', 5000, None, 5400)
    assert key == 'pose_5000' and 'inference_plan' in states and 'history_p1_19618_ninapro_db2' in states


@pytest.mark.parametrize('fault', ['context', 'mean', 'order', 'precision', 'weights', 'code', 'plan', 'control', 'candidate', 'source', 'cap'])
def test_repinned_benchmark_report_cannot_change_rules_or_identity(tmp_path, fault):
    plan, report = inference_scene(tmp_path)
    if fault == 'context': report['benchmark_context_sha256'] = 'f' * 64
    if fault == 'mean': report['records'][0]['mean_seconds']['T100-LQ'] = 1.
    if fault == 'order': report['records'][0]['calls'][0]['arm'] = 'T100-LQ'
    if fault == 'precision': report['records'][0]['precision'] = 'bf16_cuda_with_decoder'
    if fault == 'weights': report['historical_jobs']['pose_5000']['historical_window_counts']['pose'] = 1
    if fault == 'code': report['code_sha'] = 'f' * 40
    if fault == 'plan': report['plan_sha256'] = 'f' * 64
    if fault == 'control': report['control_checkpoint_sha256'] = 'f' * 64
    if fault == 'candidate': report['candidate_checkpoint_sha256'] = 'f' * 64
    if fault == 'source': report['source_sha256'] = {}
    if fault == 'cap': report['resource_proposal']['jobs']['pose_5000']['walltime_seconds'] = 3600
    plan['inference_receipt'] = write_pin(Path(plan['inference_receipt']['path']), report)
    with pytest.raises(ValueError): B.V.allocation_for(plan, 'extract-pose', 5000, None, 5400)
