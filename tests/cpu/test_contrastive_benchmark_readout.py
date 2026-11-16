import copy
from dataclasses import asdict, replace
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wearusfm.training import run as R
from wearusfm.model.fm import WearUsFM
from wearusfm.training.contrastive_objectives import ContrastiveConfig, attach_contrastive_head
from wearusfm.training.contrastive_memory import WindowIdentity
from wearusfm.training.pairing import state_sha256

SPEC = importlib.util.spec_from_file_location('benchmark_readout_test', Path(__file__).resolve().parents[2]/'scripts/contrastive_benchmark_readout.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def _records(arm='CN'):
    cc = ContrastiveConfig(nce_weight=.1, rnc_weight=.2, arm=arm, grad_every=100, grad_dense_start=513, grad_dense_end=576)
    rows = []
    for step in range(1, 577):
        wn, wr = cc.weights(step - 1)
        row = {'step': step, 'contrast_nce_weight': wn, 'contrast_rnc_weight': wr, 't_step_s': 1., 't_data_s': 100.}
        if cc.wants_gradients(step - 1):
            row.update(contrast_global_grad_base=10., contrast_global_grad_nce=1., contrast_global_grad_rnc=1., t_step_s=2.)
        rows.append(row)
    return rows, {'contrastive': asdict(cc)}


def test_measured_allocation_never_double_counts_loader_and_rounds_each_phase():
    rows, _ = _records()
    budget = B.measured_allocation({'elapsed_s': 576.}, rows)
    assert budget['normal_update_mean_s'] == 1.
    assert budget['diagnostic_update_mean_s'] == 2.
    assert budget['conservative_update_s'] == pytest.approx(1.02)
    assert budget['phase_allocation_seconds'] == [8100, 22800]
    assert budget['allocated_gpu_hours'] == 4*(8100+22800)/3600


def test_dense_gradient_ratio_uses_frozen_weight_and_checks_schedule():
    rows, cfg = _records()
    result = B.component_ratios(rows, cfg)
    assert result['nce']['active'] and result['nce']['median'] == pytest.approx(.01)
    assert result['nce']['within_screening_bounds']
    assert not result['rnc']['active'] and result['rnc']['median'] == pytest.approx(.02)
    changed = copy.deepcopy(rows)
    changed[512]['contrast_nce_weight'] = 0.
    with pytest.raises(ValueError, match='scheduled'):
        B.component_ratios(changed, cfg)
    with pytest.raises(ValueError, match='full-weight'):
        B.component_ratios(rows[:-1], cfg)


def test_numpy_rank_rng_and_optimizer_step_are_exact_even_with_float_tolerance():
    rng = {'numpy': ('MT19937', np.array([1, 2, 3], dtype=np.uint32), 42, 0, 0.), 'torch': torch.tensor([3, 7], dtype=torch.uint8)}
    B._nested_equal(rng, copy.deepcopy(rng), 'rank_rng')
    changed = copy.deepcopy(rng)
    changed['numpy'][1][0] += 1
    with pytest.raises(ValueError, match='exactly'):
        B._nested_equal(rng, changed, 'rank_rng')
    with pytest.raises(ValueError, match='exactly'):
        B._nested_equal({'step': torch.tensor(576.)}, {'step': torch.tensor(575.)}, 'optimizer', .01)
    B._nested_equal({'exp_avg': torch.tensor([1.])}, {'exp_avg': torch.tensor([1.005])}, 'optimizer', .01)


@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":1e999}', '{"x":NaN}'])
def test_metric_json_rejects_duplicate_and_nonfinite_numbers(text):
    with pytest.raises(ValueError):
        B._read_json(text)


def test_first_eight_historical_comparison_is_descriptive_without_a_new_gate(tmp_path):
    reference = json.loads(B.HISTORICAL_INITIAL_REFERENCE.read_text())
    rows = [{'step': i+1, **{key: 2.*series[i] for key, series in reference['historical_first8'].items()}} for i in range(8)]
    (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report = B.initial_calibration_description(tmp_path)
    assert report['descriptive_only'] is True and 'passed' not in report
    assert all(ratio == pytest.approx(2.) for ratio in report['mean_ratios'].values())


def test_resume_uses_saved_initial_displacement_and_exact_rng_buffers(tmp_path):
    cfg = replace(R.small_config(), contrastive=asdict(ContrastiveConfig(nce_weight=.1, rnc_weight=.2)))
    model = attach_contrastive_head(WearUsFM(cfg.model))
    initial = copy.deepcopy(model.state_dict())
    # A saved calibration buffer may differ from a fresh CPU reconstruction.
    # It is an authenticated reference, not a reason to broaden the tolerance.
    buffer_name = next(name for name, tensor in model.named_buffers() if tensor.is_floating_point())
    initial[buffer_name].view(-1)[0] += 1e-4
    parameter = sorted(name for name, _ in model.named_parameters())[0]
    trained = copy.deepcopy(initial)
    trained[parameter].view(-1)[0] += 1.
    state = {'step': 576, 'student': trained, 'teacher': copy.deepcopy(trained), 'optimizer': {'step': torch.tensor(576.)},
             'contrastive_state': {}, 'rank_rng': [{'numpy': np.random.get_state(), 'torch': torch.get_rng_state()} for _ in range(4)],
             'stream': {}, 'config': R.config_to_dict(cfg), 'data_signature': {}, 'pose_stats_sha256': None,
             'contrastive_provenance': {}, 'contrastive_initial_sha256': state_sha256(initial), 'alarm_counts': [0, 0]}
    full, resumed = tmp_path/'full.pt', tmp_path/'resumed.pt'
    torch.save(state, full)
    changed = copy.deepcopy(state)
    changed['student'][parameter].view(-1)[0] += .005
    torch.save(changed, resumed)
    result = B.compare_resume(full, resumed, cfg, state_sha256(initial), initial)
    assert result['passed'] and result['relative_to_full_displacement']['student'] == pytest.approx(.005, abs=1e-6)
    changed['student'][parameter].view(-1)[0] += .02
    torch.save(changed, resumed)
    assert not B.compare_resume(full, resumed, cfg, state_sha256(initial), initial)['passed']
    changed = copy.deepcopy(state)
    changed['rank_rng'][2]['torch'][0] ^= 1
    torch.save(changed, resumed)
    with pytest.raises(ValueError, match='exactly'):
        B.compare_resume(full, resumed, cfg, state_sha256(initial), initial)
    buffers = list(model.named_buffers())
    assert buffers
    changed = copy.deepcopy(state)
    name, _ = buffers[0]
    changed['teacher'][name].view(-1)[0] += 1
    torch.save(changed, resumed)
    with pytest.raises(ValueError, match='buffers'):
        B.compare_resume(full, resumed, cfg, state_sha256(initial), initial)


def test_receipt_authentication_recomputes_evidence_and_rejects_adjusted_budget(tmp_path, monkeypatch):
    paths = {'cohort_path': '/cohort', 'full_dirs': {}, 'resumed_dirs': {}, 'cuda_path': '/cuda',
             'cpu_audit_path': '/audit', 'calibration_path': '/calibration', 'calibration_run_dir': '/calrun', 'rank_exit_path': '/rankexit'}
    original = {'status': 'PASSED_FOR_REVIEW', 'failure_reasons': [], 'training_approved': False,
                'evidence_paths': paths, 'training_allocations_gpu_hours': {'C0': 5., 'CN': 6., 'CR': 7.}}
    calls = []
    def recalculate(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(original)
    monkeypatch.setattr(B, 'read_preflight', recalculate)
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps(original))
    assert B.authenticate_preflight(receipt) == original and calls == [paths]
    changed = copy.deepcopy(original)
    changed['training_allocations_gpu_hours']['CN'] = .1
    receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='recomputation'):
        B.authenticate_preflight(receipt)


def _run_folder(tmp_path):
    rows, contrastive = _records()
    cfg = R.config_to_dict(replace(R.small_config(), max_steps=576, batch_size=32, contrastive=contrastive['contrastive']))
    provenance = {'source_sha256': 's'*64, 'calibration_sha256': 'c'*64, 'cohort_sha256': 'b'*64,
                  'input_sha256': {'pose_stats': 'p'*64}}
    cohort = {'initial_state_sha256': 'i'*64}
    engine = B.ContrastiveRuntime(cfg['contrastive'])
    engine.expected_datasets = ('A',)
    engine.queue.commit(torch.ones(1, 2), [WindowIdentity('A', 'subject', 'session', None, 0, 1000, 1000.)], step=0)
    queue = engine.queue.state_dict()
    queue['last_step'] = queue['entries'][0]['step'] = 575
    engine.queue.load_state_dict(queue)
    engine.coverage = {'updates': 320, 'rnc_active': 320, 'datasets': {'A': [10240, 10240]}}
    state = {'step': 576, 'world_size': 4, 'config': cfg, 'contrastive_initial_sha256': cohort['initial_state_sha256'],
             'contrastive_provenance': provenance, 'rank_rng': [{}, {}, {}, {}],
             'stream': {'num_workers': 6, 'segment': cfg['stream_segment'], 'world_size': 4},
             'pose_stats_sha256': 'p'*64, 'contrastive_state': engine.state_dict(),
             'student': {'weight': torch.ones(2)}, 'teacher': {'weight': torch.ones(2)}}
    summary = {'steps': 576, 'world_size': 4, 'workers_per_rank': 6, 'batch_per_process': 8, 'accum_steps': 1, 'stopped': 'max_steps',
               'contrastive': {'initial_sha256': cohort['initial_state_sha256'], 'config': cfg['contrastive'], 'provenance': provenance,
                               'optimizer_updates': 576, 'queue_last_step': 575, 'expected_datasets': ['A'],
                               'coverage_completed_blocks': [], 'coverage_partial_block': engine.coverage_report(),
                               'final_student_sha256': state_sha256(state['student']), 'final_teacher_sha256': state_sha256(state['teacher'])}}
    for row in rows:
        row.update({k: 1. for k in (*B.ANCHORS, 'pose', 'grad_norm', 'contrast_projector_grad_norm')})
        row.update(contrast_nce=2., contrast_rnc=3., windows=32, contrastive_calibration_only=False,
                   origin=[[0, None, index] for index in range(32)])
        row['total'] = 2.+2.*row['contrast_nce_weight']+3.*row['contrast_rnc_weight']
        if 'contrast_global_grad_base' in row:
            row.update(contrast_rnc_cross_window_rank_correlation=.5, contrast_rnc_cross_window_rank_valid=1.,
                       contrast_rnc_cross_window_rank_pairs=50.)
    for name, value in [('config.json', cfg), ('summary.json', summary), ('origin_sessions.json', ['A/subject/session'])]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    torch.save(state, tmp_path/'checkpoint.pt')
    return cfg, provenance, cohort


@pytest.mark.parametrize('mutation', ['config', 'cohort', 'metric_step', 'metric_weight', 'missing_gradient', 'checkpoint_step', 'summary_hash'])
def test_run_evidence_rejects_recipe_provenance_metrics_and_checkpoint_mismatch(tmp_path, mutation):
    cfg, provenance, cohort = _run_folder(tmp_path)
    result = B._run_evidence(tmp_path, 'CN', cohort, cfg, provenance)
    assert len(result[2]) == 576 and len(result[3]) == 5
    if mutation == 'config':
        changed = copy.deepcopy(cfg)
        changed['lr'] *= 2
        (tmp_path/'config.json').write_text(json.dumps(changed))
    elif mutation in ('cohort', 'summary_hash'):
        changed = json.loads((tmp_path/'summary.json').read_text())
        if mutation == 'cohort': changed['contrastive']['provenance']['cohort_sha256'] = 'wrong'
        else: changed['contrastive']['final_student_sha256'] = 'wrong'
        (tmp_path/'summary.json').write_text(json.dumps(changed))
    elif mutation == 'checkpoint_step':
        changed = torch.load(tmp_path/'checkpoint.pt', weights_only=False)
        changed['step'] = 575
        torch.save(changed, tmp_path/'checkpoint.pt')
    else:
        rows = [json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
        if mutation == 'metric_step': rows[0]['step'] = 1.0
        elif mutation == 'metric_weight': rows[512]['contrast_nce_weight'] = 0.
        else: del rows[99]['contrast_global_grad_nce']
        (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):
        B._run_evidence(tmp_path, 'CN', cohort, cfg, provenance)


def test_cuda_receipt_binds_calibration_config_and_complete_numeric_checks(tmp_path):
    from contrastive_cuda_check import PROTOCOL, CRITERIA, TECHNICAL_OVERRIDES
    calibration_cfg = {'contrastive': {'calibration_only': True, 'nce_weight': 1., 'rnc_weight': 1.}}
    canonical = hashlib.sha256(json.dumps(calibration_cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cohort = {'source_sha256': 's'*64, 'initial_state_sha256': 'i'*64}
    inputs = {k: k+'-pin' for k in ('manifest', 'scales', 'window_seconds', 'pose_stats')}
    keys = ('total', 'pose', 'contrast_nce', 'contrast_rnc', 'contrast_global_grad_base', 'contrast_global_grad_nce',
            'contrast_global_grad_rnc', 'gradient', 'student', 'teacher')
    precisions = {}
    for name in ('fp32', 'bf16'):
        limits = {k: (.01 if name == 'bf16' or k in ('student', 'teacher') else 1e-5) for k in keys}
        precisions[name] = {'passed': True, 'updates': [{'update': step, 'passed': True, 'errors': dict.fromkeys(keys, 0.), 'limits': limits}
                                                       for step in (1, 2)]}
    report = {'passed': True, 'source_sha256': cohort['source_sha256'], 'initial_sha256': cohort['initial_state_sha256'],
              'protocol': PROTOCOL, 'criteria': CRITERIA, 'technical_overrides': TECHNICAL_OVERRIDES, 'world_size': 4,
              'inputs_unchanged': True, 'inputs': {**inputs, 'config': 'byte-pin-different-from-run-format'},
              'pose_stats_sha256': inputs['pose_stats'], 'config_canonical_sha256': canonical,
              'ranks': [{'rank': rank, 'passed': True, 'stop_checks': {'coverage': True, 'numeric': True, 'passed': True},
                         'arms': {'CN': copy.deepcopy(precisions), 'CR': copy.deepcopy(precisions)}} for rank in range(4)]}
    path = tmp_path/'cuda.json'
    path.write_text(json.dumps(report))
    assert B._cuda_evidence(path, cohort, inputs, calibration_cfg)['passed']
    for mutation in ('canonical', 'criterion', 'truncated'):
        changed = copy.deepcopy(report)
        if mutation == 'canonical': changed['config_canonical_sha256'] = 'wrong'
        elif mutation == 'criterion': changed['ranks'][0]['arms']['CN']['fp32']['updates'][0]['limits']['total'] = 1.
        else: del changed['ranks'][0]['arms']['CN']['fp32']['updates'][0]['errors']['gradient']
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            B._cuda_evidence(path, cohort, inputs, calibration_cfg)
