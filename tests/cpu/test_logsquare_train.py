"""Test the candidate's command, review gate and resumable phase controller."""
from dataclasses import replace
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch
from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('logsquare_training_test', ROOT / 'scripts/logsquare_train.py')
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
PLAN = ROOT / 'results/infra/logsquare_training_20261104/training_plan.json'
SIGNATURE = {'test': 'data'}


def candidate():
    comparison = json.loads((ROOT / 'results/infra/tokenizer_review_20261103/comparison_configurations.json').read_text())
    argv = comparison['arm_configurations']['T100-L']['validated_trainer_argv']
    args = T.CLI.build_parser().parse_args(argv[1:] + ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '.001'])
    return comparison, T.CLI.build_config(args)


def test_registered_command_changes_only_branch_output_dose_and_timeout():
    comparison, cfg = candidate()
    assert R.config_to_dict(cfg) == json.loads((ROOT / 'results/infra/logsquare_review_20261104/config_T100-LQ.json').read_text())
    for step, cap in ((5000, 5400), (19618, 14400)):
        command = T.command_for(comparison, cfg, Path('/tmp/own-candidate'), step, cap)
        assert command[:3] == ['torchrun', '--standalone', '--nproc_per_node=4']
        assert command[-5:] == ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '0.001', '--time-limit-s', str(cap - 300)]
        recovered = command[3:-5]
        original = comparison['arm_configurations']['T100-L']['validated_trainer_argv']
        recovered[recovered.index('--out-dir') + 1] = original[original.index('--out-dir') + 1]
        recovered[recovered.index('--max-steps') + 1] = '5000'
        assert recovered == original
        with pytest.raises(ValueError, match='CLI/config'):
            T.command_for(comparison, replace(cfg, lr=cfg.lr * 2), Path('/tmp/new'), step, cap)
    for flag in ('--init-from', '--branch-from', '--branch-state-from', '--tokenizer-logsquare'):
        changed = copy.deepcopy(comparison)
        changed['arm_configurations']['T100-L']['validated_trainer_argv'] += [flag, 'unexpected']
        with pytest.raises(ValueError, match='arguments'):
            T.command_for(changed, cfg, Path('/tmp/new'), 5000, 5400)


def test_fresh_initialization_matches_preflight_shared_weights_and_preserves_rng():
    _, cfg = candidate()
    # Mac/cluster PyTorch versions need not have bit-identical initialization;
    # production checks the cluster preflight hash, locally check the exact control.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        control = T.P.WearUsFM(replace(cfg.model, tokenizer_logsquare=False))
        expected = T.P.state_sha256(control.state_dict())
    before = torch.get_rng_state().clone()
    result = T.verify_initialization(cfg, expected)
    assert result['new_parameters'] == 24576 and result['logsquare_weight_zero'] is True
    assert torch.equal(torch.get_rng_state(), before)
    with pytest.raises(ValueError, match='historical T100-L'):
        T.verify_initialization(cfg, '0' * 64)


def test_review_only_plan_is_not_executable(tmp_path):
    plan = json.loads(PLAN.read_text())
    # The published campaign can advance to released; the negative fixture
    # must keep exercising an unreviewed plan independently of that state.
    plan['status'] = 'REVIEW_REQUIRED'
    plan['training_release'] = None
    pending = tmp_path / 'pending_plan.json'
    pending.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='review/release pending'):
        T.load_plan(pending)


def test_pinned_gzip_json(tmp_path):
    path = tmp_path / 'report.json.gz'
    path.write_bytes(gzip.compress(b'{"complete":true}', mtime=0))
    value, identity = T.read_pinned_json({'path': str(path), 'sha256': R.sha256_file(path)})
    assert value == {'complete': True} and identity['bytes'] == path.stat().st_size
    with pytest.raises(ValueError, match='SHA mismatch'):
        T.read_pinned_json({'path': str(path), 'sha256': '0' * 64})


@pytest.fixture
def release(tmp_path):
    plan = {'status': 'APPROVED_FOR_TRAINING'}
    for key in ('preflight_report', 'r1_analysis', 'resource_proposal'):
        plan[key] = {'sha256': key}
    review = tmp_path / 'review.md'
    review.write_text('Claude review: training approved with the registered epsilon decision.')
    value = {'training_approved': True, 'epsilon': .001,
             **{key + '_sha256': plan[key]['sha256'] for key in ('preflight_report', 'r1_analysis', 'resource_proposal')},
             'controller_sha256': {p: R.sha256_file(ROOT / p) for p in T.CONTROLLER_FILES},
             'claude_review': {'path': str(review), 'sha256': R.sha256_file(review)}}
    # status is an execution-state marker, not a pinned artifact.
    value.pop('status_sha256', None)
    path = tmp_path / 'release.json'
    def save():
        path.write_text(json.dumps(value))
        plan['training_release'] = {'path': str(path), 'sha256': R.sha256_file(path)}
    save()
    return plan, value, save


def test_approved_release_binds_review_analysis_budget_and_controller(release):
    plan, value, save = release
    result = T.validate_release(plan)
    assert set(result) == {'training_release', 'claude_review', *T.CONTROLLER_FILES}
    for key in ('preflight_report', 'r1_analysis', 'resource_proposal'):
        old = value[key + '_sha256']
        value[key + '_sha256'] = 'wrong'
        save()
        with pytest.raises(ValueError, match='release pin mismatch'):
            T.validate_release(plan)
        value[key + '_sha256'] = old
    value['controller_sha256'][T.CONTROLLER_FILES[0]] = '0' * 64
    save()
    with pytest.raises(ValueError, match='SHA mismatch'):
        T.validate_release(plan)


def test_r2_caps_measured_ratio_and_partition_guard():
    plan = json.loads(PLAN.read_text())
    T.validate_allocations(plan, plan['measured_rho_update'])
    for mutation in ('walltime_seconds', 'additional_steps', 'gpus', 'memory'):
        bad = copy.deepcopy(plan)
        bad['allocations']['5000'][mutation] = 'wrong'
        with pytest.raises(ValueError, match='R2'):
            T.validate_allocations(bad, plan['measured_rho_update'])
    with pytest.raises(ValueError, match='partition'):
        T.validate_allocations(plan, 20.)
    with pytest.raises(ValueError, match='invalid rho'):
        T.validate_allocations(plan, float('nan'))


def test_completed_preflight_checks_actual_rank_function_and_input_pins():
    plan = json.loads(PLAN.read_text())
    report, _ = T.read_pinned_json(plan['preflight_report'])
    assert T.validate_preflight(report, plan) == plan['measured_rho_update']
    for mutation in ('complete', 'pairing', 'dose', 'initial', 'code', 'plan'):
        bad = copy.deepcopy(report)
        if mutation == 'complete': bad['complete'] = 'true'
        if mutation == 'pairing': bad['paired_support'] = False
        if mutation == 'dose': bad['windows_each_arm'] = 2047
        if mutation == 'initial': bad['initial_function_per_rank']['3']['status'] = 'failed'
        if mutation == 'code': bad['code_sha'] = '0' * 40
        if mutation == 'plan': bad['inputs']['plan']['sha256'] = '0' * 64
        with pytest.raises(ValueError):
            T.validate_preflight(bad, plan)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    """Exercise the orchestration; mock only review loading and external trainer."""
    comparison, cfg = candidate()
    run_dir = tmp_path / 'run'
    outputs = {str(s): tmp_path / ('phase_' + str(s)) for s in (5000, 19618)}
    manifest, scales, windows, compare = (tmp_path / n for n in ('manifest', 'scales', 'windows', 'comparison'))
    for path, value in ((manifest, {}), (scales, {'scales': {}}), (windows, {'window_s': {}}), (compare, comparison)):
        path.write_text(json.dumps(value))
    inputs = {k: T.P.file_identity(p) for k, p in (('manifest', manifest), ('scales', scales), ('windows', windows), ('comparison', compare))}
    plan = json.loads(PLAN.read_text())
    plan.update(run_dir=str(run_dir), phase_receipts={s: str(p / 'receipt.json') for s, p in outputs.items()})
    plan_path = tmp_path / 'training_plan.json'
    plan_path.write_text(json.dumps(plan))
    release_path = tmp_path / 'release.json'
    release_path.write_text('{}')
    inputs.update(training_plan=T.P.file_identity(plan_path), training_release=T.P.file_identity(release_path))
    monkeypatch.setattr(T.P, 'code_identity', lambda expected: expected)
    monkeypatch.setattr(T, 'load_plan', lambda path: (copy.deepcopy(plan), cfg, copy.deepcopy(inputs)))
    monkeypatch.setattr(T.R, 'calibration_signature', lambda *args: SIGNATURE)
    monkeypatch.setattr(T, 'verify_initialization', lambda *args: {'checked': True})
    commands, checks, behavior = [], [], {'failure': None}
    def checkpoint(path, expected_cfg, step, signature):
        data = json.loads(Path(path).read_text())
        assert signature == SIGNATURE
        T.require(data['step'] == step and data['config'] == R.config_to_dict(expected_cfg), 'mock identity mismatch')
        checks.append((str(path), step))
        return {**T.P.file_identity(path), 'trained_state_sha256': str(step)}
    monkeypatch.setattr(T, 'checkpoint_identity', checkpoint)
    def train(command, *, cwd, check):
        assert cwd == ROOT and check is True
        commands.append(command)
        if behavior['failure'] == 'subprocess':
            raise subprocess.CalledProcessError(1, command)
        step = int(command[command.index('--max-steps') + 1])
        run_dir.mkdir(exist_ok=True)
        effective = R.config_to_dict(replace(cfg, max_steps=step))
        (run_dir / 'config.json').write_text(json.dumps(effective))
        (run_dir / 'summary.json').write_text(json.dumps({'stopped': 'time_limit' if behavior['failure'] == 'incomplete' else 'max_steps', 'steps': step}))
        for name in (f'checkpoint_step{step:07d}.pt', 'checkpoint.pt'):
            (run_dir / name).write_text(json.dumps({'step': step, 'config': effective, 'serialization': name}))
        if behavior['failure'] == 'source': manifest.write_text('{"changed":true}')
        if behavior['failure'] == 'stop': (run_dir / 'STOP').write_text('numerical failure')
        if behavior['failure'] == 'old': (run_dir / 'checkpoint_step0005000.pt').write_text('modified')
    monkeypatch.setattr(T.subprocess, 'run', train)
    return {'plan': plan, 'path': plan_path, 'outputs': outputs, 'run': run_dir, 'commands': commands,
            'checks': checks, 'behavior': behavior, 'code': 'a' * 40}


def execute(x, step):
    T.execute(x['path'], step, x['code'], x['outputs'][str(step)], x['plan']['allocations'][str(step)]['walltime_seconds'])


def test_own_continuation_keeps_5000_checkpoint_and_phase_receipt(execution):
    x = execution
    execute(x, 5000)
    first = (x['outputs']['5000'] / 'receipt.json').read_bytes()
    old = T.P.file_identity(x['run'] / 'checkpoint_step0005000.pt')
    execute(x, 19618)
    final = json.loads((x['outputs']['19618'] / 'receipt.json').read_text())
    assert final['complete'] is True and final['protocol'] == T.PROTOCOL and final['checkpoint_step'] == 19618
    assert len(x['commands']) == 2 and len(x['checks']) == 6
    assert T.P.file_identity(x['run'] / 'checkpoint_step0005000.pt') == old
    assert (x['outputs']['5000'] / 'receipt.json').read_bytes() == first


@pytest.mark.parametrize('failure', ['incomplete', 'subprocess', 'source', 'stop'])
def test_failed_phase_never_claims_completion(execution, failure):
    x = execution
    x['behavior']['failure'] = failure
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        execute(x, 5000)
    assert not (x['outputs']['5000'] / 'receipt.json').exists()


@pytest.mark.parametrize('failure', ['arm', 'code', 'plan', 'complete', 'checkpoint', 'resume', 'stop'])
def test_continuation_requires_own_completed_provenance(execution, failure):
    x = execution
    execute(x, 5000)
    path = x['outputs']['5000'] / 'receipt.json'
    rec = json.loads(path.read_text())
    if failure == 'arm': rec['arm'] = 'T100-L'
    if failure == 'code': rec['code_sha'] = 'b' * 40
    if failure == 'plan': rec['plan_sha256'] = '0' * 64
    if failure == 'complete': rec['complete'] = 'false'
    if failure == 'checkpoint': rec['checkpoint_sha256'] = '0' * 64
    if failure == 'resume': rec['resume_checkpoint_sha256'] = '0' * 64
    if failure == 'stop': (x['run'] / 'STOP').write_text('stop')
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError):
        execute(x, 19618)
    assert len(x['commands']) == 1 and not x['outputs']['19618'].exists()


def test_continuation_rejects_mutation_of_named_5000_checkpoint(execution):
    execute(execution, 5000)
    execution['behavior']['failure'] = 'old'
    with pytest.raises(ValueError, match='5000 checkpoint was modified'):
        execute(execution, 19618)
    assert not (execution['outputs']['19618'] / 'receipt.json').exists()



@pytest.fixture
def full_checkpoint(tmp_path):
    cfg = replace(R.small_config(), max_steps=5000)
    cfg = replace(cfg, model=replace(cfg.model, tokenizer_logsquare=True))
    model = T.P.WearUsFM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Only the new branch is active in this synthetic update. Missing moments
    # for genuinely inactive parameters are valid, but not missing parameters.
    model.tokenizer.logsquare_weight.grad = torch.ones_like(model.tokenizer.logsquare_weight)
    optimizer.step()
    optimizer.state[model.tokenizer.logsquare_weight]['step'].fill_(5000)
    state = {'step': 5000, 'config': R.config_to_dict(cfg), 'world_size': 4,
             'stream': {'num_workers': 6, 'segment': cfg.stream_segment, 'world_size': 4},
             'rank_rng': [{'torch_rng': torch.Generator().manual_seed(i).get_state(),
                           'cuda_rng': torch.zeros(16, dtype=torch.uint8)} for i in range(4)],
             'pose_stats_sha256': '757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e',
             'optimizer': optimizer.state_dict(), 'data_signature': SIGNATURE, 'alarm_counts': (0, 0),
             'student': model.state_dict(), 'teacher': copy.deepcopy(model.state_dict())}
    path = tmp_path / 'checkpoint.pt'
    torch.save(state, path)
    return cfg, state, path


def test_checkpoint_accepts_full_model_and_active_log_optimizer_only(full_checkpoint):
    cfg, state, path = full_checkpoint
    assert len(state['optimizer']['state']) == 1
    assert T.checkpoint_identity(path, cfg, 5000, SIGNATURE)['sha256'] == R.sha256_file(path)


@pytest.mark.parametrize('corruption', ['log_only', 'missing_weight', 'extra_weight', 'model_shape', 'model_dtype',
                                       'optimizer_inventory', 'optimizer_order', 'moment_shape', 'moment_dtype', 'log_moments'])
def test_checkpoint_rejects_incomplete_model_or_optimizer_inventory(full_checkpoint, corruption):
    cfg, state, path = full_checkpoint
    params = state['optimizer']['param_groups'][0]['params']
    log_id = next(iter(state['optimizer']['state']))
    if corruption == 'log_only':
        state['student'] = {T.L.NEW_WEIGHT: state['student'][T.L.NEW_WEIGHT]}
        state['teacher'] = {T.L.NEW_WEIGHT: state['teacher'][T.L.NEW_WEIGHT]}
    if corruption == 'missing_weight': state['teacher'].pop('tokenizer.proj.weight')
    if corruption == 'extra_weight': state['student']['foreign'] = torch.ones(1)
    if corruption == 'model_shape': state['teacher']['tokenizer.proj.weight'] = torch.ones(3)
    if corruption == 'model_dtype': state['teacher']['tokenizer.proj.weight'] = state['teacher']['tokenizer.proj.weight'].double()
    if corruption == 'optimizer_inventory': params.append(len(params))
    if corruption == 'optimizer_order': params[0], params[1] = params[1], params[0]
    if corruption == 'moment_shape':
        for k in ('exp_avg', 'exp_avg_sq'): state['optimizer']['state'][log_id][k] = torch.ones(3)
    if corruption == 'moment_dtype':
        for k in ('exp_avg', 'exp_avg_sq'): state['optimizer']['state'][log_id][k] = state['optimizer']['state'][log_id][k].double()
    if corruption == 'log_moments':
        # A different valid parameter has moments, so generic nonempty-state
        # validation succeeds but the required active log branch must fail.
        model = T.P.WearUsFM(cfg.model)
        other_id = next(i for i, (n, _) in enumerate(model.named_parameters()) if n != T.L.NEW_WEIGHT)
        other = list(model.parameters())[other_id]
        state['optimizer']['state'] = {other_id: {'step': torch.tensor(5000.), 'exp_avg': torch.zeros_like(other),
                                                'exp_avg_sq': torch.zeros_like(other)}}
    torch.save(state, path)
    with pytest.raises(ValueError):
        T.checkpoint_identity(path, cfg, 5000, SIGNATURE)
