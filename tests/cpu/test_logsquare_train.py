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
from wearusfm.model.patch_tokens import PatchTokenizer
from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('logsquare_training_test', ROOT / 'scripts/logsquare_train.py')
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
PLAN = ROOT / 'results/infra/logsquare_training_20261104/training_plan.json'
GUARD_FILE = 'results/infra/logsquare_initialization_20261105/initial_state_guard.json'
SIGNATURE = {'test': 'data'}
# The two values of the one calibrated buffer on the cluster, one ULP apart: the preflights
# (NumPy imported first) and every T100 training checkpoint (Torch imported first).
PREFLIGHT_GAIN, TRAINING_GAIN = 548.6939187567023, 548.6939187567021


def candidate():
    comparison = json.loads((ROOT / 'results/infra/tokenizer_review_20261103/comparison_configurations.json').read_text())
    argv = comparison['arm_configurations']['T100-L']['validated_trainer_argv']
    args = T.CLI.build_parser().parse_args(argv[1:] + ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '.001'])
    return comparison, T.CLI.build_config(args)


def test_registered_command_changes_only_branch_output_dose_and_timeout():
    comparison, cfg = candidate()
    assert R.config_to_dict(cfg) == json.loads((ROOT / 'results/infra/logsquare_review_20261104/config_T100-LQ.json').read_text())
    guard, records = {'path': '/code/guard.json', 'sha256': 'a' * 64}, Path('/tmp/phase/initial_state')
    for step, cap in ((5000, 5400), (19618, 14400)):
        fresh = (guard, records) if step == 5000 else ()
        command = T.command_for(comparison, cfg, Path('/tmp/own-candidate'), step, cap, *fresh)
        assert command[:3] == ['torchrun', '--standalone', '--nproc_per_node=4']
        assert command[-5:] == ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '0.001', '--time-limit-s', str(cap - 300)]
        # Only the fresh phase starts its ranks behind the guard; the trainer arguments after it are untouched.
        entry = ['scripts/logsquare_train_rank.py', '--initial-state-guard', '/code/guard.json',
                 '--initial-state-guard-sha256', 'a' * 64, '--initial-state-records', str(records)] if step == 5000 else []
        assert command[3:3 + len(entry)] == entry and '--verify-only' not in command  # the dry-run switch never trains
        with pytest.raises(ValueError, match='fresh phase'):
            T.command_for(comparison, cfg, Path('/tmp/own-candidate'), step, cap, *(() if step == 5000 else (guard, records)))
        recovered = command[3 + len(entry):-5]
        original = comparison['arm_configurations']['T100-L']['validated_trainer_argv']
        recovered[recovered.index('--out-dir') + 1] = original[original.index('--out-dir') + 1]
        recovered[recovered.index('--max-steps') + 1] = '5000'
        assert recovered == original
        with pytest.raises(ValueError, match='CLI/config'):
            T.command_for(comparison, replace(cfg, lr=cfg.lr * 2), Path('/tmp/new'), step, cap, *fresh)
    for flag in ('--init-from', '--branch-from', '--branch-state-from', '--tokenizer-logsquare'):
        changed = copy.deepcopy(comparison)
        changed['arm_configurations']['T100-L']['validated_trainer_argv'] += [flag, 'unexpected']
        with pytest.raises(ValueError, match='arguments'):
            T.command_for(changed, cfg, Path('/tmp/new'), 5000, 5400, guard, records)


def calibrated(monkeypatch, gain):
    """Stand in for a numeric context of the cluster: the constructor's calibration returns this gain.
    The Mac does not reproduce the import-order effect, so the regression injects its measured values."""
    monkeypatch.setattr(PatchTokenizer, '_calibrate_gain', lambda self, *args, **kwargs: gain)


def preflight_reference(cfg, monkeypatch):
    """SHA-256 of the control model as the preflight built it: seed, weights, and the preflight's gain."""
    calibrated(monkeypatch, PREFLIGHT_GAIN)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        return T.P.state_sha256(T.P.WearUsFM(replace(cfg.model, tokenizer_logsquare=False)).state_dict())


def expected_for(reference):
    """What verify_initialization compares against: the reviewed guard, or the bare hash before the correction
    (b2934b0, job 63532335), so that the two regressions below can be run unchanged on the old controller."""
    if not hasattr(T, 'G'):
        return reference
    guard = json.loads((ROOT / GUARD_FILE).read_text())
    guard['reference']['shared_sha256'] = reference
    assert (guard['reference']['fe_gain'], guard['training_gain']['fe_gain']) == (PREFLIGHT_GAIN, TRAINING_GAIN)
    return guard


def test_training_context_gain_passes_the_exact_initialization_check(monkeypatch):
    """Regression for job 63532335: fails on the controller of b2934b0, passes after the correction.

    The reviewed hash is the preflight's state. Controller and trainer build the same weights with the gain of
    the historical T100-L training, one ULP below: the old full-hash comparison stopped there."""
    _, cfg = candidate()
    expected = expected_for(preflight_reference(cfg, monkeypatch))
    calibrated(monkeypatch, TRAINING_GAIN)
    before = torch.get_rng_state().clone()
    result = T.verify_initialization(cfg, expected)
    assert result['new_parameters'] == 24576 and result['logsquare_weight_zero'] is True
    assert torch.equal(torch.get_rng_state(), before)


def test_one_ulp_of_one_real_parameter_is_still_rejected(monkeypatch):
    """The check is not loosened: the smallest possible change of one weight stops it, before and after."""
    _, cfg = candidate()
    expected = expected_for(preflight_reference(cfg, monkeypatch))
    calibrated(monkeypatch, TRAINING_GAIN)
    original = T.P.WearUsFM

    def changed(model_cfg):
        model = original(model_cfg)
        with torch.no_grad():
            weight = model.tokenizer.proj.weight
            weight[0, 0] = torch.nextafter(weight[0, 0], torch.tensor(float('inf')))
        return model

    monkeypatch.setattr(T.P, 'WearUsFM', changed)
    with pytest.raises(ValueError, match='historical T100-L'):
        T.verify_initialization(cfg, expected)


@pytest.mark.parametrize('case', ['preflight_gain', 'other_gain', 'parameter', 'new_weight'])
def test_rejection_reports_hashes_gain_and_where_the_difference_is(monkeypatch, case):
    """Job 63532335 said only that a hash differed. Every rejection now carries what is needed to read it."""
    _, cfg = candidate()
    guard = expected_for(preflight_reference(cfg, monkeypatch))
    other = float(torch.nextafter(torch.tensor(TRAINING_GAIN, dtype=torch.float64), torch.tensor(0., dtype=torch.float64)))
    calibrated(monkeypatch, {'preflight_gain': PREFLIGHT_GAIN, 'other_gain': other}.get(case, TRAINING_GAIN))
    original = T.P.WearUsFM

    def changed(model_cfg):
        model = original(model_cfg)
        with torch.no_grad():
            if case == 'parameter':
                model.tokenizer.proj.bias[3] += 1.
            if case == 'new_weight':
                model.tokenizer.logsquare_weight[0, 0] = 1e-6
        return model

    monkeypatch.setattr(T.P, 'WearUsFM', changed)
    with pytest.raises(T.G.InitialStateError) as caught:
        T.verify_initialization(cfg, guard)
    message, record = str(caught.value), caught.value.record
    classification = {'preflight_gain': T.G.PREFLIGHT_STATE, 'other_gain': T.G.UNREGISTERED_GAIN,
                      'parameter': T.G.OTHER_TENSORS, 'new_weight': T.G.PASSED}[case]
    assert record['classification'] == classification and record['passed'] is False
    assert 'historical T100-L: ' + classification in message and 'controller pre-check' in message
    for value in (guard['reference']['shared_sha256'], record['shared_sha256'], record['reference_gain_shared_sha256'],
                  record['fe_gain_hex'], guard['training_gain']['fe_gain_hex'], guard['reference']['fe_gain_hex'], 'reading:'):
        assert value in message
    ulps = {'preflight_gain': (1, 0), 'other_gain': (-1, -2)}.get(case, (0, -1))
    assert (record['historical_training_gain']['observed_minus_registered_ulps'],
            record['reference_gain']['observed_minus_registered_ulps']) == ulps
    assert f'observed is {ulps[0]:+d} ULP' in message
    assert record['other_tensors_match_reference'] is (case != 'parameter')
    assert ('every other tensor DIFFER' in message) is (case == 'parameter')
    assert record['logsquare_weight_zero'] is (case != 'new_weight')
    # Every observed tensor hash is in the record; the changed one is identifiable against any other record.
    tensors = record['localization']['per_tensor']
    assert len(tensors) == record['tensors'] and T.L.NEW_WEIGHT not in tensors and T.G.GAIN in tensors
    if case == 'preflight_gain':  # exactly the state the old check demanded: now refused, with the reason
        assert record['shared_sha256'] == guard['reference']['shared_sha256'] and 'not in the trainer numeric context' in message


def test_released_plan_without_guard_is_refused_and_the_guard_is_bound_to_the_plan(tmp_path):
    plan = json.loads(PLAN.read_text())
    # The release of 04/11 pins two controller files and no guard: the corrected controller cannot run it.
    with pytest.raises(ValueError, match='controller review incomplete'):
        T.load_plan(PLAN)
    with pytest.raises(ValueError, match='new release required'):
        T.plan_guard(plan)
    plan['initial_state_guard'] = {'path': GUARD_FILE, 'sha256': R.sha256_file(ROOT / GUARD_FILE)}
    identity = T.plan_guard(plan)
    assert identity['sha256'] == plan['initial_state_guard']['sha256'] and Path(identity['path']) == ROOT / GUARD_FILE
    guard = json.loads((ROOT / GUARD_FILE).read_text())
    for mutation in ('pin', 'reference', 'report', 'training_gain', 'receipt', 'groups'):
        bad_plan, bad = copy.deepcopy(plan), copy.deepcopy(guard)
        if mutation == 'pin': bad_plan['initial_state_guard']['sha256'] = '0' * 64
        if mutation == 'reference': bad_plan['shared_initial_sha256'] = '0' * 64
        if mutation == 'report': bad_plan['preflight_report'] = {**plan['preflight_report'], 'sha256': '0' * 64}
        if mutation == 'training_gain':  # a gain taken from anywhere but the historical checkpoints
            bad['training_gain'].update(fe_gain=PREFLIGHT_GAIN, fe_gain_hex=PREFLIGHT_GAIN.hex())
        if mutation == 'receipt': bad['training_gain']['sources'][1]['checkpoint_sha256'] = '0' * 64
        if mutation == 'groups': bad['group_references']['frontend_and_gain_sha256'] = '0' * 64
        if bad != guard:
            path = tmp_path / (mutation + '.json')
            path.write_text(json.dumps(bad))
            bad_plan['initial_state_guard'] = {'path': str(path), 'sha256': R.sha256_file(path)}
        with pytest.raises(ValueError):
            T.plan_guard(bad_plan)


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
    inputs.update(training_plan=T.P.file_identity(plan_path), training_release=T.P.file_identity(release_path),
                  initial_state_guard=T.P.file_identity(ROOT / GUARD_FILE))
    monkeypatch.setattr(T.P, 'code_identity', lambda expected: expected)
    monkeypatch.setattr(T, 'load_plan', lambda path: (copy.deepcopy(plan), cfg, copy.deepcopy(inputs)))
    monkeypatch.setattr(T.R, 'calibration_signature', lambda *args: SIGNATURE)
    monkeypatch.setattr(T, 'verify_initialization', lambda *args: {'checked': True})
    commands, checks, behavior = [], [], {'failure': None}
    def checkpoint(path, expected_cfg, step, signature, guard):
        data = json.loads(Path(path).read_text())
        assert signature == SIGNATURE and guard['training_gain']['fe_gain'] == TRAINING_GAIN
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
        assert (T.RANK_ENTRY in command) is (step == 5000)
        if step == 5000:  # what the four guarded ranks leave next to the launch record
            records = Path(command[command.index('--initial-state-records') + 1])
            assert records.is_dir() and not list(records.iterdir())
            student = {'role': 'student', 'passed': True, 'classification': T.G.PASSED, 'shared_sha256': 'b' * 64,
                       'reference_gain_shared_sha256': 'c' * 64, 'fe_gain': TRAINING_GAIN, 'fe_gain_hex': TRAINING_GAIN.hex()}
            counted = {'role': 'parameter_count', 'judged': False, 'fe_gain': TRAINING_GAIN, 'fe_gain_hex': TRAINING_GAIN.hex()}
            for rank in range(4):
                row = {'rank': rank, 'world_size': 4, 'guard_sha256': command[command.index('--initial-state-guard-sha256') + 1],
                       'passed': True, 'verify_only': False, 'trainer_returned': True,
                       'constructions': ([dict(counted)] if rank == 0 else []) + [dict(student)]}
                failure = behavior['failure']
                if failure == 'rank_missing' and rank == 3: continue
                if failure == 'rank_failed' and rank == 2: row['passed'] = False
                if failure == 'rank_unreturned' and rank == 1: row['trainer_returned'] = False
                if failure == 'rank_dry' and rank == 1: row.update(verify_only=True, trainer_returned=False)
                if failure == 'rank_unmarked' and rank == 2: del row['verify_only']
                if failure == 'rank_guard' and rank == 1: row['guard_sha256'] = '0' * 64
                if failure == 'rank_count' and rank == 0: row['constructions'].pop()
                if failure == 'rank_extra' and rank == 2: row['constructions'].append(dict(student))
                if failure == 'rank_unjudged' and rank == 1: row['constructions'] = [dict(counted)]
                if failure == 'rank_state' and rank == 3: row['constructions'][-1]['fe_gain_hex'] = PREFLIGHT_GAIN.hex()
                if failure == 'rank_class' and rank == 3: row['constructions'][-1]['classification'] = T.G.PREFLIGHT_STATE
                (records / f'rank{rank}.json').write_text(json.dumps(row))
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
    screening = json.loads(first)
    guard_sha = R.sha256_file(ROOT / GUARD_FILE)
    # The receipt states the state the four ranks trained, not the pre-check; the continuation carries it on.
    assert screening['trained_initial_state'] == final['trained_initial_state']
    assert screening['trained_initial_state']['classification'] == T.G.PASSED
    assert screening['trained_initial_state']['models_built_per_rank'] == [2, 1, 1, 1]
    assert screening['trained_initial_state']['fe_gain'] == TRAINING_GAIN and len(screening['trained_initial_state']['records']) == 4
    assert screening['initial_state_guard_sha256'] == final['initial_state_guard_sha256'] == guard_sha
    launch = json.loads((x['outputs']['5000'] / 'launch.json').read_text())
    assert launch['initialization'] == {'checked': True} and launch['initial_state_guard_sha256'] == guard_sha
    assert not (x['outputs']['19618'] / 'initial_state').exists()
    assert len(x['commands']) == 2 and len(x['checks']) == 6
    assert T.P.file_identity(x['run'] / 'checkpoint_step0005000.pt') == old
    assert (x['outputs']['5000'] / 'receipt.json').read_bytes() == first


@pytest.mark.parametrize('failure', ['incomplete', 'subprocess', 'source', 'stop', 'rank_missing', 'rank_failed',
                                     'rank_unreturned', 'rank_guard', 'rank_count', 'rank_extra', 'rank_unjudged',
                                     'rank_state', 'rank_class', 'rank_dry', 'rank_unmarked'])
def test_failed_phase_never_claims_completion(execution, failure):
    x = execution
    x['behavior']['failure'] = failure
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        execute(x, 5000)
    assert not (x['outputs']['5000'] / 'receipt.json').exists()


def test_dry_run_records_and_training_records_are_never_exchanged(tmp_path):
    """The diagnostic dry run (--verify-only) and a training phase leave the same record format: the reader of one
    refuses the other, so a dry run is never a training attestation and a training is never read as a dry run."""
    student = {'role': 'student', 'passed': True, 'classification': T.G.PASSED, 'shared_sha256': 'b' * 64,
               'reference_gain_shared_sha256': 'c' * 64, 'fe_gain': TRAINING_GAIN, 'fe_gain_hex': TRAINING_GAIN.hex()}
    counted = {'role': 'parameter_count', 'judged': False, 'fe_gain': TRAINING_GAIN, 'fe_gain_hex': TRAINING_GAIN.hex()}
    for name, dry in (('dry', True), ('trained', False)):
        (tmp_path / name).mkdir()
        for rank in range(4):
            row = {'rank': rank, 'world_size': 4, 'guard_sha256': 'a' * 64, 'passed': True, 'verify_only': dry,
                   'trainer_returned': not dry, 'constructions': ([counted] if rank == 0 else []) + [student]}
            (tmp_path / name / f'rank{rank}.json').write_text(json.dumps(row))
    dry = T.rank_attestation(tmp_path / 'dry', 'a' * 64, verify_only=True)
    trained = T.rank_attestation(tmp_path / 'trained', 'a' * 64)
    assert 'nothing was trained' in dry['verified_on'] and 'built and trained' in trained['verified_on']
    assert {key: dry[key] for key in ('classification', 'shared_sha256', 'fe_gain_hex', 'models_built_per_rank')} == \
        {key: trained[key] for key in ('classification', 'shared_sha256', 'fe_gain_hex', 'models_built_per_rank')}
    with pytest.raises(ValueError, match='unverified initial state: rank 0'):
        T.rank_attestation(tmp_path / 'dry', 'a' * 64)
    with pytest.raises(ValueError, match='unverified initial state: rank 0'):
        T.rank_attestation(tmp_path / 'trained', 'a' * 64, verify_only=True)


@pytest.mark.parametrize('failure', ['arm', 'code', 'plan', 'complete', 'checkpoint', 'resume', 'stop', 'initial', 'guard'])
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
    if failure == 'initial': rec['trained_initial_state'] = None
    if failure == 'guard': rec['initial_state_guard_sha256'] = '0' * 64
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


def test_checkpoint_gain_must_be_the_registered_training_gain(full_checkpoint):
    """The gain is a buffer no update touches: what a checkpoint saves is what training started with."""
    cfg, state, path = full_checkpoint
    saved = state['student'][T.G.GAIN].item()
    below = float(torch.nextafter(torch.tensor(saved, dtype=torch.float64), torch.tensor(0., dtype=torch.float64)))
    assert T.checkpoint_identity(path, cfg, 5000, SIGNATURE, {'training_gain': {'fe_gain': saved}})['sha256'] == R.sha256_file(path)
    with pytest.raises(ValueError, match='checkpoint gain is not the registered historical T100-L training gain: student'):
        T.checkpoint_identity(path, cfg, 5000, SIGNATURE, {'training_gain': {'fe_gain': below}})
    state['teacher'][T.G.GAIN] = torch.tensor(below, dtype=torch.float64)
    torch.save(state, path)
    with pytest.raises(ValueError, match='training gain: teacher'):
        T.checkpoint_identity(path, cfg, 5000, SIGNATURE, {'training_gain': {'fe_gain': saved}})
