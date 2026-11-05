"""Provenance, isolation, mutation safety and honest diagnostic classification."""
import copy
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from wearusfm.training.pairing import state_sha256

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('initialization_diagnostic_test', ROOT / 'scripts/logsquare_initialization_diagnostic.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
REFERENCE_GAIN, TRAINING_GAIN = 548.6939187567023, 548.6939187567021


@pytest.fixture
def fixture_plan(tmp_path):
    root = tmp_path / 'code'
    root.mkdir()
    def pin(name, value):
        path = root / name
        data = json.dumps(value).encode()
        if name.endswith('.gz'):
            data = gzip.compress(data, mtime=0)
        path.write_bytes(data)
        return {'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    sources = {}
    for index in range(102):
        name = f'source_{index}.py'
        data = f'# immutable scientific source {index}\n'.encode()
        (root / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    comparison = pin('comparison.json', {'arm_configurations': {}})
    candidate = pin('candidate.json', {'seed': 0, 'model': {'tokenizer_logsquare': True, 'tokenizer_logsquare_eps': .001}})
    reference = pin('reference.json', {'comparison': comparison})
    preflight = pin('preflight.json', {'reference_plan': reference, 'candidate_config': candidate})
    initial = {'shared_sha256': 'a' * 64, 'fe_gain': 548.6939187567023, 'new_weight_zero': True}
    report = pin('report.json.gz', {'complete': True, 'protocol': 'tokenizer-logsquare-preflight-20261104-v1',
                                  'code_sha': 'b' * 40, 'initial_shared_weights_rng_and_function_exact': True,
                                  'arms': {arm: {'initial': initial} for arm in ('T100-L', 'T100-LQ')}})
    training = {'protocol': 'tokenizer-logsquare-training-20261104-v1', 'arm': 'T100-LQ',
                'scientific_source_sha256': sources, 'required_scientific_sources': list(sources),
                'preflight_report': report, 'preflight_plan': preflight,
                'preflight_code_sha': 'b' * 40, 'shared_initial_sha256': 'a' * 64}
    training_pin = pin('training.json', training)
    # The guard under dry run, with every registered value readable from pinned records of this tree.
    historical = pin('historical_preflight.json', {'arms': {'T100-L': {'initial': {
        'all_sha256': 'a' * 64, 'fe_gain': REFERENCE_GAIN, 'shared_sha256': 'c' * 64, 'frontend_and_gain_sha256': 'd' * 64}}}})
    sources = []
    for step, digest in ((5000, 'e' * 64), (19618, 'f' * 64)):
        identity = {'checkpoint_step': step, 'checkpoint_sha256': digest, 'training_code_sha': '1' * 40}
        summary = {'receipt_summary': {'arm': 'T100-L', 'complete': True, 'checkpoint_step': step, 'checkpoint_sha256': digest,
                                       'checkpoint_guard': {**identity, 'fe_gain': TRAINING_GAIN}}}
        sources.append({**pin(f'receipt_{step}.json', summary), **identity})
    guard = {'protocol': D.G.PROTOCOL, 'arm': 'T100-LQ', 'gain_buffer': D.GAIN,
             'new_parameter': {'name': D.NEW_WEIGHT, 'shape': [384, 64]},
             'reference': {'shared_sha256': 'a' * 64, 'fe_gain': REFERENCE_GAIN, 'fe_gain_hex': REFERENCE_GAIN.hex(), 'source': report},
             'training_gain': {'fe_gain': TRAINING_GAIN, 'fe_gain_hex': TRAINING_GAIN.hex(), 'arm': 'T100-L', 'sources': sources},
             'group_references': {'source': historical, 'outside_frontend_and_gain_sha256': 'c' * 64,
                                  'frontend_and_gain_sha256': 'd' * 64}}
    plan = {'protocol': D.PROTOCOL, 'status': 'APPROVED_FOR_DIAGNOSTIC', 'training_plan': training_pin,
            'preflight_report': report, 'candidate_config': candidate, 'comparison': comparison,
            'initial_state_guard': pin('guard.json', guard),
            'variants': list(D.VARIANTS), 'cuda_required': False, 'child_timeout_seconds': 90,
            'rank_entry_timeout_seconds': 300, 'expected_scientific_sources': 102}
    path = root / 'diagnostic.json'
    def save():
        path.write_text(json.dumps(plan))
    save()
    return root, path, plan, training, pin, save, guard


def test_valid_plan_uses_only_local_provenance_and_all_102_sources(fixture_plan):
    root, path, plan, _, _, _, written = fixture_plan
    validated, candidate, _, expected, guard = D.validate_plan(path, root)
    assert validated == plan and candidate['seed'] == 0
    assert expected == {'shared_sha256': 'a' * 64, 'fe_gain': 548.6939187567023}
    assert guard == written and D.VARIANTS == D.CONSTRUCTION_VARIANTS + (D.RANK_VARIANT,) and len(D.VARIANTS) == 6


@pytest.mark.parametrize('mutation', ['plan_pin', 'source', 'source_inventory', 'candidate_provenance',
                                     'report_provenance', 'comparison_provenance', 'variants', 'timeout', 'status',
                                     'old_variants', 'rank_timeout', 'rank_timeout_missing', 'guard_missing', 'guard_pin',
                                     'guard_size', 'guard_outside', 'guard_reference', 'guard_reference_gain',
                                     'guard_training_gain', 'guard_group'])
def test_mutations_rejected_before_construction(fixture_plan, mutation):
    root, path, plan, training, pin, save, guard = fixture_plan
    if mutation.startswith('guard_') and mutation not in ('guard_missing', 'guard_pin', 'guard_size', 'guard_outside'):
        changed = copy.deepcopy(guard)
        if mutation == 'guard_reference':       # a guard bound to another reference hash
            changed['reference']['shared_sha256'] = 'b' * 64
        elif mutation == 'guard_reference_gain':  # one ULP away from the pinned preflight report
            changed['reference'].update(fe_gain=math.nextafter(REFERENCE_GAIN, math.inf),
                                        fe_gain_hex=math.nextafter(REFERENCE_GAIN, math.inf).hex())
        elif mutation == 'guard_training_gain':   # a gain no historical checkpoint record holds
            changed['training_gain'].update(fe_gain=REFERENCE_GAIN, fe_gain_hex=REFERENCE_GAIN.hex())
        else:
            changed['group_references']['frontend_and_gain_sha256'] = '0' * 64
        plan['initial_state_guard'] = pin('guard.json', changed)
    elif mutation == 'guard_missing':
        del plan['initial_state_guard']
    elif mutation == 'guard_pin':
        plan['initial_state_guard'] = {**plan['initial_state_guard'], 'sha256': '0' * 64}
    elif mutation == 'guard_size':
        plan['initial_state_guard'] = {**plan['initial_state_guard'], 'bytes': 1}
    elif mutation == 'guard_outside':
        outside = root.parent / 'guard.json'
        outside.write_bytes((root / 'guard.json').read_bytes())
        plan['initial_state_guard'] = {**plan['initial_state_guard'], 'path': '../guard.json'}
    elif mutation == 'old_variants':
        plan['variants'] = list(D.CONSTRUCTION_VARIANTS)
    elif mutation == 'rank_timeout':
        plan['rank_entry_timeout_seconds'] = 301
    elif mutation == 'rank_timeout_missing':
        del plan['rank_entry_timeout_seconds']
    elif mutation == 'plan_pin':
        plan['training_plan']['sha256'] = '0' * 64
    elif mutation == 'source':
        (root / 'source_27.py').write_text('changed')
    elif mutation == 'source_inventory':
        training['required_scientific_sources'].pop()
        plan['training_plan'] = pin('training.json', training)
    elif mutation.endswith('_provenance'):
        key = {'candidate_provenance': 'candidate_config', 'report_provenance': 'preflight_report',
               'comparison_provenance': 'comparison'}[mutation]
        plan[key] = {**plan[key], 'sha256': '0' * 64}
    elif mutation == 'variants':
        plan['variants'].reverse()
    elif mutation == 'timeout':
        plan['child_timeout_seconds'] = 91
    else:
        plan['status'] = 'REVIEW_REQUIRED'
    save()
    with pytest.raises(ValueError):
        D.validate_plan(path, root)


def model_with_gain(gain=2.0, logsquare=True):
    model = torch.nn.Module()
    model.tokenizer = torch.nn.Module()
    model.tokenizer.register_buffer('fe_gain', torch.tensor(gain, dtype=torch.float64))
    model.tokenizer.register_parameter('weight', torch.nn.Parameter(torch.tensor([[1., 2.]])))
    if logsquare:
        model.tokenizer.register_parameter('logsquare_weight', torch.nn.Parameter(torch.zeros(2, 64)))
    return model


def expected_for(model):
    return {'shared_sha256': state_sha256({key: value for key, value in model.state_dict().items() if key != D.NEW_WEIGHT}),
            'fe_gain': model.tokenizer.fe_gain.item()}


def tiny_guard(reference_gain=2.0):
    """A guard for the two-tensor model: the reference holds `reference_gain`, the training gain is one ULP below."""
    training = math.nextafter(reference_gain, 0.)
    return {'new_parameter': {'name': D.NEW_WEIGHT, 'shape': [2, 64]},
            'reference': {**expected_for(model_with_gain(reference_gain)), 'fe_gain_hex': reference_gain.hex()},
            'training_gain': {'fe_gain': training, 'fe_gain_hex': training.hex()}}


@pytest.mark.parametrize('case', ['training', 'preflight', 'other_gain', 'parameter'])
def test_guard_verdict_is_recorded_for_the_nonlinear_arm_and_agrees_with_the_diagnostic(case):
    guard = tiny_guard()
    expected = {key: guard['reference'][key] for key in ('shared_sha256', 'fe_gain')}
    gain = {'training': guard['training_gain']['fe_gain'], 'preflight': 2.0, 'other_gain': math.nextafter(2.0, math.inf),
            'parameter': guard['training_gain']['fe_gain']}[case]
    for arm in ('T100-LQ', 'T100-L'):
        model = model_with_gain(gain, logsquare=arm == 'T100-LQ')
        if case == 'parameter':
            with torch.no_grad():
                model.tokenizer.weight[0, 0] += .01
        before = state_sha256(model.state_dict())
        row = D.inspect_state(model, expected)
        D.guard_observations(model, row, arm, guard)
        assert state_sha256(model.state_dict()) == before
        assert row['fe_gain_hex'] == gain.hex()
        assert row['gain_ulps'] == {'training': {'from_preflight_reference': -1, 'from_historical_training': 0},
                                    'preflight': {'from_preflight_reference': 0, 'from_historical_training': 1},
                                    'other_gain': {'from_preflight_reference': 1, 'from_historical_training': 2},
                                    'parameter': {'from_preflight_reference': -1, 'from_historical_training': 0}}[case]
        if arm == 'T100-L':
            assert 'guard' not in row  # the guard judges the arm that is launched
            continue
        assert row['guard']['classification'] == {'training': D.G.PASSED, 'preflight': D.G.PREFLIGHT_STATE,
                                                  'other_gain': D.G.UNREGISTERED_GAIN, 'parameter': D.G.OTHER_TENSORS}[case]
        assert row['guard']['passed'] is (case == 'training')
        assert row['diagnostic_classification'] == {
            'training': 'only_gain_substitution_recovers_exact_hash', 'preflight': 'exact_original_match',
            'other_gain': 'only_gain_substitution_recovers_exact_hash',
            'parameter': 'difference_not_explained_by_gain_alone'}[case]
        # The two tools must agree: a row that tells another story is an error of the diagnostic, not an observation.
        row['diagnostic_classification'] = 'exact_original_match' if case != 'preflight' else 'difference_not_explained_by_gain_alone'
        with pytest.raises(ValueError, match='disagree'):
            D.guard_observations(model, row, arm, guard)


def test_gain_only_difference_identified_without_mutating_model():
    model = model_with_gain()
    expected = expected_for(model)
    model.tokenizer.fe_gain.copy_(torch.nextafter(model.tokenizer.fe_gain, torch.tensor(float('inf'))))
    before = state_sha256(model.state_dict())
    result = D.inspect_state(model, expected)
    assert not result['original_matches_historical']
    assert result['hypothetical_pinned_gain_matches_historical']
    assert result['diagnostic_classification'] == 'only_gain_substitution_recovers_exact_hash'
    assert result['full_sha256_before'] == result['full_sha256_after'] == before == state_sha256(model.state_dict())
    assert set(result['per_tensor']) == {'tokenizer.weight', 'tokenizer.fe_gain'}


def test_parameter_difference_cannot_be_hidden_by_gain_substitution():
    model = model_with_gain()
    expected = expected_for(model)
    with torch.no_grad():
        model.tokenizer.weight[0, 0] += .01
        model.tokenizer.fe_gain += .01
    result = D.inspect_state(model, expected)
    assert not result['hypothetical_pinned_gain_matches_historical']
    assert result['diagnostic_classification'] == 'difference_not_explained_by_gain_alone'


def test_exact_match_and_zero_branch_observations():
    linear, nonlinear = model_with_gain(logsquare=False), model_with_gain()
    expected = expected_for(linear)
    for model in (linear, nonlinear):
        result = D.inspect_state(model, expected)
        assert result['diagnostic_classification'] == 'exact_original_match'
        assert result['logsquare_weight_zero']
    with torch.no_grad():
        nonlinear.tokenizer.logsquare_weight[0, 0] = 1
    assert not D.inspect_state(nonlinear, expected)['logsquare_weight_zero']


@pytest.mark.parametrize('gain', [float('nan'), float('inf'), 0., -1.])
def test_invalid_gain_rejected(gain):
    with pytest.raises(ValueError, match='gain'):
        D.inspect_state(model_with_gain(gain), {'shared_sha256': 'a' * 64, 'fe_gain': 2.})


def test_parent_import_does_not_load_numeric_libraries():
    code = ('import importlib.util,sys; '
            f's=importlib.util.spec_from_file_location("diag",{str(ROOT / "scripts/logsquare_initialization_diagnostic.py")!r}); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'assert not any(k in sys.modules for k in ("torch","numpy","scipy","sklearn"))')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_parent_retains_errors_and_launches_each_context_separately(tmp_path, monkeypatch):
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text('{}')
    out = tmp_path / 'results'
    monkeypatch.setattr(D, 'code_identity', lambda expected: expected)
    monkeypatch.setattr(D, 'validate_plan', lambda path: ({'child_timeout_seconds': 2, 'rank_entry_timeout_seconds': 7},
                                                          {}, {}, {}, {}))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        variant = command[command.index('--child') + 1]
        destination = Path(command[command.index('--child-output') + 1])
        assert kwargs['timeout'] == (7 if variant == D.RANK_VARIANT else 2) and kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        assert destination.parent == out
        if variant == 'trainer':
            raise subprocess.TimeoutExpired(command, 2, output=b'partial', stderr=b'timeout evidence')
        row = {'variant': variant, 'original_matches_historical': False}
        if variant == 'preflight':
            row['error'] = 'retained exception'
        destination.write_text(json.dumps(row))
        return subprocess.CompletedProcess(command, int('error' in row), stdout='diagnostic stdout', stderr='diagnostic stderr')
    monkeypatch.setattr(D.subprocess, 'run', run)
    result = D.execute(plan_path, 'a' * 40, out)
    assert len(commands) == 6 and len({tuple(command) for command in commands}) == 6
    assert [command[command.index('--child') + 1] for command in commands] == list(D.VARIANTS)
    assert not result['complete'] and not result['training_authorized_by_report']
    # Failed contexts: no reading, and never an authorization.
    assert result['reading']['computable'] is False and not result['reading']['cause_demonstrated']
    assert not result['reading']['fix_dry_run_passed'] and result['reading']['training_authorized_by_reading'] is False
    assert result['variants']['preflight']['error'] == 'retained exception'
    assert result['variants']['trainer']['error'] == 'child timed out'
    assert (out / 'trainer.stdout').read_bytes() == b'partial'
    assert (out / 'trainer.stderr').read_bytes() == b'timeout evidence'
    assert json.loads((out / 'report.json').read_text()) == result
    with pytest.raises(ValueError, match='new absolute'):
        D.execute(plan_path, 'a' * 40, out)


def test_false_historical_match_is_descriptive_not_process_failure(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    monkeypatch.setattr(D, 'code_identity', lambda expected: expected)
    monkeypatch.setattr(D, 'validate_plan', lambda path: ({'child_timeout_seconds': 2, 'rank_entry_timeout_seconds': 7},
                                                          {}, {}, {}, {}))
    def run(command, **kwargs):
        Path(command[command.index('--child-output') + 1]).write_text(json.dumps({'original_matches_historical': False}))
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(D.subprocess, 'run', run)
    report = D.execute(path, 'a' * 40, tmp_path / 'results')
    assert report['complete'] and report['diagnostic_only']
    assert not report['training_authorized_by_report']


@pytest.mark.parametrize('variant', D.CONSTRUCTION_VARIANTS)
def test_child_construction_order_warmup_and_late_environment_inspection(monkeypatch, variant):
    import wearusfm.model.fm as fm

    @dataclass(frozen=True)
    class ModelConfig:
        tokenizer_logsquare: bool = True
        dim: int = 2

    events = []
    cfg = SimpleNamespace(model=ModelConfig(), seed=0)
    candidate = {'test': 'exact candidate'}
    comparison = {'arm_configurations': {'T100-L': {'validated_trainer_argv': ['scripts/train_jepa.py']}}}
    plan = {'cuda_required': False}
    expected = expected_for(model_with_gain())
    monkeypatch.setattr(D, 'code_identity', lambda expected_code: expected_code)
    monkeypatch.setattr(D, 'validate_plan', lambda path: (plan, candidate, comparison, expected, tiny_guard()))
    seed = torch.manual_seed
    def tracked_seed(value):
        events.append(('seed', value))
        return seed(value)
    monkeypatch.setattr(torch, 'manual_seed', tracked_seed)
    def construct(model_cfg):
        events.append(('construct', model_cfg.tokenizer_logsquare))
        return model_with_gain(logsquare=model_cfg.tokenizer_logsquare)
    monkeypatch.setattr(fm, 'WearUsFM', construct)
    def warmup(model_cfg):
        events.append(('warmup', model_cfg.tokenizer_logsquare))
        # Real count_parameters constructs a fresh model without seeding it.
        return sum(parameter.numel() for parameter in construct(model_cfg).parameters())
    cli = SimpleNamespace(R=SimpleNamespace(config_to_dict=lambda value: candidate, count_parameters=warmup),
                          build_config=lambda args: cfg,
                          build_parser=lambda: SimpleNamespace(parse_args=lambda args: None))
    def imported(relative, name):
        events.append(('import', relative))
        return cli if relative == 'scripts/train_jepa.py' else SimpleNamespace(CLI=cli)
    monkeypatch.setattr(D, 'import_file', imported)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: events.append(('set_threads', count)))
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda count: events.append(('set_interop', count)))
    def full_environment(runtime):
        events.append(('full_environment', None))
        return {'inspected_after_construction': True}
    monkeypatch.setattr(D, 'environment', full_environment)
    # These must never be consulted by the pre-construction snapshot.
    def forbidden_query():
        raise AssertionError('numeric runtime queried before construction')
    monkeypatch.setattr(torch, 'get_num_threads', forbidden_query)
    monkeypatch.setattr(torch, 'get_num_interop_threads', forbidden_query)
    result = D.child(Path('/unused'), variant, 'a' * 40)
    assert events[0] == ('import', D.MODULES[variant])
    assert events[-1] == ('full_environment', None)
    expected_flags = [arm == 'T100-LQ' for arm in D.CONSTRUCTION_ORDER[variant]]
    construction_flags = [value for event, value in events if event == 'construct']
    if variant == 'trainer_rank0':
        assert construction_flags == [True, *expected_flags]
        assert next(i for i, row in enumerate(events) if row[0] == 'warmup') < next(i for i, row in enumerate(events) if row[0] == 'seed')
        assert result['unseeded_parameter_count_warmups'] == 1
    else:
        assert construction_flags == expected_flags
        assert result['unseeded_parameter_count_warmups'] == 0
    assert result['construction_order'] == list(D.CONSTRUCTION_ORDER[variant])
    assert result['paired_shared_state_exact'] and result['paired_cpu_rng_exact']
    assert result['environment_after']['inspected_after_construction']
    # The dry run of the correction: exact gain distances on both arms, the guard's verdict on the launched arm.
    for arm, row in result['arms'].items():
        assert row['gain_ulps'] == {'from_preflight_reference': 0, 'from_historical_training': 1}
        assert ('guard' in row) is (arm == 'T100-LQ')
    assert result['arms']['T100-LQ']['guard']['classification'] == D.G.PREFLIGHT_STATE


def test_run_bounded_returns_the_status_and_kills_the_whole_process_tree(tmp_path):
    out, err = tmp_path / 'out', tmp_path / 'err'
    assert D.run_bounded([sys.executable, '-c', 'import sys; print("done"); sys.stderr.write("note")'], 30, out, err) == (0, False)
    assert out.read_text() == 'done\n' and err.read_text() == 'note'
    assert D.run_bounded([sys.executable, '-c', 'raise SystemExit(3)'], 30, out, err) == (3, False)
    # A launcher that starts a worker and waits, as torchrun does: the bound must also end the worker.
    marker = tmp_path / 'worker.pid'
    launcher = ('import subprocess, sys, time; '
                'worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"]); '
                f'open({str(marker)!r}, "w").write(str(worker.pid)); time.sleep(600)')
    started = time.monotonic()
    code, timed_out = D.run_bounded([sys.executable, '-c', launcher], 3, out, err)
    assert timed_out and code != 0 and time.monotonic() - started < 30
    worker = int(marker.read_text())
    for _ in range(100):
        try:
            os.kill(worker, 0)
        except ProcessLookupError:
            break
        time.sleep(.05)
    else:
        os.kill(worker, 9)
        raise AssertionError('worker survived the bound')


def approved_copy(tmp_path):
    """The real diagnostic plan, approved only inside this test; every pin is read from the repository."""
    plan = json.loads((ROOT / 'results/infra/logsquare_initialization_20261105/plan.json').read_text())
    assert plan['status'] == 'REVIEW_REQUIRED' and plan['cuda_required'] is True  # the published plan stays unapproved
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps({**plan, 'status': 'APPROVED_FOR_DIAGNOSTIC', 'cuda_required': False}))
    return path, plan


def rank_rows(records, guard_sha256, mode):
    student = {'role': 'student', 'passed': True, 'classification': D.G.PASSED, 'shared_sha256': 'b' * 64,
               'reference_gain_shared_sha256': 'c' * 64, 'fe_gain': TRAINING_GAIN, 'fe_gain_hex': TRAINING_GAIN.hex()}
    counted = {'role': 'parameter_count', 'judged': False, 'fe_gain': TRAINING_GAIN, 'fe_gain_hex': TRAINING_GAIN.hex()}
    for rank in range(4):
        if mode == 'missing' and rank == 3:
            continue
        row = {'rank': rank, 'world_size': 4, 'guard_sha256': guard_sha256, 'passed': True, 'verify_only': True,
               'trainer_returned': False, 'constructions': ([dict(counted)] if rank == 0 else []) + [dict(student)]}
        if mode == 'rejected':
            row['passed'] = False
            row['constructions'][-1].update(passed=False, classification=D.G.PREFLIGHT_STATE)
        if mode == 'trained' and rank == 2:
            row.update(verify_only=False, trainer_returned=True)
        (records / f'rank{rank}.json').write_text(json.dumps(row))


@pytest.mark.parametrize('mode', ['passed', 'rejected', 'missing', 'timeout', 'trained'])
def test_rank_entry_dry_run_uses_the_production_command_and_the_controller_reader(tmp_path, monkeypatch, mode):
    """Real plan, real pins, real controller: only torchrun is replaced, by the records four ranks would leave."""
    plan_path, plan = approved_copy(tmp_path)
    workspace = tmp_path / 'out'
    workspace.mkdir()
    monkeypatch.setattr(D, 'code_identity', lambda expected: expected)
    seen = {}
    def torchrun(command, seconds, stdout, stderr):
        seen.update(command=command, seconds=seconds)
        stdout.write_text('')
        stderr.write_text('rank output')
        records = Path(command[command.index('--initial-state-records') + 1])
        assert records.is_dir() and not list(records.iterdir())
        rank_rows(records, command[command.index('--initial-state-guard-sha256') + 1], mode)
        Path(command[command.index('--out-dir') + 1]).mkdir()
        return (1 if mode == 'rejected' else 0, mode == 'timeout')
    monkeypatch.setattr(D, 'run_bounded', torchrun)
    # The controller's pre-check builds the real model: on this machine it is refused (another platform, other
    # tensors), which must be recorded without stopping the ranks. For the accepted case the real model is still
    # built and only the verdict of the controller's guard is replaced.
    if mode == 'passed':
        original_import = D.import_file
        def imported(relative, name):
            module = original_import(relative, name)
            if relative == 'scripts/logsquare_train.py':
                monkeypatch.setattr(module.G, 'verify', lambda state, guard, context: {
                    'context': context, 'classification': D.G.PASSED, 'passed': True, 'tensors': len(state) - 1})
            return module
        monkeypatch.setattr(D, 'import_file', imported)
    result = D.rank_entry_child(plan_path, 'a' * 40, workspace)
    command = seen['command']
    precheck = result['controller_precheck']
    if mode == 'passed':
        assert precheck['passed'] is True and precheck['classification'] == D.G.PASSED and precheck['tensors'] == 269
        assert precheck['context'] == 'controller pre-check, temporary CPU model' and 'temporary model' in precheck['scope']
    else:
        assert precheck['passed'] is False and precheck['classification'] == D.G.OTHER_TENSORS
        assert precheck['message'].startswith('fresh shared initialization differs from historical T100-L: ')
        assert len(precheck['localization']['per_tensor']) == precheck['tensors']
    # The controller's own command for the fresh phase, with its production allocation; only the directories are the
    # diagnostic's, and the one added word is the switch that keeps every model away from the trainer.
    controller = D.import_file('scripts/logsquare_train.py', 'logsquare_diagnostic_test_controller')
    comparison = json.loads((ROOT / plan['comparison']['path']).read_text())
    argv = list(comparison['arm_configurations']['T100-L']['validated_trainer_argv'])
    cfg = controller.CLI.build_config(controller.CLI.build_parser().parse_args(
        argv[1:] + ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '.001']))
    training = json.loads((ROOT / plan['training_plan']['path']).read_text())
    pin = {'path': str(ROOT / plan['initial_state_guard']['path']), 'sha256': plan['initial_state_guard']['sha256']}
    production = controller.command_for(comparison, cfg, workspace / (D.RANK_VARIANT + '_run'), 5000,
                                        training['allocations']['5000']['walltime_seconds'], pin,
                                        workspace / (D.RANK_VARIANT + '_initial_state'))
    assert [word for word in command if word != '--verify-only'] == production
    assert command[command.index(controller.RANK_ENTRY) + 1] == '--verify-only' and command.count('--verify-only') == 1
    assert seen['seconds'] == plan['rank_entry_timeout_seconds'] - 30 == 270
    assert result['command'] == command and result['torchrun_stderr_tail'] == 'rank output'
    assert result['run_dir_entries'] == [] and result['variant'] == D.RANK_VARIANT
    if mode == 'passed':
        assert 'error' not in result and result['attestation']['passed'] is True
        assert result['attestation']['classification'] == D.G.PASSED and 'nothing was trained' in result['attestation']['verified_on']
        assert set(result['ranks']) == {'0', '1', '2', '3'} and all(row['verify_only'] for row in result['ranks'].values())
    elif mode in ('rejected', 'trained'):  # complete observations that the reader refuses: descriptive, not a process failure
        assert 'error' not in result and result['attestation']['passed'] is False
        assert 'unverified initial state' in result['attestation']['reason']
    elif mode == 'missing':
        assert result['error'].startswith('not every rank judged') and result['ranks']['3'] is None
        assert result['attestation']['passed'] is False
    else:
        assert result['error'] == 'torchrun timed out' and result['torchrun_timed_out'] is True
    with pytest.raises(ValueError, match='new rank-entry workspace'):
        D.rank_entry_child(plan_path, 'a' * 40, workspace)


def reading_rows():
    """A report in which the cause is demonstrated and the dry run passed."""
    def row(match, reference_ulps, training_ulps, guard=None):
        value = {'original_matches_historical': match, 'hypothetical_pinned_gain_matches_historical': True,
                 'diagnostic_classification': 'exact_original_match' if match else 'only_gain_substitution_recovers_exact_hash',
                 'shared_sha256': 'a' * 64 if match else 'b' * 64,
                 'gain_ulps': {'from_preflight_reference': reference_ulps, 'from_historical_training': training_ulps}}
        return value if guard is None else {**value, 'guard': {'classification': guard, 'passed': guard == D.G.PASSED}}
    training = lambda: {'arms': {'T100-L': row(False, -1, 0), 'T100-LQ': row(False, -1, 0, D.G.PASSED)}}  # noqa: E731
    return {'preflight': {'arms': {'T100-L': row(True, 0, 1), 'T100-LQ': row(True, 0, 1, D.G.PREFLIGHT_STATE)}},
            'controller': training(), 'trainer': training(), 'trainer_rank0': training(), 'controller_threads1': training(),
            D.RANK_VARIANT: {'torchrun_returncode': 0, 'run_dir_entries': [],
                             'controller_precheck': {'passed': True, 'classification': D.G.PASSED},
                             'attestation': {'passed': True, 'shared_sha256': 'b' * 64, 'classification': D.G.PASSED}}}


def test_reading_states_cause_trained_state_and_dry_run_and_never_authorizes():
    result = D.reading(reading_rows())
    assert result['computable'] and result['cause_demonstrated'] and result['trainer_builds_the_registered_training_state']
    assert result['fix_dry_run_passed'] and result['guard_refuses_the_preflight_context']
    assert result['training_authorized_by_reading'] is False
    assert result['informative_controller_one_thread']['gain_ulps'] == {'from_preflight_reference': -1, 'from_historical_training': 0}


@pytest.mark.parametrize('change, cause, trained, dry', [
    # The preflight context no longer reproduces the reference: nothing explains the failed gate.
    (lambda v: v['preflight']['arms']['T100-L'].update(original_matches_historical=False), False, True, True),
    # The controller builds the preflight state: the gate would not have failed for this reason.
    (lambda v: v['controller']['arms']['T100-LQ'].update(original_matches_historical=True), False, True, True),
    # The controller gain is neither registered value.
    (lambda v: v['controller']['arms']['T100-L']['gain_ulps'].update(from_historical_training=-1), False, True, True),
    # A tensor other than the gain differs in the controller context.
    (lambda v: v['controller']['arms']['T100-LQ'].update(hypothetical_pinned_gain_matches_historical=False), False, True, True),
    # The trainer context left the gain of the historical checkpoints: cause shown, the relaunch would stop.
    (lambda v: v['trainer_rank0']['arms']['T100-LQ']['gain_ulps'].update(from_historical_training=1), True, False, True),
    (lambda v: v['trainer']['arms']['T100-L'].update(hypothetical_pinned_gain_matches_historical=False), True, False, True),
    # The guard refuses a trainer-context model.
    (lambda v: v['trainer']['arms']['T100-LQ']['guard'].update(passed=False), True, True, False),
    # The controller pre-check of the corrected launch refuses its temporary model.
    (lambda v: v[D.RANK_VARIANT]['controller_precheck'].update(passed=False), True, True, False),
    # The real ranks: refused records, a non-zero launcher, something written by the trainer, or another state.
    (lambda v: v[D.RANK_VARIANT]['attestation'].update(passed=False), True, True, False),
    (lambda v: v[D.RANK_VARIANT].update(torchrun_returncode=1), True, True, False),
    (lambda v: v[D.RANK_VARIANT].update(run_dir_entries=['checkpoint.pt']), True, True, False),
    (lambda v: v[D.RANK_VARIANT]['attestation'].update(shared_sha256='e' * 64), True, True, False),
])
def test_reading_is_falsifiable_on_each_statement(change, cause, trained, dry):
    variants = reading_rows()
    change(variants)
    result = D.reading(variants)
    assert result['computable'] is True and result['training_authorized_by_reading'] is False
    assert (result['cause_demonstrated'], result['trainer_builds_the_registered_training_state'],
            result['fix_dry_run_passed']) == (cause, trained, dry)


@pytest.mark.parametrize('broken', ['preflight', 'controller', 'trainer', 'trainer_rank0', 'controller_threads1', 'rank'])
def test_reading_needs_every_context(broken):
    variants = reading_rows()
    variants[D.RANK_VARIANT if broken == 'rank' else broken] = {'error': 'child timed out'}
    result = D.reading(variants)
    assert result == {'computable': False, 'reason': result['reason'], 'cause_demonstrated': False,
                      'trainer_builds_the_registered_training_state': False, 'fix_dry_run_passed': False,
                      'training_authorized_by_reading': False}
    assert 'failed contexts: ' + (D.RANK_VARIANT if broken == 'rank' else broken) in result['reason']
    # A context that reported an error gives no reading even when every field of the reading is present.
    variants = reading_rows()
    variants[D.RANK_VARIANT]['error'] = 'torchrun timed out'
    assert D.reading(variants)['computable'] is False
    del variants['trainer']
    assert 'trainer' in D.reading(variants)['reason']


def test_published_plan_fixes_the_reading_resources_and_stays_unapproved():
    base = ROOT / 'results/infra/logsquare_initialization_20261105'
    plan = json.loads((base / 'plan.json').read_text())
    resources = json.loads((base / 'resources.json').read_text())
    assert plan['status'] == resources['status'] == 'REVIEW_REQUIRED' and plan['training_authorized_by_report'] is False
    assert plan['variants'] == list(D.VARIANTS) and plan['child_timeout_seconds'] == 90
    assert set(plan['reading']['statements']) == {'cause_demonstrated', 'trainer_builds_the_registered_training_state',
                                                  'fix_dry_run_passed'}
    assert set(plan['reading']['statements']) < set(D.reading(reading_rows()))
    # Five construction contexts, the bounded torchrun dry run, and the registered margin fill the allocation.
    run = resources['jobs'][1]
    assert 5 * plan['child_timeout_seconds'] + plan['rank_entry_timeout_seconds'] + 150 == run['walltime_seconds'] == 900
    assert run['allocated_gpu_hours_cap'] == run['gpus'] * run['walltime_seconds'] / 3600 == resources['total_gpu_hours_cap'] == 1.0
    assert resources['total_core_hours_cap'] == sum(job['allocated_core_hours_cap'] for job in resources['jobs'])
    launcher = (ROOT / 'scripts/slurm/logsquare_initialization_diagnostic.sbatch').read_text()
    assert '#SBATCH --time=00:15:00' in launcher and "!= '00:15:00'" in launcher and 'rank_entry_timeout_seconds' in launcher
    guard = D.G.load_guard(ROOT / plan['initial_state_guard']['path'], plan['initial_state_guard']['sha256'])
    assert D.G.validate_provenance(guard) is guard and guard['reference']['shared_sha256'] == \
        json.loads((ROOT / plan['training_plan']['path']).read_text())['shared_initial_sha256']
