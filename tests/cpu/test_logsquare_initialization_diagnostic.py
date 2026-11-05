"""Provenance, isolation, mutation safety and honest diagnostic classification."""
import copy
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from wearusfm.training.pairing import state_sha256

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/logsquare_initialization_diagnostic.py'
SPEC = importlib.util.spec_from_file_location('initialization_diagnostic_test', SCRIPT)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
REFERENCE_GAIN, TRAINING_GAIN = 548.6939187567023, 548.6939187567021
STATUSES = ('REVIEW_REQUIRED', 'APPROVED_FOR_DIAGNOSTIC')


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
            'variants': list(D.VARIANTS), 'cuda_required': False, 'child_timeout_seconds': 75,
            'rank_entry_timeout_seconds': 285, 'cleanup_bound_seconds': 15, 'expected_scientific_sources': 102}
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
                                     'old_variants', 'rank_timeout', 'rank_timeout_missing', 'cleanup_bound',
                                     'cleanup_bound_missing', 'cleanup_bound_type', 'guard_missing', 'guard_pin',
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
    elif mutation == 'rank_timeout':    # one second more than the declared 300 s limit less the cleanup
        plan['rank_entry_timeout_seconds'] = 286
    elif mutation == 'rank_timeout_missing':
        del plan['rank_entry_timeout_seconds']
    elif mutation == 'cleanup_bound':   # a plan that declares another cleanup than the supervisor performs
        plan['cleanup_bound_seconds'] = 14
    elif mutation == 'cleanup_bound_missing':
        del plan['cleanup_bound_seconds']
    elif mutation == 'cleanup_bound_type':
        plan['cleanup_bound_seconds'] = 15.0
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
    elif mutation == 'timeout':         # one second more than the declared 90 s limit less the cleanup
        plan['child_timeout_seconds'] = 76
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
    commands, records = [], {}
    def bounded(command, seconds, stdout, stderr, record):
        commands.append(command)
        variant = command[command.index('--child') + 1]
        destination = Path(command[command.index('--child-output') + 1])
        assert seconds == (7 if variant == D.RANK_VARIANT else 2)
        assert destination.parent == out and (stdout, stderr) == (out / (variant + '.stdout'), out / (variant + '.stderr'))
        record.update(forced=False, timed_out=False, terminated=[], killed=[], survivors=[], names={},
                      process_table_unavailable=False)
        records[variant] = record
        if variant == 'trainer':  # the bound expired: the supervisor ended the child; what it wrote stays in the files
            stdout.write_bytes(b'partial')
            stderr.write_bytes(b'timeout evidence')
            record.update(forced=True, timed_out=True, terminated=[41], names={'41': 'python'})
            return -signal.SIGTERM, True
        stdout.write_text('diagnostic stdout')
        stderr.write_text('diagnostic stderr')
        row = {'variant': variant, 'original_matches_historical': False}
        if variant == 'preflight':
            row['error'] = 'retained exception'
        elif variant == 'trainer_rank0':  # it returned, but a process of its own was still running and had to be ended
            record.update(terminated=[42], names={'42': 'python'})
        elif variant == 'controller_threads1':  # a process that not even SIGKILL ended inside the cleanup bound
            record.update(terminated=[43], killed=[43], survivors=[43], names={'43': 'python'})
        elif variant == D.RANK_VARIANT:  # the process table was lost during the cleanup: descendants unverified
            record.update(process_table_unavailable=True)
        destination.write_text(json.dumps(row))
        return int('error' in row), False
    monkeypatch.setattr(D, 'run_bounded', bounded)
    result = D.execute(plan_path, 'a' * 40, out)
    assert len(commands) == 6 and len({tuple(command) for command in commands}) == 6
    assert [command[command.index('--child') + 1] for command in commands] == list(D.VARIANTS)
    assert not result['complete'] and not result['training_authorized_by_report']
    # Failed contexts: no reading, and never an authorization.
    assert result['reading']['computable'] is False and not result['reading']['cause_demonstrated']
    assert not result['reading']['fix_dry_run_passed'] and result['reading']['training_authorized_by_reading'] is False
    rows = result['variants']
    assert all(rows[variant]['supervision'] == records[variant] for variant in D.VARIANTS)
    assert 'error' not in rows['controller'] and 'output_tails' not in rows['controller']
    assert rows['preflight']['error'] == 'retained exception' and rows['preflight']['process_returncode'] == 1
    # The error of a failed context comes with the end of what it wrote; the whole output stays in the files.
    assert rows['preflight']['output_tails'] == {'preflight.stdout': 'diagnostic stdout', 'preflight.stderr': 'diagnostic stderr'}
    assert rows['trainer']['error'] == 'child timed out' and rows['trainer']['timeout_seconds'] == 2
    assert rows['trainer']['process_returncode'] == -signal.SIGTERM and 'cleanup_error' not in rows['trainer']
    assert rows['trainer']['output_tails'] == {'trainer.stdout': 'partial', 'trainer.stderr': 'timeout evidence'}
    assert (out / 'trainer.stdout').read_bytes() == b'partial'
    assert (out / 'trainer.stderr').read_bytes() == b'timeout evidence'
    # A context that left something behind is failed, even with a clean status and a complete child report.
    assert rows['trainer_rank0']['process_returncode'] == 0
    assert rows['trainer_rank0']['error'] == rows['trainer_rank0']['cleanup_error'] == \
        'processes outlived the command and were ended: 42 (python)'
    assert rows['controller_threads1']['error'] == 'processes survived the cleanup: 43 (python)'
    assert rows[D.RANK_VARIANT]['error'] == 'process table unavailable during the cleanup: descendants were not verified'
    assert json.loads((out / 'report.json').read_text()) == result
    with pytest.raises(ValueError, match='new absolute'):
        D.execute(plan_path, 'a' * 40, out)
    # Nothing is launched by a parent that cannot see its processes.
    monkeypatch.setattr(D, 'process_table', lambda *seconds: None)
    with pytest.raises(ValueError, match='cannot supervise'):
        D.execute(plan_path, 'a' * 40, tmp_path / 'unsupervised')
    assert len(commands) == 6 and not (tmp_path / 'unsupervised').exists()


def test_false_historical_match_is_descriptive_not_process_failure(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    monkeypatch.setattr(D, 'code_identity', lambda expected: expected)
    monkeypatch.setattr(D, 'validate_plan', lambda path: ({'child_timeout_seconds': 2, 'rank_entry_timeout_seconds': 7},
                                                          {}, {}, {}, {}))
    def bounded(command, seconds, stdout, stderr, record):
        Path(command[command.index('--child-output') + 1]).write_text(json.dumps({'original_matches_historical': False}))
        stdout.write_text('')
        stderr.write_text('')
        record.update(forced=False, timed_out=False, terminated=[], killed=[], survivors=[], process_table_unavailable=False)
        return 0, False
    monkeypatch.setattr(D, 'run_bounded', bounded)
    report = D.execute(path, 'a' * 40, tmp_path / 'results')
    assert report['complete'] and report['diagnostic_only']
    assert all('error' not in row and 'output_tails' not in row for row in report['variants'].values())
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
    """The real diagnostic plan, approved and CPU-only inside this test; every pin is read from the repository."""
    plan = json.loads((ROOT / 'results/infra/logsquare_initialization_20261105/plan.json').read_text())
    assert plan['status'] in STATUSES and plan['cuda_required'] is True  # the status is the reviewer's to set
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


@pytest.mark.parametrize('mode', ['passed', 'rejected', 'missing', 'timeout', 'trained', 'left_behind'])
def test_rank_entry_dry_run_uses_the_production_command_and_the_controller_reader(tmp_path, monkeypatch, mode):
    """Real plan, real pins, real controller: only torchrun is replaced, by the records four ranks would leave."""
    plan_path, plan = approved_copy(tmp_path)
    workspace = tmp_path / 'out'
    workspace.mkdir()
    monkeypatch.setattr(D, 'code_identity', lambda expected: expected)
    seen = {}
    def torchrun(command, seconds, stdout, stderr, record):
        seen.update(command=command, seconds=seconds, record=record)
        stdout.write_text('')
        stderr.write_text('rank output')
        records = Path(command[command.index('--initial-state-records') + 1])
        assert records.is_dir() and not list(records.iterdir())
        rank_rows(records, command[command.index('--initial-state-guard-sha256') + 1], mode)
        Path(command[command.index('--out-dir') + 1]).mkdir()
        record.update(forced=mode == 'timeout', timed_out=mode == 'timeout', terminated=[], killed=[], survivors=[],
                      names={}, process_table_unavailable=False)
        if mode == 'timeout':
            record.update(terminated=[51, 52, 53, 54, 55])
        if mode == 'left_behind':  # torchrun returned with four accepted records, but a rank was still running
            record.update(terminated=[56], names={'56': 'python'})
        return (1 if mode == 'rejected' else 0, mode == 'timeout')
    monkeypatch.setattr(D, 'run_bounded', torchrun)
    # The controller's pre-check builds the real model: on this machine it is refused (another platform, other
    # tensors), which must be recorded without stopping the ranks. For the accepted case the real model is still
    # built and only the verdict of the controller's guard is replaced.
    if mode in ('passed', 'left_behind'):
        original_import = D.import_file
        def imported(relative, name):
            module = original_import(relative, name)
            if relative == 'scripts/logsquare_train.py':
                monkeypatch.setattr(module.G, 'verify', lambda state, guard, context: {
                    'context': context, 'classification': D.G.PASSED, 'passed': True, 'tensors': len(state) - 1})
            return module
        monkeypatch.setattr(D, 'import_file', imported)
    before = time.monotonic()
    result = D.rank_entry_child(plan_path, 'a' * 40, workspace)
    preparation = time.monotonic() - before
    command = seen['command']
    precheck = result['controller_precheck']
    if mode in ('passed', 'left_behind'):
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
    # torchrun gets what the preparation left of the context's timeout, less the cleanup bound and the time kept for
    # the report: its bound expires before the parent's whatever the preparation took.
    whole = plan['rank_entry_timeout_seconds'] - D.CLEANUP_BOUND_SECONDS - D.REPORT_SECONDS
    assert whole == 250 and 0 < whole - seen['seconds'] <= preparation
    assert result['torchrun_bound_seconds'] == round(seen['seconds'], 3) and result['torchrun_supervision'] is seen['record']
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
    elif mode == 'timeout':
        assert result['error'] == 'torchrun timed out' and result['torchrun_timed_out'] is True
        assert 'cleanup_error' not in result  # ending the ranks at the bound is the cleanup doing its work
    else:  # accepted records do not excuse a rank that outlived torchrun
        assert result['attestation']['passed'] is True and result['torchrun_returncode'] == 0
        assert result['error'] == result['cleanup_error'] == 'processes outlived the command and were ended: 56 (python)'
        assert D.reading({**reading_rows(), D.RANK_VARIANT: result})['computable'] is False
    with pytest.raises(ValueError, match='new rank-entry workspace'):
        D.rank_entry_child(plan_path, 'a' * 40, workspace)


def test_rank_entry_dry_run_launches_nothing_when_the_preparation_used_its_time(tmp_path, monkeypatch):
    """The race with the parent's bound: torchrun is started only if its own bound can expire, and be cleaned up, first."""
    plan_path, plan = approved_copy(tmp_path)
    workspace = tmp_path / 'out'
    workspace.mkdir()
    monkeypatch.setattr(D, 'code_identity', lambda expected: expected)
    monkeypatch.setattr(D, 'run_bounded', lambda *args, **kwargs: pytest.fail('torchrun was launched'))
    whole = plan['rank_entry_timeout_seconds'] - D.CLEANUP_BOUND_SECONDS - D.REPORT_SECONDS
    # The child began so long ago that fewer than MIN_LAUNCH_SECONDS would remain for torchrun.
    result = D.rank_entry_child(plan_path, 'a' * 40, workspace, started=time.monotonic() - (whole - D.MIN_LAUNCH_SECONDS + 1))
    assert result['error'].startswith('nothing launched: the preparation left ') and result['torchrun_bound_seconds'] < D.MIN_LAUNCH_SECONDS
    assert result['torchrun_returncode'] is None and result['torchrun_timed_out'] is False and 'torchrun_supervision' not in result
    assert result['ranks'] == {str(rank): None for rank in range(4)} and result['attestation']['passed'] is False
    assert result['run_dir_entries'] is None and 'controller_precheck' in result  # what was observed is still reported
    assert D.reading({**reading_rows(), D.RANK_VARIANT: result})['computable'] is False


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


def test_published_plan_fixes_the_reading_and_resources_and_never_authorizes_training():
    base = ROOT / 'results/infra/logsquare_initialization_20261105'
    plan = json.loads((base / 'plan.json').read_text())
    resources = json.loads((base / 'resources.json').read_text())
    # Approving the diagnostic is the reviewer's step and must not break this file; training is never authorized here.
    assert plan['status'] in STATUSES and plan['training_authorized_by_report'] is False
    assert resources['training_submissions'] is False
    assert plan['variants'] == list(D.VARIANTS)
    assert set(plan['reading']['statements']) == {'cause_demonstrated', 'trainer_builds_the_registered_training_state',
                                                  'fix_dry_run_passed'}
    assert set(plan['reading']['statements']) < set(D.reading(reading_rows()))
    # The declared limit of a context includes the cleanup of its processes: the timeout is when waiting stops, and
    # the cleanup has one absolute deadline that holds all of its steps, also after a command that returned.
    timing = D.SUPERVISION
    assert plan['cleanup_bound_seconds'] == D.CLEANUP_BOUND_SECONDS == 15
    assert sum(timing[step + '_seconds'] for step in ('settle', 'orderly', 'term_grace', 'kill_wait')) <= D.CLEANUP_BOUND_SECONDS
    assert 'first to the launcher of the context alone' in plan['time_limits']
    construction = plan['child_timeout_seconds'] + plan['cleanup_bound_seconds']
    dry_run = plan['rank_entry_timeout_seconds'] + plan['cleanup_bound_seconds']
    assert (construction, dry_run) == (90, 300) and 'cleanup included' in plan['time_limits']
    # Five construction contexts, the bounded torchrun dry run, and the registered margin fill the allocation.
    run = resources['jobs'][1]
    assert 5 * construction + dry_run + 150 == run['walltime_seconds'] == 900
    # Inside the dry run torchrun still has several times its minimum, after the cleanup bound and the report.
    assert plan['rank_entry_timeout_seconds'] - D.CLEANUP_BOUND_SECONDS - D.REPORT_SECONDS >= 4 * D.MIN_LAUNCH_SECONDS
    assert run['allocated_gpu_hours_cap'] == run['gpus'] * run['walltime_seconds'] / 3600 == resources['total_gpu_hours_cap'] == 1.0
    assert resources['total_core_hours_cap'] == sum(job['allocated_core_hours_cap'] for job in resources['jobs'])
    launcher = (ROOT / 'scripts/slurm/logsquare_initialization_diagnostic.sbatch').read_text()
    assert '#SBATCH --time=00:15:00' in launcher and "!= '00:15:00'" in launcher
    for key in ('child_timeout_seconds', 'rank_entry_timeout_seconds', 'cleanup_bound_seconds'):
        assert f"p.get('{key}') != {plan[key]}" in launcher
    guard = D.G.load_guard(ROOT / plan['initial_state_guard']['path'], plan['initial_state_guard']['sha256'])
    assert D.G.validate_provenance(guard) is guard and guard['reference']['shared_sha256'] == \
        json.loads((ROOT / plan['training_plan']['path']).read_text())['shared_initial_sha256']


# --- Supervision of a context's processes (review of GPT on fd874d7, R-TIMEOUT) ---------------------------------------
# Real processes. A launcher starts its worker in a session of its own, as torchrun's elastic agent does, so the worker
# is outside the launcher's process group. What is checked is that the worker stops working (its heartbeat stops), not
# that a PID disappears: a zombie has a PID and does nothing. Every process a test starts writes its PID to a file, and
# the fixture kills whatever is left, also when the test fails.

HEARTBEAT = ('import os, signal, sys, time\n'
             '{setup}'
             'open({pid!r}, "w").write(str(os.getpid()))\n'
             'while True:\n'
             '    open({beat!r}, "w").write(str(time.monotonic_ns()))\n'
             '    time.sleep(.02)\n')


@pytest.fixture
def own_processes():
    files = []
    yield files
    for path in files:
        if path.is_file() and path.read_text().strip().isdigit():
            pid = int(path.read_text())
            assert pid > 1 and pid != os.getpid() and pid != os.getpgid(0)
            for kill in (os.killpg, os.kill):  # the group it leads, if it leads one; otherwise the process alone
                try:
                    kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def elastic_like(directory, own_processes, worker_setup='', launcher_setup='', launcher_end='worker.wait()'):
    """Scripts of a launcher and of the worker it starts in a session of its own; the files they write."""
    files = {name: directory / name for name in ('launcher.pid', 'worker.pid', 'worker.heartbeat')}
    own_processes += [files['launcher.pid'], files['worker.pid']]
    worker = HEARTBEAT.format(setup=worker_setup, pid=str(files['worker.pid']), beat=str(files['worker.heartbeat']))
    launcher = ('import os, signal, subprocess, sys, time\n' + launcher_setup +
                f'open({str(files["launcher.pid"])!r}, "w").write(str(os.getpid()))\n'
                f'worker = subprocess.Popen([sys.executable, "-c", {worker!r}], start_new_session=True)\n' +
                launcher_end + '\n')
    return launcher, files


def still_working(heartbeat):
    assert heartbeat.is_file(), 'the worker never started: this scenario proves nothing'
    before = heartbeat.read_text()
    time.sleep(.3)
    return heartbeat.read_text() != before


def pids(files):
    return {name.split('.')[0]: int(path.read_text()) for name, path in files.items() if name.endswith('.pid')}


def test_supervision_bound_ends_a_worker_in_a_session_of_its_own(tmp_path, own_processes):
    """First scenario of GPT: the bound of the command expires while its worker runs outside its process group."""
    launcher, files = elastic_like(tmp_path, own_processes)
    record, started = {}, time.monotonic()
    status, timed_out = D.run_bounded([sys.executable, '-c', launcher], 1, tmp_path / 'stdout', tmp_path / 'stderr', record)
    elapsed = time.monotonic() - started
    launched = pids(files)
    assert timed_out and status == -signal.SIGTERM
    assert not still_working(files['worker.heartbeat']), 'the worker kept working after the bound'
    # SIGTERM was enough. The launcher was asked first; it has no handler and left its worker behind, so the worker
    # was asked next. The two sessions were both seen as this invocation's.
    assert record['forced'] is True and record['timed_out'] is True and record['bound_seconds'] == 1
    assert record['terminated'] == [launched['launcher'], launched['worker']]
    assert record['killed'] == record['survivors'] == [] and record['process_table_unavailable'] is False
    assert record['observed_processes'] == 2 and set(record['names']) == {str(pid) for pid in launched.values()}
    # The names are the ones the processes had while they ran, not the placeholder of an entry that is being removed.
    assert all(name and not name.startswith(('<', '(')) for name in record['names'].values())
    assert D.supervision_problem(record) is None
    assert 1 <= elapsed < 1 + D.CLEANUP_BOUND_SECONDS and record['cleanup_seconds'] < D.SUPERVISION['orderly_seconds']


def test_supervision_asks_the_launcher_first_so_that_it_closes_its_worker_in_order(tmp_path, own_processes):
    """torchrun's agent ends its workers itself when it receives SIGTERM: the cleanup lets a launcher do that."""
    # As torch does: the handler only raises, and the launcher closes its worker where it catches the exception.
    asked = ('class Asked(Exception):\n'
             '    pass\n'
             'def ask(signum, frame):\n'
             '    raise Asked(signum)\n'
             'signal.signal(signal.SIGTERM, ask)\n')
    closing = ('try:\n'
               '    worker.wait()\n'
               'except Asked as asked:\n'
               '    os.killpg(worker.pid, asked.args[0])\n'  # the group of each worker, with the signal it received
               '    worker.wait()\n'
               '    print("closed in order, worker status", worker.returncode, flush=True)\n'
               '    raise SystemExit(7)\n')
    launcher, files = elastic_like(tmp_path, own_processes, launcher_setup=asked, launcher_end=closing)
    record = {}
    status, timed_out = D.run_bounded([sys.executable, '-c', launcher], 1, tmp_path / 'stdout', tmp_path / 'stderr', record)
    launched = pids(files)
    assert timed_out and status == 7  # the launcher's own status, after its own closing
    assert (tmp_path / 'stdout').read_text() == f'closed in order, worker status {-signal.SIGTERM}\n'
    assert not still_working(files['worker.heartbeat'])
    # Only the launcher was asked: its worker was ended by the launcher itself, and nothing was killed.
    assert record['terminated'] == [launched['launcher']] and record['killed'] == record['survivors'] == []
    assert record['observed_processes'] == 2 and D.supervision_problem(record) is None
    assert record['cleanup_seconds'] < D.SUPERVISION['orderly_seconds']


def test_supervision_lets_the_elastic_launcher_of_the_installed_torch_close_its_workers(tmp_path, monkeypatch, own_processes):
    """The real process context of torch's elastic agent, which torchrun uses for its workers.

    Its workers run in sessions of their own, and SIGTERM to the launcher is its orderly shutdown: the handler torch
    installs raises SignalException, and the agent closes the workers with the signal it received. Asking the
    workers at the same moment breaks that closing, so the launcher is asked alone. The cleanup is triggered by
    interrupting the supervisor as soon as both workers are working, because the time torch takes to import is not
    this test's subject; an interruption and an expired bound enter the same forced cleanup.
    """
    elastic = pytest.importorskip('torch.distributed.elastic.multiprocessing')
    if 'logs_specs' not in inspect.signature(elastic.start_processes).parameters:
        pytest.skip('the elastic process context of this torch is started with another signature')
    files = {f'worker{rank}.{kind}': tmp_path / f'worker{rank}.{kind}' for rank in range(2) for kind in ('pid', 'heartbeat')}
    files['launcher.pid'] = tmp_path / 'launcher.pid'
    own_processes += [path for name, path in files.items() if name.endswith('.pid')]
    workers = {rank: HEARTBEAT.format(setup='', pid=str(files[f'worker{rank}.pid']),
                                      beat=str(files[f'worker{rank}.heartbeat'])) for rank in range(2)}
    launcher = ('import os, signal, sys\n'
                'from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, SignalException, start_processes\n'
                f'open({str(files["launcher.pid"])!r}, "w").write(str(os.getpid()))\n'
                f'workers = {workers!r}\n'
                'context = start_processes(name="rank", entrypoint=sys.executable,\n'
                '                          args={rank: ("-c", code) for rank, code in workers.items()},\n'
                '                          envs={rank: {} for rank in workers},\n'
                f'                          logs_specs=DefaultLogsSpecs(log_dir={str(tmp_path / "logs")!r}))\n'
                'try:\n'
                '    while context.wait(1) is None:\n'
                '        pass\n'
                'except SignalException as error:\n'  # what the agent does: close the workers with that signal
                '    context.close(error.sigval)\n'
                '    print("closed in order after", error.sigval.name, flush=True)\n'
                '    raise SystemExit(7)\n')
    real, interrupted, record = D.process_table, [], {}
    def table(*seconds):
        if not interrupted and all(files[f'worker{rank}.heartbeat'].is_file() for rank in range(2)):
            interrupted.append(True)
            raise KeyboardInterrupt
        return real(*seconds)
    monkeypatch.setattr(D, 'process_table', table)
    with pytest.raises(KeyboardInterrupt):
        D.run_bounded([sys.executable, '-c', launcher], 120, tmp_path / 'stdout', tmp_path / 'stderr', record)
    launched = pids(files)
    assert len(set(launched.values())) == 3
    assert not still_working(files['worker0.heartbeat']) and not still_working(files['worker1.heartbeat'])
    assert (tmp_path / 'stdout').read_text() == 'closed in order after SIGTERM\n', (tmp_path / 'stderr').read_text()[-2000:]
    assert record['forced'] is True and record['terminated'] == [launched['launcher']]
    assert record['killed'] == record['survivors'] == [] and record['observed_processes'] >= 3
    assert record['cleanup_seconds'] < D.SUPERVISION['orderly_seconds'] and D.supervision_problem(record) is None


@pytest.mark.parametrize('child', ['waiting', 'cleaning'])
def test_supervision_parent_bound_ends_the_nested_launcher_and_its_worker(tmp_path, monkeypatch, own_processes, child):
    """Second scenario of GPT: the parent's bound expires first, and the child cannot finish its own supervision.

    The real execute and the real run_bounded on both levels. The helper stands for the rank-entry child. 'waiting':
    its preparation used more than the margin, so its own bound (30 s) is far away when the parent's expires.
    'cleaning': its own bound expired first, but its worker ignores SIGTERM and the child is still inside its cleanup.
    A child that is stopped cannot clean up after itself, so the parent must end the helper, its launcher and the worker.
    """
    out = tmp_path / 'output'
    bound, timing = ('30', '') if child == 'waiting' else ('.5', ', term_grace_seconds=30, kill_wait_seconds=30')
    launcher, files = elastic_like(tmp_path, own_processes,
                                   worker_setup='' if child == 'waiting' else 'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n',
                                   launcher_setup='print("launcher output", flush=True)\n'
                                                  'sys.stderr.write("launcher note"); sys.stderr.flush()\n')
    files['helper.pid'] = tmp_path / 'helper.pid'
    own_processes.append(files['helper.pid'])
    nested = [str(out / (D.RANK_VARIANT + '.torchrun.' + suffix)) for suffix in ('stdout', 'stderr')]
    helper = tmp_path / 'rank_entry_helper.py'
    helper.write_text('import importlib.util, os, sys\n'
                      f'open({str(files["helper.pid"])!r}, "w").write(str(os.getpid()))\n'
                      'print("child output", flush=True)\n'
                      'sys.stderr.write("child note"); sys.stderr.flush()\n'
                      f's = importlib.util.spec_from_file_location("nested_diagnostic", {str(SCRIPT)!r})\n'
                      'd = importlib.util.module_from_spec(s); s.loader.exec_module(d)\n'
                      f'd.run_bounded([sys.executable, "-c", {launcher!r}], {bound}, {nested[0]!r}, {nested[1]!r}'
                      f'{timing})\n')
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(D, 'VARIANTS', (D.RANK_VARIANT,))
    monkeypatch.setattr(D, '__file__', str(helper))
    monkeypatch.setattr(D, 'code_identity', lambda expected: expected)
    monkeypatch.setattr(D, 'validate_plan',
                        lambda _: ({'child_timeout_seconds': 1, 'rank_entry_timeout_seconds': 1.5}, None, None, {}, None))
    # The parent's own steps, shortened: the nested helper keeps the timings of the file it loads.
    monkeypatch.setitem(D.SUPERVISION, 'orderly_seconds', 1.)
    monkeypatch.setitem(D.SUPERVISION, 'term_grace_seconds', 1.)
    started = time.monotonic()
    report = D.execute(plan, 'a' * 40, out)
    elapsed = time.monotonic() - started
    row = report['variants'][D.RANK_VARIANT]
    assert report['complete'] is False and report['reading']['computable'] is False
    assert row['error'] == 'child timed out' and row['timeout_seconds'] == 1.5 and 'cleanup_error' not in row
    assert not still_working(files['worker.heartbeat']), 'the nested worker kept working after the parent returned'
    supervision, launched = row['supervision'], pids(files)
    assert supervision['forced'] is True and supervision['timed_out'] is True and supervision['survivors'] == []
    assert supervision['observed_processes'] >= 3 and row['process_returncode'] == -signal.SIGTERM
    asked = [pid for pid in supervision['terminated'] if pid in launched.values()]
    if child == 'waiting':  # SIGTERM was enough: each of the three was asked when the one above it was gone
        assert asked == [launched['helper'], launched['launcher'], launched['worker']] and supervision['killed'] == []
        assert supervision['cleanup_seconds'] < 1 and elapsed < 1.5 + 1
    else:  # the child had already ended its launcher; the parent ends the child and, with SIGKILL, the deaf worker
        assert asked == [launched['helper'], launched['worker']]
        assert supervision['killed'] == [launched['worker']]
        assert 1 + 1 <= supervision['cleanup_seconds'] < D.CLEANUP_BOUND_SECONDS and elapsed < 1.5 + D.CLEANUP_BOUND_SECONDS
    # Output and error stay: the files of the child and of its launcher, and their ends inside the report.
    assert row['output_tails'] == {D.RANK_VARIANT + '.stdout': 'child output\n', D.RANK_VARIANT + '.stderr': 'child note',
                                   D.RANK_VARIANT + '.torchrun.stdout': 'launcher output\n',
                                   D.RANK_VARIANT + '.torchrun.stderr': 'launcher note'}
    assert (out / (D.RANK_VARIANT + '.stderr')).read_text() == 'child note'
    assert json.loads((out / 'report.json').read_text()) == report


def test_supervision_normal_return_keeps_status_and_output_and_signals_nothing(tmp_path, own_processes):
    out, err, record = tmp_path / 'out', tmp_path / 'err', {}
    code = ('import os, sys; print(os.environ["PYTHONDONTWRITEBYTECODE"], os.getsid(0) == os.getpid() == os.getpgid(0)); '
            'sys.stderr.write("note")')
    started = time.monotonic()
    assert D.run_bounded([sys.executable, '-c', code], 30, out, err, record) == (0, False)
    assert time.monotonic() - started < 5 and out.read_text() == '1 True\n' and err.read_text() == 'note'
    clean = {'forced': False, 'timed_out': False, 'terminated': [], 'killed': [], 'survivors': [], 'names': {},
             'process_table_unavailable': False, 'bound_seconds': 30}
    assert {key: record[key] for key in clean} == clean and D.supervision_problem(record) is None
    # A worker in its own session that is still running when the launcher returns, and ends by itself a moment later:
    # it is seen, it is given the time to end, and nothing is signalled.
    worker_setup = ('import threading\n'
                    'threading.Timer(1.8, lambda: os._exit(0)).start()\n')
    launcher, files = elastic_like(tmp_path, own_processes, worker_setup=worker_setup, launcher_end='time.sleep(1.2)')
    record = {}
    assert D.run_bounded([sys.executable, '-c', launcher], 30, out, err, record) == (0, False)
    assert {key: record[key] for key in clean} == clean and record['observed_processes'] == 2
    assert 0 < record['cleanup_seconds'] < D.SUPERVISION['settle_seconds'] and D.supervision_problem(record) is None
    assert not still_working(files['worker.heartbeat'])


def test_supervision_failed_command_keeps_its_error_and_leaves_no_worker(tmp_path, own_processes):
    out, err, record = tmp_path / 'out', tmp_path / 'err', {}
    assert D.run_bounded([sys.executable, '-c', 'import sys; sys.stderr.write("broken"); raise SystemExit(3)'],
                         30, out, err, record) == (3, False)
    assert err.read_text() == 'broken' and record['terminated'] == [] and D.supervision_problem(record) is None
    # A launcher that fails and leaves its worker running in another session: the status and the error are the
    # launcher's own, the worker is ended, and the context is reported as one that left something behind.
    launcher, files = elastic_like(tmp_path, own_processes,
                                   launcher_end='time.sleep(1.2); sys.stderr.write("launcher failed"); raise SystemExit(3)')
    record, started = {}, time.monotonic()
    assert D.run_bounded([sys.executable, '-c', launcher], 30, out, err, record, settle_seconds=.3) == (3, False)
    assert time.monotonic() - started < 1.2 + D.CLEANUP_BOUND_SECONDS and err.read_text() == 'launcher failed'
    assert not still_working(files['worker.heartbeat']), 'the worker outlived its failed launcher'
    worker = pids(files)['worker']
    assert record['forced'] is False and record['timed_out'] is False
    assert record['terminated'] == [worker] and record['killed'] == record['survivors'] == []
    assert D.supervision_problem(record).startswith(f'processes outlived the command and were ended: {worker} (')


@pytest.mark.parametrize('deaf', ['worker', 'launcher', 'launcher and worker'])
def test_supervision_escalates_step_by_step_on_processes_that_ignore_sigterm(tmp_path, own_processes, deaf):
    """Launcher first; after orderly_seconds everyone still running; after term_grace_seconds more, SIGKILL."""
    # A handler that does nothing is not inherited by the worker; an ignored signal would be.
    launcher, files = elastic_like(
        tmp_path, own_processes, worker_setup='signal.signal(signal.SIGTERM, signal.SIG_IGN)\n' if 'worker' in deaf else '',
        launcher_setup='signal.signal(signal.SIGTERM, lambda *_: None)\n' if 'launcher' in deaf else '')
    record, started = {}, time.monotonic()
    status, timed_out = D.run_bounded([sys.executable, '-c', launcher], 1, tmp_path / 'stdout', tmp_path / 'stderr', record,
                                      orderly_seconds=.5, term_grace_seconds=.5)
    elapsed = time.monotonic() - started
    assert not still_working(files['worker.heartbeat']), 'a worker kept working after the cleanup'
    launched = pids(files)
    # A deaf launcher does not close its worker: the worker is asked after the orderly step, ends, and the launcher
    # then returns by itself. A deaf worker is killed after the grace, and so is a deaf launcher that still waits for it.
    expected = {'worker': (-signal.SIGTERM, {launched['worker']}, 1.), 'launcher': (0, set(), .5),
                'launcher and worker': (-signal.SIGKILL, set(launched.values()), 1.)}[deaf]
    assert timed_out and (status, set(record['killed'])) == expected[:2]
    assert record['terminated'] == [launched['launcher'], launched['worker']]  # everyone was asked, the launcher first
    assert record['survivors'] == [] and D.supervision_problem(record) is None
    assert expected[2] <= record['cleanup_seconds'] < expected[2] + 2 and elapsed < 1 + expected[2] + 2


def test_supervision_is_bounded_and_reports_survivors_when_signals_have_no_effect(tmp_path, monkeypatch, own_processes):
    """Nothing waits for a process that cannot be ended: the cleanup returns at its deadline and names what is left."""
    launcher, files = elastic_like(tmp_path, own_processes)
    record, signalled, started = {}, [], time.monotonic()
    with monkeypatch.context() as patched:
        patched.setattr(D.os, 'kill', lambda pid, signum: signalled.append((pid, signum)))
        status, timed_out = D.run_bounded([sys.executable, '-c', launcher], 1, tmp_path / 'stdout', tmp_path / 'stderr',
                                          record, orderly_seconds=.4, term_grace_seconds=.4, kill_wait_seconds=.4)
    elapsed = time.monotonic() - started
    launched = set(pids(files).values())
    # Three steps of .4 s after the bound of 1 s; the recorded time is rounded, so the lower limits leave .05 s.
    assert timed_out and status is None and 2.15 <= elapsed < 4 and 1.15 <= record['cleanup_seconds'] < 3
    assert set(record['survivors']) == set(record['terminated']) == set(record['killed']) == launched
    assert {pid for pid, signum in signalled if signum == signal.SIGTERM} == launched
    assert {pid for pid, signum in signalled if signum == signal.SIGKILL} == launched
    assert D.supervision_problem(record).startswith('processes survived the cleanup: ')
    assert still_working(files['worker.heartbeat'])  # they really were left running: the fixture ends them


def test_supervision_signals_only_the_processes_of_its_own_invocation(tmp_path, own_processes):
    """Bystanders of the same user, the same parent and the same program are never signalled."""
    bystanders = []
    try:
        for name, session in (('same_group', False), ('own_session', True)):
            files = {key: tmp_path / f'{name}.{key}' for key in ('pid', 'heartbeat')}
            own_processes.append(files['pid'])
            code = HEARTBEAT.format(setup='', pid=str(files['pid']), beat=str(files['heartbeat']))
            bystanders.append((subprocess.Popen([sys.executable, '-c', code], start_new_session=session), files['heartbeat']))
        launcher, files = elastic_like(tmp_path, own_processes, worker_setup='signal.signal(signal.SIGTERM, signal.SIG_IGN)\n')
        record = {}
        status, timed_out = D.run_bounded([sys.executable, '-c', launcher], 1, tmp_path / 'stdout', tmp_path / 'stderr', record,
                                          orderly_seconds=.5, term_grace_seconds=.5)
        assert timed_out and not still_working(files['worker.heartbeat'])
        named = {*record['terminated'], *record['killed'], *record['survivors']}
        assert named == set(pids(files).values()) and not named & {proc.pid for proc, _ in bystanders}
        assert os.getpid() not in named and record['observed_processes'] == 2
        for proc, heartbeat in bystanders:
            assert proc.poll() is None and still_working(heartbeat), 'a process outside the invocation was signalled'
    finally:
        for proc, _ in bystanders:
            proc.kill()
            proc.wait()


def test_supervision_cleans_up_when_the_supervisor_itself_is_interrupted(tmp_path, monkeypatch, own_processes):
    launcher, files = elastic_like(tmp_path, own_processes)
    real, calls, record = D.process_table, [], {}
    def table(*seconds):
        calls.append(seconds)
        if len(calls) == 3:  # the second table taken while waiting: by then the worker is running
            raise KeyboardInterrupt
        return real(*seconds)
    monkeypatch.setattr(D, 'process_table', table)
    with pytest.raises(KeyboardInterrupt):
        D.run_bounded([sys.executable, '-c', launcher], 30, tmp_path / 'stdout', tmp_path / 'stderr', record)
    assert not still_working(files['worker.heartbeat']), 'an interrupted supervisor left its worker running'
    assert record['forced'] is True and record['timed_out'] is False and record['survivors'] == []
    assert record['terminated'] == [pids(files)['launcher'], pids(files)['worker']] and record['killed'] == []


def test_supervision_without_a_process_table_starts_nothing_and_never_claims_a_verified_cleanup(tmp_path, monkeypatch,
                                                                                                own_processes):
    marker = tmp_path / 'started'
    with monkeypatch.context() as patched:
        patched.setattr(D, 'process_table', lambda *seconds: None)
        with pytest.raises(ValueError, match='no command is started unsupervised'):
            D.run_bounded([sys.executable, '-c', f'open({str(marker)!r}, "w")'], 30, tmp_path / 'stdout', tmp_path / 'stderr')
    time.sleep(.3)
    assert not marker.exists()
    # The table is lost after the start. The root's own group is still ended, because its number cannot have been
    # reused while the root is not reaped; the worker in another session cannot be named, and the record says so.
    launcher, files = elastic_like(tmp_path, own_processes)
    real, calls, record = D.process_table, [], {}
    def table(*seconds):
        calls.append(seconds)
        return real(*seconds) if len(calls) == 1 else None
    monkeypatch.setattr(D, 'process_table', table)
    status, timed_out = D.run_bounded([sys.executable, '-c', launcher], 1, tmp_path / 'stdout', tmp_path / 'stderr', record)
    launched = pids(files)
    assert timed_out and status == -signal.SIGTERM and record['terminated'] == [launched['launcher']]
    assert record['process_table_unavailable'] is True and record['survivors'] == []
    assert D.supervision_problem(record) == 'process table unavailable during the cleanup: descendants were not verified'
    assert still_working(files['worker.heartbeat'])  # unverified means unverified: the fixture ends it


def test_supervision_tracks_groups_by_membership_and_never_adopts_a_foreign_one():
    """The rule itself, on tables: members are named by their group; a group is adopted only from a member's child."""
    own = os.getpgid(0)
    invocation = D.Invocation.__new__(D.Invocation)
    invocation.root, invocation.groups, invocation.names, invocation.never = 100, {100}, {}, {0, 1, own}
    table = {1: (0, 1, 'Ss', 'init'), 100: (50, 100, 'Ss', 'launcher'), 101: (100, 100, 'S', 'helper'),
             200: (100, 200, 'Ss', 'worker'), 201: (200, 200, 'S', 'loader'), 300: (201, 300, 'Ss', 'detached'),
             400: (1, 400, 'Ss', 'bystander'), 401: (400, 400, 'S', 'bystander child'),
             500: (100, own, 'S', 'cannot happen'), 50: (1, own, 'S', 'supervisor')}
    assert invocation.alive(table, False) == [100, 101, 200, 201, 300] and invocation.groups == {100, 200, 300}
    # The launcher is gone and its orphans now belong to init: they are still members, by their groups.
    orphaned = {pid: ((1,) + row[1:] if row[0] == 100 else row) for pid, row in table.items() if pid != 100}
    assert invocation.alive(orphaned, True) == [101, 200, 201, 300] and invocation.groups == {100, 200, 300}
    # A zombie is a member that no longer runs; an emptied group is forgotten, so its number can never name a later one.
    later = {1: table[1], 200: (1, 200, 'Z', '<defunct>'), 300: (1, 300, 'Ss', 'detached'), 100: (1, 100, 'Ss', 'reused number')}
    assert invocation.alive({key: later[key] for key in (1, 200, 300)}, True) == [300] and invocation.groups == {200, 300}
    assert invocation.alive(later, True) == [300] and 100 not in invocation.groups
    # Every member ever seen keeps the name it had while it ran: the placeholder of a zombie does not replace it.
    assert invocation.names == {100: 'launcher', 101: 'helper', 200: 'worker', 201: 'loader', 300: 'detached'}
    # A command that has no session of its own is refused before anything is tracked.
    with pytest.raises(ValueError, match='session of its own'):
        D.Invocation(os.getpid())


def test_process_table_lists_this_process_with_its_parent_group_and_state():
    table = D.process_table()
    ppid, pgid, state, name = table[os.getpid()]
    assert (ppid, pgid) == (os.getppid(), os.getpgid(0)) and state and not state.startswith('Z') and name
    assert all(type(pid) is int and len(row) == 4 for pid, row in table.items())
