"""Provenance, isolation, mutation safety and honest diagnostic classification."""
import copy
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from wearusfm.training.pairing import state_sha256

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('initialization_diagnostic_test', ROOT / 'scripts/logsquare_initialization_diagnostic.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


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
    plan = {'protocol': D.PROTOCOL, 'status': 'APPROVED_FOR_DIAGNOSTIC', 'training_plan': training_pin,
            'preflight_report': report, 'candidate_config': candidate, 'comparison': comparison,
            'variants': list(D.VARIANTS), 'cuda_required': False, 'child_timeout_seconds': 90,
            'expected_scientific_sources': 102}
    path = root / 'diagnostic.json'
    def save():
        path.write_text(json.dumps(plan))
    save()
    return root, path, plan, training, pin, save


def test_valid_plan_uses_only_local_provenance_and_all_102_sources(fixture_plan):
    root, path, plan, _, _, _ = fixture_plan
    validated, candidate, _, expected = D.validate_plan(path, root)
    assert validated == plan and candidate['seed'] == 0
    assert expected == {'shared_sha256': 'a' * 64, 'fe_gain': 548.6939187567023}


@pytest.mark.parametrize('mutation', ['plan_pin', 'source', 'source_inventory', 'candidate_provenance',
                                     'report_provenance', 'comparison_provenance', 'variants', 'timeout', 'status'])
def test_mutations_rejected_before_construction(fixture_plan, mutation):
    root, path, plan, training, pin, save = fixture_plan
    if mutation == 'plan_pin':
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
    monkeypatch.setattr(D, 'validate_plan', lambda path: ({'child_timeout_seconds': 2}, {}, {}, {}))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        variant = command[command.index('--child') + 1]
        destination = Path(command[command.index('--child-output') + 1])
        assert kwargs['timeout'] == 2 and kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
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
    assert len(commands) == 5 and len({tuple(command) for command in commands}) == 5
    assert not result['complete'] and not result['training_authorized_by_report']
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
    monkeypatch.setattr(D, 'validate_plan', lambda path: ({'child_timeout_seconds': 2}, {}, {}, {}))
    def run(command, **kwargs):
        Path(command[command.index('--child-output') + 1]).write_text(json.dumps({'original_matches_historical': False}))
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(D.subprocess, 'run', run)
    report = D.execute(path, 'a' * 40, tmp_path / 'results')
    assert report['complete'] and report['diagnostic_only']
    assert not report['training_authorized_by_report']


@pytest.mark.parametrize('variant', D.VARIANTS)
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
    monkeypatch.setattr(D, 'validate_plan', lambda path: (plan, candidate, comparison, expected))
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
