"""The exact initial-state rule of T100-LQ and the historical records it is read from."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('logsquare_initial_state_test', ROOT / 'scripts/logsquare_initial_state.py')
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)
GUARD_FILE = ROOT / 'results/infra/logsquare_initialization_20261105/initial_state_guard.json'


def neighbour(value, steps):
    """The float64 `steps` ULP away."""
    target = torch.tensor(float('inf') if steps > 0 else 0., dtype=torch.float64)
    result = torch.tensor(value, dtype=torch.float64)
    for _ in range(abs(steps)):
        result = torch.nextafter(result, target)
    return result.item()


def small_model(seed=0):
    cfg = R.small_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return WearUsFM(replace(cfg.model, tokenizer_logsquare=True))


def local_guard(model, preflight_ulps=1):
    """A guard for this machine's model with the cluster's geometry: the trainer builds the registered
    training gain, and the reference hash holds the same tensors with a gain `preflight_ulps` ULP away."""
    state = model.state_dict()
    natural = state[G.GAIN].item()
    reference_gain = neighbour(natural, preflight_ulps)
    shared = {key: value for key, value in state.items() if key != G.NEW_WEIGHT}
    shared[G.GAIN] = torch.tensor(reference_gain, dtype=torch.float64)
    frontend = {key: value for key, value in shared.items() if key.startswith(G.FRONTEND) or key == G.GAIN}
    guard = json.loads(GUARD_FILE.read_text())
    guard['new_parameter']['shape'] = list(state[G.NEW_WEIGHT].shape)
    guard['reference'].update(shared_sha256=state_sha256(shared), fe_gain=reference_gain, fe_gain_hex=reference_gain.hex())
    guard['training_gain'].update(fe_gain=natural, fe_gain_hex=natural.hex())
    guard['group_references'].update(
        outside_frontend_and_gain_sha256=state_sha256({k: v for k, v in shared.items() if k not in frontend}),
        frontend_and_gain_sha256=state_sha256(frontend))
    return G.validate_guard(guard)


def with_gain(model, gain):
    state = dict(model.state_dict())
    state[G.GAIN] = torch.tensor(gain, dtype=torch.float64)
    return state


def test_guard_module_loads_no_numeric_library():
    """Controller, ranks and diagnostic parent import it before or without Torch: it must not decide the order."""
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('g', sys.argv[1]); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print([n for n in ('torch', 'numpy', 'scipy') if n in sys.modules])")
    out = subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts/logsquare_initial_state.py')],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]'


def test_registered_values_are_reread_from_pinned_historical_records():
    guard = G.load_guard(GUARD_FILE, R.sha256_file(GUARD_FILE))
    assert G.validate_provenance(guard) is guard
    reference, training = guard['reference'], guard['training_gain']
    assert reference['shared_sha256'] == '38cb4cce85f9ba01cfd9ceceaff10ff202136089e8f6757562f684efab5345ea'
    # One buffer, one ULP: the preflight above the historical T100-L training checkpoints.
    assert G.bits(reference['fe_gain']) - G.bits(training['fe_gain']) == 1
    assert [s['checkpoint_step'] for s in training['sources']] == [5000, 19618]
    with pytest.raises(ValueError, match='differs from its pin'):
        G.load_guard(GUARD_FILE, '0' * 64)


@pytest.mark.parametrize('mutation', ['reference_hash', 'reference_gain', 'training_gain', 'checkpoint', 'training_code',
                                      'receipt_pin', 'report_pin', 'outside_group', 'frontend_group'])
def test_value_not_backed_by_its_record_is_refused(mutation):
    guard = json.loads(GUARD_FILE.read_text())
    reference, training, groups = guard['reference'], guard['training_gain'], guard['group_references']
    below = neighbour(training['fe_gain'], -1)
    if mutation == 'reference_hash': reference['shared_sha256'] = 'a' * 64
    if mutation == 'reference_gain': reference.update(fe_gain=below, fe_gain_hex=below.hex())
    if mutation == 'training_gain': training.update(fe_gain=below, fe_gain_hex=below.hex())
    if mutation == 'checkpoint': training['sources'][0]['checkpoint_sha256'] = 'a' * 64
    if mutation == 'training_code': training['sources'][1]['training_code_sha'] = 'a' * 40
    if mutation == 'receipt_pin': training['sources'][0]['sha256'] = 'a' * 64
    if mutation == 'report_pin': reference['source']['sha256'] = 'a' * 64
    if mutation == 'outside_group': groups['outside_frontend_and_gain_sha256'] = 'a' * 64
    if mutation == 'frontend_group': groups['frontend_and_gain_sha256'] = 'a' * 64
    with pytest.raises(ValueError):
        G.validate_provenance(G.validate_guard(guard))


@pytest.mark.parametrize('mutation', ['protocol', 'buffer', 'branch', 'hex', 'endpoints', 'arm', 'hash'])
def test_malformed_guard_is_refused(mutation):
    guard = json.loads(GUARD_FILE.read_text())
    if mutation == 'protocol': guard['protocol'] = 'other'
    if mutation == 'buffer': guard['gain_buffer'] = 'tokenizer.proj.bias'
    if mutation == 'branch': guard['new_parameter']['name'] = 'tokenizer.proj.weight'
    if mutation == 'hex': guard['training_gain']['fe_gain_hex'] = guard['reference']['fe_gain_hex']
    if mutation == 'endpoints': guard['training_gain']['sources'].pop()
    if mutation == 'arm': guard['training_gain']['arm'] = 'T100-G'
    if mutation == 'hash': guard['reference']['shared_sha256'] = 'XYZ'
    with pytest.raises(ValueError):
        G.validate_guard(guard)


def test_exact_bits_and_invalid_gains():
    assert G.bits(548.6939187567023) - G.bits(548.6939187567021) == 1
    assert G.bits(982.4250666375573) - G.bits(982.425066637557) == 2  # T25-L: the same two contexts, 14/10 and 03/11
    for value in (float('nan'), float('inf'), 0., -1., 1):
        with pytest.raises(ValueError):
            G.bits(value)


def test_only_the_historical_training_state_passes_and_nothing_is_modified():
    model = small_model()
    guard = local_guard(model)
    before = state_sha256(model.state_dict())
    record = G.verify(model.state_dict(), guard, 'test')
    assert record['passed'] is True and record['classification'] == G.PASSED and record['context'] == 'test'
    assert record['other_tensors_match_reference'] is True and record['logsquare_weight_zero'] is True
    assert record['reference_gain_shared_sha256'] == guard['reference']['shared_sha256'] != record['shared_sha256']
    assert record['historical_training_gain']['observed_minus_registered_ulps'] == 0
    assert record['reference_gain']['observed_minus_registered_ulps'] == -1
    assert 'localization' not in record and state_sha256(model.state_dict()) == before
    assert model.state_dict()[G.GAIN].item() == guard['training_gain']['fe_gain']  # the hypothetical gain stayed on the copy


@pytest.mark.parametrize('steps,classification', [(1, 'preflight_reference_state'), (-1, 'reference_tensors_with_unregistered_gain'),
                                                  (2, 'reference_tensors_with_unregistered_gain')])
def test_any_other_gain_is_refused_with_its_distance(steps, classification):
    model = small_model()
    guard = local_guard(model)
    state = with_gain(model, neighbour(guard['training_gain']['fe_gain'], steps))
    with pytest.raises(G.InitialStateError) as caught:
        G.verify(state, guard, 'gain test')
    record = caught.value.record
    assert record['classification'] == classification and record['other_tensors_match_reference'] is True
    assert record['historical_training_gain']['observed_minus_registered_ulps'] == steps
    assert f'observed is {steps:+d} ULP' in str(caught.value) and 'every other tensor MATCH' in str(caught.value)
    local = record['localization']
    assert local['outside_frontend_and_gain']['matches_reference'] and local['frontend_with_reference_gain']['matches_reference']
    # The preflight state is the one whose own hash equals the reference: refused all the same.
    assert (record['shared_sha256'] == guard['reference']['shared_sha256']) is (steps == 1)


@pytest.mark.parametrize('where', ['parameter', 'frontend', 'seed'])
def test_a_tensor_difference_is_never_explained_by_the_gain(where):
    model = small_model()
    guard = local_guard(model)
    changed = small_model(seed=1 if where == 'seed' else 0)
    name = next(key for key in changed.state_dict() if key.startswith(G.FRONTEND)) if where == 'frontend' else 'tokenizer.proj.bias'
    if where != 'seed':
        with torch.no_grad():
            tensor = changed.state_dict()[name]
            flat = tensor.reshape(-1)
            flat[0] = torch.nextafter(flat[0], torch.tensor(float('inf'), dtype=tensor.dtype))  # one ULP of one element
    with pytest.raises(G.InitialStateError) as caught:
        G.verify(changed.state_dict(), guard, 'tensor test')
    record, message = caught.value.record, str(caught.value)
    assert record['classification'] == G.OTHER_TENSORS and record['other_tensors_match_reference'] is False
    assert record['historical_training_gain']['observed_minus_registered_ulps'] == 0  # the gain alone is the registered one
    local = record['localization']
    assert local['outside_frontend_and_gain']['matches_reference'] is (where == 'frontend')
    # The front-end coefficients do not depend on the seed: another seed changes only the tensors outside it.
    assert local['frontend_with_reference_gain']['matches_reference'] is (where != 'frontend')
    assert 'every other tensor DIFFER' in message and 'Not the import-order effect' in message
    reference = {key: state_sha256(value) for key, value in model.state_dict().items() if key != G.NEW_WEIGHT}
    differing = [key for key, row in local['per_tensor'].items() if row['sha256'] != reference[key]]
    assert differing == [name] if where != 'seed' else len(differing) > 10


def test_nonzero_branch_wrong_shape_and_invalid_gain_are_refused():
    model = small_model()
    guard = local_guard(model)
    state = dict(model.state_dict())
    state[G.NEW_WEIGHT] = state[G.NEW_WEIGHT].clone()
    state[G.NEW_WEIGHT][0, 0] = 1e-9
    with pytest.raises(G.InitialStateError, match='new log-square weight is not zero'):
        G.verify(state, guard, 'branch')
    other = copy.deepcopy(guard)
    other['new_parameter']['shape'] = [32, 64]
    with pytest.raises(ValueError, match='shape'):
        G.verify(model.state_dict(), other, 'shape')
    for bad in (torch.tensor(float('nan'), dtype=torch.float64), torch.tensor(2., dtype=torch.float32), torch.ones(1, dtype=torch.float64)):
        broken = dict(model.state_dict())
        broken[G.GAIN] = bad
        with pytest.raises(ValueError, match='invalid calibrated gain'):
            G.verify(broken, guard, 'gain')
    without = {key: value for key, value in model.state_dict().items() if key != G.NEW_WEIGHT}
    with pytest.raises(ValueError, match='without log-square weight'):
        G.verify(without, guard, 'control arm')
