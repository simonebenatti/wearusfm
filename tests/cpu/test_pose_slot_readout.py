"""Fixed sketch, masked ridge and train-only dispatch oracles; no remote inputs."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import pose_slot_features as F
import pose_slot_readout as S


@pytest.fixture(autouse=True)
def projection_artifact(tmp_path, monkeypatch):
    path = tmp_path / 'projections.npz'
    np.savez(path, **F.generate_projection_arrays())
    monkeypatch.setattr(F, 'PROJECTION_PATH', path)
    return path


def data(seed=20, users=('a', 'b', 'c', 'd'), windows=5, patches=16, joints=2):
    rng = np.random.default_rng(seed)
    n = len(users) * windows
    mean = rng.normal(size=(n, patches, 384)).astype(np.float16)
    residual = (rng.normal(size=(n, patches, 336)) * .12).astype(np.float32)
    lengths = np.full(n, patches, dtype=np.int64)
    x = F.reader_features(mean, residual, 'M48', lengths)
    weights = rng.normal(size=(48, joints)) * .15
    target = (x @ weights + rng.normal(size=(n, patches, joints))).astype(np.float32)
    mask = rng.random(target.shape) > .1
    meta = [{'user': u, 'n_patches': patches, 'recording': i, 'start': 0}
            for i, u in enumerate(np.repeat(users, windows))]
    return {'lat': mean, 'lengths': lengths, 'target': target, 'mask': mask,
            'central_mask': S.T.central_mask(mask, lengths), 'meta': meta,
            'std': np.full((n, joints), .7, np.float32),
            'pose_var': np.tile([.1, .2, S.THRESHOLD, .8, .9], len(users)).astype(np.float32)}, residual


def test_projection_basis_is_fixed_orthogonal_and_not_generated_at_runtime(monkeypatch):
    expected = F.generate_projection_arrays()
    monkeypatch.setattr(F, 'generate_projection_arrays', lambda: pytest.fail('runtime QR generation'))
    got = F.projection_arrays()
    np.testing.assert_array_equal(got['slot_weights'], expected['slot_weights'])
    np.testing.assert_array_equal(got['channel_projection'], expected['channel_projection'])
    np.testing.assert_array_equal(got['slot_weights'].sum(1), np.zeros(7))
    np.testing.assert_allclose(got['slot_weights'] @ got['slot_weights'].T, np.eye(7) / 64, atol=0, rtol=0)
    np.testing.assert_allclose(got['channel_projection'].astype(float).T @ got['channel_projection'].astype(float),
                               np.eye(48), atol=2e-7, rtol=0)
    assert F.recipe()['non_dc_rows_in_order'] == [13, 26, 3, 44, 40, 43, 51]
    got['channel_projection'][:] = 0
    assert np.count_nonzero(F.projection_arrays()['channel_projection']) > 0


def test_missing_or_corrupt_projection_artifact_fails_without_fallback(projection_artifact):
    projection_artifact.unlink()
    with pytest.raises(FileNotFoundError):
        F.projection_arrays()
    arrays = F.generate_projection_arrays()
    arrays['channel_projection'] *= 2
    np.savez(projection_artifact, **arrays)
    with pytest.raises(ValueError, match='orthonormal'):
        F.projection_arrays()
    arrays = F.generate_projection_arrays()
    arrays['slot_weights'][0] *= -1
    np.savez(projection_artifact, **arrays)
    with pytest.raises(ValueError, match='Hadamard'):
        F.projection_arrays()


def test_projected_residuals_match_float64_oracle_and_preserve_common_mean():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(2, 11, 64, 384)).astype(np.float32)
    original = z.copy()
    matrices = F.projection_arrays()
    expected = (np.einsum('qk,spkd->spqd', matrices['slot_weights'].astype(float),
                          z.astype(float) - z.astype(float).mean(-2, keepdims=True))
                @ matrices['channel_projection'].astype(float)).reshape(2, 11, 336)
    got = F.projected_residuals(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(got, expected, atol=3e-7, rtol=2e-5)
    shift = rng.normal(size=(2, 11, 1, 384)).astype(np.float32)
    shifted = F.projected_residuals(torch.from_numpy(z + shift)).numpy()
    np.testing.assert_allclose(shifted, got, atol=3e-7, rtol=2e-5)
    np.testing.assert_array_equal(z, original)
    assert got.shape == (2, 11, 336) and got.dtype == np.float32


def test_identical_slots_have_exactly_zero_residuals_even_under_autocast():
    values = torch.randn(2, 11, 1, 384).to(torch.bfloat16)
    slots = values.expand(-1, -1, 64, -1)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        residual = F.projected_residuals(slots)
    assert residual.dtype == torch.float32 and not bool(residual.count_nonzero())
    with pytest.raises(ValueError, match='nonfinite'):
        F.projected_residuals(torch.full((1, 1, 64, 384), float('nan')))


def test_views_preserve_historical_mean_and_use_common_denominator():
    d, residual = data()
    mean = d['lat']
    normalized = S.T.normalized_inputs(mean, d['lengths'])
    tensor = torch.from_numpy(mean.astype(np.float32))
    den = torch.sqrt(torch.var(tensor, dim=-1, unbiased=False, keepdim=True) + 1e-5).numpy()
    q = F.projection_arrays()['channel_projection']
    expected = {'M384': normalized, 'M48': normalized @ q,
                'S384': np.concatenate((normalized @ q, residual / den), -1),
                'S720': np.concatenate((normalized, residual / den), -1)}
    for reader in F.READERS:
        got = F.reader_features(mean, residual, reader, d['lengths'])
        np.testing.assert_array_equal(got, expected[reader])
        assert got.shape == (*mean.shape[:2], F.WIDTHS[reader])
    zeros = np.zeros_like(residual)
    np.testing.assert_array_equal(F.reader_features(mean, zeros, 'S384', d['lengths'])[..., 48:], zeros)
    np.testing.assert_array_equal(F.reader_features(mean, zeros, 'S720', d['lengths'])[..., 384:], zeros)


def test_padding_is_zero_and_real_inputs_do_not_depend_on_target_masks():
    d, residual = data()
    lengths = d['lengths'].copy()
    lengths[0] = 11
    d['lat'][0, 11:] = np.nan
    residual[0, 11:] = np.inf
    for reader in F.READERS:
        first = F.reader_features(d['lat'], residual, reader, lengths)
        d['mask'][:] = False
        d['target'][:] = np.nan
        np.testing.assert_array_equal(first, F.reader_features(d['lat'], residual, reader, lengths))
        assert not np.any(first[0, 11:])
    residual[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite real'):
        F.reader_features(d['lat'], residual, 'M384', lengths)


def test_statistics_use_identical_support_for_all_four_views():
    d, residual = data(windows=2)
    for reader in F.READERS:
        parts, dim, joints = S.grouped_stats(d, residual, reader, batch=3)
        assert dim == F.WIDTHS[reader]
        for user, groups in parts.items():
            pick = np.asarray([m['user'] == user for m in d['meta']])
            for group in groups:
                for joint in group['joints']:
                    assert group['n'] == d['central_mask'][pick, :, joint].sum()


def test_cv_and_predictions_cannot_use_masked_or_evaluation_targets():
    train, residual = data(seed=61)
    weights, bias, first = S.fit_reader(train, residual, 'M48', [['a', 'c'], ['b', 'd']])
    changed = copy.deepcopy(train)
    changed['target'][~changed['mask']] = np.nan
    w2, b2, second = S.fit_reader(changed, residual, 'M48')
    np.testing.assert_array_equal(weights, w2)
    np.testing.assert_array_equal(bias, b2)
    assert first['selection'] == second['selection']
    val, val_residual = data(seed=42, users=('v', 'w'))
    prediction = S.predict(val['lat'], val_residual, val['lengths'], 'M48', weights, bias)
    val['target'][:] = 1e8
    val['mask'][:] = False
    val['pose_var'][:] = 1e8
    np.testing.assert_array_equal(prediction, S.predict(val['lat'], val_residual, val['lengths'], 'M48', weights, bias))
    with pytest.raises(ValueError, match='fold users differ'):
        S.fit_reader(train, residual, 'M48', [['a', 'b'], ['c', 'd']])


def test_zero_residual_extra_dimensions_cannot_improve_fixed_alpha_fit(monkeypatch):
    train, residual = data(seed=61, windows=2)
    residual[:] = 0
    monkeypatch.setattr(S.E, 'cross_validate', lambda *a: {'alpha': 100., 'boundary_unresolved': False})
    w0, b0, _ = S.fit_reader(train, residual, 'M48')
    w1, b1, _ = S.fit_reader(train, residual, 'S384')
    p0 = S.predict(train['lat'], residual, train['lengths'], 'M48', w0, b0)
    p1 = S.predict(train['lat'], residual, train['lengths'], 'S384', w1, b1)
    np.testing.assert_allclose(p0, p1, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(w1[48:], 0, atol=1e-12)


def cells():
    d, _ = data()
    factors = {'B/M384': .6, 'B/M48': .5, 'B/S384': .9, 'B/S720': .95,
               'R3c/M384': .5, 'R3c/M48': .4, 'R3c/S384': .7, 'R3c/S720': .8}
    return {key: {'analysis': S.T.summarize(d, d['target'][:, 4:-4] * f)} for key, f in factors.items()}


def test_eight_cell_bootstrap_and_incremental_comparisons_share_user_draws():
    c = cells()
    got = S.paired_comparison(c)
    primary = got['dynamic']
    assert len(primary['contrasts']) == 9
    assert primary['contrasts']['B_S384_minus_M48']['delta_ratio'] == pytest.approx(.01 - .25, abs=1e-7)
    item = primary['contrasts']['S384_minus_M384_difference_of_differences']
    assert item['delta_ratio'] == pytest.approx(.01, abs=1e-7)
    users = primary['users']
    draw = np.random.default_rng(S.SEED).integers(0, len(users), (2000, len(users)))
    num = np.asarray([item['per_user'][u]['delta_centered_sse'] for u in users])
    den = np.asarray([item['per_user'][u]['target_variance_sum'] for u in users])
    np.testing.assert_array_equal(item['ci95'], np.percentile(num[draw].sum(1) / den[draw].sum(1), [2.5, 97.5]))
    c['R3c/S720']['analysis']['groups']['dynamic']['centered']['per_user'].pop(users[0])
    with pytest.raises(ValueError, match='paired users missing'):
        S.paired_comparison(c)


def test_pairing_rejects_targets_and_missing_cells():
    c = cells()
    c['R3c/S720']['analysis']['targets_sha256'] = 'changed'
    with pytest.raises(ValueError, match='support differs'):
        S.paired_comparison(c)
    c.pop('R3c/S720')
    with pytest.raises(ValueError, match='eight cells'):
        S.paired_comparison(c)


def test_residuals_are_read_only_mmaps_with_fixed_shape(tmp_path):
    path = tmp_path / 'residual.npy'
    np.save(path, np.zeros((2, 16, 336), np.float32))
    got = S.read_residual(path, (2, 16, 384))
    assert isinstance(got, np.memmap) and not got.flags.writeable
    with pytest.raises(ValueError, match='shape/dtype'):
        S.read_residual(path, (3, 16, 384))
    np.save(path, np.zeros((2, 16, 336), np.float16))
    with pytest.raises(ValueError, match='shape/dtype'):
        S.read_residual(path, (2, 16, 384))


def test_alpha_boundary_stops_before_predictions(monkeypatch):
    train, residual = data(windows=2)
    monkeypatch.setattr(S.E, 'cross_validate', lambda *a: {'boundary_unresolved': True})
    with pytest.raises(ValueError, match='boundary unresolved'):
        S.fit_reader(train, residual, 'M48')


def test_benchmark_cli_opens_only_B_train_and_pins_receipt(tmp_path, monkeypatch):
    train, residual = data()
    plan = {'sources': {'file': 'hash'}, 'fold_users': [['a', 'c'], ['b', 'd']],
            'output_root': str(tmp_path), 'caches': {'B': {'train': {'marker': 'B/train'}}}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    touched = []
    monkeypatch.setattr(S.P, 'validate_plan', lambda p: None)
    monkeypatch.setattr(S.P, 'runtime_identity', lambda sha: {'sources': plan['sources']})
    monkeypatch.setattr(S.P, 'validate_reference', lambda p, m, s: touched.append(('reference', m, s)) or
                        {'users': ['a', 'b', 'c', 'd'], 'states': {}})
    def inspect(p, model, split, **kwargs):
        assert (model, split) == ('B', 'train')
        assert kwargs['expected_code_sha'] == 'frozen'
        assert kwargs['expected_plan_sha256'] == S.hashlib.sha256(path.read_bytes()).hexdigest()
        touched.append(('residual', model, split))
        return {'path': 'residual', 'states': {}}
    monkeypatch.setattr(S.P, 'validate_residual', inspect)
    monkeypatch.setattr(S.T, 'read_data', lambda pin: touched.append(pin['marker']) or train)
    monkeypatch.setattr(S, 'read_residual', lambda *args: residual)
    monkeypatch.setattr(S, 'benchmark_train', lambda *args: {'timing': 1})
    out = tmp_path / 'benchmark.json'
    assert S.main(['benchmark', '--plan', str(path), '--expected-code-sha', 'frozen', '--out', str(out)]) == 0
    assert touched == [('reference', 'B', 'train'), ('residual', 'B', 'train'), 'B/train']
    assert json.loads(out.read_text())['status'] == 'train_only_timing'


def test_full_run_has_eight_cells_same_users_and_support(monkeypatch, tmp_path):
    train, tr = data(seed=61)
    val, vr = data(seed=85, users=('v', 'w'))
    sets = {'B/train': train, 'R3c/train': copy.deepcopy(train), 'B/val': val, 'R3c/val': copy.deepcopy(val)}
    plan = {'fold_users': [['a', 'c'], ['b', 'd']], 'output_root': str(tmp_path), 'caches': {
        model: {split: {'fixture': f'{model}/{split}'} for split in ('train', 'val')} for model in S.MODELS}}
    monkeypatch.setattr(S.T, 'read_data', lambda pin: copy.deepcopy(sets[pin['fixture']]))
    monkeypatch.setattr(S, 'read_residual', lambda p, shape: tr if 'train' in str(p) else vr)
    seen = []
    def fit(d, residual, reader, folds):
        assert {m['user'] for m in d['meta']} == {'a', 'b', 'c', 'd'}
        seen.append(reader)
        return np.zeros((F.WIDTHS[reader], 2)), np.zeros(2), {'fit_support_sha256': S.T.arrays_hash(d['central_mask'])}
    monkeypatch.setattr(S, 'fit_reader', fit)
    result, coefficients = S.run(plan)
    assert len(result['cells']) == 8 and len(coefficients) == 16
    assert seen == list(F.READERS) * 2
    assert len({r['fit']['fit_support_sha256'] for r in result['cells'].values()}) == 1
    assert all(c['delta_ratio'] == 0 for c in result['paired']['dynamic']['contrasts'].values())
    sets['R3c/train']['target'][0, 4, 0] += 1
    with pytest.raises(ValueError, match='target/support mismatch'):
        S.run(plan)
