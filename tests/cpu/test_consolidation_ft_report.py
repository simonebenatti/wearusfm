"""Meaningful controls for a read-only analysis of the frozen EPN fits."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('consolidation_ft_report_tested', REPO / 'scripts/consolidation_ft_report.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def raw_json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


@pytest.fixture
def complete_fixture(tmp_path, monkeypatch):
    """Synthetic native-label data; no real predictions or scores are read."""
    plan = json.loads((REPO / 'results/infra/consolidation_20261104/finetune_plan.json').read_text())
    root = tmp_path / 'epn'
    root.mkdir()
    subjects = [f'user_{i:03}' for i in range(314)]
    roles = R.fold_roles(subjects)
    rows = np.arange(47100)
    s = np.repeat(np.asarray(subjects), 150)
    y = np.tile(np.asarray(['fist', 'noGesture']), len(rows) // 2)
    remote = Path(plan['fit_output_root'])
    manifest = {'protocol': R.MANIFEST_PROTOCOL, 'code_sha': R.FIT_CODE, 'source_sha256': R.FIT_SOURCE,
                'plan_sha256': R.FIT_PLAN, 'input_root_remote': str(remote), 'files': [], 'checkpoints': [],
                'expected_roles': roles, 'expected_subjects': subjects, 'expected_n_rows': len(rows)}

    def write(key, raw):
        p = root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        pin = {'relative_path': key, 'remote_path': str(remote / key), 'sha256': R.sha(raw), 'bytes': len(raw)}
        manifest['files'].append(pin)
        return pin

    historical = {'complete': True, 'kind': 'aggregate', 'arm': 'T100-G', 'task': 'p2', 'step': 19618,
                  'datasets': {'epn612': {'n_oof_subjects': 314, 'n_oof_windows': len(rows), 'folds': []}}}
    for fold in range(4):
        historical['datasets']['epn612']['folds'].append({'fold': fold, **roles[fold],
            **{f'n_{role}': 150 * len(ids) for role, ids in roles[fold].items()}})
    hist_raw = raw_json(historical)
    (tmp_path / 'history.json').write_bytes(hist_raw)
    manifest['historical_reference'] = {'path': 'history.json', 'sha256': R.sha(hist_raw), 'dataset': 'epn612'}
    monkeypatch.setattr(R, 'ROOT', tmp_path)

    for regime in R.REGIMES:
        for fold in range(4):
            provenance = {'protocol': R.FIT_PROTOCOL, 'recipe_sha256': R.RECIPE, 'source_sha256': R.FIT_SOURCE,
                          'checkpoint_sha256': plan['checkpoint']['sha256'], 'input_sha256': plan['input_sha256'],
                          'fold': fold, 'regime': regime, 'roles': roles[fold], 'classes': ['fist', 'noGesture']}
            previous = previous_report_pin = None
            for index in (0, 1):
                directory = f'fold{fold}/{regime}/candidate{index}'
                count = int(np.ceil(150 * len(roles[fold]['train']) / 8))
                val = [.4 + .1 * index, .45 + .1 * index, .5 + .1 * index, .5 + .1 * index, .5 + .1 * index]
                history = [{'epoch': e + 1, 'step': count * (e + 1), 'loss': 1. / (e + 1),
                            'last_gradient_norm': .1, 'val_bacc': value} for e, value in enumerate(val)]
                pt = f'{directory}/candidate_{index}.pt'
                pt_pin = {'relative_path': pt, 'remote_path': str(remote / pt), 'sha256': 'a' * 64, 'bytes': 100}
                manifest['checkpoints'].append(pt_pin)
                candidate = {'complete': True, 'convergence_claim': False, 'provenance': provenance,
                             'candidate': index, 'lr': plan['lr_grid'][regime][index],
                             'head_lr': plan['lr_grid'][regime][index] * plan['head_lr_multiplier'][regime],
                             'history': history, 'best_epoch': 3, 'val_bacc': val[2], 'updates': 5 * count,
                             'planned_updates': 5 * count, 'stopped': 'fixed_budget',
                             'checkpoint': pt_pin['remote_path'], 'checkpoint_sha256': pt_pin['sha256']}
                cp = write(f'{directory}/candidate_{index}.json', raw_json(candidate))
                write(f'{directory}/metrics.jsonl', b''.join(raw_json({'candidate': index, 'lr': candidate['lr'], **h}).replace(b'\n', b'') + b'\n' for h in history))
                report = {'protocol': R.FIT_PROTOCOL, 'mode': 'fit', 'code_sha': R.FIT_CODE, 'source_sha256': R.FIT_SOURCE,
                          'recipe_sha256': R.RECIPE, 'plan_sha256': R.FIT_PLAN,
                          'checkpoint_sha256': plan['checkpoint']['sha256'], 'input_sha256': plan['input_sha256'],
                          'fold': fold, 'regime': regime, 'complete': True, 'convergence_claim': False,
                          'fit_allocation_by_regime': plan['fit_allocation_by_regime'],
                          'fit_packaging_policy': plan['fit_packaging_policy'],
                          'fit_limit_seconds_by_regime': {'frozen': 14100, 'finetune': 24900, 'scratch': 24900},
                          'elapsed_seconds': 100, 'roles': roles[fold], 'candidate_provenance': provenance,
                          'candidate_file_sha256': {str(index): cp['sha256']}, 'candidates': [candidate]}
                if index == 0:
                    report.update(candidate_only=0, scientific_status='first_candidate_recorded', held_evaluated=False)
                else:
                    report['candidates'] = [{**previous, 'source_receipt': {'path': previous_report_pin['remote_path'], 'sha256': previous_report_pin['sha256']}}, candidate]
                    mask = np.isin(s, roles[fold]['held'])
                    targets = y[mask]
                    pred = targets.copy() if regime == 'finetune' else np.full(len(targets), 'fist') if regime == 'frozen' else np.where(targets == 'fist', 'noGesture', 'fist')
                    buf = io.BytesIO()
                    np.savez(buf, rows=rows[mask], y=targets, pred=pred, s=s[mask], fold=np.full(mask.sum(), fold), input_sha256=plan['input_sha256'])
                    hp = write(f'{directory}/held.npz', buf.getvalue())
                    report.update(scientific_status='held_evaluated', held_evaluated=True, selected_candidate=1,
                                  selected_lr=candidate['lr'], selected_best_epoch=3, n_held=int(mask.sum()),
                                  held_bacc=R.score(targets, pred), held_sha256=hp['sha256'])
                previous_report_pin = write(f'{directory}/report.json', raw_json(report))
                previous = candidate
    plan_path, manifest_path = tmp_path / 'plan.json', tmp_path / 'manifest.json'
    plan_path.write_bytes(raw_json(plan))
    manifest_path.write_bytes(raw_json(manifest))
    return root, plan_path, manifest_path


def rewrite_fixture_file(fixture, key, mutate, repin=True):
    root, _, path = fixture
    value = json.loads((root / key).read_text())
    mutate(value)
    raw = raw_json(value)
    (root / key).write_bytes(raw)
    if repin:
        manifest = json.loads(path.read_text())
        for item in manifest['files']:
            if item['relative_path'] == key:
                item.update(sha256=R.sha(raw), bytes=len(raw))
        path.write_bytes(raw_json(manifest))


def test_complete_native_label_matrix_has_expected_paired_contrasts(complete_fixture):
    report = R.run(*complete_fixture)
    assert report['scientifically_complete']
    assert report['verified_files'] == 84 and report['verified_remote_checkpoint_hashes'] == 24
    assert len(report['candidate_curves']) == 24 and len(report['choices']) == 12
    stat = report['statistics']
    assert stat['n_subjects'] == 314 and stat['n_rows'] == 47100
    assert stat['contrasts']['finetune_minus_frozen']['delta_pp'] == 50
    assert stat['contrasts']['finetune_minus_scratch']['delta_pp'] == 100
    assert stat['contrasts']['finetune_minus_frozen']['ci95_pp'] == [50, 50]
    assert report['scratch_rule']['flag'] is False
    assert not report['checkpoint_payloads_loaded']


@pytest.mark.parametrize('failure', ['hash', 'selection', 'split_source', 'early_held', 'truncated_dose'])
def test_receipt_tampering_is_rejected(complete_fixture, failure):
    base = 'fold0/frozen/candidate1/report.json'
    if failure == 'hash':
        rewrite_fixture_file(complete_fixture, base, lambda x: x.update(held_bacc=.123), repin=False)
        message = 'hash/size'
    elif failure == 'selection':
        rewrite_fixture_file(complete_fixture, base, lambda x: x.update(selected_candidate=0, selected_lr=.001))
        message = 'validation selection'
    elif failure == 'split_source':
        rewrite_fixture_file(complete_fixture, base, lambda x: x['candidates'][0]['source_receipt'].update(sha256='b' * 64))
        message = 'source receipt'
    elif failure == 'early_held':
        rewrite_fixture_file(complete_fixture, 'fold0/frozen/candidate0/report.json', lambda x: x.update(held_evaluated=True))
        message = 'prematurely'
    else:
        rewrite_fixture_file(complete_fixture, 'fold0/frozen/candidate0/candidate_0.json', lambda x: x.update(stopped='max_updates'))
        message = 'candidate JSON pin'
    with pytest.raises(ValueError, match=message):
        R.run(*complete_fixture)


def test_candidate_validation_rejects_wrong_dose_and_honors_divergence(complete_fixture):
    root, plan_path, _ = complete_fixture
    candidate = json.loads((root / 'fold0/frozen/candidate0/candidate_0.json').read_text())
    plan = json.loads(plan_path.read_text())
    metrics = [{'candidate': 0, 'lr': candidate['lr'], **h} for h in candidate['history']]
    changed = copy.deepcopy(candidate)
    changed['planned_updates'] -= 1
    with pytest.raises(ValueError, match='dose mismatch'):
        R.candidate_check(changed, changed['provenance'], 'frozen', 0, plan, 31650, metrics)
    changed = copy.deepcopy(candidate)
    changed.update(stopped='nonfinite', best_epoch=None, val_bacc=None, history=changed['history'][:2], numeric_failure={'kind': 'gradient'})
    changed.pop('checkpoint')
    changed.pop('checkpoint_sha256')
    assert R.candidate_check(changed, changed['provenance'], 'frozen', 0, plan, 31650, metrics[:2]) is False
    assert R.select([changed, candidate]) is candidate


def cell(subjects, y, pred):
    return {'rows': np.arange(len(y)), 'y': np.asarray(y), 'pred': np.asarray(pred), 's': np.asarray(subjects), 'fold': np.zeros(len(y), dtype=int)}


def test_pooled_macro_recall_is_not_average_subject_score():
    y = ['a'] * 9 + ['b'] + ['a'] + ['b'] * 9
    pred = ['a'] * 10 + ['b'] * 10
    a = cell(['s0'] * 10 + ['s1'] * 10, y, pred)
    result = R.statistics({r: a for r in R.REGIMES}, n_boot=80)
    assert result['cells']['frozen']['balanced_accuracy'] == .9
    assert [s['balanced_accuracy']['frozen'] for s in result['per_subject']] == [.5, .5]
    assert result['contrasts']['finetune_minus_frozen']['ci95_pp'] == [0, 0]


def test_bootstrap_matches_direct_resampling_and_retains_missing_classes():
    subjects = ['s0', 's0', 's1', 's2', 's2', 's2']
    y = ['a', 'a', 'b', 'c', 'c', 'c']
    cells = {'frozen': cell(subjects, y, ['a', 'b', 'a', 'c', 'a', 'b']),
             'finetune': cell(subjects, y, y), 'scratch': cell(subjects, y, ['b'] * 6)}
    n_boot, seed = 80, 73
    result = R.statistics(cells, n_boot=n_boot, seed=seed)
    unique = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    deltas, missing = [], 0
    for picks in rng.integers(len(unique), size=(n_boot, len(unique))):
        indices = np.concatenate([np.flatnonzero(np.asarray(subjects) == unique[p]) for p in picks])
        supported = sorted(set(np.asarray(y)[indices]))
        scores = []
        for r in R.REGIMES:
            target, predicted = cells[r]['y'][indices], cells[r]['pred'][indices]
            scores.append(sum(np.sum((target == c) & (predicted == c)) / np.sum(target == c) for c in supported) / len(supported))
        deltas.append(100 * (scores[1] - scores[0]))
        missing += len(supported) < 3
    assert np.allclose(result['contrasts']['finetune_minus_frozen']['ci95_pp'], np.quantile(deltas, [.025, .975]))
    assert result['bootstrap']['draws_with_missing_classes'] == missing > 0
    assert result['bootstrap']['discarded_draws'] == 0


@pytest.mark.parametrize('key', ['rows', 'y', 's', 'fold'])
def test_unpaired_support_is_rejected(key):
    a = cell(['s0', 's1'], ['fist', 'noGesture'], ['fist', 'noGesture'])
    cells = {r: copy.deepcopy(a) for r in R.REGIMES}
    if key in ('rows', 'fold'):
        cells['scratch'][key][0] += 1
    else:
        cells['scratch'][key][0] = cells['scratch'][key][1]
    with pytest.raises(ValueError, match='pairing differs'):
        R.statistics(cells, n_boot=20)


def test_scratch_rule_is_conservative_and_incomplete_is_not_false():
    def candidate(best, last_delta=0):
        return {'stopped': 'fixed_budget', 'best_epoch': best,
                'history': [{'epoch': i, 'val_bacc': .5 + (last_delta if i == 5 else 0)} for i in range(1, 6)]}
    assert not R.scratch_rule([candidate(4) for _ in range(4)])['flag']
    assert R.scratch_rule([candidate(5), candidate(5), candidate(5), candidate(4)])['flag']
    assert R.scratch_rule([candidate(5, .01), candidate(4, .01), candidate(4, .01), candidate(4)])['flag']
    result = R.scratch_rule([candidate(5), candidate(5), candidate(5), None])
    assert result['flag'] is None and not result['evaluable']


def test_ties_choose_first_epoch_and_lower_lr(complete_fixture):
    root, plan_path, _ = complete_fixture
    c = json.loads((root / 'fold0/frozen/candidate0/candidate_0.json').read_text())
    plan = json.loads(plan_path.read_text())
    c['best_epoch'] = 5  # same best score since epoch3: choosing late is invalid
    metrics = [{'candidate': 0, 'lr': c['lr'], **h} for h in c['history']]
    with pytest.raises(ValueError, match='best epoch'):
        R.candidate_check(c, c['provenance'], 'frozen', 0, plan, 31650, metrics)
    lower = {'stopped': 'fixed_budget', 'candidate': 0, 'val_bacc': .7}
    higher = {'stopped': 'fixed_budget', 'candidate': 1, 'val_bacc': .7}
    assert R.select([higher, lower]) is lower
