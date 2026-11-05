"""Synthetic controls for fifteen-epoch receipt audit and paired dose reads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ft15_report_tested', REPO / 'scripts/consolidation_ft15_report.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
support_spec = importlib.util.spec_from_file_location('ft5_synthetic_fixture_support', Path(__file__).with_name('test_consolidation_ft_report.py'))
SUPPORT = importlib.util.module_from_spec(support_spec)
support_spec.loader.exec_module(SUPPORT)
SUPPORT.R = R.FIVE
complete_fixture = SUPPORT.complete_fixture
raw_json = SUPPORT.raw_json


@pytest.fixture
def fifteen_fixture(complete_fixture, monkeypatch):
    """Reuse synthetic native-label receipts, never real scientific predictions.

    git_bytes is a fake object store: tests exercise exact snapshot validation
    without making commits or changing any real repository.
    """
    root5, plan5_path, manifest5_path = complete_fixture
    tmp = root5.parent
    report5 = R.FIVE.run(*complete_fixture)
    report5_raw = raw_json(report5)
    (tmp / 'prior_report.json').write_bytes(report5_raw)
    monkeypatch.setattr(R, 'PRIOR_REPORT_SHA', R.sha(report5_raw))
    monkeypatch.setattr(R, 'ROOT', tmp)
    plan = json.loads((REPO / R.PLAN_PATH).read_text())
    plan.update(status='claude_review_converged', fits_status='claude_review_converged')
    plan['review']['commit'] = 'ab' * 20
    plan['prior_five_results'].update(
        report={'path': 'prior_report.json', 'sha256': R.sha(report5_raw)},
        manifest={'path': 'manifest.json', 'sha256': R.sha(manifest5_path.read_bytes())})
    plan['reference_five']['plan'] = {'path': 'plan.json', 'sha256': R.sha(plan5_path.read_bytes())}
    plan['reference_five']['manifest'] = plan['prior_five_results']['manifest']
    plan_path = tmp / 'plan15.json'
    plan_raw = raw_json(plan)
    plan_path.write_bytes(plan_raw)
    root15 = tmp / 'epn15'
    shutil.copytree(root5, root15)
    manifest = json.loads(manifest5_path.read_text())
    for key in ('code_sha', 'source_sha256', 'plan_sha256'):
        manifest.pop(key)
    manifest.update(protocol=R.MANIFEST_PROTOCOL, input_root_remote=plan['fit_output_root'], files=[], checkpoints=[])
    extras = ('scripts/consolidation_finetune15.py', 'scripts/consolidation_finetune.py', 'scripts/calibration_probe.py',
              'scripts/tokenizer_train.py', 'scripts/tokenizer_preflight.py', 'scripts/slurm/consolidation_finetune15.sbatch',
              'scripts/slurm/lib_code_dir.sh')
    blobs = {p: f'# synthetic {p}\n'.encode() for p in extras}
    blobs[extras[0]] = ('SOURCE_EXTRA_PATHS = ' + repr(extras) + '\nraise AssertionError("must never execute snapshot source")\n').encode()
    blobs['src/model.py'] = b'# synthetic scientific model\n'
    blobs[R.PLAN_PATH] = plan_raw
    source = R.sha(json.dumps({p: R.sha(blobs[p]) for p in ['src/model.py', *extras]}, sort_keys=True).encode())
    code = hashlib.sha1(b'synthetic object-store commit').hexdigest()
    identity = {'code_sha': code, 'plan_sha256': R.sha(plan_raw), 'source_sha256': source, 'recipe_sha256': plan['recipe_sha256']}
    manifest['run_identity'] = identity

    def fake_git(*args):
        if args == ('rev-parse', f'{code}^{{commit}}'):
            return (code + '\n').encode()
        if args == ('ls-tree', '-r', '--name-only', code, 'src'):
            return b'src/model.py\n'
        if args[0] == 'show' and args[1].startswith(code + ':'):
            return blobs[args[1][len(code) + 1:]]
        raise ValueError('unknown object-store request')
    monkeypatch.setattr(R, 'git_bytes', fake_git)
    remote = Path(plan['fit_output_root'])
    def write(key, raw):
        (root15 / key).write_bytes(raw)
        pin = {'relative_path': key, 'remote_path': str(remote / key), 'sha256': R.sha(raw), 'bytes': len(raw)}
        manifest['files'].append(pin)
        return pin
    for regime in R.REGIMES:
        for fold in range(4):
            previous = previous_report = None
            for index in (0, 1):
                directory = f'fold{fold}/{regime}/candidate{index}'
                candidate = json.loads((root5 / directory / f'candidate_{index}.json').read_text())
                report = json.loads((root5 / directory / 'report.json').read_text())
                provenance = copy.deepcopy(report['candidate_provenance'])
                provenance.update(protocol=R.FIT_PROTOCOL, source_sha256=source, recipe_sha256=plan['recipe_sha256'])
                step = candidate['planned_updates'] // 5
                history = [{'epoch': epoch, 'step': epoch * step, 'loss': 1. / epoch,
                            'last_gradient_norm': .1, 'val_bacc': .1 + .01 * epoch + .05 * index} for epoch in range(1, 16)]
                ptkey = f'{directory}/candidate_{index}.pt'
                manifest['checkpoints'].append({'relative_path': ptkey, 'remote_path': str(remote / ptkey),
                                               'sha256': candidate['checkpoint_sha256'], 'bytes': 100})
                candidate.update(provenance=provenance, history=history, best_epoch=15, val_bacc=history[-1]['val_bacc'],
                                 updates=15 * step, planned_updates=15 * step, checkpoint=str(remote / ptkey))
                cp = write(f'{directory}/candidate_{index}.json', raw_json(candidate))
                write(f'{directory}/metrics.jsonl', b''.join(json.dumps({'candidate': index, 'lr': candidate['lr'], **h}).encode() + b'\n' for h in history))
                report.update(protocol=R.FIT_PROTOCOL, **identity, candidate_provenance=provenance,
                              candidate_file_sha256={str(index): cp['sha256']},
                              initialization='fresh from teacher19618 or random seed0; new 15-epoch cosine; no resume from five epochs')
                for key in ('fit_allocation_by_regime', 'fit_packaging_policy', 'resource_projection_policy', 'reference_five', 'reference_source_sha256'):
                    report[key] = plan[key]
                report['fit_limit_seconds_by_regime'] = {r: plan['fit_allocation_by_regime'][r]['seconds'] - 300 for r in R.REGIMES}
                report['candidates'] = [candidate]
                if index == 1:
                    report['candidates'] = [{**previous, 'source_receipt': {'path': previous_report['remote_path'], 'sha256': previous_report['sha256']}}, candidate]
                    report['selected_best_epoch'] = 15
                    hp = write(f'{directory}/held.npz', (root5 / directory / 'held.npz').read_bytes())
                    report['held_sha256'] = hp['sha256']
                previous_report = write(f'{directory}/report.json', raw_json(report))
                previous = candidate
    manifest_path = tmp / 'manifest15.json'
    manifest_path.write_bytes(raw_json(manifest))
    return root15, root5, plan_path, manifest_path, blobs


def test_complete_fifteen_matrix_reuses_verified_five_and_pairs_all_six(fifteen_fixture):
    report = R.run(*fifteen_fixture[:4])
    assert report['scientifically_complete']
    assert report['verified_files'] == 84 and report['verified_remote_checkpoint_hashes'] == 24
    assert report['statistics']['contrasts']['finetune_minus_frozen']['delta_pp'] == 50
    assert all(x['delta_pp'] == 0 and x['ci95_pp'] == [0, 0] for x in report['dose_comparison']['contrasts'].values())
    assert all(x['flag'] is True for x in report['validation_tail'].values())
    assert report['dose_comparison']['bootstrap']['same_draw_all_six_cells']
    assert len(report['candidate_curves']) == 24


def test_all_candidates_diverged_keeps_failed_cell_and_blocks_all_bootstraps(fifteen_fixture):
    root15, root5, plan_path, manifest_path, _ = fifteen_fixture
    manifest = json.loads(manifest_path.read_text())
    base = 'fold0/scratch'
    def save(key, value):
        raw = raw_json(value)
        (root15 / key).write_bytes(raw)
        for entry in manifest['files']:
            if entry['relative_path'] == key:
                entry.update(sha256=R.sha(raw), bytes=len(raw))
        return R.sha(raw)
    previous = previous_hash = None
    for index in (0, 1):
        directory = f'{base}/candidate{index}'
        ck, rk, mk = f'{directory}/candidate_{index}.json', f'{directory}/report.json', f'{directory}/metrics.jsonl'
        candidate = json.loads((root15 / ck).read_text())
        candidate.update(stopped='nonfinite', val_bacc=None, best_epoch=None,
                         history=candidate['history'][:2], numeric_failure={'kind': 'gradient'})
        for key in ('checkpoint', 'checkpoint_sha256', 'updates', 'planned_updates'):
            candidate.pop(key)
        candidate_hash = save(ck, candidate)
        raw = b''.join(json.dumps({'candidate': index, 'lr': candidate['lr'], **h}).encode() + b'\n' for h in candidate['history'])
        (root15 / mk).write_bytes(raw)
        for entry in manifest['files']:
            if entry['relative_path'] == mk:
                entry.update(sha256=R.sha(raw), bytes=len(raw))
        report = json.loads((root15 / rk).read_text())
        report.update(candidate_file_sha256={str(index): candidate_hash}, candidates=[candidate])
        if index == 1:
            report['candidates'] = [{**previous, 'source_receipt': {
                'path': str(Path(manifest['input_root_remote']) / base / 'candidate0/report.json'), 'sha256': previous_hash}}, candidate]
            report.update(scientific_status='all_candidates_diverged', held_evaluated=False, selected_lr=None)
            for key in ('held_sha256', 'n_held', 'held_bacc', 'selected_best_epoch', 'selected_candidate'):
                report.pop(key)
        previous_hash = save(rk, report)
        previous = candidate
    held = f'{base}/candidate1/held.npz'
    (root15 / held).unlink()
    manifest['files'] = [x for x in manifest['files'] if x['relative_path'] != held]
    manifest['checkpoints'] = [x for x in manifest['checkpoints'] if not x['relative_path'].startswith(base + '/')]
    manifest_path.write_bytes(raw_json(manifest))
    report = R.run(root15, root5, plan_path, manifest_path)
    assert report['complete'] and not report['scientifically_complete']
    assert len(report['failed_cells']) == 1 and report['failed_cells'][0]['regime'] == 'scratch'
    assert report['statistics'] is None and report['dose_comparison'] is None
    assert report['validation_tail']['scratch']['flag'] is None
    assert report['verified_remote_checkpoint_hashes'] == 22


@pytest.mark.parametrize('key', ['code_sha', 'source_sha256', 'plan_sha256', 'recipe_sha256'])
def test_placeholder_run_identities_rejected_before_reading_results(fifteen_fixture, key):
    _, _, plan_path, manifest_path, _ = fifteen_fixture
    manifest, plan = json.loads(manifest_path.read_text()), json.loads(plan_path.read_text())
    manifest['run_identity'][key] = None
    with pytest.raises(ValueError, match='placeholder'):
        R.verify_identity(manifest, plan, plan_path.read_bytes())


def test_published_plan_and_source_blobs_are_checked_without_execution(fifteen_fixture):
    _, _, plan_path, manifest_path, blobs = fifteen_fixture
    manifest, plan = json.loads(manifest_path.read_text()), json.loads(plan_path.read_text())
    assert R.verify_identity(manifest, plan, plan_path.read_bytes()) == manifest['run_identity']
    # The synthetic controller contains a top-level raise; AST inspection must
    # never execute it. Changing one model blob must invalidate the source pin.
    blobs['src/model.py'] += b'changed\n'
    with pytest.raises(ValueError, match='source snapshot hash'):
        R.verify_identity(manifest, plan, plan_path.read_bytes())
    with pytest.raises(ValueError, match='published fit snapshot'):
        R.verify_identity(manifest, plan, plan_path.read_bytes() + b' ')


def test_five_raw_inputs_are_mandatory_and_five_report_cannot_be_replaced(fifteen_fixture):
    root15, root5, plan_path, manifest_path, _ = fifteen_fixture
    with pytest.raises(ValueError, match='five-epoch OOF inputs required'):
        R.run(root15, root5 / 'missing', plan_path, manifest_path)
    prior_path = root5.parent / 'prior_report.json'
    prior_path.write_bytes(prior_path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='pinned JSON hash'):
        R.run(root15, root5, plan_path, manifest_path)


def test_five_epochs_cannot_masquerade_as_fifteen(fifteen_fixture):
    root15, _, plan_path, _, _ = fifteen_fixture
    plan = json.loads(plan_path.read_text())
    c = json.loads((root15 / 'fold0/frozen/candidate0/candidate_0.json').read_text())
    c.update(history=c['history'][:5], updates=c['updates'] // 3, planned_updates=c['planned_updates'] // 3,
             best_epoch=5, val_bacc=c['history'][4]['val_bacc'])
    metrics = [{'candidate': 0, 'lr': c['lr'], **h} for h in c['history']]
    with pytest.raises(ValueError, match='fifteen full epochs'):
        R.candidate_check(c, c['provenance'], 'frozen', 0, plan, 31650, metrics)


def test_fifteen_ties_and_nonfinite_candidates_preserve_selection_rules(fifteen_fixture):
    root15, _, plan_path, _, _ = fifteen_fixture
    plan = json.loads(plan_path.read_text())
    c = json.loads((root15 / 'fold0/frozen/candidate0/candidate_0.json').read_text())
    c['history'][14]['val_bacc'] = c['history'][13]['val_bacc']
    c['val_bacc'] = c['history'][13]['val_bacc']
    metrics = [{'candidate': 0, 'lr': c['lr'], **h} for h in c['history']]
    with pytest.raises(ValueError, match='best epoch'):
        R.candidate_check(c, c['provenance'], 'frozen', 0, plan, 31650, metrics)
    c['best_epoch'] = 14
    assert R.candidate_check(c, c['provenance'], 'frozen', 0, plan, 31650, metrics)
    c.update(stopped='nonfinite', best_epoch=None, val_bacc=None, numeric_failure={'kind': 'gradient'}, history=c['history'][:2])
    c.pop('checkpoint')
    assert R.candidate_check(c, c['provenance'], 'frozen', 0, plan, 31650, metrics[:2]) is False


def test_six_cell_bootstrap_matches_explicit_subject_resampling():
    cell = SUPPORT.cell
    subjects, labels = ['u0', 'u0', 'u1', 'u2'], ['a', 'a', 'b', 'c']
    five = {r: cell(subjects, labels, ['a', 'b', 'a', 'c']) for r in R.REGIMES}
    fifteen = {r: cell(subjects, labels, labels) for r in R.REGIMES}
    output = R.dose_statistics(fifteen, five, n_boot=80, seed=19)
    rng = np.random.default_rng(19)
    differences = []
    for draw in rng.integers(3, size=(80, 3)):
        take = np.concatenate([np.flatnonzero(np.asarray(subjects) == f'u{i}') for i in draw])
        y = np.asarray(labels)[take]
        pred = five['frozen']['pred'][take]
        old = np.mean([np.mean(pred[y == label] == label) for label in np.unique(y)])
        differences.append(100 * (1 - old))
    expected = np.quantile(differences, [.025, .975])
    for value in output['contrasts'].values():
        assert np.allclose(value['ci95_pp'], expected)
    assert output['bootstrap']['draws_with_missing_classes'] > 0
    assert output['bootstrap']['discarded_draws'] == 0


@pytest.mark.parametrize('key', ['rows', 'y', 's', 'fold'])
def test_cross_dose_support_must_match_even_when_within_dose_pairing_is_valid(key):
    a = SUPPORT.cell(['u0', 'u1'], ['fist', 'noGesture'], ['fist', 'noGesture'])
    old = {r: copy.deepcopy(a) for r in R.REGIMES}
    new = {r: copy.deepcopy(a) for r in R.REGIMES}
    for r in R.REGIMES:
        new[r][key][0] = new[r][key][1] if key in ('s', 'y') else new[r][key][0] + 1
    with pytest.raises(ValueError, match='15-versus-5 paired support'):
        R.dose_statistics(new, old, n_boot=20)


def test_tail_diagnostic_is_distinct_from_five_epoch_rule_and_not_an_extension():
    def selected(best, delta=0):
        return {'stopped': 'fixed_budget', 'best_epoch': best,
                'history': [{'epoch': i, 'val_bacc': .5 + (delta if i == 15 else 0)} for i in range(1, 16)]}
    assert not R.learning_diagnostic([selected(5) for _ in range(4)])['flag']
    yes = R.learning_diagnostic([selected(15), selected(15), selected(15), selected(14)])
    assert yes['flag'] and not yes['automatic_extension']
    assert R.learning_diagnostic([selected(14, .01) for _ in range(4)])['flag']
    missing = R.learning_diagnostic([selected(15), selected(15), selected(15), None])
    assert missing['flag'] is None and not missing['evaluable']
    negative = R.learning_diagnostic([selected(14) for _ in range(4)])
    assert negative['non_trigger_does_not_establish_convergence']
