#!/usr/bin/env python3
"""Verify and describe the frozen five-epoch EPN fine-tuning experiment.

Read-only inputs; no fitting, inference, network calls or checkpoint loading.
PT identities are checked against a remote-hash manifest, not deserialized here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'consolidation-ft-report-20261105-v1'
MANIFEST_PROTOCOL = 'consolidation-ft-results-manifest-20261105-v1'
FIT_PROTOCOL = 'consolidation-finetune-epn-20261104-v1'
FIT_CODE = 'a2eeb755676a0ab780fc269f0679684eb41cd6f2'
FIT_SOURCE = 'aa857fb321bae83fd4b6463092c89dd2dba27e248cbe063f25a30032bf529b98'
FIT_PLAN = '2ce4927fa44f2d6a7442100562b8bebde1bd8d59a5172ff2f01fa4837167d7f6'
RECIPE = '2ba1b7f523385342535d89e0132c34bd97bf26f9d900677872d7aac92301b13a'
REGIMES = ('frozen', 'finetune', 'scratch')
N_BOOT, SEED = 2000, 20261102
RECIPE_KEYS = ('protocol', 'checkpoint', 'epn_roots', 'input_sha256', 'batch', 'epochs',
               'max_updates', 'seed', 'lr_grid', 'head_lr_multiplier', 'scratch_learning_rule')
SCRATCH_RULE = {
    'folds': 4, 'best_epoch5_folds_at_least': 3,
    'or_median_validation_bacc_epoch5_minus4_at_least': .005,
    'followup_epochs': 15, 'followup_regimes': ['finetune', 'scratch'],
    'followup_initialization': 'fresh from same original teacher19618/random seed0; new 15-epoch horizon, not resume from selected5epoch',
    'followup_gate': 'preregistered direction; publish measured resources and validate implementation before any extension',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode_json(raw):
    def invalid(value):
        raise ValueError(f'nonstandard JSON number: {value}')
    return json.loads(raw, parse_constant=invalid)


def number(value, lo=None, hi=None):
    return (type(value) in (int, float) and math.isfinite(value)
            and (lo is None or value >= lo) and (hi is None or value <= hi))


def close(a, b):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def relative(value):
    p = Path(value)
    require(not p.is_absolute() and p.parts and '..' not in p.parts and str(p) == value,
            f'unsafe relative path: {value}')
    return p


class Inputs:
    def __init__(self, root, manifest):
        self.root = Path(root).resolve()
        self.remote = Path(manifest['input_root_remote'])
        require(self.remote.is_absolute(), 'remote input root must be absolute')
        self.files = self.index(manifest['files'])
        self.checkpoints = self.index(manifest['checkpoints'])
        require(not (set(self.files) & set(self.checkpoints)), 'duplicate file/checkpoint entry')
        self.used, self.used_pt = set(), set()
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file()}
        require(actual == set(self.files), 'local input inventory differs from manifest')

    def index(self, entries):
        out = {}
        for entry in entries:
            key = entry['relative_path']
            p = relative(key)
            require(key not in out, f'duplicate manifest entry: {key}')
            require(entry['remote_path'] == str(self.remote / p), 'remote path differs from input layout')
            require(type(entry['bytes']) is int and entry['bytes'] > 0, 'invalid pinned byte count')
            require(type(entry['sha256']) is str and len(entry['sha256']) == 64
                    and all(c in '0123456789abcdef' for c in entry['sha256']), 'invalid SHA256')
            out[key] = entry
        return out

    def read(self, key):
        require(key in self.files, f'missing manifest input: {key}')
        path = self.root / relative(key)
        require(path.resolve().is_relative_to(self.root) and not path.is_symlink(), 'input symlink/escape')
        raw = path.read_bytes()
        entry = self.files[key]
        require(len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], f'input hash/size mismatch: {key}')
        self.used.add(key)
        return raw

    def json(self, key):
        return decode_json(self.read(key))

    def checkpoint(self, key, candidate):
        require(key in self.checkpoints, f'missing remote checkpoint identity: {key}')
        pin = self.checkpoints[key]
        require(candidate['checkpoint'] == pin['remote_path'] and candidate['checkpoint_sha256'] == pin['sha256'],
                'candidate checkpoint differs from remotely hashed file')
        self.used_pt.add(key)

    def finish(self):
        require(self.used == set(self.files) and self.used_pt == set(self.checkpoints),
                'unconsumed or unexpected manifest inputs')


def fold_roles(subjects):
    """Independently reproduce the already frozen CP.fold_roles convention."""
    subjects = sorted(subjects)
    require(len(subjects) >= 4 and len(set(subjects)) == len(subjects), 'invalid subject universe')
    seed = 20261015
    order = [subjects[i] for i in np.random.default_rng(seed).permutation(len(subjects))]
    answer = []
    for fold in range(4):
        held = sorted(order[fold::4])
        rest = [s for s in subjects if s not in held]
        nval = max(1, math.ceil(.1 * len(rest)))
        require(nval < len(rest), 'too few train subjects')
        val = sorted(np.random.default_rng([seed, fold]).choice(rest, nval, replace=False).tolist())
        answer.append({'train': [s for s in rest if s not in val], 'val': val, 'held': held})
    return answer


def historical_support(manifest):
    ref = manifest['historical_reference']
    path = ROOT / relative(ref['path'])
    raw = path.read_bytes()
    require(sha(raw) == ref['sha256'], 'historical reference hash mismatch')
    report = decode_json(raw)
    require(report['complete'] is True and report['kind'] == 'aggregate' and report['arm'] == 'T100-G'
            and report['task'] == 'p2' and report['step'] == 19618 and ref['dataset'] == 'epn612',
            'wrong historical reference')
    dataset = report['datasets']['epn612']
    folds = sorted(dataset['folds'], key=lambda x: x['fold'])
    require([f['fold'] for f in folds] == list(range(4)), 'historical fold coverage mismatch')
    roles = [{key: f[key] for key in ('train', 'val', 'held')} for f in folds]
    subjects = sorted({s for r in roles for s in r['held']})
    require(roles == fold_roles(subjects), 'historical roles differ from frozen split algorithm')
    require(manifest['expected_roles'] == roles and manifest['expected_subjects'] == subjects,
            'manifest subject roles disagree with pinned history')
    require(dataset['n_oof_subjects'] == len(subjects) == 314, 'unexpected EPN population')
    require(dataset['n_oof_windows'] == manifest['expected_n_rows'] == 47100, 'unexpected EPN row count')
    require(sum(f['n_held'] for f in folds) == dataset['n_oof_windows'], 'historical held count mismatch')
    for f in folds:
        require(f['n_train'] + f['n_val'] + f['n_held'] == dataset['n_oof_windows'], 'historical role row counts differ')
    return subjects, roles, folds


def native_strings(values, name):
    values = np.asarray(values)
    require(values.ndim == 1 and values.dtype.kind in 'SU' and len(values) > 0, f'invalid native strings: {name}')
    values = np.char.decode(values, 'utf-8') if values.dtype.kind == 'S' else values
    require(all(str(x).strip() for x in values), f'blank string: {name}')
    return values


def load_held(raw, fold, roles, classes, input_sha):
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        require(set(data.files) == {'rows', 'y', 'pred', 's', 'fold', 'input_sha256'}, 'wrong held array keys')
        require(data['input_sha256'].shape == () and str(data['input_sha256']) == input_sha, 'held input identity mismatch')
        result = {k: data[k] for k in ('rows', 'y', 'pred', 's', 'fold')}
    for key in ('y', 'pred', 's'):
        result[key] = native_strings(result[key], key)
    n = len(result['rows'])
    require(n > 0 and all(a.shape == (n,) for a in result.values()), 'held array shapes differ')
    require(result['rows'].dtype.kind in 'iu' and np.all(result['rows'] >= 0)
            and len(np.unique(result['rows'])) == n, 'invalid/duplicate held rows')
    require(result['fold'].dtype.kind in 'iu' and np.all(result['fold'] == fold), 'held fold mismatch')
    require(set(result['s']) == set(roles['held']), 'held subject support differs from roles')
    require(np.isin(result['y'], classes).all() and np.isin(result['pred'], classes).all(), 'held labels outside fit vocabulary')
    order = np.argsort(result['rows'], kind='stable')
    return {k: v[order] for k, v in result.items()}


def score(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def pair(cells):
    require(set(cells) == set(REGIMES), 'incomplete regime support')
    reference = cells['frozen']
    for regime in REGIMES:
        for key in ('rows', 'y', 's', 'fold'):
            require(np.array_equal(cells[regime][key], reference[key]), f'OOF pairing differs: {regime}/{key}')


def candidate_check(candidate, provenance, regime, index, plan, n_train, metrics):
    require(candidate['complete'] is True and candidate['convergence_claim'] is False
            and candidate['provenance'] == provenance, 'candidate provenance/completion mismatch')
    require(candidate['candidate'] == index and candidate['lr'] == plan['lr_grid'][regime][index]
            and candidate['head_lr'] == candidate['lr'] * plan['head_lr_multiplier'][regime], 'candidate LR identity mismatch')
    history = candidate['history']
    require(metrics == [{'candidate': index, 'lr': candidate['lr'], **row} for row in history], 'metrics differ from persisted history')
    require([r['epoch'] for r in history] == list(range(1, len(history) + 1)) and len(history) <= 5,
            'duplicate/incomplete epoch history')
    per_epoch = math.ceil(n_train / plan['batch'])
    for row in history:
        require(row['step'] == row['epoch'] * per_epoch, 'epoch update count differs from frozen dose')
        require(number(row['loss'], 0) and number(row['last_gradient_norm'], 0)
                and number(row['val_bacc'], 0, 1), 'nonfinite or invalid candidate history')
    if candidate['stopped'] == 'nonfinite':
        require(candidate['val_bacc'] is None and candidate['best_epoch'] is None and 'numeric_failure' in candidate
                and 'checkpoint' not in candidate, 'invalid divergence outcome')
        return False
    require(candidate['stopped'] == 'fixed_budget' and len(history) == 5, 'candidate did not finish five epochs')
    require(candidate['updates'] == candidate['planned_updates'] == 5 * per_epoch, 'candidate dose mismatch')
    best = max(history, key=lambda row: row['val_bacc'])  # first epoch wins ties
    require(close(candidate['val_bacc'], best['val_bacc']) and candidate['best_epoch'] == best['epoch'],
            'candidate best epoch/score differs from validation history')
    return True


def select(candidates):
    finite = [c for c in candidates if c['stopped'] == 'fixed_budget']
    return min(finite, key=lambda c: (-c['val_bacc'], c['candidate'])) if finite else None


def scratch_rule(selected):
    if len(selected) != 4 or any(c is None or c['stopped'] != 'fixed_budget' or len(c['history']) != 5 for c in selected):
        return {'evaluable': False, 'flag': None, 'reason': 'four complete selected scratch histories required'}
    at_five = sum(c['best_epoch'] == 5 for c in selected)
    deltas = [c['history'][4]['val_bacc'] - c['history'][3]['val_bacc'] for c in selected]
    median = float(np.median(deltas))
    return {'evaluable': True, 'flag': at_five >= 3 or median >= .005,
            'folds_best_at_epoch5': at_five, 'median_val_bacc_epoch5_minus4': median,
            'fold_val_bacc_epoch5_minus4': deltas, 'rule': SCRATCH_RULE,
            'non_trigger_does_not_establish_convergence': True, 'automatic_extension': False}


def statistics(cells, n_boot=N_BOOT, seed=SEED):
    """Pooled macro recall, paired subject-cluster draws; no discarded draws."""
    pair(cells)
    ref = cells['frozen']
    subjects, si = np.unique(ref['s'], return_inverse=True)
    classes, ci = np.unique(ref['y'], return_inverse=True)
    ns, nc = len(subjects), len(classes)
    require(type(n_boot) is int and n_boot >= 2 and ns >= 2 and nc >= 2, 'insufficient bootstrap support')
    flat = si * nc + ci
    den = np.bincount(flat, minlength=ns * nc).reshape(ns, nc)
    hit = np.asarray([np.bincount(flat[cells[r]['pred'] == ref['y']], minlength=ns * nc).reshape(ns, nc) for r in REGIMES])
    points = (hit.sum(1) / den.sum(0)).mean(1)
    draws = np.empty((n_boot, len(REGIMES)))
    rng = np.random.default_rng(seed)
    missing = single = 0
    for start in range(0, n_boot, 128):
        size = min(128, n_boot - start)
        pick = rng.integers(ns, size=(size, ns))
        multiplicity = np.zeros((size, ns), dtype=np.int64)
        np.add.at(multiplicity, (np.arange(size)[:, None], pick), 1)
        totals = multiplicity @ den
        correct = (multiplicity @ hit.transpose(1, 0, 2).reshape(ns, -1)).reshape(size, len(REGIMES), nc)
        present = totals > 0
        recalls = np.divide(correct, totals[:, None, :], out=np.zeros(correct.shape, dtype=float), where=present[:, None, :])
        draws[start:start + size] = recalls.sum(2) / present.sum(1)[:, None]
        missing += int(np.sum(present.sum(1) < nc))
        single += int(np.sum(present.sum(1) == 1))
    contrasts = {}
    for name, left, right in (('finetune_minus_frozen', 1, 0), ('finetune_minus_scratch', 1, 2)):
        delta = 100 * (draws[:, left] - draws[:, right])
        contrasts[name] = {'delta_pp': float(100 * (points[left] - points[right])),
                           'ci95_pp': np.quantile(delta, [.025, .975]).tolist(),
                           'paired_subject_se_pp': float(np.std(delta, ddof=1))}
    per_subject = []
    for i, subject in enumerate(subjects):
        present = den[i] > 0
        values = {r: float(np.mean(hit[j, i, present] / den[i, present])) for j, r in enumerate(REGIMES)}
        per_subject.append({'subject': str(subject), 'n': int(den[i].sum()), 'class_counts': den[i].tolist(),
                            'correct_by_class': {r: hit[j, i].tolist() for j, r in enumerate(REGIMES)},
                            'balanced_accuracy': values,
                            'finetune_minus_frozen_pp': 100 * (values['finetune'] - values['frozen']),
                            'finetune_minus_scratch_pp': 100 * (values['finetune'] - values['scratch'])})
    return {'n_rows': len(ref['y']), 'n_subjects': ns, 'classes': classes.tolist(),
            'cells': {r: {'balanced_accuracy': float(points[j]), 'balanced_accuracy_percent': float(100 * points[j]),
                          'ci95_percent': (100 * np.quantile(draws[:, j], [.025, .975])).tolist(),
                          'recall_by_class': (hit[j].sum(0) / den.sum(0)).tolist()} for j, r in enumerate(REGIMES)},
            'contrasts': contrasts, 'per_subject': per_subject,
            'bootstrap': {'replicates': n_boot, 'seed': seed, 'unit': 'subject', 'same_draw_all_regimes': True,
                          'draws_with_missing_classes': missing, 'single_class_draws': single, 'discarded_draws': 0,
                          'estimand': 'pooled macro recall; present classes on each shared draw',
                          'interval': 'descriptive percentile 95%; conditional on fitted models/folds, no refits or seed uncertainty'}}


def run(input_root, plan_path, manifest_path):
    manifest_raw, plan_raw = Path(manifest_path).read_bytes(), Path(plan_path).read_bytes()
    manifest, plan = decode_json(manifest_raw), decode_json(plan_raw)
    require(manifest['protocol'] == MANIFEST_PROTOCOL and manifest['code_sha'] == FIT_CODE
            and manifest['source_sha256'] == FIT_SOURCE and manifest['plan_sha256'] == FIT_PLAN, 'unreviewed fit identity')
    require(plan['protocol'] == FIT_PROTOCOL and plan['status'] == plan['fits_status'] == 'claude_review_converged', 'unreviewed fit plan')
    require(sha(json.dumps({k: plan[k] for k in RECIPE_KEYS}, sort_keys=True).encode()) == plan['recipe_sha256'] == RECIPE,
            'scientific recipe differs from frozen fit')
    require(plan['scratch_learning_rule'] == SCRATCH_RULE and plan['epochs'] == 5 and plan['batch'] == 8
            and plan['max_updates'] is None and plan['seed'] == 0, 'frozen dose/rule changed')
    inputs = Inputs(input_root, manifest)
    require(str(inputs.remote) == plan['fit_output_root'], 'remote output root differs from frozen layout')
    subjects, roles, historical_folds = historical_support(manifest)
    curves, choices, failed, arrays, selected_scratch = [], [], [], {r: [] for r in REGIMES}, []
    for regime in REGIMES:
        for fold in range(4):
            base = f'fold{fold}/{regime}'
            reports, candidates = [], []
            for index in (0, 1):
                directory = f'{base}/candidate{index}'
                report_key, candidate_key = f'{directory}/report.json', f'{directory}/candidate_{index}.json'
                report, candidate = inputs.json(report_key), inputs.json(candidate_key)
                expected = {'protocol': FIT_PROTOCOL, 'mode': 'fit', 'code_sha': FIT_CODE, 'plan_sha256': FIT_PLAN,
                            'recipe_sha256': RECIPE, 'source_sha256': FIT_SOURCE,
                            'checkpoint_sha256': plan['checkpoint']['sha256'], 'input_sha256': plan['input_sha256'],
                            'fold': fold, 'regime': regime}
                require(all(report.get(k) == v for k, v in expected.items()) and report['complete'] is True
                        and report['convergence_claim'] is False, f'fit report identity differs: {directory}')
                for key in ('fit_allocation_by_regime', 'fit_packaging_policy'):
                    require(report[key] == plan[key], 'resource policy differs from approved plan')
                require(report['fit_limit_seconds_by_regime'] == {r: plan['fit_allocation_by_regime'][r]['seconds'] - 300 for r in REGIMES}, 'fit limits differ')
                require(number(report['elapsed_seconds'], 0), 'invalid elapsed time')
                require(report['roles'] == roles[fold], 'fit roles differ from pinned historical roles')
                provenance = report['candidate_provenance']
                expected_prov = {k: expected[k] for k in ('protocol', 'recipe_sha256', 'source_sha256', 'checkpoint_sha256', 'input_sha256', 'fold', 'regime')}
                expected_prov.update(roles=roles[fold], classes=provenance['classes'])
                require(provenance == expected_prov, 'candidate provenance fields differ')
                classes = native_strings(np.asarray(provenance['classes']), 'fit classes')
                require(classes.tolist() == sorted(set(classes.tolist())) and len(classes) >= 2, 'fit vocabulary not unique/sorted')
                require(report['candidate_file_sha256'] == {str(index): inputs.files[candidate_key]['sha256']}, 'candidate JSON pin mismatch')
                metric_key = f'{directory}/metrics.jsonl'
                metrics = [decode_json(line) for line in inputs.read(metric_key).splitlines() if line] if metric_key in inputs.files else []
                finite = candidate_check(candidate, provenance, regime, index, plan, historical_folds[fold]['n_train'], metrics)
                if finite:
                    inputs.checkpoint(f'{directory}/candidate_{index}.pt', candidate)
                require(report['candidates'][-1] == candidate and len(report['candidates']) == index + 1, 'embedded candidate differs from persisted result')
                curves.append({'fold': fold, 'regime': regime, **candidate})
                reports.append(report)
                candidates.append(candidate)
            c0, c1 = reports
            require(c0.get('candidate_only') == 0 and c0['scientific_status'] == 'first_candidate_recorded'
                    and c0['held_evaluated'] is False, 'candidate0 evaluated held prematurely')
            c0pin = {'path': str(inputs.remote / f'{base}/candidate0/report.json'),
                     'sha256': inputs.files[f'{base}/candidate0/report.json']['sha256']}
            require(c1['candidates'][0] == {**candidates[0], 'source_receipt': c0pin}, 'split candidate0/source receipt mismatch')
            require(candidates[0]['provenance'] == candidates[1]['provenance'], 'split provenance mismatch')
            chosen = select(candidates)
            if regime == 'scratch':
                selected_scratch.append(chosen)
            choice = {'fold': fold, 'regime': regime, 'status': c1['scientific_status'],
                      'selected_candidate': chosen['candidate'] if chosen else None,
                      'selected_lr': chosen['lr'] if chosen else None,
                      'selected_best_epoch': chosen['best_epoch'] if chosen else None}
            choices.append(choice)
            if chosen is None or c1['scientific_status'] == 'selected_model_nonfinite_on_held':
                require(c1['held_evaluated'] is False and 'held_sha256' not in c1, 'invalid failed held outcome')
                if chosen is None:
                    require(c1['scientific_status'] == 'all_candidates_diverged' and c1['selected_lr'] is None, 'invalid all-diverged outcome')
                else:
                    require(c1['selected_lr'] == chosen['lr'] and 'numeric_failure' in c1, 'held failure changed selected LR')
                failed.append(choice)
                continue
            require(c1['scientific_status'] == 'held_evaluated' and c1['held_evaluated'] is True
                    and all(c1[k] == choice[k] for k in ('selected_candidate', 'selected_lr', 'selected_best_epoch')),
                    'held candidate differs from validation selection')
            held_key = f'{base}/candidate1/held.npz'
            require(inputs.files[held_key]['sha256'] == c1['held_sha256'], 'held receipt hash mismatch')
            held = load_held(inputs.read(held_key), fold, roles[fold], classes, plan['input_sha256'])
            require(len(held['rows']) == c1['n_held'] == historical_folds[fold]['n_held'], 'held sample count mismatch')
            require(close(score(held['y'], held['pred']), c1['held_bacc']), 'held score does not reproduce')
            choice['held_bacc'] = c1['held_bacc']
            arrays[regime].append(held)
    inputs.finish()
    result = {'protocol': PROTOCOL, 'complete': True, 'scientifically_complete': not failed,
              'manifest_sha256': sha(manifest_raw), 'analysis_plan_sha256': sha(plan_raw),
              'fit_code_sha': FIT_CODE, 'fit_plan_sha256': FIT_PLAN, 'source_sha256': FIT_SOURCE,
              'recipe_sha256': RECIPE, 'historical_reference': manifest['historical_reference'],
              'verified_files': len(inputs.files), 'verified_remote_checkpoint_hashes': len(inputs.checkpoints),
              'checkpoint_payloads_loaded': False, 'choices': choices, 'candidate_curves': curves,
              'failed_cells': failed, 'scratch_rule': scratch_rule(selected_scratch), 'statistics': None,
              'scope': 'EPN development; five supervised epochs; one seed; calibrated new subjects; no official test evaluation',
              'limitations': ['No regression, dose, depth or general-FM conclusion.',
                              'Historical logistic score is a separate reader, not this frozen-head baseline.',
                              'No V-R1 P1/P2 or pose adoption gate is applied to EPN fine-tuning.',
                              'F-R2 does not launch an extension; a negative flag does not establish scratch convergence.']}
    if not failed:
        cells = {}
        for regime in REGIMES:
            cell = {key: np.concatenate([a[key] for a in arrays[regime]]) for key in arrays[regime][0]}
            order = np.argsort(cell['rows'], kind='stable')
            cell = {key: value[order] for key, value in cell.items()}
            require(np.array_equal(cell['rows'], np.arange(manifest['expected_n_rows'])), 'OOF rows duplicated or missing')
            require(sorted(set(cell['s'])) == subjects, 'OOF subject universe differs from pinned history')
            for fold in range(4):
                for role in ('train', 'val', 'held'):
                    require(int(np.isin(cell['s'], roles[fold][role]).sum()) == historical_folds[fold][f'n_{role}'], 'OOF role sample counts differ')
                train_classes = sorted(set(cell['y'][np.isin(cell['s'], roles[fold]['train'])]))
                require(train_classes == curves[REGIMES.index(regime) * 8 + fold * 2]['provenance']['classes'], 'fit vocabulary not reproducible from train targets')
            cells[regime] = cell
        pair(cells)
        result['statistics'] = statistics(cells)
        result['paired_support_sha256'] = sha(b''.join(np.ascontiguousarray(cells['frozen'][k]).tobytes() for k in ('rows', 'y', 's', 'fold')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and not args.out.is_relative_to(args.input_root), 'new output outside input directory required')
    result = run(args.input_root, args.plan, args.manifest)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(encoded)


if __name__ == '__main__':
    main()
