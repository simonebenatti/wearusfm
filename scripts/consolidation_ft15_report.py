#!/usr/bin/env python3
"""Read-only report for fresh 15-epoch EPN fits and paired 15-minus-5 results.

The five-epoch analysis stays immutable. No fit, inference, SSH or submission.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import consolidation_ft_report as FIVE

require, sha, decode_json = FIVE.require, FIVE.sha, FIVE.decode_json
REGIMES, N_BOOT, SEED = FIVE.REGIMES, FIVE.N_BOOT, FIVE.SEED
PROTOCOL = 'consolidation-ft15-report-20261105-v1'
MANIFEST_PROTOCOL = 'consolidation-ft15-results-manifest-20261105-v1'
FIT_PROTOCOL = 'consolidation-finetune15-epn-20261105-v1'
PRIOR_REPORT_SHA = '3c4b3601c12836ba7e13a7012458810ca450cfa373a83ba880c4fa36d4d7b87a'
PLAN_PATH = 'results/infra/finetune15_20261105/plan.json'
RECIPE = '975fa69d633a8504a2589f720f7f5b3f19f2e4ffdd66a0f68500c1256376e56b'
RECIPE_KEYS = ('protocol', 'checkpoint', 'epn_roots', 'input_sha256', 'batch', 'epochs',
               'max_updates', 'seed', 'lr_grid', 'head_lr_multiplier', 'reading_rule15')
READING_RULE15 = {
    'epochs': 15, 'primary_contrasts': ['finetune_minus_scratch', 'finetune_minus_frozen'],
    'secondary_contrasts': ['frozen15_minus_frozen5', 'finetune15_minus_finetune5', 'scratch15_minus_scratch5'],
    'bootstrap': {'replicates': 2000, 'seed': 20261102, 'unit': 'subject', 'paired': True},
    'validation_tail': {'regimes': ['frozen', 'finetune', 'scratch'], 'folds': 4,
        'best_epoch15_folds_at_least': 3, 'or_median_validation_bacc_epoch15_minus14_at_least': .005,
        'missing_or_diverged': 'not_evaluable', 'false_does_not_establish_convergence': True, 'automatic_extension': False},
    'interpretation': 'development; fixed15epoch recipe including new cosine horizon; not isolated dose effect; historical logistic reference descriptive only; no V-R1 adoption gate',
}


def valid_hash(value, length=64):
    return (isinstance(value, str) and len(value) == length
            and all(c in '0123456789abcdef' for c in value) and len(set(value)) > 1)


def pinned_json(pin):
    require(valid_hash(pin['sha256']), 'missing or placeholder pin')
    path = ROOT / FIVE.relative(pin['path'])
    raw = path.read_bytes()
    require(sha(raw) == pin['sha256'], f'pinned JSON hash mismatch: {pin["path"]}')
    return decode_json(raw), raw


def candidate_check(candidate, provenance, regime, index, plan, n_train, metrics):
    """The same validation selection, with the new fixed fifteen-epoch dose."""
    require(candidate['complete'] is True and candidate['convergence_claim'] is False
            and candidate['provenance'] == provenance, 'candidate provenance/completion mismatch')
    require(candidate['candidate'] == index and candidate['lr'] == plan['lr_grid'][regime][index]
            and candidate['head_lr'] == candidate['lr'] * plan['head_lr_multiplier'][regime], 'candidate LR mismatch')
    history = candidate['history']
    require(metrics == [{'candidate': index, 'lr': candidate['lr'], **h} for h in history], 'metrics/history mismatch')
    require([h['epoch'] for h in history] == list(range(1, len(history) + 1)) and len(history) <= 15,
            'duplicate or incomplete epoch history')
    per_epoch = math.ceil(n_train / plan['batch'])
    for h in history:
        require(h['step'] == h['epoch'] * per_epoch, 'epoch update count differs from fifteen-epoch dose')
        require(FIVE.number(h['loss'], 0) and FIVE.number(h['last_gradient_norm'], 0)
                and FIVE.number(h['val_bacc'], 0, 1), 'invalid candidate history')
    if candidate['stopped'] == 'nonfinite':
        require(candidate['val_bacc'] is None and candidate['best_epoch'] is None
                and 'numeric_failure' in candidate and 'checkpoint' not in candidate, 'invalid numerical outcome')
        return False
    require(candidate['stopped'] == 'fixed_budget' and len(history) == 15, 'fifteen full epochs required')
    require(candidate['updates'] == candidate['planned_updates'] == 15 * per_epoch, 'fifteen-epoch update budget mismatch')
    best = max(history, key=lambda h: h['val_bacc'])
    require(candidate['best_epoch'] == best['epoch'] and FIVE.close(candidate['val_bacc'], best['val_bacc']),
            'best epoch/score differs from validation history')
    return True


def learning_diagnostic(selected):
    """Validation-only descriptive flag, separately for all three regimes."""
    if (len(selected) != 4 or any(c is None or c.get('stopped') != 'fixed_budget'
            or [h['epoch'] for h in c.get('history', [])] != list(range(1, 16))
            or c.get('best_epoch') not in range(1, 16) for c in selected)):
        return {'evaluable': False, 'flag': None, 'reason': 'four complete selected fifteen-epoch histories required',
                'automatic_extension': False, 'non_trigger_does_not_establish_convergence': True}
    deltas = [c['history'][14]['val_bacc'] - c['history'][13]['val_bacc'] for c in selected]
    require(all(FIVE.number(x) for x in deltas), 'nonfinite terminal validation delta')
    best_last = sum(c['best_epoch'] == 15 for c in selected)
    median = float(np.median(deltas))
    return {'evaluable': True, 'flag': best_last >= 3 or median >= .005,
            'folds_best_at_epoch15': best_last, 'median_val_bacc_epoch15_minus14': median,
            'fold_val_bacc_epoch15_minus14': deltas, 'automatic_extension': False,
            'non_trigger_does_not_establish_convergence': True}


def dose_statistics(fifteen, five, n_boot=N_BOOT, seed=SEED):
    """Secondary contrasts; the same subject draw is used for all six cells."""
    FIVE.pair(fifteen)
    FIVE.pair(five)
    ref = fifteen['frozen']
    for r in REGIMES:
        for k in ('rows', 'y', 's', 'fold'):
            require(np.array_equal(fifteen[r][k], five[r][k]), f'15-versus-5 paired support mismatch: {r}/{k}')
    subjects, si = np.unique(ref['s'], return_inverse=True)
    classes, ci = np.unique(ref['y'], return_inverse=True)
    ns, nc = len(subjects), len(classes)
    require(type(n_boot) is int and n_boot >= 2 and ns >= 2 and nc >= 2, 'insufficient paired bootstrap support')
    flat = si * nc + ci
    den = np.bincount(flat, minlength=ns * nc).reshape(ns, nc)
    names = [f'{dose}/{r}' for dose in (15, 5) for r in REGIMES]
    cells = [table[r] for table in (fifteen, five) for r in REGIMES]
    hit = np.asarray([np.bincount(flat[c['pred'] == ref['y']], minlength=ns * nc).reshape(ns, nc) for c in cells])
    points = (hit.sum(1) / den.sum(0)).mean(1)
    draws = np.empty((n_boot, 6))
    rng = np.random.default_rng(seed)
    missing = single = 0
    for start in range(0, n_boot, 128):
        size = min(128, n_boot - start)
        picked = rng.integers(ns, size=(size, ns))
        multiplicity = np.zeros((size, ns), dtype=np.int64)
        np.add.at(multiplicity, (np.arange(size)[:, None], picked), 1)
        totals = multiplicity @ den
        correct = (multiplicity @ hit.transpose(1, 0, 2).reshape(ns, -1)).reshape(size, 6, nc)
        present = totals > 0
        recalls = np.divide(correct, totals[:, None, :], out=np.zeros(correct.shape, dtype=float), where=present[:, None, :])
        draws[start:start + size] = recalls.sum(2) / present.sum(1)[:, None]
        missing += int(np.sum(present.sum(1) < nc))
        single += int(np.sum(present.sum(1) == 1))
    contrasts = {}
    for i, r in enumerate(REGIMES):
        delta = 100 * (draws[:, i] - draws[:, i + 3])
        contrasts[f'{r}15_minus_{r}5'] = {'delta_pp': float(100 * (points[i] - points[i + 3])),
                                      'ci95_pp': np.quantile(delta, [.025, .975]).tolist(),
                                      'paired_subject_se_pp': float(np.std(delta, ddof=1))}
    return {'secondary': True, 'interpretation': 'fresh fifteen-epoch cosine versus fresh five-epoch cosine; not an isolated dose effect',
            'cells': {name: {'balanced_accuracy': float(points[i])} for i, name in enumerate(names)},
            'contrasts': contrasts, 'n_rows': len(ref['y']), 'n_subjects': ns,
            'bootstrap': {'replicates': n_boot, 'seed': seed, 'same_draw_all_six_cells': True,
                          'unit': 'subject', 'draws_with_missing_classes': missing, 'single_class_draws': single,
                          'discarded_draws': 0, 'estimand': 'pooled macro recall; common present classes on each draw',
                          'interval': 'descriptive paired percentile 95%; fitted models/folds held fixed'}}


def git_bytes(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.PIPE)


def verify_identity(manifest, plan, plan_raw):
    """Verify actual published blobs; never execute/import historical source."""
    require(manifest['protocol'] == MANIFEST_PROTOCOL, 'wrong results manifest protocol')
    identity = manifest['run_identity']
    require(set(identity) == {'code_sha', 'source_sha256', 'plan_sha256', 'recipe_sha256'}, 'incomplete run identity')
    require(valid_hash(identity['code_sha'], 40) and all(valid_hash(identity[k]) for k in identity if k != 'code_sha'),
            'run identity has missing or placeholder SHA')
    code = identity['code_sha']
    require(git_bytes('rev-parse', f'{code}^{{commit}}').decode().strip() == code, 'run code is not an actual commit')
    frozen_raw = git_bytes('show', f'{code}:{PLAN_PATH}')
    require(sha(frozen_raw) == sha(plan_raw) == identity['plan_sha256'], 'plan differs from published fit snapshot')
    require(decode_json(frozen_raw) == plan, 'published fit plan differs')
    require(plan['protocol'] == FIT_PROTOCOL and plan['status'] == plan['fits_status'] == 'claude_review_converged',
            'fifteen-epoch plan has not converged')
    require(valid_hash(plan['review']['commit'], 40), 'plan has no concrete review commit')
    recipe = sha(json.dumps({k: plan[k] for k in RECIPE_KEYS}, sort_keys=True).encode())
    require(recipe == identity['recipe_sha256'] == plan['recipe_sha256'] == RECIPE, 'fifteen-epoch recipe mismatch')
    require(plan['reading_rule15'] == READING_RULE15 and plan['epochs'] == 15 and plan['batch'] == 8
            and plan['seed'] == 0 and plan['max_updates'] is None, 'unreviewed dose or reading rule')
    require(plan['fit_packaging_policy']['mode'] == 'split_only', 'only reviewed split packaging accepted')
    controller = 'scripts/consolidation_finetune15.py'
    tree = ast.parse(git_bytes('show', f'{code}:{controller}').decode())
    assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'SOURCE_EXTRA_PATHS' for t in node.targets)]
    require(len(assignments) == 1, 'source snapshot does not declare one literal extra-path list')
    extras = ast.literal_eval(assignments[0])
    require(isinstance(extras, (tuple, list)) and all(isinstance(x, str) for x in extras), 'nonliteral source inventory')
    required_extras = (controller, 'scripts/consolidation_finetune.py', 'scripts/calibration_probe.py',
                       'scripts/tokenizer_train.py', 'scripts/tokenizer_preflight.py',
                       'scripts/slurm/consolidation_finetune15.sbatch', 'scripts/slurm/lib_code_dir.sh')
    require(tuple(extras) == required_extras, 'source inventory differs from reviewed controller')
    source_files = sorted(x for x in git_bytes('ls-tree', '-r', '--name-only', code, 'src').decode().splitlines() if x.endswith('.py'))
    require(source_files, 'empty scientific source snapshot')
    paths = source_files + list(extras)
    require(len(set(paths)) == len(paths), 'duplicate scientific source path')
    fingerprints = {str(FIVE.relative(p)): sha(git_bytes('show', f'{code}:{p}')) for p in paths}
    require(sha(json.dumps(fingerprints, sort_keys=True).encode()) == identity['source_sha256'], 'source snapshot hash mismatch')
    return identity


def combine(arrays, expected_n_rows, subjects, roles, historical_folds, classes_by_fold):
    cells = {}
    for regime in REGIMES:
        require(len(arrays[regime]) == 4, 'four held folds required; no failed-fold pruning')
        cell = {k: np.concatenate([a[k] for a in arrays[regime]]) for k in arrays[regime][0]}
        order = np.argsort(cell['rows'], kind='stable')
        cell = {k: v[order] for k, v in cell.items()}
        require(np.array_equal(cell['rows'], np.arange(expected_n_rows)), 'OOF rows duplicated or missing')
        require(sorted(set(cell['s'])) == subjects, 'OOF subject population differs')
        for fold in range(4):
            for role in ('train', 'val', 'held'):
                require(int(np.isin(cell['s'], roles[fold][role]).sum()) == historical_folds[fold][f'n_{role}'], 'role row count differs')
            require(sorted(set(cell['y'][np.isin(cell['s'], roles[fold]['train'])])) == classes_by_fold[regime, fold],
                    'train vocabulary does not reproduce from OOF targets')
        cells[regime] = cell
    FIVE.pair(cells)
    return cells


def verify_fifteen(input_root, plan, manifest, identity):
    inputs = FIVE.Inputs(input_root, manifest)
    require(str(inputs.remote) == plan['fit_output_root'], 'fifteen-epoch output layout mismatch')
    subjects, roles, historical = FIVE.historical_support(manifest)
    curves, choices, failed = [], [], []
    selected = {r: [] for r in REGIMES}
    arrays, vocabularies = {r: [] for r in REGIMES}, {}
    for regime in REGIMES:
        for fold in range(4):
            base = f'fold{fold}/{regime}'
            reports, candidates = [], []
            for index in (0, 1):
                directory = f'{base}/candidate{index}'
                rk, ck = f'{directory}/report.json', f'{directory}/candidate_{index}.json'
                report, candidate = inputs.json(rk), inputs.json(ck)
                expected = {'protocol': FIT_PROTOCOL, 'mode': 'fit', **identity,
                            'checkpoint_sha256': plan['checkpoint']['sha256'], 'input_sha256': plan['input_sha256'],
                            'fold': fold, 'regime': regime}
                require(all(report.get(k) == v for k, v in expected.items()) and report['complete'] is True
                        and report['convergence_claim'] is False, f'fit report provenance mismatch: {directory}')
                for key in ('fit_allocation_by_regime', 'fit_packaging_policy', 'resource_projection_policy',
                            'reference_five', 'reference_source_sha256'):
                    require(report[key] == plan[key], 'fit resource policy mismatch')
                require(report['initialization'] == 'fresh from teacher19618 or random seed0; new 15-epoch cosine; no resume from five epochs',
                        'wrong fifteen-epoch initialization declaration')
                limits = {r: plan['fit_allocation_by_regime'][r]['seconds'] - plan['fit_packaging_policy']['seconds_reserved_per_job'] for r in REGIMES}
                require(report['fit_limit_seconds_by_regime'] == limits, 'fit resource limits mismatch')
                require(FIVE.number(report['elapsed_seconds'], 0), 'invalid elapsed time')
                require(report['roles'] == roles[fold], 'fit roles differ from pinned historical support')
                provenance = report['candidate_provenance']
                expected_prov = {k: expected[k] for k in ('protocol', 'recipe_sha256', 'source_sha256', 'checkpoint_sha256', 'input_sha256', 'fold', 'regime')}
                expected_prov.update(roles=roles[fold], classes=provenance['classes'])
                require(provenance == expected_prov, 'candidate provenance fields differ')
                classes = FIVE.native_strings(np.asarray(provenance['classes']), 'fit vocabulary')
                require(classes.tolist() == sorted(set(classes.tolist())) and len(classes) >= 2, 'fit vocabulary invalid')
                vocabularies[regime, fold] = classes.tolist()
                require(report['candidate_file_sha256'] == {str(index): inputs.files[ck]['sha256']}, 'candidate JSON pin mismatch')
                mk = f'{directory}/metrics.jsonl'
                metrics = [decode_json(x) for x in inputs.read(mk).splitlines() if x] if mk in inputs.files else []
                finite = candidate_check(candidate, provenance, regime, index, plan, historical[fold]['n_train'], metrics)
                if finite:
                    inputs.checkpoint(f'{directory}/candidate_{index}.pt', candidate)
                require(report['candidates'][-1] == candidate and len(report['candidates']) == index + 1, 'embedded candidate mismatch')
                curves.append({'fold': fold, 'regime': regime, **candidate})
                reports.append(report)
                candidates.append(candidate)
            c0, c1 = reports
            require(c0.get('candidate_only') == 0 and c0['scientific_status'] == 'first_candidate_recorded'
                    and c0['held_evaluated'] is False, 'candidate0 evaluated held prematurely')
            c0pin = {'path': str(inputs.remote / f'{base}/candidate0/report.json'),
                     'sha256': inputs.files[f'{base}/candidate0/report.json']['sha256']}
            require(c1['candidates'][0] == {**candidates[0], 'source_receipt': c0pin}, 'split source receipt mismatch')
            require(candidates[0]['provenance'] == candidates[1]['provenance'], 'split candidate provenance differs')
            chosen = FIVE.select(candidates)
            selected[regime].append(chosen)
            choice = {'fold': fold, 'regime': regime, 'status': c1['scientific_status'],
                      'selected_candidate': chosen['candidate'] if chosen else None,
                      'selected_lr': chosen['lr'] if chosen else None, 'selected_best_epoch': chosen['best_epoch'] if chosen else None}
            choices.append(choice)
            if chosen is None or c1['scientific_status'] == 'selected_model_nonfinite_on_held':
                require(c1['held_evaluated'] is False and 'held_sha256' not in c1, 'invalid failed held outcome')
                if chosen is None:
                    require(c1['scientific_status'] == 'all_candidates_diverged' and c1['selected_lr'] is None, 'invalid all-diverged outcome')
                else:
                    require(c1['selected_lr'] == chosen['lr'] and 'numeric_failure' in c1, 'held failure changed selected LR')
                # A nonfinite selected model also makes its regime's terminal
                # diagnostic nonassessable; it never authorizes a fallback LR.
                selected[regime][-1] = None
                failed.append(choice)
                continue
            require(c1['scientific_status'] == 'held_evaluated' and c1['held_evaluated'] is True
                    and all(c1[k] == choice[k] for k in ('selected_candidate', 'selected_lr', 'selected_best_epoch')),
                    'held selection differs from validation-only choice')
            hk = f'{base}/candidate1/held.npz'
            require(inputs.files[hk]['sha256'] == c1['held_sha256'], 'held pin mismatch')
            held = FIVE.load_held(inputs.read(hk), fold, roles[fold], classes, plan['input_sha256'])
            require(len(held['rows']) == c1['n_held'] == historical[fold]['n_held'], 'held row count mismatch')
            require(FIVE.close(FIVE.score(held['y'], held['pred']), c1['held_bacc']), 'held score does not reproduce')
            choice['held_bacc'] = c1['held_bacc']
            arrays[regime].append(held)
    inputs.finish()
    cells = None if failed else combine(arrays, manifest['expected_n_rows'], subjects, roles, historical, vocabularies)
    audit = {'verified_files': len(inputs.files), 'verified_remote_checkpoint_hashes': len(inputs.checkpoints),
             'checkpoint_payloads_loaded': False, 'choices': choices, 'candidate_curves': curves, 'failed_cells': failed,
             'validation_tail': {r: learning_diagnostic(selected[r]) for r in REGIMES}}
    return audit, cells


def verify_prior(input_root, plan):
    require(input_root is not None and Path(input_root).is_dir(), 'five-epoch OOF inputs required; recover the pinned WORK results')
    pins = plan['prior_five_results']
    require(pins['report']['sha256'] == PRIOR_REPORT_SHA, 'prior five-epoch report pin changed')
    published, _ = pinned_json(pins['report'])
    manifest, _ = pinned_json(pins['manifest'])
    prior_plan, _ = pinned_json(plan['reference_five']['plan'])
    require(published['complete'] is True and published['scientifically_complete'] is True
            and published['manifest_sha256'] == pins['manifest']['sha256'], 'prior report is not the complete pinned analysis')
    fresh = FIVE.run(input_root, ROOT / plan['reference_five']['plan']['path'], ROOT / pins['manifest']['path'])
    require(fresh['statistics'] == published['statistics'] and fresh['choices'] == published['choices']
            and fresh['scratch_rule'] == published['scratch_rule'] and fresh['paired_support_sha256'] == published['paired_support_sha256'],
            'raw five-epoch data do not reproduce published results')
    inputs = FIVE.Inputs(input_root, manifest)
    subjects, roles, historical = FIVE.historical_support(manifest)
    arrays, vocabularies = {r: [] for r in REGIMES}, {}
    # FIVE.run has already audited every file, receipt, checkpoint identity,
    # candidate and fold. Read only the pinned final OOF to return the six-cell
    # pairing inputs; no second checkpoint selection or model evaluation.
    for r in REGIMES:
        for fold in range(4):
            directory = f'fold{fold}/{r}/candidate1'
            report = inputs.json(f'{directory}/report.json')
            classes = report['candidate_provenance']['classes']
            vocabularies[r, fold] = classes
            arrays[r].append(FIVE.load_held(inputs.read(f'{directory}/held.npz'), fold, roles[fold], classes, prior_plan['input_sha256']))
    cells = combine(arrays, manifest['expected_n_rows'], subjects, roles, historical, vocabularies)
    return published, cells


def run(input_root, input_root_five, plan_path, manifest_path):
    raw_plan, raw_manifest = Path(plan_path).read_bytes(), Path(manifest_path).read_bytes()
    plan, manifest = decode_json(raw_plan), decode_json(raw_manifest)
    identity = verify_identity(manifest, plan, raw_plan)
    if 'prior_five' in manifest:
        require(manifest['prior_five'] == {k: plan['prior_five_results'][k] for k in ('report', 'manifest')}, 'duplicate prior pins disagree')
    prior, five = verify_prior(input_root_five, plan)
    audit, fifteen = verify_fifteen(input_root, plan, manifest, identity)
    report = {'protocol': PROTOCOL, 'complete': True, 'scientifically_complete': fifteen is not None,
              'run_identity': identity, 'manifest_sha256': sha(raw_manifest), 'analysis_plan_sha256': sha(raw_plan),
              'prior_five_report_sha256': PRIOR_REPORT_SHA, 'reading_rule15': READING_RULE15,
              **audit, 'statistics': None, 'dose_comparison': None,
              'scope': 'EPN development; fresh fifteen-epoch recipes; one seed; calibrated subjects; no official test evaluation',
              'limitations': ['No isolated dose effect: the new cosine horizon changes the optimization trajectory.',
                              'No general FM, regression, P1/P2 or V-R1 adoption conclusion.',
                              'Validation diagnostics do not authorize an automatic extension or establish convergence when negative.',
                              'Historical logistic score remains a separate descriptive reader.']}
    if fifteen is not None:
        # The cross-dose equality is checked before publishing any final scores.
        report['dose_comparison'] = dose_statistics(fifteen, five)
        report['statistics'] = FIVE.statistics(fifteen)
        require(report['statistics']['bootstrap']['seed'] == prior['statistics']['bootstrap']['seed'] == SEED,
                'bootstrap convention differs from five epochs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--input-root-five', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(not args.out.exists() and all(not out.is_relative_to(p.resolve()) for p in (args.input_root, args.input_root_five)),
            'new output outside both input roots required')
    result = run(args.input_root, args.input_root_five, args.plan, args.manifest)
    raw = json.dumps(result, indent=2, allow_nan=False) + '\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(raw)


if __name__ == '__main__':
    main()
