#!/usr/bin/env python3
"""Paired subject bootstrap of existing pool768 and PSD3 pool128 predictions.

No fitting, feature extraction or network access. This compares existing pipelines,
not the causal effect of PCA alone. The production CLI fixes 2000 draws/20261102.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wearusfm.harness.downstream_ab import fingerprint  # noqa: E402

N_BOOT, SEED = 2000, 20261102
MODELS = ('B', 'R3c')
PIPELINES = ('pool768', 'pool128')
CELLS = tuple(f'{model}/{pipeline}' for model in MODELS for pipeline in PIPELINES)
WEIGHTS = {'p1': {'ninapro_db2': 8 / 13, 'ninapro_db3': 3 / 13, 'ninapro_db6': 2 / 13},
           'p2': {'epn612': .5, 'uci_emg': .5}}
DATASETS = tuple(ds for weights in WEIGHTS.values() for ds in weights)
REPORT_BASE = 'results/infra/pilot_posa/sonde_convergenza_20261029/B'
PSD3_REPORT = 'results/infra/psd3_20261029/report_completo.json'
CONTRASTS = {
    'pool128_minus_pool768/B': (-1., 1., 0., 0.),
    'pool128_minus_pool768/R3c': (0., 0., -1., 1.),
    'B_minus_R3c/pool768': (1., 0., -1., 0.),
    'B_minus_R3c/pool128': (0., 1., 0., -1.),
    'difference_of_B_minus_R3c': (-1., 1., 1., -1.),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _git_blob(repo_root, source):
    return subprocess.check_output(['git', 'show', f"{source['ref']}:{source['path']}"], cwd=repo_root)


def _pinned(raw, expected, name):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None,
            f'{name}: invalid SHA-256 pin')
    require(digest(raw) == expected, f'{name}: SHA-256 pin mismatch')
    return {'path': str(name), 'sha256': expected, 'bytes': len(raw)}


def _labels(values, name):
    """Native labels only: UTF-8 byte strings decode; numeric labels never stringify."""
    arr = np.asarray(values)
    require(arr.ndim == 1 and len(arr) > 0, f'{name}: empty or non-vector labels')
    if all(isinstance(v, (str, bytes, np.str_, np.bytes_)) for v in values):
        vals = [v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values]
        require(all(v.strip() for v in vals), f'{name}: blank label')
        return np.asarray(vals), 'text'
    require(all(isinstance(v, (Real, np.bool_)) for v in values), f'{name}: mixed or invalid labels')
    require(arr.dtype.kind in 'biuf' and np.isfinite(arr).all(), f'{name}: nonfinite numeric labels')
    return arr, 'numeric'


def _subjects(values, name):
    arr, family = _labels(values, name)
    require(family == 'text', f'{name}: subjects must be nonempty strings')
    return arr


def _integers(values, name, maximum=None):
    arr = np.asarray(values)
    require(arr.ndim == 1 and len(arr) > 0 and arr.dtype.kind in 'iu', f'{name}: invalid integer vector')
    require(all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) for v in values)
            and np.all(arr >= 0) and np.all(arr <= np.iinfo(np.int64).max), f'{name}: invalid indices')
    if maximum is not None:
        require(np.all(arr <= maximum), f'{name}: index out of range')
    return arr.astype(np.int64)


def _roles(folds, expected_subjects, name):
    require(isinstance(folds, list) and len(folds) == 4, f'{name}: need four folds')
    by_fold = {}
    held_to_fold = {}
    for f in folds:
        i = f.get('fold')
        require(type(i) is int and 0 <= i < 4 and i not in by_fold, f'{name}: duplicate/invalid fold')
        raw = f.get('roles', f)
        roles = {}
        for role in ('train', 'val', 'held'):
            names = _subjects(raw[role], f'{name}/{i}/{role}').tolist()
            require(len(names) == len(set(names)), f'{name}: duplicate subject in {role}')
            roles[role] = sorted(names)
        groups = [set(roles[r]) for r in ('train', 'val', 'held')]
        require(not any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))),
                f'{name}: subject appears in multiple roles')
        require(set.union(*groups) == expected_subjects, f'{name}: missing or additional subject in roles')
        for subject in roles['held']:
            require(subject not in held_to_fold, f'{name}: subject held in multiple folds')
            held_to_fold[subject] = i
        by_fold[i] = roles
    require(set(held_to_fold) == expected_subjects, f'{name}: missing held subject')
    return [by_fold[i] for i in range(4)], held_to_fold


def _normalize(rows, y, subjects, pred, folds, name):
    rows = _integers(rows, name + '/rows')
    require(len(set(rows.tolist())) == len(rows), f'{name}: duplicate row')
    y, family = _labels(y, name + '/targets')
    pred, pred_family = _labels(pred, name + '/predictions')
    subjects = _subjects(subjects, name + '/subjects')
    folds = _integers(folds, name + '/folds', maximum=3)
    require(family == pred_family, f'{name}: target/prediction label family mismatch')
    require(all(len(v) == len(rows) for v in (y, pred, subjects, folds)), f'{name}: inconsistent row lengths')
    require(len(np.unique(y)) >= 2, f'{name}: fewer than two target classes')
    order = np.argsort(rows, kind='stable')
    return {'rows': rows[order], 'y': y[order], 'subjects': subjects[order],
            'pred': pred[order], 'fold': folds[order], 'label_family': family}


def load_and_validate(manifest_path, oof_dir, shard_dir, repo_root=ROOT):
    """Verify every pinned input and exact four-cell pairing before scoring anything."""
    manifest_path, oof_dir, shard_dir = map(Path, (manifest_path, oof_dir, shard_dir))
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    require(manifest.get('protocol') == 'exploratory-existing-pool-pipelines-20261102-v1', 'unexpected manifest protocol')
    provenance = {'manifest': {'path': str(manifest_path), 'sha256': digest(raw), 'bytes': len(raw)},
                  'git_inputs': [], 'shards': [], 'oof': []}
    git_sources, local_sources = {}, {}
    for source in manifest['sources']:
        if source['kind'] == 'git_blob':
            key = source['path']
            require(key not in git_sources, 'duplicate git source pin')
            require(source['ref'] == manifest['source_commit'], 'git source commit mismatch')
            data = _git_blob(repo_root, source)
            info = _pinned(data, source['sha256'], key)
            require(len(data) == source['bytes'], f'{key}: byte-count mismatch')
            provenance['git_inputs'].append(dict(info, ref=source['ref']))
            git_sources[key] = data
        elif source['kind'] == 'local_file':
            key = Path(source['path']).name
            require(key not in local_sources, 'duplicate shard pin')
            local_sources[key] = source
        else:
            raise ValueError('unknown manifest source kind')
    require(set(local_sources) == {f'{m}_{ds}_pool.json' for m in MODELS for ds in DATASETS}, 'incomplete shard pins')
    complete = json.loads(git_sources[PSD3_REPORT])
    pins = {}
    for p in manifest['local_oof_inventory']['missing']:
        key = (p['model'], p['task'])
        require(key not in pins, 'duplicate OOF pin')
        pins[key] = p
    require(set(pins) == {(m, t) for m in MODELS for t in WEIGHTS}, 'incomplete OOF pins')
    reference_scores = {(d['model'], d['dataset']): d for d in manifest['dataset_scores']}
    require(len(reference_scores) == len(manifest['dataset_scores']) == 10, 'duplicate/incomplete dataset reference scores')
    grid = {ds: {} for ds in DATASETS}
    recipes = set()
    identities = {m: set() for m in MODELS}
    for model in MODELS:
        for task, weights in WEIGHTS.items():
            report_path = f'{REPORT_BASE}/{model}_{task}_pool.json'
            old = json.loads(git_sources[report_path])
            require(old['complete'] is True and old['phase'] == 'B' and old['encoder'] == model and old['task'] == task,
                    f'{report_path}: not a complete matching phase B report')
            require(old['weights'] == weights and set(old['datasets']) == set(weights), 'changed fixed dataset weights/support')
            require(old['readout']['layer'] == 'pool' and old['readout']['k_requested'] == 0 and old['max_iter'] == 12000,
                    'unexpected original pool768 recipe')
            pin = pins[(model, task)]
            require(pin['remote_path_from_report'] == old['oof'] and pin['sha256_from_report'] == old['oof_sha256'],
                    'OOF report/manifest pin mismatch')
            path = oof_dir / Path(old['oof']).name
            npz_raw = path.read_bytes()
            info = _pinned(npz_raw, old['oof_sha256'], path)
            provenance['oof'].append(dict(info, model=model, task=task, original_path=old['oof']))
            with np.load(io.BytesIO(npz_raw), allow_pickle=False) as z:
                ds_names = _subjects(z['datasets'], 'NPZ/datasets').tolist()
                require(len(ds_names) == len(set(ds_names)) and set(ds_names) == set(weights), 'NPZ dataset support mismatch')
                require(set(z.files) == {'datasets'} | {f'{ds}_{k}' for ds in weights for k in ('row', 'y', 's', 'pred', 'fold')},
                        'unexpected/missing OOF arrays')
                for ds in weights:
                    name = f'{model}/{ds}'
                    row, y, subjects, pred, fold = (z[f'{ds}_{k}'] for k in ('row', 'y', 's', 'pred', 'fold'))
                    od = old['datasets'][ds]
                    require(fingerprint(row) == od['oof_rows_sha256'], f'{name}: original ordered row fingerprint mismatch')
                    require(len(row) == od['n_oof_windows'], f'{name}: original OOF count mismatch')
                    expected_subjects = _subjects(od['subjects'], name + '/report subjects').tolist()
                    require(len(set(expected_subjects)) == len(expected_subjects), f'{name}: duplicate report subject')
                    roles, held_map = _roles(od['folds'], set(expected_subjects), name + '/old')
                    old_cell = _normalize(row, y, subjects, pred, fold, name + '/pool768')
                    require(set(old_cell['subjects']) == set(expected_subjects), f'{name}: missing or additional OOF subject')
                    expected_folds = np.asarray([held_map[s] for s in old_cell['subjects']])
                    require(np.array_equal(old_cell['fold'], expected_folds), f'{name}: original row/subject/fold mismatch')
                    for f in od['folds']:
                        require(np.count_nonzero(old_cell['fold'] == f['fold']) == f['n_held'], f'{name}: held row count mismatch')
                    file_name = f'{model}_{ds}_pool.json'
                    source = local_sources[file_name]
                    shard_raw = (shard_dir / file_name).read_bytes()
                    sinfo = _pinned(shard_raw, source['sha256'], shard_dir / file_name)
                    require(len(shard_raw) == source['bytes'], f'{name}: shard byte-count mismatch')
                    provenance['shards'].append(sinfo)
                    shard = json.loads(shard_raw)
                    require(shard['status'] == 'completed' and shard['model'] == model and shard['dataset'] == ds
                            and shard['task'] == task and shard['view'] == 'pool', f'{name}: incomplete/wrong shard identity')
                    require(shard['recipe']['features']['k'] == 128, f'{name}: wrong PSD3 dimension')
                    require(shard['recipe']['logistic']['c_grid'] == old['solver_recipe']['c_grid'], f'{name}: C grid mismatch')
                    require(shard['checkpoint_sha256'] == old['checkpoint_sha256'], f'{name}: checkpoint mismatch')
                    require(shard['input_sha256'] == od['input_sha256'], f'{name}: input mismatch')
                    require(shard['provenance']['sha256'] == old['cache_pins'][ds]['sha256'], f'{name}: cache mismatch')
                    recipes.add(shard['recipe_sha256'])
                    identities[model].add(shard['checkpoint_sha256'])
                    new_roles, new_map = _roles(shard['folds'], set(expected_subjects), name + '/new')
                    require(roles == new_roles and held_map == new_map, f'{name}: fold roles mismatch')
                    require(fingerprint([f['roles'] for f in shard['folds']]) == shard['roles_sha256'], f'{name}: role fingerprint mismatch')
                    require(set(shard['predictions']['logistic']) == {'0'}, f'{name}: unexpected logistic seed')
                    new_subjects = _subjects(shard['subjects'], name + '/new subjects')
                    require(set(new_subjects) == set(expected_subjects), f'{name}: missing or additional PSD3 subject')
                    new_fold = [new_map[s] for s in new_subjects]
                    new_cell = _normalize(shard['rows'], shard['y'], new_subjects,
                                          shard['predictions']['logistic']['0'], new_fold, name + '/pool128')
                    require(fingerprint(shard['rows'], shard['y'], shard['subjects']) == shard['provenance']['oof_data_sha256'],
                            f'{name}: PSD3 OOF data fingerprint mismatch')
                    for key in ('rows', 'y', 'subjects', 'fold'):
                        require(np.array_equal(old_cell[key], new_cell[key]), f'{name}: paired {key} mismatch')
                    require(old_cell['label_family'] == new_cell['label_family'], f'{name}: target label family mismatch')
                    for f in shard['folds']:
                        classes, family = _labels(f['logistic']['classes'], name + '/train classes')
                        require(family == old_cell['label_family'], f'{name}: train class mapping mismatch')
                        mask = old_cell['fold'] == f['fold']
                        for cell in (old_cell, new_cell):
                            require(np.isin(cell['y'][mask], classes).all() and np.isin(cell['pred'][mask], classes).all(),
                                    f'{name}: prediction or target outside actual fold train classes')
                    for pipeline, cell, score in (('pool768', old_cell, od['s']),
                                                   ('pool128', new_cell, complete['cells'][f'{name}/pool']['bacc_oof']['logistic']['mean'])):
                        reference = reference_scores[(model, ds)]
                        require(abs(score - reference[f'{pipeline}_bacc']) < 1e-12, f'{name}: manifest point mismatch')
                        cell['reported_bacc'] = score
                        cell['checkpoint_sha256'] = shard['checkpoint_sha256']
                        cell['input_sha256'] = shard['input_sha256']
                        grid[ds][f'{model}/{pipeline}'] = cell
    require(len(recipes) == 1 and all(len(s) == 1 for s in identities.values()), 'inconsistent recipe/checkpoint identity')
    provenance['psd3_recipe_sha256'] = recipes.pop()
    provenance['checkpoints'] = {m: next(iter(s)) for m, s in identities.items()}
    provenance['label_mapping'] = 'PSD3 stored predictions are already mapped back to native classes. Decode UTF-8 bytes; preserve numeric labels; never infer integer-to-string mappings.'
    provenance['existing_pipeline_differences'] = manifest['actual_pipeline_differences']
    return grid, provenance


def _summary(point, draws):
    cells = {name: {'bacc': float(point[i]), 'ci95': np.quantile(draws[:, i], [.025, .975]).tolist()}
             for i, name in enumerate(CELLS)}
    contrasts = {}
    for name, values in CONTRASTS.items():
        c = np.asarray(values)
        samples = 100 * (draws @ c)
        contrasts[name] = {'delta_pp': float(100 * (point @ c)),
                           'ci95_pp': np.quantile(samples, [.025, .975]).tolist(),
                           'paired_subject_se_pp': float(np.std(samples, ddof=1)),
                           'coefficients': dict(zip(CELLS, values))}
    return {'cells': cells, 'contrasts': contrasts}


def compare_cells(grid, n_boot=N_BOOT, seed=SEED, batch_size=128):
    """Pooled macro recall; the same subject draw is used for all four cells."""
    require(type(n_boot) is int and n_boot >= 2 and type(seed) is int and seed >= 0
            and type(batch_size) is int and batch_size > 0, 'invalid bootstrap configuration')
    require(set(grid) == set(DATASETS), 'incomplete dataset grid')
    rng = np.random.default_rng(seed)
    results, points, samples = {}, {}, {}
    for ds in DATASETS:
        cells = grid[ds]
        require(set(cells) == set(CELLS), f'{ds}: incomplete model/pipeline grid')
        ref = cells[CELLS[0]]
        for name, cell in cells.items():
            for key in ('rows', 'y', 'subjects', 'fold'):
                require(np.array_equal(cell[key], ref[key]), f'{ds}/{name}: paired {key} mismatch')
            require(cell['label_family'] == ref['label_family'], f'{ds}: label family mismatch')
        classes, cls_index = np.unique(ref['y'], return_inverse=True)
        subjects, sub_index = np.unique(ref['subjects'], return_inverse=True)
        ns, nc = len(subjects), len(classes)
        flat = sub_index * nc + cls_index
        n = np.bincount(flat, minlength=ns * nc).reshape(ns, nc)
        hits = np.asarray([np.bincount(flat[cells[name]['pred'] == ref['y']], minlength=ns * nc).reshape(ns, nc)
                           for name in CELLS], dtype=np.float64)
        point = (hits.sum(1) / n.sum(0)).mean(1)
        for i, name in enumerate(CELLS):
            if 'reported_bacc' in cells[name]:
                require(abs(point[i] - cells[name]['reported_bacc']) < 1e-12, f'{ds}/{name}: OOF score differs from pinned report')
        draws = np.empty((n_boot, len(CELLS)))
        missing = single = 0
        for start in range(0, n_boot, batch_size):
            size = min(batch_size, n_boot - start)
            pick = rng.integers(ns, size=(size, ns))
            multiplicity = np.zeros((size, ns), dtype=np.int64)
            np.add.at(multiplicity, (np.arange(size)[:, None], pick), 1)
            den = multiplicity @ n
            correct = (multiplicity @ hits.transpose(1, 0, 2).reshape(ns, -1)).reshape(size, len(CELLS), nc)
            present = den > 0
            recall = np.divide(correct, den[:, None, :], out=np.zeros_like(correct), where=present[:, None, :])
            draws[start:start + size] = recall.sum(2) / present.sum(1)[:, None]
            missing += int(np.count_nonzero(present.sum(1) < nc))
            single += int(np.count_nonzero(present.sum(1) == 1))
        result = _summary(point, draws)
        result.update(n_rows=len(ref['y']), n_subjects=ns, n_classes=nc,
                      paired_rows_targets_subjects_folds_sha256=fingerprint(ref['rows'], ref['y'], ref['subjects'], ref['fold']),
                      counts={'classes': classes.tolist(), 'subjects': subjects.tolist(), 'denominators': n.tolist(),
                              'correct': {name: hits[i].astype(np.int64).tolist() for i, name in enumerate(CELLS)}},
                      bootstrap_draws_with_missing_classes=missing, bootstrap_single_class_draws=single)
        results[ds], points[ds], samples[ds] = result, point, draws
    tasks = {}
    for task, weights in WEIGHTS.items():
        point = sum(w * points[ds] for ds, w in weights.items())
        draws = sum(w * samples[ds] for ds, w in weights.items())
        tasks[task] = dict(_summary(point, draws), weights=weights)
    return {'protocol': 'existing-pool-pipelines-paired-subject-20261102-v1', 'descriptive_only': True,
            'comparison': 'Existing pool768 and PSD3 pool128 pipelines; NOT a causal PCA-only ablation.',
            'n_boot': n_boot, 'seed': seed, 'datasets': results, 'tasks': tasks,
            'bootstrap': 'Resample subjects within each dataset; one shared draw for all pipelines and encoders; independent draws across datasets. Pooled macro recall, fixed P1/P2 weights. Keep all draws, including those missing classes, with the same present-class support for every cell.',
            'uncertainty': 'Conditional on one fixed checkpoint per encoder, fitted readers and folds; no refits or pretraining-seed uncertainty. Descriptive percentile intervals, no multiplicity adjustment and no adoption gate.',
            'units': {'bacc': 'fraction', 'ci95': 'fraction', 'contrasts': 'percentage points'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--oof-dir', type=Path, required=True)
    parser.add_argument('--shard-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'output already exists; never overwrite an analysis')
    grid, provenance = load_and_validate(args.manifest, args.oof_dir, args.shard_dir, args.repo_root)
    result = compare_cells(grid)
    result['provenance'] = provenance
    result['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__,
                         'script_sha256': digest(Path(__file__).read_bytes())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': digest(args.output.read_bytes()),
                      'complete': True, 'n_boot': N_BOOT, 'seed': SEED}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
