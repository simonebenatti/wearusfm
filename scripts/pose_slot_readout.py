"""Frozen pose ridge on mean384, mean48, slot384 and nested slot720.

Only projected residuals are opened, as a read-only NPY memory map. The encoder
is not instantiated. All model selection uses the original 40 training users.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import pose_slot_features as F
import pose_slot_plan as P
import pose_temporal_readout as T

E, D = T.E, T.D
require = D.require
MODELS, READERS = T.MODELS, F.READERS
OFFSET, SEED, N_BOOT, THRESHOLD, BATCH = T.OFFSET, T.SEED, T.N_BOOT, T.THRESHOLD, T.BATCH
PROTOCOL = 'pose-slot-readout-20261102-v1'


def recipe():
    return {'features': F.recipe(), 'common_centers': T.recipe()['common_centers'],
            'target_support': T.recipe()['target_support'], 'cv': T.recipe()['cv'],
            'fit': T.recipe()['fit'], 'dynamic_threshold': THRESHOLD,
            'dynamic_selection': T.recipe()['dynamic_selection'],
            'bootstrap': {'unit': 'user', 'replicates': N_BOOT, 'seed': SEED},
            'batch_windows': BATCH,
            'primary': 'S384 minus M384 for B/R3c and difference of these contrasts, same feature width; not residual-only attribution',
            'compressed_mean_control': 'S384 minus M48 for B/R3c and difference of these contrasts, adding residuals to the same compressed mean',
            'secondary': 'S720 minus M384 for B/R3c and difference of these contrasts; larger reader preserving full mean',
            'scope': 'Only seven fixed slot contrasts projected to 48 channels; a negative result does not exclude other slot information.'}


def read_residual(path, mean_shape):
    before = T.file_state(path)
    result = np.load(Path(path), mmap_mode='r', allow_pickle=False)
    require(isinstance(result, np.memmap) and result.dtype == np.float32
            and result.shape == (*mean_shape[:2], F.Q * F.CHANNELS), 'residual NPY shape/dtype mismatch')
    require(not result.flags.writeable and T.file_state(path) == before, 'residual map must be stable and read-only')
    return result


def grouped_stats(train, residual, reader, batch=BATCH):
    require(reader in READERS and batch > 0, 'invalid reader/batch')
    require(residual.shape == (*train['lat'].shape[:2], F.Q * F.CHANNELS), 'mean/residual shape mismatch')
    users = np.asarray([m['user'] for m in train['meta']])
    dim, joints = F.WIDTHS[reader], train['target'].shape[-1]
    parts = {}
    for start in range(0, len(users), batch):
        sl = slice(start, start + batch)
        x = F.reader_features(train['lat'][sl], residual[sl], reader, train['lengths'][sl])[:, OFFSET:-OFFSET]
        for user in sorted(set(users[sl])):
            pick = users[sl] == user
            stats = E.sufficient(x[pick].reshape(-1, dim), train['target'][sl][pick, OFFSET:-OFFSET].reshape(-1, joints),
                                 train['central_mask'][sl][pick].reshape(-1, joints))
            parts[user] = E.merge_stats([parts[user], stats], joints) if user in parts else stats
    return parts, dim, joints


def fit_reader(train, residual, reader, expected_folds=None):
    start = perf_counter()
    parts, dim, joints = grouped_stats(train, residual, reader)
    statistics_seconds = perf_counter() - start
    actual_folds = [sorted(parts)[::2], sorted(parts)[1::2]]
    require(expected_folds is None or actual_folds == expected_folds, 'training fold users differ')
    start = perf_counter()
    cv = E.cross_validate(parts, dim, joints)
    require(not cv['boundary_unresolved'], 'alpha boundary unresolved; no scientific report')
    weights, bias = E.prepare_solver(E.merge_stats(list(parts.values()), joints), dim, joints)(cv['alpha'])
    return weights, bias, {'selection': cv, 'n_features': dim, 'n_users': len(parts),
                           'statistics_seconds': statistics_seconds, 'selection_refit_seconds': perf_counter() - start,
                           'fit_support_sha256': T.arrays_hash(train['central_mask']),
                           'coefficients_sha256': T.arrays_hash(weights, bias)}


def predict(mean, residual, lengths, reader, weights, bias, batch=BATCH):
    """No targets, target masks or dynamic labels can affect this transform."""
    require(weights.shape == (F.WIDTHS[reader], len(bias)), 'coefficient shape mismatch')
    result = np.empty((len(mean), mean.shape[1] - 2 * OFFSET, len(bias)), dtype=np.float64)
    for start in range(0, len(mean), batch):
        sl = slice(start, start + batch)
        x = F.reader_features(mean[sl], residual[sl], reader, lengths[sl])[:, OFFSET:-OFFSET]
        result[sl] = x.astype(np.float64) @ weights + bias
    require(np.isfinite(result).all(), 'nonfinite prediction')
    return result


def contrast_definitions():
    definitions = {}
    for reader, baseline, level in (('S384', 'M384', 'primary_equal_width'),
                                    ('S384', 'M48', 'incremental_compressed_mean'),
                                    ('S720', 'M384', 'incremental_full_mean')):
        for model in MODELS:
            definitions[f'{model}_{reader}_minus_{baseline}'] = {
                'level': level, 'coefficients': {f'{model}/{reader}': 1., f'{model}/{baseline}': -1.}}
        definitions[f'{reader}_minus_{baseline}_difference_of_differences'] = {
            'level': level, 'coefficients': {f'B/{reader}': 1., f'B/{baseline}': -1.,
                                            f'R3c/{reader}': -1., f'R3c/{baseline}': 1.}}
    return definitions


def paired_comparison(cells, n_boot=N_BOOT, seed=SEED):
    require(set(cells) == {f'{m}/{r}' for m in MODELS for r in READERS}, 'need all eight cells')
    require(type(n_boot) is int and n_boot >= 2 and type(seed) is int and seed >= 0, 'invalid bootstrap configuration')
    reference = cells['B/M384']['analysis']
    for cell in cells.values():
        require(cell['analysis']['support_sha256'] == reference['support_sha256']
                and cell['analysis']['targets_sha256'] == reference['targets_sha256'], 'paired metric support differs')
    output = {}
    for group in ('dynamic', 'all', 'static'):
        rows = {key: value['analysis']['groups'][group]['centered']['per_user'] for key, value in cells.items()}
        base, users = rows['B/M384'], sorted(rows['B/M384'])
        require(len(users) >= 2 and all(sorted(r) == users for r in rows.values()), 'paired users missing')
        denominator = np.asarray([base[u]['sums']['vary'] for u in users])
        for values in rows.values():
            for user in users:
                require(all(values[user][key] == base[user][key] for key in ('n_windows', 'n_window_joints', 'n_valid_elements')),
                        'paired per-user support differs')
                require(values[user]['sums']['vary'] == base[user]['sums']['vary'], 'paired target variance differs')
        draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
        den = denominator[draws].sum(1)
        require(np.isfinite(den).all() and (den > 0).all(), 'invalid bootstrap denominator; no replicas discarded')
        result = {'users': users, 'n_bootstrap': n_boot, 'seed': seed, 'contrasts': {}, 'cell_ratio_ci95': {}}
        errors = {key: np.asarray([values[u]['sums']['centered'] for u in users]) for key, values in rows.items()}
        for key, error in errors.items():
            require(np.isfinite(error).all(), 'nonfinite per-user errors')
            result['cell_ratio_ci95'][key] = np.percentile(error[draws].sum(1) / den, [2.5, 97.5]).tolist()
        for label, definition in contrast_definitions().items():
            coefficients = definition['coefficients']
            numerator = sum(coef * errors[key] for key, coef in coefficients.items())
            values = numerator[draws].sum(1) / den
            require(np.isfinite(values).all(), 'nonfinite bootstrap contrast')
            result['contrasts'][label] = {'delta_ratio': float(numerator.sum() / denominator.sum()),
                                          'ci95': np.percentile(values, [2.5, 97.5]).tolist(), **definition,
                                          'per_user': {u: {'delta_centered_sse': float(v), 'target_variance_sum': float(d)}
                                                       for u, v, d in zip(users, numerator, denominator)}}
        output[group] = result
    return output


def benchmark_train(train, residual):
    """Train only: construct sufficient statistics/three solves; no CV or val metrics."""
    result = {}
    for reader in READERS:
        start = perf_counter()
        parts, dim, joints = grouped_stats(train, residual, reader)
        stat_seconds = perf_counter() - start
        users = sorted(parts)
        start = perf_counter()
        for selected in (users[::2], users[1::2], users):
            E.prepare_solver(E.merge_stats([parts[u] for u in selected], joints), dim, joints)(1e5)
        result[reader] = {'statistics_seconds': stat_seconds, 'three_eigensystems_seconds': perf_counter() - start,
                          'n_features': dim, 'n_users': len(users), 'valid_fit_elements': int(train['central_mask'].sum()),
                          'one_gram_bytes': 8 * dim * dim,
                          'note': 'timing only; fixed alpha 1e5, no validation inference or CV score'}
    return result


def run(plan):
    """Plan supplies original cache pins and projected residual paths; CLI validates receipts."""
    cells, coefficients, identities, previous = {}, {}, {}, None
    for model in MODELS:
        train = T.read_data(plan['caches'][model]['train'])
        val = T.read_data(plan['caches'][model]['val'])
        train_residual = read_residual(Path(plan['output_root']) / model / 'train' / 'residual.npy', train['lat'].shape)
        val_residual = read_residual(Path(plan['output_root']) / model / 'val' / 'residual.npy', val['lat'].shape)
        train_users, val_users = [{m['user'] for m in d['meta']} for d in (train, val)]
        require(train_users.isdisjoint(val_users), 'train/val user leakage')
        require(float(np.nanmedian(val['pose_var'])) == THRESHOLD, 'original dynamic threshold changed')
        signatures = {'train': T.target_signature(train), 'val': T.target_signature(val)}
        require(previous is None or signatures == previous, 'B/R3c target/support mismatch')
        previous, identities[model] = signatures, signatures
        for reader in READERS:
            weights, bias, fit = fit_reader(train, train_residual, reader, plan['fold_users'])
            start = perf_counter()
            prediction = predict(val['lat'], val_residual, val['lengths'], reader, weights, bias)
            key = f'{model}/{reader}'
            cells[key] = {'fit': fit, 'prediction_seconds': perf_counter() - start, 'analysis': T.summarize(val, prediction)}
            coefficients[f'{model}_{reader}_weights'], coefficients[f'{model}_{reader}_bias'] = weights, bias
            del prediction
        del train, val, train_residual, val_residual
    return {'status': 'completed', 'cells': cells, 'paired': paired_comparison(cells), 'target_identities': identities,
            'scope': 'Exploratory reused val users; fixed encoders and folds, one fixed sketch. Same-width recipe comparison; incremental slot comparisons retain the compressed or full mean but add reader capacity. No endpoint gate or automatic FM continuation.'}, coefficients


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validate', 'benchmark', 'run'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-code-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    plan_bytes = args.plan.read_bytes()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    P.validate_plan(plan)
    P.fresh_output(args.out, plan)
    coefficient_path = args.out.with_suffix('.coefficients.npz')
    if args.stage == 'run':
        P.fresh_output(coefficient_path, plan)
    identity = P.runtime_identity(args.expected_code_sha)
    require(identity['sources'] == plan['sources'], 'source pins mismatch')
    inspections, states = {}, {}
    models = ('B',) if args.stage == 'benchmark' else MODELS
    splits = ('train',) if args.stage == 'benchmark' else ('train', 'val')
    for model in models:
        for split in splits:
            reference = P.validate_reference(plan, model, split)
            residual = P.validate_residual(plan, model, split, expected_code_sha=args.expected_code_sha,
                                           expected_plan_sha256=plan_sha)
            inspections[f'{model}/{split}'] = {'reference': reference, 'residual': residual}
            states.update(reference['states'])
            states.update(residual['states'])
        train_users = inspections[f'{model}/train']['reference']['users']
        require(train_users == sorted(sum(plan['fold_users'], [])), 'train user identity mismatch')
        if 'val' in splits:
            require(set(train_users).isdisjoint(inspections[f'{model}/val']['reference']['users']), 'train/val leakage')
    if args.stage == 'validate':
        result = {'status': 'validated_hashes_headers_metadata', 'scientific_metrics_computed': False}
    elif args.stage == 'benchmark':
        train = T.read_data(plan['caches']['B']['train'])
        residual = read_residual(inspections['B/train']['residual']['path'], train['lat'].shape)
        result = {'status': 'train_only_timing', 'model': 'B', 'timings': benchmark_train(train, residual)}
    else:
        result, coefficients = run(plan)
    require(args.plan.read_bytes() == plan_bytes and P.runtime_identity(args.expected_code_sha) == identity,
            'source/plan/numerical environment changed during execution')
    require(states == {path: P.file_state(path) for path in states}, 'input artifact changed during execution')
    result.update(protocol=PROTOCOL, recipe=recipe(), runtime=identity, plan_sha256=plan_sha,
                  inspections=inspections, input_pins=plan['caches'], slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'run':
        with coefficient_path.open('xb') as stream:
            np.savez_compressed(stream, **coefficients)
        result['coefficients'] = {'path': str(coefficient_path), 'sha256': D.file_sha256(coefficient_path)}
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'out': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
