"""Frozen B5000 post-LayerNorm masked ridge; train-only CV, paired dynamics.

This standalone diagnostic never instantiates the encoder or trains a model.
Only the frozen PoseHead class and the adjacent unchanged pose_dynamics module
are imported. All ridge sufficient statistics use float64, without scaling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pose_dynamics as D

VAL_SHA = '70bb551309b502565d38072abf41ffee322a06de14589d36b720570f1cfbfa46'
CHECKPOINT_SHA = '815956b98d27cd2ff71f1255d0e220e2b0e08e467e7bedf93a54d0984b10783e'
STATS_SHA = '757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e'
CODE_SHA = '24139ab901c052357a46a65bc20466f2eb1e8358'
CONFIG_SHA = 'f55fa6a5f117ad8898c20bbaf8b20f288c169a9a9a12e7bf3bfdad26fef1d157'
DYNAMICS_SHA = '5e5ff2c8b27d5692a064534f793924ddf5463c7bab13ccf75356c0950d9dd5b8'
M2_MAX_ABS, M2_RMSE = .005, .0005
require = D.require


def load_head(path):
    import torch
    from wearusfm.model.pose_head import PoseHead
    require(D.file_sha256(path) == CHECKPOINT_SHA, 'checkpoint SHA mismatch')
    config_path = path.parent / 'config.json'
    require(D.file_sha256(config_path) == CONFIG_SHA, 'checkpoint config file SHA mismatch')
    config = json.loads(config_path.read_text())
    # This pickle is trusted only after matching the frozen B SHA.
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state.get('step') == 5000 and state.get('pose_stats_sha256') == STATS_SHA, 'checkpoint step/stats mismatch')
    require(state.get('config') == config and config['model'].get('pose_head') is True, 'checkpoint config mismatch')
    head = PoseHead(config['model']['dim']).float().eval()
    weights = {k.removeprefix('pose_head.'): v for k, v in state['student'].items() if k.startswith('pose_head.')}
    head.load_state_dict(weights, strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in head.state_dict().values()), 'nonfinite head')
    require(bool(head.out.weight.count_nonzero()), 'untrained zero head')
    return head


def load_cache(path):
    data = D.load_cache(path)
    with np.load(path, allow_pickle=False) as archive:
        data['lat'] = archive['lat']
    lat = data['lat']
    require(lat.ndim == 3 and lat.shape[:2] == data['mask'].shape[:2] and lat.dtype == np.float16, 'expected fp16 cached latents')
    valid = data['mask'].any(axis=2)
    require(np.isfinite(lat[valid]).all(), 'nonfinite valid latent')
    require(np.isfinite(data['target'][data['mask']]).all() and np.isfinite(data['pred'][data['mask']]).all(), 'nonfinite valid pose')
    return data


def check_cache(data, path, provenance, split, checkpoint):
    cfg = data['config']
    for key, value in {'which': 'student', 'split': split, 'code_sha': CODE_SHA,
                       'pose_stats_sha256': STATS_SHA, 'checkpoint_pose_stats_sha256': STATS_SHA,
                       'checkpoint': str(checkpoint), 'n_windows': 10, 'window_s': 4.0, 'seed': 20261026}.items():
        require(cfg.get(key) == value, 'cache config mismatch: ' + key)
    for key, value in {'checkpoint_sha256': CHECKPOINT_SHA, 'checkpoint_step': 5000,
                       'pose_stats_sha256': STATS_SHA, 'which': 'student', 'split': split}.items():
        require(provenance.get(key) == value, 'provenance mismatch: ' + key)
    require(provenance.get('execution_code_sha', provenance.get('training_code_sha')) == CODE_SHA, 'provenance code mismatch')
    source = provenance.get('cache') if split == 'val' else provenance.get('cache', provenance.get('output'))
    require(source is not None and Path(source).resolve() == path.resolve(), 'provenance cache path mismatch')
    expected_records, expected_users = (200, 40) if split == 'train' else (1950, 15)
    require(data['n_recordings'] == expected_records, 'recording count mismatch')
    users = {m['user'].removeprefix('u') for m in data['meta']}
    supported = {m['user'].removeprefix('u') for m, ok in zip(data['meta'], data['mask'].any(axis=(1, 2))) if ok}
    require(len(users) == expected_users and users == supported, 'user coverage mismatch')
    if split == 'val':
        D.check_provenance(data, path, VAL_SHA, provenance)
    else:
        require(provenance.get('mode') == 'extract' and provenance.get('config') == cfg, 'train extraction config mismatch')
        require(provenance.get('cache_bytes') == path.stat().st_size, 'train cache bytes mismatch')
        for key, value in {'n_recordings': 200, 'n_users': 40, 'n_windows': len(data['meta']),
                           'config_file_sha256': CONFIG_SHA}.items():
            require(provenance.get(key) == value, 'train provenance count/config mismatch: '+key)
        rows = data['mask'].any(axis=2)
        audit = {'selected_any_rows': int(rows.sum()),
                 'partial_joint_rows': int((rows & ~data['mask'].all(axis=2)).sum()),
                 'valid_elements': int(data['mask'].sum()),
                 'invalid_elements_excluded': int(data['mask'].size-data['mask'].sum())}
        require(provenance.get('mask_audit') == audit, 'train mask audit mismatch')
        require(provenance.get('r3c_subset', {}).get('identical_window_metadata') is True, 'train subset comparison missing')
    return users


def features_and_check(head, lat, pred, mask):
    """FP32 torch matches the head's transform; check each supplied block."""
    import torch
    valid = mask.any(axis=-1)
    require(np.isfinite(lat[valid]).all(), 'nonfinite valid latent')
    safe = np.where(valid[..., None], lat, 0).astype(np.float32)
    with torch.no_grad():
        x = head.norm(torch.from_numpy(safe))
        reconstructed = head.out(x).numpy()
    diff = reconstructed[mask].astype(np.float64) - pred[mask].astype(np.float64)
    require(np.isfinite(diff).all(), 'M2 nonfinite difference')
    maximum = float(np.abs(diff).max(initial=0))
    sse, n = float(diff @ diff), len(diff)
    rmse = float(np.sqrt(sse / n)) if n else 0.0
    require(maximum <= M2_MAX_ABS and rmse <= M2_RMSE, f'M2 failed: max_abs={maximum}, RMSE={rmse}')
    return x.numpy(), {'max_abs': maximum, 'rmse': rmse, 'sse': sse, 'n': n}


def mask_groups(mask):
    groups = {}
    for joint in range(mask.shape[1]):
        groups.setdefault(np.packbits(mask[:, joint]).tobytes(), []).append(joint)
    return list(groups.values())


def sufficient(x, y, mask):
    """Exact row masks per joint; identical masks share the expensive X'X."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    require(x.ndim == y.ndim == 2 and mask.shape == y.shape and len(x) == len(y), 'stat shape mismatch')
    require(np.isfinite(x[mask.any(axis=1)]).all() and np.isfinite(y[mask]).all(), 'nonfinite supported fit data')
    groups = []
    for joints in mask_groups(mask):
        sel = mask[:, joints[0]]
        xx, yy = x[sel], y[np.ix_(sel, joints)]
        groups.append({'joints': joints, 'n': int(sel.sum()), 'sx': xx.sum(axis=0), 'xx': xx.T @ xx,
                       'sy': yy.sum(axis=0), 'xy': xx.T @ yy, 'yy': (yy * yy).sum(axis=0)})
    return groups


def merge_stats(stats, n_joints):
    # Joints may share masks in one user but not another. Group by the complete
    # signature across selected users, preserving exact general masked fits.
    lookup = [{j: (i, g['joints'].index(j)) for i, g in enumerate(s) for j in g['joints']} for s in stats]
    signatures = {}
    for j in range(n_joints):
        signatures.setdefault(tuple(d[j][0] for d in lookup), []).append(j)
    out = []
    for signature, joints in signatures.items():
        selected = [s[i] for s, i in zip(stats, signature)]
        row = {'joints': joints, 'n': sum(g['n'] for g in selected)}
        for k in ('sx', 'xx'):
            row[k] = sum((g[k] for g in selected), np.zeros_like(selected[0][k]))
        for k in ('sy', 'xy', 'yy'):
            values = [g[k][..., [g['joints'].index(j) for j in joints]] for g in selected]
            row[k] = sum(values, np.zeros_like(values[0]))
        out.append(row)
    return out


def prepare_solver(stats, dim, n_joints):
    eigensystems = []
    for g in stats:
        require(g['n'] > 0, 'joint without fitting support')
        mx, my = g['sx'] / g['n'], g['sy'] / g['n']
        gram = g['xx'] - np.outer(g['sx'], g['sx']) / g['n']
        gram = (gram + gram.T) / 2
        eigen, vectors = np.linalg.eigh(gram)
        tol = 1e-10 * max(1., float(np.abs(eigen).max()))
        require(float(eigen.min()) >= -tol, 'centered Gram not PSD within float64 tolerance')
        eigen = np.maximum(eigen, 0)
        cross = g['xy'] - np.outer(g['sx'], my)
        eigensystems.append((g['joints'], mx, my, eigen, vectors, vectors.T @ cross))
    def solve(alpha):
        require(np.isfinite(alpha) and alpha > 0, 'alpha must be positive')
        weights, bias = np.empty((dim, n_joints)), np.empty(n_joints)
        for joints, mx, my, eigen, vectors, cross in eigensystems:
            w = vectors @ (cross / (eigen[:, None] + alpha))
            weights[:, joints], bias[joints] = w, my - mx @ w
        require(np.isfinite(weights).all() and np.isfinite(bias).all(), 'nonfinite ridge coefficients')
        return weights, bias
    return solve


def score(stats, weights, bias):
    sse, count = 0., 0
    for g in stats:
        w, b = weights[:, g['joints']], bias[g['joints']]
        terms = (g['yy'], np.sum(w * (g['xx'] @ w), axis=0), g['n'] * b*b,
                 -2*np.sum(w*g['xy'], axis=0), -2*b*g['sy'], 2*b*(g['sx'] @ w))
        value = sum(terms)
        tolerance = 1e-9 * np.maximum(1., sum(np.abs(t) for t in terms))
        require(np.isfinite(value).all() and (value >= -tolerance).all(), 'invalid validation SSE')
        sse += float(np.maximum(value, 0).sum())
        count += g['n'] * len(g['joints'])
    require(count > 0, 'validation fold has no support')
    return sse, count


def choose_alpha(evaluate, max_extensions=3):
    values = {exponent: float(evaluate(10.**exponent)) for exponent in range(-2, 5)}
    extended = {'lower': 0, 'upper': 0}
    while True:
        require(all(np.isfinite(v) for v in values.values()), 'nonfinite CV score')
        best = min(sorted(values), key=lambda e: values[e])
        side = 'lower' if best == min(values) else 'upper' if best == max(values) else None
        if side is None or extended[side] == max_extensions:
            break
        exponent = min(values)-1 if side == 'lower' else max(values)+1
        values[exponent] = float(evaluate(10.**exponent))
        extended[side] += 1
    return {'alpha': 10.**best, 'cv_mse': values[best], 'grid': [{'alpha': 10.**e, 'mse': values[e]} for e in sorted(values)],
            'extensions': extended, 'boundary_unresolved': side is not None, 'boundary': side,
            'objective': 'pooled masked standardized SSE / valid elements across both held-out folds; raw-SSE ridge penalty',
            'tie_break': 'smallest alpha'}


def cross_validate(per_user, dim, n_joints):
    users = sorted(per_user)
    require(len(users) >= 2, 'need two training users')
    halves = [users[::2], users[1::2]]
    aggregates = [merge_stats([per_user[u] for u in half], n_joints) for half in halves]
    solvers = [prepare_solver(s, dim, n_joints) for s in aggregates]
    def evaluate(alpha):
        scores = [score(aggregates[1-i], *solvers[i](alpha)) for i in range(2)]
        return sum(s[0] for s in scores) / sum(s[1] for s in scores)
    result = choose_alpha(evaluate)
    result['fold_users'] = halves
    return result


def paired_groups(head, ridge, n_boot=D.N_BOOT, seed=D.SEED):
    out = {}
    for group in ('dynamic', 'all', 'static'):
        h = head['groups'][group]['centered_support_n_ge_2']['per_user']
        r = ridge['groups'][group]['centered_support_n_ge_2']['per_user']
        users = sorted(h)
        require(users == sorted(r), 'paired users differ')
        for u in users:
            require(all(h[u][k] == r[u][k] for k in ('n_windows', 'n_window_joints', 'n_valid_elements')), 'paired supports differ')
            require(np.isclose(h[u]['sums']['vary'], r[u]['sums']['vary'], rtol=1e-12, atol=1e-12), 'paired target variance differs')
        den = np.array([h[u]['sums']['vary'] for u in users])
        delta = np.array([r[u]['sums']['centered'] - h[u]['sums']['centered'] for u in users])
        row = {'users': users, 'n_bootstrap': n_boot, 'seed': seed, 'estimand': '(pooled centered MSE ridge - head) / pooled target temporal variance',
               'delta_ratio': float(delta.sum()/den.sum()) if den.sum() > 0 else None,
               'per_user': {u: {'delta_centered_sse': float(d), 'target_variance_sum': float(v), 'delta_ratio': float(d/v) if v > 0 else None} for u, d, v in zip(users, delta, den)}}
        if len(users) < 2:
            row.update(evaluable=False, reason='fewer than two users')
        else:
            draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
            denominators, numerators = den[draws].sum(axis=1), delta[draws].sum(axis=1)
            valid = (denominators > 0) & np.isfinite(denominators) & np.isfinite(numerators)
            if not valid.all():
                row.update(evaluable=False, reason='invalid replica; no replicas dropped', invalid_replicates=int((~valid).sum()))
            else:
                ci = np.percentile(numerators/denominators, [2.5, 97.5]).tolist()
                row.update(evaluable=True, delta_ratio_ci95=ci, interpretation='ridge lower' if ci[1] < 0 else 'ridge higher' if ci[0] > 0 else 'inconclusive, not equivalence')
        out[group] = row
    return out


def check_reference(head, reference):
    for k in ('n_windows', 'n_users', 'n_dynamic'):
        require(head[k] == reference[k], 'reference count differs: '+k)
    for group in ('all', 'dynamic', 'static'):
        for support in ('total_support_n_ge_1', 'centered_support_n_ge_2'):
            a, b = head['groups'][group][support], reference['groups'][group][support]
            require(set(a['per_user']) == set(b['per_user']), 'reference users differ')
            for u in a['per_user']:
                aa, bb = a['per_user'][u], b['per_user'][u]
                require(all(aa[k] == bb[k] for k in ('n_windows', 'n_window_joints', 'n_valid_elements')), 'reference support differs')
                require(all(np.isclose(v, bb['sums'][k], rtol=1e-6, atol=1e-8) for k, v in aa['sums'].items()), 'reference moments differ')
    return {'consistent': True, 'rtol': 1e-6, 'atol': 1e-8, 'checked': 'all per-user moment sums and support counts'}


def process_blocks(data, head, callback, batch=128):
    checks = []
    for start in range(0, len(data['meta']), batch):
        sl = slice(start, start+batch)
        x, check = features_and_check(head, data['lat'][sl], data['pred'][sl], data['mask'][sl])
        checks.append(check)
        callback(sl, x)
    count = sum(c['n'] for c in checks)
    return {'blocks': len(checks), 'batch_windows': batch, 'max_abs': max(c['max_abs'] for c in checks),
            'rmse': float(np.sqrt(sum(c['sse'] for c in checks)/count)), 'n_valid_elements': count,
            'thresholds_per_block': {'max_abs': M2_MAX_ABS, 'rmse': M2_RMSE}, 'passed': True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'val', 'checkpoint', 'out', 'predictions-out', 'reference-dynamics', 'train-provenance', 'val-provenance'):
        ap.add_argument('--'+name, type=Path, required=True)
    for name in ('expected-train-sha256', 'analysis-source-sha256', 'wrapper-sha256'):
        ap.add_argument('--'+name)
    args = ap.parse_args(argv)
    D.fresh_destination(args.out)
    D.fresh_destination(args.predictions_out)
    require(args.out.resolve() != args.predictions_out.resolve(), 'outputs must differ')
    for key in ('analysis_source_sha256', 'wrapper_sha256', 'expected_train_sha256'):
        value = getattr(args, key)
        require(value is None or D.is_sha(value), 'invalid '+key)
    source_sha = D.file_sha256(__file__)
    require(args.analysis_source_sha256 in (None, source_sha), 'analysis source SHA mismatch')
    require(D.file_sha256(D.__file__) == DYNAMICS_SHA, 'frozen dynamics source SHA mismatch')
    tp, vp = [json.loads(p.read_text()) for p in (args.train_provenance, args.val_provenance)]
    train_sha = args.expected_train_sha256 or tp.get('cache_sha256')
    require(D.is_sha(train_sha) and D.file_sha256(args.train) == train_sha, 'train cache SHA mismatch or unavailable')
    require(tp.get('cache_sha256', train_sha) == train_sha, 'train provenance SHA mismatch')
    require(D.file_sha256(args.val) == VAL_SHA, 'val cache SHA mismatch')
    head = load_head(args.checkpoint)
    train, val = load_cache(args.train), load_cache(args.val)
    train_users = check_cache(train, args.train, tp, 'train', args.checkpoint)
    val_users = check_cache(val, args.val, vp, 'val', args.checkpoint)
    require(train_users.isdisjoint(val_users), 'train/val user leakage')
    require(train['lat'].shape[2] == val['lat'].shape[2] == head.out.in_features, 'feature dimension mismatch')
    for key in ('pose_root', 'per_side'):
        require(train['config'].get(key) == val['config'].get(key), 'train/val preprocessing mismatch: '+key)
    per_user_parts = {}
    users = np.asarray([m['user'] for m in train['meta']])
    def collect(sl, x):
        for u in sorted(set(users[sl].tolist())):
            selected = users[sl] == u
            yy, mm = train['target'][sl][selected], train['mask'][sl][selected]
            stats = sufficient(x[selected].reshape(-1, x.shape[-1]), yy.reshape(-1, 20), mm.reshape(-1, 20))
            if u in per_user_parts:
                stats = merge_stats([per_user_parts[u], stats], 20)
            per_user_parts[u] = stats
    m2_train = process_blocks(train, head, collect)
    dim = head.out.in_features
    selection = cross_validate(per_user_parts, dim, 20)
    total_stats = merge_stats(list(per_user_parts.values()), 20)
    weights, bias = prepare_solver(total_stats, dim, 20)(selection['alpha'])
    target_means = np.empty(20)
    for row in total_stats:
        target_means[row['joints']] = row['sy']/row['n']
    train_support = {'n_users': len(per_user_parts), 'n_recordings_requested': train['n_recordings'],
                     'n_windows': len(users), 'n_valid_elements': int(train['mask'].sum()),
                     'n_rows_any_valid': int(train['mask'].any(axis=2).sum()),
                     'n_rows_partial_joints': int((train['mask'].any(axis=2) & ~train['mask'].all(axis=2)).sum()),
                     'n_valid_per_joint': train['mask'].sum(axis=(0, 1)).tolist(),
                     'target_mean_per_joint': target_means.tolist()}
    del train
    pred = np.empty(val['pred'].shape, dtype=np.float64)
    def predict(sl, x):
        pred[sl] = x.astype(np.float64) @ weights + bias
    m2_val = process_blocks(val, head, predict)
    del val['lat']
    analysis_head = D.analyze(val)
    reference = json.loads(args.reference_dynamics.read_text())
    require(reference['provenance']['cache_sha256'] == VAL_SHA and reference['provenance']['checkpoint_sha256'] == CHECKPOINT_SHA, 'reference provenance mismatch')
    reference_check = check_reference(analysis_head, reference)
    analysis_ridge = D.analyze({**val, 'pred': pred})
    analysis_constant = D.analyze({**val, 'pred': np.broadcast_to(target_means, pred.shape)})
    analysis_ridge['scope'] = 'Descriptive decomposition of masked ridge predictions fitted on the fixed train subset; no pass/fail criterion.'
    analysis_constant['scope'] = 'Secondary descriptive constant: masked target mean per joint fitted on the fixed train subset; no pass/fail criterion.'
    val_rows = val['mask'].any(axis=2)
    result = {'schema_version': 1, 'scope': 'Descriptive B5000 accessibility; no continuation gate and no equivalence claim.',
              'features': 'FP32 frozen student PoseHead LayerNorm on cached fp16 latents; float64 ridge sufficient statistics; no feature scaling',
              'layernorm_eps': head.norm.eps, 'alpha_selection': selection, 'train_support': train_support,
              'val_mask_audit': {'n_rows_any_valid': int(val_rows.sum()), 'n_rows_all_joints_valid': int(val['mask'].all(axis=2).sum()),
                                 'n_rows_partial_joints': int((val_rows & ~val['mask'].all(axis=2)).sum()),
                                 'n_valid_elements': int(val['mask'].sum()), 'n_valid_per_joint': val['mask'].sum(axis=(0, 1)).tolist()},
              'm2': {'train': m2_train, 'val': m2_val}, 'reference_head_check': reference_check,
              'head': analysis_head, 'ridge': analysis_ridge, 'fit_subset_constant_secondary': analysis_constant,
              'paired': paired_groups(analysis_head, analysis_ridge),
              'provenance': {'checkpoint_sha256': CHECKPOINT_SHA, 'checkpoint_step': 5000, 'pose_stats_sha256': STATS_SHA,
                             'train_cache_sha256': train_sha, 'val_cache_sha256': VAL_SHA,
                             'analysis_source_sha256': source_sha, 'dynamics_source_sha256': D.file_sha256(D.__file__),
                             'wrapper_sha256': args.wrapper_sha256, 'numpy_version': np.__version__,
                             'inputs': {k: {'path': str(getattr(args, k).resolve()), 'sha256': D.file_sha256(getattr(args, k))}
                                        for k in ('train_provenance', 'val_provenance', 'reference_dynamics')}}}
    # Complete all scientific checks before any output creation. Exclusive modes
    # prevent overwrites; the prediction archive deliberately omits huge latents.
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open('xb') as f:
        np.savez_compressed(f, pred=pred, pred_head=val['pred'], target=val['target'], mask=val['mask'], std=val['std'],
                            pose_var=val['pose_var'], has_head=np.asarray(True), n_recordings=np.asarray(val['n_recordings']),
                            meta_json=np.asarray(json.dumps(val['meta'])), config_json=np.asarray(json.dumps(val['config'])),
                            provenance_json=np.asarray(json.dumps(result['provenance'])), ridge_weights=weights, ridge_bias=bias)
    result['provenance']['predictions'] = {'path': str(args.predictions_out), 'sha256': D.file_sha256(args.predictions_out)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(args.out), 'alpha': selection['alpha'], 'paired_dynamic': result['paired']['dynamic']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
