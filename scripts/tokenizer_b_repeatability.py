#!/usr/bin/env python3
"""Local descriptive T25-L minus B19618 comparison of already fitted OOF.

Never fits models or accesses the network. A single pair of training runs cannot
estimate a training-noise floor. Shared draws follow the frozen tokenizer report.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/infra/tokenizer_evaluation_20261103'
ARMS = ('T25-L', 'B19618')
WEIGHTS = {'p1': {'ninapro_db2': 8/13, 'ninapro_db3': 3/13, 'ninapro_db6': 2/13},
           'p2': {'epn612': .5, 'uci_emg': .5}}
C_GRID = [.001, .01, .1, 1., 10., 100.]
N_BOOT, SEED = 2000, 20261102


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(*arrays):
    # Reproduce the published dtype/shape/bytes convention, independently.
    h = hashlib.sha256()
    for a in arrays:
        require(not a.dtype.hasobject, 'object array not accepted')
        h.update(json.dumps([a.dtype.str, a.shape]).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
        h.update(b'\0')
    return h.hexdigest()


def pinned(path, expected):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, f'SHA256 mismatch: {path}')
    return raw


def labels(a, name):
    require(a.ndim == 1 and len(a) > 0, f'invalid {name}')
    if a.dtype.kind in 'SU':
        a = a.astype(str)
        require(all(s.strip() for s in a), f'blank {name}')
    else:
        require(a.dtype.kind in 'biuf' and np.isfinite(a).all(), f'invalid numeric {name}')
    return a


def normalize(a):
    require(set(a) == {'row', 'y', 'pred', 's', 'fold'}, 'wrong array keys')
    for k in ('y', 'pred', 's'):
        a[k] = labels(a[k], k)
    require(a['s'].dtype.kind == 'U', 'subjects must be strings')
    require((a['y'].dtype.kind == 'U') == (a['pred'].dtype.kind == 'U'), 'native label family mismatch')
    n = len(a['row'])
    require(n > 0 and all(v.shape == (n,) for v in a.values()), 'array shapes differ')
    require(a['row'].dtype.kind in 'iu' and np.all(a['row'] >= 0)
            and len(np.unique(a['row'])) == n, 'invalid/duplicate rows')
    require(a['fold'].dtype.kind in 'iu' and np.isin(a['fold'], range(4)).all(), 'invalid folds')
    order = np.argsort(a['row'], kind='stable')
    return {k: v[order] for k, v in a.items()}


def paired(a, b):
    for k in ('row', 'y', 's', 'fold'):
        require(np.array_equal(a[k], b[k]), f'OOF pairing differs: {k}')


def roles_and_audit(report, a, ds, records):
    folds = report['datasets'][ds]['folds']
    require(sorted(f['fold'] for f in folds) == list(range(4)), 'four unique folds required')
    universe = set(a['s'])
    output = []
    for f in sorted(folds, key=lambda f: f['fold']):
        roles = {k: sorted(f[k]) for k in ('train', 'val', 'held')}
        sets = [set(roles[k]) for k in roles]
        require(all(len(roles[k]) == len(set(roles[k])) for k in roles), 'duplicate role subject')
        require(set.union(*sets) == universe and not any(sets[i] & sets[j] for i,j in ((0,1),(0,2),(1,2))), 'invalid role partition')
        held = a['fold'] == f['fold']
        require(set(a['s'][held]) == sets[2] and np.sum(held) == f['n_held'], 'held subjects/count mismatch')
        fits = [r for r in records if r['dataset'] == ds and r['fold'] == f['fold']]
        require(len(fits) == 7, 'need six candidate fits and refit')
        candidates = [r for r in fits if r['fit_phase'] == 'candidate']
        refits = [r for r in fits if r['fit_phase'] == 'refit']
        require(sorted(r['C'] for r in candidates) == C_GRID and len(refits) == 1, 'candidate/refit coverage mismatch')
        selected = max(candidates, key=lambda r: (r['val_bacc'], -r['C']))
        require(selected['C'] == f['c'] == refits[0]['C'], 'selection/refit mismatch')
        for r in fits:
            require(r['roles_sha256'] == f['roles_sha256'] and r['scaler_sha256'] == f['scaler_sha256'], 'fit role/scaler mismatch')
            require(r['converged_by_criterion'] is True and not r['reached_limit']
                    and r['convergence_warnings'] == 0 and not r['anomalous_stop'] and r['exception'] is None, 'unconverged fit')
            require(r['max_iter'] == 12000 and 0 < r['n_iter'] < 12000 and r['n_features'] == 768
                    and r['dtype'] == 'float32' and r['scale'] is True and r['n_train'] == f['n_train'], 'fit recipe differs')
            params = r['solver_params']
            require(all(params[k] == v for k,v in {'solver':'lbfgs','penalty':'l2','class_weight':'balanced','tol':1e-4,'random_state':0,'max_iter':12000,'C':r['C']}.items()), 'solver params differ')
            require(np.isin(a['y'][held], r['classes']).all() and np.isin(a['pred'][held], r['classes']).all(), 'held class outside train class mapping')
        output.append(roles)
    return output


def count(a):
    users, ui = np.unique(a['s'], return_inverse=True)
    classes, ci = np.unique(a['y'], return_inverse=True)
    den = np.zeros((len(users), len(classes)), dtype=np.int64)
    hit = np.zeros_like(den)
    np.add.at(den, (ui, ci), 1)
    np.add.at(hit, (ui, ci), (a['pred'] == a['y']).astype(np.int64))
    stats = {'classes': classes.tolist(), 'subjects': {s: {'n': den[i].tolist(), 'correct': hit[i].tolist()} for i,s in enumerate(users)}}
    return users.tolist(), den, hit, stats


def summary(points, draws):
    return {'cells': {arm: {'balanced_accuracy': float(points[i]), 'balanced_accuracy_percent': float(100*points[i]),
                             'ci95_percent': (100*np.quantile(draws[:,i], [.025,.975])).tolist()} for i,arm in enumerate(ARMS)},
            'contrast': {'name': 'T25-L_minus_B19618', 'delta_pp': float(100*(points[0]-points[1])),
                         'ci95_pp': (100*np.quantile(draws[:,0]-draws[:,1], [.025,.975])).tolist()}}


def run(manifest_path, frozen_path):
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    frozen_raw = frozen_path.read_bytes()
    frozen = json.loads(frozen_raw)
    require(frozen['complete'] and frozen['n_bootstrap'] == N_BOOT and frozen['seed'] == SEED, 'frozen comparison differs')
    require(len(manifest['files']) == 4 and {(x['arm'],x['task']) for x in manifest['files']} == {(a,t) for a in ARMS for t in WEIGHTS}, 'four source cells required')
    data, reports, all_roles, audits, sources = {}, {}, {}, {}, []
    for entry in manifest['files']:
        arm, task = entry['arm'], entry['task']
        report_path = ROOT / entry['report_path'] if arm == 'T25-L' else manifest_path.parent / f'B_{task}_pool.json'
        rr = pinned(report_path, entry['report_sha256']); r = json.loads(rr)
        require(r['complete'] is True and r['task'] == task and r['weights'] == WEIGHTS[task], 'report identity/weights differ')
        require(r['oof_sha256'] == entry['sha256'], 'report OOF hash differs')
        if arm == 'T25-L':
            require(r['arm'] == arm and r['step'] == 19618 and r['kind'] == 'aggregate', 'T25 report differs')
            require(r['recipe'] == frozen['recipe'], 'T25 recipe differs')
            require(any(x['sha256'] == entry['report_sha256'] for x in frozen['reports']), 'T25 report not in frozen comparison')
            audit_path, audit_hash = report_path.with_suffix('.audit.jsonl'), r['audit_sha256']
        else:
            require(r['encoder'] == 'B' and r['phase'] == 'B' and r['max_iter'] == 12000
                    and r['readout']['layer'] == 'pool' and r['readout']['k_requested'] == 0, 'B report differs')
            audit_path, audit_hash = manifest_path.parent / f'B_{task}_pool.json.audit.jsonl', r['audit']['sha256']
        raw_audit = pinned(audit_path, audit_hash)
        records = [json.loads(line) for line in raw_audit.splitlines() if line]
        require(len(records) == 28*len(WEIGHTS[task]), 'audit record count differs')
        npz_raw = pinned(entry['path'], entry['sha256'])
        require(len(npz_raw) == entry['bytes'], 'OOF byte count differs')
        with np.load(io.BytesIO(npz_raw), allow_pickle=False) as z:
            expected = {f'{ds}_{k}' for ds in WEIGHTS[task] for k in ('row','y','pred','s','fold')}
            require(set(z.files) == expected | ({'datasets'} if arm == 'B19618' else set()), 'unexpected OOF keys')
            if arm == 'B19618':
                require(set(z['datasets']) == set(WEIGHTS[task]), 'dataset list differs')
            for ds in WEIGHTS[task]:
                arrays = {k: z[f'{ds}_{k}'] for k in ('row','y','pred','s','fold')}
                rd = r['datasets'][ds]
                require(len(arrays['row']) == rd['n_oof_windows'], 'report row count differs')
                if arm == 'T25-L':
                    require(fingerprint(*(arrays[k] for k in ('row','y','s','fold'))) == rd['identity_sha256'], 'T25 identity hash differs')
                else:
                    require(fingerprint(arrays['row']) == rd['oof_rows_sha256'], 'B rows hash differs')
                a = normalize(arrays)
                data[(arm,ds)] = a
                all_roles[(arm,ds)] = roles_and_audit(r,a,ds,records)
        reports[(arm,task)] = r
        audits[f'{arm}/{task}'] = {'fits':len(records), 'converged_by_criterion':True,
                                  'max_iterations':max(x['n_iter'] for x in records), 'audit_sha256':audit_hash}
        sources.append({'arm':arm,'task':task,'oof_sha256':entry['sha256'],'oof_bytes':len(npz_raw),
                        'report_path':str(report_path.relative_to(ROOT)),'report_sha256':sha(rr),
                        'audit_path':str(audit_path.relative_to(ROOT)),'audit_sha256':audit_hash})
    rng = np.random.default_rng(SEED)
    datasets, points, samples = {}, {}, {}
    for ds in sorted(frozen['datasets']):
        a,b = (data[(arm,ds)] for arm in ARMS)
        paired(a,b)
        require(all_roles[(ARMS[0],ds)] == all_roles[(ARMS[1],ds)], 'cross-model roles differ')
        rows = [count(x) for x in (a,b)]
        users, den, _, _ = rows[0]
        require(rows[1][0] == users and np.array_equal(rows[1][1],den), 'subject counts differ')
        draw = rng.integers(0,len(users),size=(N_BOOT,len(users)))
        require(fingerprint(draw) == frozen['datasets'][ds]['draws_sha256'], 'frozen bootstrap draws differ')
        weights = np.zeros((N_BOOT,len(users)),dtype=np.int64)
        np.add.at(weights,(np.repeat(np.arange(N_BOOT),len(users)),draw.ravel()),1)
        denominators = weights @ den
        present = denominators > 0
        values = np.column_stack([np.divide(weights @ row[2],denominators,out=np.zeros_like(denominators,dtype=float),where=present).sum(1)/present.sum(1) for row in rows])
        p = np.asarray([np.mean(row[2].sum(0)/den.sum(0)) for row in rows])
        block = summary(p,values)
        require(np.allclose(block['cells']['T25-L']['ci95_percent'], frozen['datasets'][ds]['cells']['T25-L']['ci95_percent'],rtol=0,atol=1e-12), 'T25 frozen CI differs')
        require(abs(p[0]-frozen['datasets'][ds]['cells']['T25-L']['balanced_accuracy']) < 1e-12, 'T25 frozen point differs')
        task = next(t for t,w in WEIGHTS.items() if ds in w)
        for i,arm in enumerate(ARMS):
            expected = reports[(arm,task)]['datasets'][ds]['balanced_accuracy' if arm=='T25-L' else 's']
            require(abs(p[i]-expected) < 1e-12, f'{arm} source point differs')
        block.update({'subjects':users,'classes':rows[0][3]['classes'],
                      'n_oof_windows':len(a['row']), 'paired_identity_verified':True, 'roles_identical':True,
                      'draws_sha256':fingerprint(draw),'paired_counts':{arm:row[3] for arm,row in zip(ARMS,rows)},
                      'replicates_missing_classes':int(np.sum(present.sum(1)<den.shape[1])),
                      'single_class_replicates':int(np.sum(present.sum(1)==1)),
                      'missing_class_rule':'macro recall over target classes present in each draw; retain all draws'})
        datasets[ds], points[ds], samples[ds] = block,p,values
    tasks = {}
    for task,ws in WEIGHTS.items():
        p = sum(w*points[ds] for ds,w in ws.items())
        draws = sum(w*samples[ds] for ds,w in ws.items())
        tasks[task] = summary(p,draws)
        require(abs(p[0]-reports[('T25-L',task)]['balanced_accuracy']) < 1e-12, 'T25 task point differs')
        require(abs(p[1]-reports[('B19618',task)]['S']) < 1e-12, 'B S fraction differs')
        require(np.allclose(tasks[task]['cells']['T25-L']['ci95_percent'],frozen['tasks'][task]['cells']['T25-L']['ci95_percent'],rtol=0,atol=1e-12), 'T25 task CI differs')
    # Exercise the actual pairing guard, not a duplicate check.
    mutated = {k:v.copy() for k,v in next(iter(data.values())).items()}
    original = next(iter(data.values()))
    mutated['fold'][0] = (mutated['fold'][0]+1)%4
    try:
        paired(original,mutated)
    except ValueError as exc:
        guard_check = str(exc)
    else:
        raise AssertionError('pair mismatch was accepted')
    return {'protocol':'descriptive-existing-oof-t25-minus-b19618-20261104-v1','complete':True,
            'n_bootstrap':N_BOOT,'seed':SEED,'dataset_draw_order':sorted(datasets),'weights':WEIGHTS,
            'scope':'post hoc descriptive comparison of two existing fixed trained models and converged probes; no new fits or jobs',
            'uncertainty':'paired subject bootstrap conditional on one pair of fixed checkpoints and fitted probes; development validation; no multiplicity adjustment; does not estimate between-training variance or establish a noise floor',
            'decision_rule':'none; no noninferiority, promotion or new noise-floor threshold',
            'provenance':{'manifest_sha256':sha(raw_manifest),'frozen_comparison_sha256':sha(frozen_raw),
                          'script_sha256':sha(Path(__file__).read_bytes()),'sources':sources},
            'verification':{'all_280_fits_converged':sum(v['fits'] for v in audits.values())==280,
                            'all_pairing_roles_hashes_and_source_points_pass':True,
                            'T25_points_CIs_and_draws_match_frozen_report':True,
                            'B_S_stored_in_fraction_units_verified':True,
                            'pair_mismatch_guard_exercised':guard_check},
            'audit_summary':audits,'datasets':datasets,'tasks':tasks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=BASE/'t25_b_repeatability_20261104/collection_receipt.json')
    parser.add_argument('--frozen-report',type=Path,default=BASE/'probe_cpu_19618/final/comparison.json')
    parser.add_argument('--out',type=Path,default=BASE/'t25_b_repeatability_20261104/paired_report.json')
    args=parser.parse_args()
    require(not args.out.exists(),'output already exists; use a new output path')
    result=run(args.manifest,args.frozen_report)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'verification':result['verification'],'contrasts':{g:{k:v['contrast'] for k,v in result[g].items()} for g in ('datasets','tasks')}},indent=2))

if __name__=='__main__':
    main()
