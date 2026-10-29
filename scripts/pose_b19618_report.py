"""Frozen intermediate pose comparisons: B19618/B5000 and B19618/R3c19618.

No training or tuning on evaluation users. The common ridge transform is a
stateless FP32 LayerNorm, not either model's learned head transformation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pose_dynamics as D
import pose_e3_b as E

CODE_SHA = E.CODE_SHA
STATS_SHA = E.STATS_SHA
EPS = 1e-5
require = D.require


def same_targets(a, b):
    require(a['meta'] == b['meta'], 'paired window metadata differ')
    require(a['n_recordings'] == b['n_recordings'], 'paired recording counts differ')
    for key in ('target', 'mask', 'std', 'pose_var'):
        x, y = a[key], b[key]
        require(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True),
                'paired targets/support differ: ' + key)


def load_checked(path, model, split):
    from wearusfm.harness import pose_eval as PE
    ppath = Path(str(path) + '.provenance.json')
    p = json.loads(ppath.read_text())
    for key, value in {'model': model, 'checkpoint_step': 19618, 'code_sha': CODE_SHA,
                       'which': 'student', 'split': split, 'stats_sha256': STATS_SHA, 'mode': 'pose'}.items():
        require(p.get(key) == value, 'cache provenance mismatch: ' + key)
    require(Path(p['cache']).resolve() == path.resolve(), 'cache provenance path mismatch')
    require(D.file_sha256(path) == p['cache_sha256'] and path.stat().st_size == p['cache_bytes'], 'cache SHA/size mismatch')
    require(p.get('same_metadata_target_mask_std_pose_var_as_B5000') is True, 'missing paired extraction verification')
    receipt_path = path.parent/'extraction_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('complete') is True and receipt.get('checkpoint_unchanged') is True, 'extraction incomplete')
    require(receipt['outputs'][split]['sha256'] == p['cache_sha256'], 'extraction receipt/cache mismatch')
    require(receipt['outputs'][split]['provenance_sha256'] == D.file_sha256(ppath), 'provenance changed after extraction')
    for key in ('checkpoint', 'checkpoint_sha256', 'checkpoint_step', 'model', 'code_sha'):
        require(receipt[key] == p[key], 'extraction identity mismatch: ' + key)
    data = PE.load_npz(path)
    require(data['config'] == p['config'] and data['config']['which'] == 'student', 'cached config mismatch')
    require(data['config']['pose_stats_sha256'] == STATS_SHA, 'target statistics differ')
    require(data['config']['checkpoint_pose_stats_sha256'] == (STATS_SHA if model == 'B' else None), 'head statistics differ')
    require(data['has_head'] == (model == 'B'), 'unexpected head presence')
    pred, target, mask, lat = (data[k] for k in ('pred','target','mask','lat'))
    require(pred.shape == target.shape == mask.shape and mask.ndim == 3 and mask.shape[-1] == 20 and mask.dtype == bool,
            'invalid pose arrays')
    require(lat.shape == (*mask.shape[:2], 384) and lat.dtype == np.float16, 'invalid latent array')
    require(data['std'].shape == (len(pred),20) and np.isfinite(data['std']).all() and (data['std'] > 0).all(), 'invalid std')
    require(np.isfinite(target).all() and np.isfinite(pred).all(), 'nonfinite cached target/prediction')
    require(data['n_recordings'] == (200 if split == 'train' else 1950), 'wrong recording count')
    users = {m['user'] for m in data['meta']}
    require(len(users) == (40 if split == 'train' else 15), 'wrong user count')
    return data, {**p, 'provenance_sha256': D.file_sha256(ppath), 'extraction_receipt_sha256': D.file_sha256(receipt_path)}


def paired_sums(users, numerator, denominator, label, n_boot=1000, seed=20261009):
    """Ratio of pooled paired differences; never drop invalid bootstrap draws."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    require(numerator.shape == denominator.shape == (len(users),), 'paired sum shape mismatch')
    require(np.isfinite(numerator).all() and np.isfinite(denominator).all() and (denominator >= 0).all(), 'invalid paired sums')
    result = {'estimand': label, 'users': users, 'n_bootstrap': n_boot, 'seed': seed,
              'per_user': {u: {'numerator': float(n), 'denominator': float(d), 'ratio': float(n/d) if d > 0 else None}
                           for u,n,d in zip(users,numerator,denominator)}}
    result['delta'] = float(numerator.sum()/denominator.sum()) if denominator.sum() > 0 else None
    if len(users) < 2:
        return {**result, 'evaluable': False, 'reason': 'fewer than two users'}
    draws = np.random.default_rng(seed).integers(0,len(users),(n_boot,len(users)))
    den, num = denominator[draws].sum(1), numerator[draws].sum(1)
    valid = (den > 0) & np.isfinite(den) & np.isfinite(num)
    if not valid.all():
        return {**result, 'evaluable': False, 'reason': 'invalid replica; no replicas dropped',
                'invalid_replicates': int((~valid).sum())}
    ci = np.percentile(num/den,[2.5,97.5]).tolist()
    return {**result, 'evaluable': True, 'ci95': ci,
            'direction': 'negative' if ci[1] < 0 else 'positive' if ci[0] > 0 else 'inconclusive, not equivalence'}


def paired_e1(reference, candidate):
    a, b = [x['e1_dynamic_control']['per_user'] for x in (reference,candidate)]
    users = sorted(a)
    require(users == sorted(b), 'E1 paired users differ')
    for u in users:
        require(all(a[u][k] == b[u][k] for k in ('n_windows','n_valid_elements')), 'E1 supports differ')
        require(np.isclose(a[u]['constant_sum_rad'],b[u]['constant_sum_rad'],rtol=1e-12,atol=1e-12), 'E1 constants differ')
    return paired_sums(users,[b[u]['error_sum_rad']-a[u]['error_sum_rad'] for u in users],
                       [a[u]['constant_sum_rad'] for u in users], 'candidate minus reference E1 MAE/constant, dynamic windows')


def centered_pair(reference,candidate):
    result = E.paired_groups(reference,candidate)
    for row in result.values():
        row['estimand'] = '(candidate centered SSE - reference centered SSE) / target variance sum'
        if 'interpretation' in row:
            row['interpretation'] = row['interpretation'].replace('ridge lower','candidate lower error').replace('ridge higher','candidate higher error')
    return result


def descriptive_head(data, step):
    from wearusfm.harness import pose_eval as PE
    # PE.report without a ridge uses lat only for its first dimension. A zero-width
    # placeholder avoids reading B5000 latents; predictions/targets are unchanged.
    report = PE.report({**data, 'lat': np.empty((len(data['meta']),0,0),dtype=np.float16)})
    for key in ('learned','learned_substantially'):
        report['E1'].pop(key,None)
    report['E2'].pop('misaligned_ok',None)
    report.pop('rule',None)
    report['scope'] = f'Descriptive student head at step {step}; endpoint gates not applied.'
    support = report['E2'].get('paired_support',{}).get('per_user',{})
    users = sorted(support)
    report['E2']['gap_with_ci'] = paired_sums(users,
        [support[u]['gap']*support[u]['n_valid_elements'] for u in users],
        [support[u]['n_valid_elements'] for u in users], 'misaligned minus aligned MAE, radians, common dynamic support')
    return report


def paired_e2(reference,candidate):
    a, b = [x['E2'].get('paired_support',{}).get('per_user',{}) for x in (reference,candidate)]
    require(reference['E2'].get('pairing') == candidate['E2'].get('pairing'), 'E2 pair selection differs')
    users = sorted(a)
    require(users == sorted(b), 'E2 paired users differ')
    for u in users:
        require(all(a[u][k] == b[u][k] for k in ('n_windows','n_valid_elements','n_same_recording','n_other_session')),
                'E2 paired support differs')
    return paired_sums(users,[(b[u]['gap']-a[u]['gap'])*a[u]['n_valid_elements'] for u in users],
                       [a[u]['n_valid_elements'] for u in users], 'candidate minus reference E2 gap; positive is greater association, not a quality gate')


def common_features(lat,mask):
    import torch
    valid = mask.any(axis=-1)
    require(np.isfinite(lat[valid]).all(),'nonfinite valid features')
    safe = np.where(valid[...,None],lat,0).astype(np.float32)
    with torch.no_grad():
        return torch.nn.functional.layer_norm(torch.from_numpy(safe),(safe.shape[-1],),eps=EPS).numpy()


def load_b_head(provenance):
    import torch
    from wearusfm.model.pose_head import PoseHead
    path = Path(provenance['checkpoint'])
    require(D.file_sha256(path) == provenance['checkpoint_sha256'], 'B checkpoint changed')
    require(D.file_sha256(path.parent/'config.json') == provenance['config_file_sha256'], 'B config changed')
    state = torch.load(path,map_location='cpu',weights_only=False)
    require(state['step'] == 19618 and state['pose_stats_sha256'] == STATS_SHA, 'B head identity mismatch')
    head = PoseHead(state['config']['model']['dim']).float().eval()
    head.load_state_dict({k.removeprefix('pose_head.'):v for k,v in state['student'].items() if k.startswith('pose_head.')})
    require(all(bool(torch.isfinite(v).all()) for v in head.state_dict().values()), 'nonfinite head')
    return head


def fit_common(train,val,head=None,batch=128):
    train_users = np.asarray([m['user'] for m in train['meta']])
    val_users = {m['user'] for m in val['meta']}
    require(set(train_users).isdisjoint(val_users),'train/val leakage')
    dim,joints = train['lat'].shape[-1],train['target'].shape[-1]
    require(val['lat'].shape[-1] == dim,'feature dimension differs')
    parts,checks = {},{'train':[],'val':[]}
    for start in range(0,len(train_users),batch):
        sl = slice(start,start+batch)
        x = common_features(train['lat'][sl],train['mask'][sl])
        if head is not None:
            _,check = E.features_and_check(head,train['lat'][sl],train['pred'][sl],train['mask'][sl])
            checks['train'].append(check)
        for u in sorted(set(train_users[sl])):
            sel = train_users[sl] == u
            stats = E.sufficient(x[sel].reshape(-1,dim),train['target'][sl][sel].reshape(-1,joints),train['mask'][sl][sel].reshape(-1,joints))
            parts[u] = E.merge_stats([parts[u],stats],joints) if u in parts else stats
    selection = E.cross_validate(parts,dim,joints)
    weights,bias = E.prepare_solver(E.merge_stats(list(parts.values()),joints),dim,joints)(selection['alpha'])
    pred = np.empty(val['pred'].shape,dtype=np.float64)
    for start in range(0,len(val['meta']),batch):
        sl = slice(start,start+batch)
        x = common_features(val['lat'][sl],val['mask'][sl])
        if head is not None:
            _,check = E.features_and_check(head,val['lat'][sl],val['pred'][sl],val['mask'][sl])
            checks['val'].append(check)
        pred[sl] = x.astype(np.float64)@weights+bias
    m2 = {}
    for split,values in checks.items():
        if not values:
            m2[split] = {'applicable':False,'reason':'R3c has no pose head'}
        else:
            n = sum(v['n'] for v in values)
            m2[split] = {'applicable':True,'passed':True,'blocks':len(values),'n':n,
                         'max_abs':max(v['max_abs'] for v in values),
                         'rmse':float(np.sqrt(sum(v['sse'] for v in values)/n)),
                         'thresholds_per_block':{'max_abs':E.M2_MAX_ABS,'rmse':E.M2_RMSE}}
    return pred,selection,m2


def analyze(data,label):
    result = D.analyze(data)
    result['scope'] = label + '; descriptive, no endpoint gate or automatic continuation.'
    return result


def prediction_cache(path,data,pred):
    with path.open('xb') as f:
        np.savez_compressed(f,pred=pred,target=data['target'],mask=data['mask'],std=data['std'],pose_var=data['pose_var'],
            meta_json=np.asarray(json.dumps(data['meta'])),config_json=np.asarray(json.dumps({**data['config'],
            'prediction_source':'common stateless LayerNorm then train-only ridge','normalization_eps':EPS})),
            has_head=np.asarray(bool(data['has_head'])),n_recordings=np.asarray(data['n_recordings']))


def json_safe(value):
    if isinstance(value,dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('b-dir','r3c-dir','baseline','out','predictions-dir'):
        ap.add_argument('--'+key,type=Path,required=True)
    for key in ('source-sha256','wrapper-sha256'):
        ap.add_argument('--'+key,required=True)
    a = ap.parse_args(argv)
    require(os.environ.get('WEARUSFM_CODE_SHA') == CODE_SHA,'wrong frozen execution code')
    require(D.file_sha256(__file__) == a.source_sha256 and D.is_sha(a.wrapper_sha256),'invalid source/wrapper SHA')
    require(D.file_sha256(D.__file__) == E.DYNAMICS_SHA,'dynamics source changed')
    require(D.file_sha256(E.__file__) == '859dc19b7867a9597fe7fe868cde8833471a1db7da018113809e6a6fda831782','ridge source changed')
    D.fresh_destination(a.out)
    D.fresh_destination(a.predictions_dir)
    require(a.out.resolve() != a.predictions_dir.resolve() and a.predictions_dir.resolve() not in a.out.resolve().parents,
            'overlapping outputs')
    require(D.file_sha256(a.baseline) == E.VAL_SHA,'B5000 baseline changed')
    baseline = D.load_cache(a.baseline)
    baseline_analysis = analyze(baseline,'B5000 student head')
    baseline_head = descriptive_head(baseline,5000)
    btrain,btrain_prov = load_checked(a.b_dir/'train_subset.npz','B','train')
    bval,bval_prov = load_checked(a.b_dir/'val.npz','B','val')
    same_targets(baseline,bval)
    b_analysis = analyze(bval,'B19618 student head')
    b_head = descriptive_head(bval,19618)
    bp,bcv,bm2 = fit_common(btrain,bval,load_b_head(bval_prov))
    b_ridge = analyze({**bval,'pred':bp},'B19618 common-normalized ridge')
    rtrain,rtrain_prov = load_checked(a.r3c_dir/'train_subset.npz','R3c','train')
    same_targets(btrain,rtrain)
    del btrain
    rval,rval_prov = load_checked(a.r3c_dir/'val.npz','R3c','val')
    same_targets(bval,rval)
    rp,rcv,rm2 = fit_common(rtrain,rval)
    del rtrain
    require(bcv['fold_users'] == rcv['fold_users'],'ridge fold users differ')
    r_ridge = analyze({**rval,'pred':rp},'R3c19618 common-normalized ridge')
    result = {'schema_version':1,'scope':'Exploratory intermediate endpoint; no adoption, equivalence, plateau or automatic continuation claim.',
        'normalization':{'kind':'stateless LayerNorm per instant after mean K','affine':False,'eps':EPS,'dtype':'float32',
                         'limitation':'removes instantaneous mean/norm; not all raw linear accessibility'},
        'baseline_head':baseline_head,'candidate_head':b_head,
        'head_dynamics':{'B5000':baseline_analysis,'B19618':b_analysis},
        'head_evolution':{'E1':paired_e1(baseline_analysis,b_analysis),'centered':centered_pair(baseline_analysis,b_analysis),
                          'E2_gap':paired_e2(baseline_head,b_head)},
        'accessibility':{'B':{'selection':bcv,'analysis':b_ridge,'m2':bm2},'R3c':{'selection':rcv,'analysis':r_ridge,'m2':rm2},
                         'B_minus_R3c':{'E1':paired_e1(r_ridge,b_ridge),'centered':centered_pair(r_ridge,b_ridge)}},
        'provenance':{'source_sha256':a.source_sha256,'wrapper_sha256':a.wrapper_sha256,'code_sha':CODE_SHA,
                      'B5000_cache_sha256':E.VAL_SHA,'B_train':btrain_prov,'B_val':bval_prov,'R3c_train':rtrain_prov,'R3c_val':rval_prov,
                      'slurm_job_id':os.environ.get('SLURM_JOB_ID')}}
    # All scientific checks precede output creation; exclusive paths never overwrite.
    a.predictions_dir.mkdir(parents=True,exist_ok=False)
    result['prediction_outputs'] = {}
    for label,data,pred in [('B',bval,bp),('R3c',rval,rp)]:
        path = a.predictions_dir/(label+'_ridge.npz')
        prediction_cache(path,data,pred)
        result['prediction_outputs'][label] = {'path':str(path),'sha256':D.file_sha256(path)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:
        json.dump(json_safe(result),f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({'completed':True,'output':str(a.out),'head_E1_delta':result['head_evolution']['E1']['delta'],
                      'head_dynamic_delta':result['head_evolution']['centered']['dynamic']['delta_ratio'],
                      'accessibility_dynamic_delta':result['accessibility']['B_minus_R3c']['centered']['dynamic']['delta_ratio']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
