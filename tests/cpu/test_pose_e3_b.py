"""Small mathematical regressions for the standalone B ridge diagnostic."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import pose_e3_b as E
from wearusfm.model.pose_head import PoseHead


def dataset(seed=3, rows=100, dim=4, joints=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, dim))
    w, b = rng.normal(size=(dim, joints)), rng.normal(size=joints)
    y = x @ w + b
    mask = rng.random(y.shape) > .25
    return x, y, mask, w, b


def test_masked_exact_recovery_and_invalid_targets_irrelevant():
    x, y, mask, w, b = dataset()
    stats = E.sufficient(x, y, mask)
    got = E.prepare_solver(stats, 4, 3)(1e-10)
    np.testing.assert_allclose(got[0], w, atol=1e-9)
    np.testing.assert_allclose(got[1], b, atol=1e-9)
    changed = y.copy()
    changed[~mask] = np.nan
    other = E.prepare_solver(E.sufficient(x, changed, mask), 4, 3)(1e-10)
    np.testing.assert_array_equal(got[0], other[0])
    np.testing.assert_array_equal(got[1], other[1])
    sse, n = E.score(stats, *got)
    assert sse < 1e-9 and n == mask.sum()


def test_mask_groups_merge_exact_and_score_matches_direct():
    x, y, mask, _, _ = dataset()
    mask[:, 1] = mask[:, 0]
    chunks = [E.sufficient(x[:31], y[:31], mask[:31]), E.sufficient(x[31:], y[31:], mask[31:])]
    merged = E.merge_stats(chunks, 3)
    assert len(merged) == 2
    w, b = E.prepare_solver(merged, 4, 3)(2.)
    raw = E.prepare_solver(E.sufficient(x, y, mask), 4, 3)(2.)
    np.testing.assert_allclose(w, raw[0], atol=1e-13)
    np.testing.assert_allclose(b, raw[1], atol=1e-13)
    sse, n = E.score(merged, w, b)
    assert sse == pytest.approx(np.square(x@w+b-y)[mask].sum(), abs=1e-10)
    assert n == mask.sum()


def test_merge_different_masks_between_chunks():
    x, y, mask, _, _ = dataset()
    mask[:50, 1] = mask[:50, 0]
    stats = E.merge_stats([E.sufficient(x[:50], y[:50], mask[:50]), E.sufficient(x[50:], y[50:], mask[50:])], 3)
    w, b = E.prepare_solver(stats, 4, 3)(1.)
    ref = E.prepare_solver(E.sufficient(x, y, mask), 4, 3)(1.)
    np.testing.assert_allclose(w, ref[0], atol=1e-13)
    np.testing.assert_allclose(b, ref[1], atol=1e-13)


def test_ln_nonidentity_and_fp16_head_check_wrong_weights_fail():
    head = PoseHead(4, 3).eval()
    lat = np.array([[[2., 4., 8., 16.], [3., 7., 9., 20.]]], dtype=np.float16)
    with torch.no_grad():
        head.out.weight.copy_(torch.tensor([[.1,.2,.3,.4], [.3,-.2,.1,.6], [.2,.2,-.1,.1]]))
        pred = head(torch.tensor(lat.astype(np.float32))).numpy()
    mask = np.ones(pred.shape, dtype=bool)
    x, check = E.features_and_check(head, lat, pred, mask)
    assert not np.allclose(x, lat)
    np.testing.assert_allclose(x.mean(axis=-1), 0., atol=1e-7)
    manual = (lat.astype(np.float64)-lat.mean(axis=-1, keepdims=True))/np.sqrt(lat.astype(np.float64).var(axis=-1, keepdims=True)+head.norm.eps)
    np.testing.assert_allclose(x, manual, rtol=1e-6, atol=1e-7)
    assert check['max_abs'] == 0
    with torch.no_grad():
        head.out.bias.add_(.02)
    with pytest.raises(ValueError, match='M2 failed'):
        E.features_and_check(head, lat, pred, mask)


def test_ln_learned_affine_and_m2_ignores_invalid_entries():
    head = PoseHead(4, 3)
    with torch.no_grad():
        head.norm.weight.copy_(torch.tensor([2.,3.,4.,5.]))
        head.norm.bias.copy_(torch.tensor([.1,.2,.3,.4]))
        head.out.weight.fill_(.01)
    lat = np.arange(8, dtype=np.float16).reshape(1,2,4)
    with torch.no_grad():
        pred = head(torch.tensor(lat.astype(np.float32))).numpy()
    mask = np.ones(pred.shape, bool)
    mask[:,1,:] = False
    lat[:,1,:], pred[:,1,:] = np.nan, np.nan
    x, c = E.features_and_check(head, lat, pred, mask)
    assert c['max_abs'] == 0 and np.isfinite(x).all()


@pytest.mark.parametrize('optimum,alpha,unresolved,side', [(2.,100.,False,None), (8.,1e7,True,'upper'), (-8.,1e-5,True,'lower'), (5.,1e5,False,None)])
def test_alpha_boundaries(optimum, alpha, unresolved, side):
    r = E.choose_alpha(lambda a: (np.log10(a)-optimum)**2)
    assert r['alpha'] == alpha and r['boundary_unresolved'] == unresolved and r['boundary'] == side
    assert all(n <= 3 for n in r['extensions'].values())


def test_cv_fold_centering_train_only_matches_direct_refits():
    users = {}
    raws = {}
    for i in range(4):
        x, y, mask, _, _ = dataset(i)
        x += i*20
        y += i*7
        raws[str(i)] = x, y, mask
        users[str(i)] = E.sufficient(x, y, mask)
    cv = E.cross_validate(users, 4, 3)
    assert cv['fold_users'] == [['0','2'], ['1','3']]
    for entry in cv['grid']:
        total, n = 0., 0
        for i in (0,1):
            fit, held = cv['fold_users'][i], cv['fold_users'][1-i]
            x, y, m = [np.concatenate([raws[u][k] for u in fit]) for k in range(3)]
            xv, yv, mv = [np.concatenate([raws[u][k] for u in held]) for k in range(3)]
            for j in range(3):
                a, target = x[m[:,j]], y[m[:,j],j]
                mx, my = a.mean(axis=0), target.mean()
                centered = a-mx
                w = np.linalg.solve(centered.T@centered+entry['alpha']*np.eye(4), centered.T@(target-my))
                pred = (xv-mx)@w+my
                total += ((pred-yv[:,j])[mv[:,j]]**2).sum()
                n += mv[:,j].sum()
        assert entry['mse'] == pytest.approx(total/n, rel=1e-9)


def analyses(pred, target, mask):
    data = {'pred': pred, 'target': target, 'mask': mask, 'std': np.ones((len(pred),pred.shape[-1])),
            'pose_var': np.array([1.,2.,1.,2.]), 'meta': [{'user':u} for u in ('a','a','b','b')]}
    return E.D.analyze(data)


def test_paired_ci_manual_and_inconclusive_not_equivalent():
    y = np.arange(24.).reshape(4,3,2)
    mask = np.ones(y.shape, bool)
    head = analyses(y*.5, y, mask)
    ridge = analyses(y*.75, y, mask)
    paired = E.paired_groups(head, ridge)
    for row in paired.values():
        assert row['delta_ratio'] == pytest.approx(-.1875)
        np.testing.assert_allclose(row['delta_ratio_ci95'], [-.1875,-.1875])
        assert row['interpretation'] == 'ridge lower'
    same = E.paired_groups(head, head)
    assert same['dynamic']['interpretation'] == 'inconclusive, not equivalence'


def test_paired_zero_variance_no_dropped_replicas():
    y = np.ones((4,3,2))
    a = analyses(y,y,np.ones(y.shape,bool))
    r = E.paired_groups(a,a)
    assert not r['dynamic']['evaluable'] and r['dynamic']['invalid_replicates'] == 1000


def test_psd_violation_rejected_and_missing_joint_support():
    x,y,m,_,_ = dataset()
    s = E.sufficient(x,y,m)
    s[0]['xx'] *= -1
    with pytest.raises(ValueError, match='PSD'):
        E.prepare_solver(s,4,3)
    m[:,0] = False
    with pytest.raises(ValueError, match='without fitting support'):
        E.prepare_solver(E.sufficient(x,y,m),4,3)


def test_complete_cli_tiny_cache_provenance_and_prediction_archive(tmp_path, monkeypatch):
    import json
    rng = np.random.default_rng(89)
    head = PoseHead(4)
    with torch.no_grad():
        head.out.weight.copy_(torch.from_numpy(rng.normal(0,.1,(20,4)).astype(np.float32)))
    checkpoint = tmp_path/'checkpoint.pt'
    cfg = {'model': {'dim': 4, 'pose_head': True}}
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    torch.save({'step':5000, 'pose_stats_sha256':E.STATS_SHA, 'config':cfg,
                'student':{'pose_head.'+k:v for k,v in head.state_dict().items()}},checkpoint)
    monkeypatch.setattr(E,'CHECKPOINT_SHA',E.D.file_sha256(checkpoint))
    monkeypatch.setattr(E,'CONFIG_SHA',E.D.file_sha256(tmp_path/'config.json'))
    paths = {}
    for split, count, records in [('train',40,200),('val',15,1950)]:
        lat = rng.normal(size=(count,4,4)).astype(np.float16)
        with torch.no_grad():
            pred = head(torch.tensor(lat.astype(np.float32))).numpy()
        target = (pred*1.5+rng.normal(0,.1,pred.shape)).astype(np.float32)
        mask = np.ones(pred.shape,bool)
        mask[::2,0,0] = False
        config = {'which':'student','split':split,'code_sha':E.CODE_SHA,'pose_stats_sha256':E.STATS_SHA,
                  'checkpoint_pose_stats_sha256':E.STATS_SHA,'checkpoint':str(checkpoint),
                  'n_windows':10,'window_s':4.,'seed':20261026,'per_side':False,'pose_root':'/test'}
        meta = [{'user':split+str(i), 'recording':i} for i in range(count)]
        path = tmp_path/(split+'.npz')
        np.savez(path,lat=lat,pred=pred,target=target,mask=mask,std=np.ones((count,20)),
                 pose_var=np.arange(count,dtype=float),meta_json=np.asarray(json.dumps(meta)),
                 config_json=np.asarray(json.dumps(config)),has_head=np.asarray(True),n_recordings=np.asarray(records))
        provenance = {'mode':'extract' if split=='train' else 'report','cache':str(path),
                      'cache_sha256':E.D.file_sha256(path),'cache_bytes':path.stat().st_size,
                      'checkpoint_sha256':E.CHECKPOINT_SHA,'checkpoint_step':5000,'checkpoint':str(checkpoint),
                      'pose_stats_sha256':E.STATS_SHA,'which':'student','split':split,'execution_code_sha':E.CODE_SHA,
                      'config':config,'n_recordings':records,'n_users':count,'n_windows':count,
                      'config_file_sha256':E.CONFIG_SHA,'mask_audit':{'selected_any_rows':int(mask.any(axis=2).sum()),
                      'partial_joint_rows':int((mask.any(axis=2)&~mask.all(axis=2)).sum()),
                      'valid_elements':int(mask.sum()),'invalid_elements_excluded':int(mask.size-mask.sum())},
                      'r3c_subset':{'identical_window_metadata':True}, 'n_recordings_requested':records,
                      'n_windows_with_valid_pose':count,'n_users_with_valid_pose':count,'n_recordings_with_windows':count}
        pp = tmp_path/(split+'_provenance.json')
        pp.write_text(json.dumps(provenance))
        paths[split] = path
    monkeypatch.setattr(E,'VAL_SHA',E.D.file_sha256(paths['val']))
    reference = E.D.analyze(E.D.load_cache(paths['val']))
    reference['provenance'] = {'cache_sha256':E.VAL_SHA,'checkpoint_sha256':E.CHECKPOINT_SHA}
    rp = tmp_path/'reference.json'
    rp.write_text(json.dumps(reference))
    out, predictions = tmp_path/'result.json', tmp_path/'predictions.npz'
    args = ['--train',str(paths['train']),'--val',str(paths['val']),'--checkpoint',str(checkpoint),
            '--out',str(out),'--predictions-out',str(predictions),'--reference-dynamics',str(rp),
            '--train-provenance',str(tmp_path/'train_provenance.json'),
            '--val-provenance',str(tmp_path/'val_provenance.json')]
    assert E.main(args) == 0
    report = json.loads(out.read_text())
    assert report['reference_head_check']['consistent'] and report['m2']['train']['passed']
    assert report['val_mask_audit']['n_rows_partial_joints'] == 8
    assert report['train_support']['n_rows_partial_joints'] == 20
    assert 'fitted' in report['ridge']['scope']
    with np.load(predictions) as z:
        assert 'lat' not in z.files and 'pred_head' in z.files
    assert E.D.load_cache(predictions)['pred'].dtype == np.float64
    with pytest.raises(ValueError,match='already exists'):
        E.main(args)


def test_wrong_train_hash_stops_before_loading_cache(tmp_path, monkeypatch):
    import json
    train = tmp_path/'train.npz'
    train.write_bytes(b'not an archive')
    provenance = tmp_path/'prov.json'
    provenance.write_text(json.dumps({'cache_sha256':'0'*64}))
    with pytest.raises(ValueError,match='train cache SHA'):
        E.main(['--train',str(train),'--val',str(train),'--checkpoint',str(train),
                '--out',str(tmp_path/'out.json'),'--predictions-out',str(tmp_path/'pred.npz'),
                '--reference-dynamics',str(provenance),'--train-provenance',str(provenance),
                '--val-provenance',str(provenance)])
    assert not (tmp_path/'out.json').exists() and not (tmp_path/'pred.npz').exists()
