"""Numerical tests of the intermediate report; no checkpoint/model training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('b19618_report',SCRIPTS/'pose_b19618_report.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_paired_ratio_is_pooled_and_bootstrap_matches_manual():
    users=['a','b','c']
    numerator=np.array([-2.,3.,-10.])
    denominator=np.array([2.,30.,100.])
    got=R.paired_sums(users,numerator,denominator,'test')
    assert got['delta']==pytest.approx(-9/132)
    assert got['delta']!=pytest.approx(np.mean(numerator/denominator))
    rng=np.random.default_rng(20261009)
    values=[]
    for _ in range(1000):
        chosen=rng.choice(3,3,replace=True)
        values.append(numerator[chosen].sum()/denominator[chosen].sum())
    np.testing.assert_array_equal(got['ci95'],np.percentile(values,[2.5,97.5]))
    assert got['direction']=='inconclusive, not equivalence'


def test_paired_invalid_replicas_not_dropped():
    result=R.paired_sums(['a','b'],[0.,1.],[0.,1.],'test')
    assert not result['evaluable'] and result['invalid_replicates']>0
    assert 'no replicas dropped' in result['reason']
    assert not R.paired_sums(['a'],[1.],[1.],'test')['evaluable']
    for nums,dens in (([np.nan,1.],[1.,1.]),([1.,1.],[-1.,1.])):
        with pytest.raises(ValueError,match='invalid paired sums'):
            R.paired_sums(['a','b'],nums,dens,'test')


def test_paired_e1_sign_and_support_guard():
    def result(errors):
        return {'e1_dynamic_control':{'per_user':{u:{'n_windows':i+1,'n_valid_elements':10*(i+1),
                'error_sum_rad':errors[i],'constant_sum_rad':10*(i+1)} for i,u in enumerate(['a','b'])}}}
    reference,candidate=result([8.,16.]),result([4.,8.])
    got=R.paired_e1(reference,candidate)
    assert got['delta']==pytest.approx(-.4) and got['direction']=='negative'
    candidate['e1_dynamic_control']['per_user']['a']['n_valid_elements']+=1
    with pytest.raises(ValueError,match='supports differ'):
        R.paired_e1(reference,candidate)


def e2(gaps):
    return {'E2':{'pairing':{'n_kept':3,'seed':20261026},'paired_support':{'per_user':{
        u:{'gap':gap,'n_windows':i+1,'n_valid_elements':10*(i+1),'n_same_recording':i+1,'n_other_session':0}
        for i,(u,gap) in enumerate(zip(['a','b'],gaps))}}}}


def test_e2_candidate_minus_reference_positive_and_pairing_guards():
    reference,candidate=e2([.1,.2]),e2([.3,.3])
    got=R.paired_e2(reference,candidate)
    assert got['delta']==pytest.approx((.2*10+.1*20)/30)
    assert got['direction']=='positive'
    changed=deepcopy(candidate)
    changed['E2']['pairing']['n_kept']=4
    with pytest.raises(ValueError,match='pair selection differs'):
        R.paired_e2(reference,changed)
    changed=deepcopy(candidate)
    changed['E2']['paired_support']['per_user']['a']['n_other_session']=1
    with pytest.raises(ValueError,match='paired support differs'):
        R.paired_e2(reference,changed)


def test_stateless_ln_matches_torch_nonidentity_and_invalid_rows_are_zero():
    lat=np.array([[[1.,3.,8.,12.],[np.nan]*4]],np.float16)
    mask=np.array([[[True,False],[False,False]]])
    features=R.common_features(lat,mask)
    expected=torch.nn.functional.layer_norm(torch.tensor([[[1.,3.,8.,12.],[0.,0.,0.,0.]]]),(4,),eps=1e-5).numpy()
    np.testing.assert_array_equal(features,expected)
    assert features.dtype==np.float32 and not np.allclose(features[0,0],lat[0,0])
    mask[0,1,0]=True
    with pytest.raises(ValueError,match='nonfinite valid features'):
        R.common_features(lat,mask)


def fit_data(seed=4):
    rng=np.random.default_rng(seed)
    weights=rng.normal(size=(4,3))
    bias=rng.normal(size=3)
    def data(users):
        n=len(users)
        lat=rng.normal(size=(n,8,4)).astype(np.float16)
        mask=rng.random((n,8,3))>.2
        mask[:,0,:]=True
        x=R.common_features(lat,mask).astype(float)
        target=x@weights+bias
        return {'lat':lat,'target':target,'mask':mask,'pred':np.zeros(target.shape),
                'meta':[{'user':u} for u in users]}
    return data(['a','b','c','d']),data(['v','w'])


def test_common_fit_per_joint_direct_solution_no_val_leakage_and_headless_adapter():
    train,val=fit_data()
    pred,cv,m2=R.fit_common(train,val,batch=2)
    assert cv['fold_users']==[['a','c'],['b','d']]
    assert not m2['train']['applicable'] and not m2['val']['applicable']
    x=R.common_features(train['lat'],train['mask']).reshape(-1,4).astype(float)
    xv=R.common_features(val['lat'],val['mask']).astype(float)
    y,m=train['target'].reshape(-1,3),train['mask'].reshape(-1,3)
    for j in range(3):
        xx,yy=x[m[:,j]],y[m[:,j],j]
        mx,my=xx.mean(0),yy.mean()
        xc=xx-mx
        w=np.linalg.solve(xc.T@xc+cv['alpha']*np.eye(4),xc.T@(yy-my))
        np.testing.assert_allclose(pred[:,:,j],(xv-mx)@w+my,rtol=1e-9,atol=1e-9)
    changed_train,changed_val=deepcopy(train),deepcopy(val)
    changed_train['target'][~train['mask']]=np.nan
    changed_val['target'][:]=1e8
    p2,cv2,_=R.fit_common(changed_train,changed_val,batch=2)
    np.testing.assert_array_equal(pred,p2)
    assert cv==cv2
    changed_val['meta'][0]['user']='a'
    with pytest.raises(ValueError,match='leakage'):
        R.fit_common(train,changed_val)


def test_same_targets_rejects_single_changed_joint_support():
    train,val=fit_data()
    data={'meta':val['meta'],'n_recordings':2,'target':val['target'],'mask':val['mask'],
          'std':np.ones((2,3)),'pose_var':np.array([1.,2.])}
    R.same_targets(data,deepcopy(data))
    changed=deepcopy(data)
    changed['mask'][0,0,0]=False
    with pytest.raises(ValueError,match='support differ: mask'):
        R.same_targets(data,changed)


def head_data():
    rng=np.random.default_rng(92)
    n=12
    target=rng.normal(size=(n,5,20)).astype(np.float32)
    meta=[{'user':str(i//4),'session':str(i//4),'stage':'test','side':'left','recording':i//4,
           'start':(i%4)*10000,'fs':1000.} for i in range(n)]
    return {'pred':target*.7,'target':target,'mask':np.ones(target.shape,bool),'std':np.ones((n,20),np.float32),
            'pose_var':np.arange(n,dtype=np.float32),'meta':meta,'n_recordings':3,'config':{},'has_head':True}


def test_descriptive_head_frozen_pairing_prediction_independent_and_no_gate():
    data=head_data()
    first=R.descriptive_head(data,5000)
    second=R.descriptive_head({**data,'pred':data['target']*.9},19618)
    assert 'rule' not in first and 'learned' not in first['E1'] and 'learned_substantially' not in first['E1']
    assert 'misaligned_ok' not in first['E2']
    assert first['E2']['pairing']==second['E2']['pairing']
    paired=R.paired_e2(first,second)
    rows1,rows2=[r['E2']['paired_support']['per_user'] for r in (first,second)]
    expected=sum((rows2[u]['gap']-rows1[u]['gap'])*rows1[u]['n_valid_elements'] for u in rows1)/sum(r['n_valid_elements'] for r in rows1.values())
    assert paired['delta']==pytest.approx(expected)
    assert first['E2']['gap_with_ci']['delta']==pytest.approx(first['E2']['paired_support']['gap_mae'],abs=1e-6)


def test_centered_candidate_sign_and_no_equivalence_label():
    data=head_data()
    reference=R.analyze(data,'reference')
    candidate=R.analyze({**data,'pred':data['target']*.9},'candidate')
    paired=R.centered_pair(reference,candidate)
    assert paired['dynamic']['delta_ratio']==pytest.approx(.01-.09,abs=1e-8)
    assert paired['dynamic']['interpretation']=='candidate lower error'
    assert R.centered_pair(reference,reference)['dynamic']['interpretation']=='inconclusive, not equivalence'


def test_load_checked_headless_cache_and_receipt_hash_guard(tmp_path):
    n=15
    path=tmp_path/'val.npz'
    cfg={'which':'student','pose_stats_sha256':R.STATS_SHA,'checkpoint_pose_stats_sha256':None}
    np.savez(path,lat=np.ones((n,2,384),np.float16),pred=np.zeros((n,2,20),np.float32),
             target=np.ones((n,2,20),np.float32),mask=np.ones((n,2,20),bool),std=np.ones((n,20),np.float32),
             pose_var=np.ones(n,np.float32),meta_json=np.asarray(json.dumps([{'user':str(i)} for i in range(n)])),
             config_json=np.asarray(json.dumps(cfg)),has_head=np.asarray(False),n_recordings=np.asarray(1950),dim=np.asarray(384))
    prov={'model':'R3c','checkpoint_step':19618,'code_sha':R.CODE_SHA,'which':'student','split':'val',
          'stats_sha256':R.STATS_SHA,'mode':'pose','cache':str(path),'cache_sha256':R.D.file_sha256(path),
          'cache_bytes':path.stat().st_size,'same_metadata_target_mask_std_pose_var_as_B5000':True,
          'checkpoint':'/reference/step19618.pt','checkpoint_sha256':'a'*64,'config':cfg}
    pp=Path(str(path)+'.provenance.json')
    pp.write_text(json.dumps(prov))
    receipt={k:prov[k] for k in ('checkpoint','checkpoint_sha256','checkpoint_step','model','code_sha')}
    receipt.update(complete=True,checkpoint_unchanged=True,outputs={'val':{'sha256':prov['cache_sha256'],
                   'provenance_sha256':R.D.file_sha256(pp)}})
    (tmp_path/'extraction_receipt.json').write_text(json.dumps(receipt))
    data,identity=R.load_checked(path,'R3c','val')
    assert not data['has_head'] and data['lat'].shape==(15,2,384)
    assert identity['checkpoint_step']==19618
    prov['cache_bytes']+=1
    pp.write_text(json.dumps(prov))
    with pytest.raises(ValueError,match='SHA/size'):
        R.load_checked(path,'R3c','val')


def test_report_payload_reproduces_and_preserves_frozen_sources():
    import hashlib
    import subprocess
    spec=importlib.util.spec_from_file_location('report_builder',SCRIPTS/'build_pose_b19618_report_payload.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    payload=builder.build()
    assert payload==(SCRIPTS/'slurm/pose_b19618_report.sbatch').read_bytes()
    subprocess.run(['bash','-n'],input=payload,check=True)
    text=payload.decode()
    for name,expected in builder.FROZEN.items():
        delimiter='WEARUSFM_B19618_'+name.upper().replace('.','_')
        source=text.split("<<'"+delimiter+"'\n")[1].split('\n'+delimiter+'\n')[0]+'\n'
        assert hashlib.sha256(source.encode()).hexdigest()==expected
        assert source==(SCRIPTS/name).read_text()
    assert '#SBATCH --time=01:00:00' in text
    assert 'B19618_REPORT_ARGS_B64' in text and 'validate=True' in text
