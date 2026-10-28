"""Offline checks for receipt-bound B/R3c extraction; no model inference."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('b19618_eval_guard',ROOT/'scripts/pose_b19618_eval_guard.py')
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def receipt_b(path):
    return {'complete':True,'expected_step':19618,'source_unchanged':True,'code_sha':G.CODE_SHA,
            'outputs':{path.name:{'sha256':'a'*64},'config.json':{'sha256':'b'*64}}}


def receipt_r3c(path):
    return {'completed':True,'provenance':{'environment_code_sha':G.CODE_SHA},
            'R3c19618':{'path':str(path),'sha256':'a'*64,'step':19618,
                       'config_path':str(path.parent/'config.json'),'config_file_sha256':'b'*64}}


@pytest.mark.parametrize('model', ['B','R3c'])
def test_receipt_identity_model_step_path_and_complete(tmp_path,model):
    path = tmp_path/'checkpoint_step0019618.pt'
    receipt = receipt_b(path) if model=='B' else receipt_r3c(path)
    assert G.receipt_identity(receipt,path,model,tmp_path/'receipt.json') == ('a'*64,'b'*64)
    with pytest.raises(ValueError):
        G.receipt_identity(receipt,tmp_path/'another.pt',model,tmp_path/'receipt.json')
    receipt['complete' if model=='B' else 'completed'] = False
    with pytest.raises(ValueError,match='incomplete'):
        G.receipt_identity(receipt,path,model,tmp_path/'receipt.json')


@pytest.mark.parametrize('model',['B','R3c'])
def test_checkpoint_hash_step_head_and_optional_r3c_stats(tmp_path,model):
    path = tmp_path/'checkpoint_step0019618.pt'
    cfg = {'max_steps':49046,'model':{'pose_head':model=='B'}}
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    weights = {'w':torch.ones(2)}
    if model=='B':
        weights['pose_head.out.weight'] = torch.ones((20,2))
    state = {'step':19618,'config':cfg,'pose_stats_sha256':G.STATS_SHA if model=='B' else None,
             'student':weights,'teacher':weights}
    torch.save(state,path)
    receipt = receipt_b(path) if model=='B' else receipt_r3c(path)
    if model=='B':
        receipt['outputs'][path.name]['sha256']=G.sha(path)
        receipt['outputs']['config.json']['sha256']=G.sha(tmp_path/'config.json')
    else:
        receipt['R3c19618']['sha256']=G.sha(path)
        receipt['R3c19618']['config_file_sha256']=G.sha(tmp_path/'config.json')
    assert G.check_checkpoint(path,model,receipt,tmp_path/'receipt.json')[0]==G.sha(path)
    path.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='SHA'):
        G.check_checkpoint(path,model,receipt,tmp_path/'receipt.json')


def make_cache(path,checkpoint,split,head=True):
    n=40 if split=='train' else 15
    target=np.arange(n*2*20,dtype=np.float32).reshape(n,2,20)/100
    mask=np.ones(target.shape,bool)
    mask[0,0,:]=False
    cfg={'checkpoint':str(checkpoint),'which':'student','split':split,'code_sha':G.CODE_SHA,
         'pose_stats_sha256':G.STATS_SHA,'checkpoint_pose_stats_sha256':G.STATS_SHA if head else None,
         'n_windows':10,'window_s':4.,'seed':20261026,'pose_root':'/pose','per_side':False}
    data={'lat':np.ones((n,2,384),np.float16),'pred':target*.8 if head else np.zeros_like(target),'target':target,'mask':mask,'std':np.ones((n,20),np.float32),
          'pose_var':np.arange(n,dtype=np.float32),'meta_json':np.asarray(json.dumps([{'user':str(i),'recording':i} for i in range(n)])),
          'config_json':np.asarray(json.dumps(cfg)),'has_head':np.asarray(head),'n_recordings':np.asarray(200 if split=='train' else 1950)}
    # Only the new latent array is checked; reference latent values can differ.
    np.savez(path,**data)
    return data


@pytest.mark.parametrize('model',['B','R3c'])
@pytest.mark.parametrize('split',['train','val'])
def test_pose_cache_same_support_and_reference_on_both_models(tmp_path,model,split):
    checkpoint=tmp_path/'checkpoint_step0019618.pt'
    reference=tmp_path/'reference.npz'
    path=tmp_path/'new.npz'
    make_cache(reference,tmp_path/'B5000.pt',split)
    make_cache(path,checkpoint,split,model=='B')
    result=G.verify_pose_cache(path,reference,model,split,checkpoint)
    assert result['same_metadata_target_mask_std_pose_var_as_B5000']
    assert result['has_head']==(model=='B')
    assert result['n_users']==(40 if split=='train' else 15)


@pytest.mark.parametrize('key',['target','mask','std','pose_var','meta_json'])
def test_pose_cache_changed_pairing_or_target_fails(tmp_path,key):
    checkpoint=tmp_path/'checkpoint.pt'
    reference,path=tmp_path/'ref.npz',tmp_path/'new.npz'
    make_cache(reference,checkpoint,'val')
    data=make_cache(path,checkpoint,'val')
    if key=='meta_json':
        meta=json.loads(str(data[key]))
        meta[0]['user']='changed'
        data[key]=np.asarray(json.dumps(meta))
    elif key=='mask':
        data[key][0,0,0]=True
    else:
        data[key].flat[0]+=1
    np.savez(path,**data)
    with pytest.raises(ValueError,match='differs'):
        G.verify_pose_cache(path,reference,'B','val',checkpoint)


def test_frozen_commands_explicit_student_pose_teacher_channel_and_roots():
    code,checkpoint,out,work,scratch=map(Path,('/code','/ckpt','/out','/work','/scratch'))
    train=G.pose_command(code,checkpoint,out,'train',work,scratch)
    val=G.pose_command(code,checkpoint,out,'val',work,scratch)
    assert train[train.index('--which')+1]=='student'
    assert train[train.index('--n-users')+1]=='40' and train[train.index('--n-recordings')+1]=='200'
    assert '--n-users' not in val and '--n-recordings' not in val
    p1=G.channel_command(code,checkpoint,out,'p1',work,scratch)
    p2=G.channel_command(code,checkpoint,out,'p2',work,scratch)
    assert p1[p1.index('--datasets')+1:]==G.DATASETS['p1']
    assert p2.count('--epn-root')==2 and '--uci-root' in p2
    assert '--which' not in p1 # frozen channel extractor hardcodes teacher
    assert '/work/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON' in p2


def test_channel_provenance_teacher_and_nonfinite_feature_guard(tmp_path):
    path=tmp_path/'ninapro_db2.npz'
    checkpoint=tmp_path/'checkpoint.pt'
    data={'checkpoint':str(checkpoint),'checkpoint_sha256':'a'*64,'task':'p1','code_sha':G.CODE_SHA,
          'protocol':'readout_per_canale_v1','extraction_recipe':json.dumps({'encoder':'teacher'}),
          'y':np.array([0,1]),'s':np.array(['a','b']),'subjects':np.array(['a','b']),
          'input_sha256':'b'*64,'splits_sha256':'c'*64,
          **{k:np.ones((2,4)) for k in ('x_local','x_dec','x_pool')}}
    np.savez(path,**data)
    assert G.verify_channel_cache(path,'p1',checkpoint,'a'*64)['which']=='teacher'
    data['x_pool'][0,0]=np.nan
    np.savez(path,**data)
    with pytest.raises(ValueError,match='feature'):
        G.verify_channel_cache(path,'p1',checkpoint,'a'*64)
    data['x_pool'][0,0]=1
    data['extraction_recipe']=json.dumps({'encoder':'student'})
    np.savez(path,**data)
    with pytest.raises(ValueError,match='teacher'):
        G.verify_channel_cache(path,'p1',checkpoint,'a'*64)


def test_fresh_path_no_overwrite_no_code(tmp_path):
    G.fresh(tmp_path/'new',[tmp_path/'code'])
    for path in (tmp_path,tmp_path/'code'/'new',Path('relative')):
        with pytest.raises(ValueError):
            G.fresh(path,[tmp_path/'code'])


def probe_fixture(tmp_path,model='B',task='p1'):
    path=tmp_path/(model+'_'+task+'_pool.json')
    checkpoint=tmp_path/'checkpoint_step0019618.pt'
    identity={'checkpoint':str(checkpoint),'checkpoint_sha256':'a'*64}
    oof=path.with_suffix('.oof.npz')
    fields={'datasets':np.array(G.DATASETS[task])}
    for ds in G.DATASETS[task]:
        fields.update({ds+'_y':np.array([0,1]),ds+'_pred':np.array([0,1]),ds+'_s':np.array(['a','b']),ds+'_row':np.array([0,1])})
    np.savez(oof,**fields)
    inputs={ds:{'input_sha256':'b'*64,'splits_sha256':'c'*64} for ds in G.DATASETS[task]}
    report={**identity,'protocol':'readout_per_canale_v1','task':task,'probe_seed':0,'code_sha':G.CODE_SHA,
            'feature_set':'channel_readout/pool','readout':{'layer':'pool','k_requested':0},
            'extraction_recipe':{'encoder':'teacher'},'oof':str(oof),'oof_sha256':G.sha(oof),'splits_sha256':'c'*64,
            'datasets':{ds:{'n_oof_windows':2,'input_sha256':'b'*64} for ds in G.DATASETS[task]}}
    path.write_text(json.dumps(report))
    return path,identity,inputs,report


def test_pool_probe_argv_and_report_oof_identity(tmp_path):
    command=G.probe_command(Path('/code'),Path('/features'),Path('/report.json'),'p2')
    assert command[command.index('--layer')+1]=='pool'
    assert command[command.index('--k')+1]=='0'
    assert command[command.index('--seed')+1]=='0'
    path,identity,inputs,report=probe_fixture(tmp_path)
    G.check_probe_report(path,'B','p1',identity,inputs)
    report['readout']['k_requested']=768
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='full pool width'):
        G.check_probe_report(path,'B','p1',identity,inputs)
    report['readout']['k_requested']=0
    path.write_text(json.dumps(report))
    path.with_suffix('.oof.npz').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='OOF'):
        G.check_probe_report(path,'B','p1',identity,inputs)


def test_guarded_report_chain_and_tamper(tmp_path):
    path,identity,inputs,report=probe_fixture(tmp_path)
    origin=tmp_path/'continuation_receipt.json'
    origin.write_text(json.dumps(receipt_b(Path(identity['checkpoint']))))
    extraction=tmp_path/'extraction_receipt.json'
    common={**identity,'model':'B','checkpoint_step':19618,'config_file_sha256':'b'*64,'code_sha':G.CODE_SHA}
    extraction.write_text(json.dumps({**common,'complete':True,'checkpoint_unchanged':True,
                                     'receipt':str(origin),'receipt_sha256':G.sha(origin)}))
    prov={**common,'mode':'probe','complete':True,'task':'p1','which':'teacher','layer':'pool','k':0,'seed':0,
          'report':str(path),'report_sha256':G.sha(path),'oof':report['oof'],'oof_sha256':report['oof_sha256'],
          'extraction_receipt':str(extraction),'extraction_receipt_sha256':G.sha(extraction),'feature_inputs':inputs}
    Path(str(path)+'.provenance.json').write_text(json.dumps(prov))
    assert G.guarded_report(path,'B','p1')['checkpoint_sha256']=='a'*64
    origin.write_text('{}')
    with pytest.raises(ValueError,match='original checkpoint receipt changed'):
        G.guarded_report(path,'B','p1')


def test_compare_runs_both_tasks_correct_direction_and_refuses_overwrite(tmp_path,monkeypatch):
    from types import SimpleNamespace
    for task in ('p1','p2'):
        for model in ('B','R3c'):
            path=tmp_path/(model+'_'+task+'_pool.json')
            path.write_text('{}')
            path.with_suffix('.oof.npz').write_bytes(b'OOF')
    def guarded(path,model,task):
        return {'path':str(path),'sha256':G.sha(path),'oof':str(path.with_suffix('.oof.npz')),
                'oof_sha256':G.sha(path.with_suffix('.oof.npz')),'checkpoint_sha256':('a' if model=='B' else 'b')*64}
    monkeypatch.setattr(G,'guarded_report',guarded)
    calls=[]
    def run(command,check):
        calls.append(command)
        assert '--single-seed' in command and '--models' in command
        assert 'R3c_' in command[command.index('--reference')+1]
        assert '/B_' in command[command.index('--candidate')+1]
        output=Path(command[command.index('--out')+1])
        output.write_text(json.dumps({'task':output.stem.removeprefix('compare_'),'seeds':1,
                                     'reference':[{'checkpoint_sha256':'b'*64}],
                                     'candidate':[{'checkpoint_sha256':'a'*64}]}))
    monkeypatch.setattr(G.subprocess,'run',run)
    args=SimpleNamespace(out=tmp_path,model=None,task=None,features=None,receipt=None,checkpoint=None,
                         reference_train=None,reference_val=None,source_sha256='c'*64,wrapper_sha256='d'*64)
    assert G.run_compare(args,Path('/code'),[Path('/code')])==0 and len(calls)==2
    assert json.loads((tmp_path/'compare_p2.json.provenance.json').read_text())['candidate']=='B'
    with pytest.raises(ValueError,match='new'):
        G.run_compare(args,Path('/code'),[Path('/code')])


def test_eval_payload_regenerates_syntax_and_passes_json_as_literal_argv(tmp_path):
    import base64
    import hashlib
    import os
    import subprocess
    import sys
    spec=importlib.util.spec_from_file_location('builder',ROOT/'scripts/build_pose_b19618_eval_payloads.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    payload=builder.build()
    assert payload==(ROOT/'scripts/slurm/pose_b19618_eval.sbatch').read_bytes()
    subprocess.run(['bash','-n'],input=payload,check=True)
    text=payload.decode()
    driver=text.split("<<'WEARUSFM_B19618_EVAL_DRIVER'\n")[1].split('\nWEARUSFM_B19618_EVAL_DRIVER')[0]
    stub=tmp_path/'stub.py'
    stub.write_text('import json,sys; print(json.dumps(sys.argv[1:]))\n')
    digest=hashlib.sha256(stub.read_bytes()).hexdigest()
    args=['--mode','probe','--out','/a path/$(touch nope); echo nope.json']
    env={**os.environ,'B19618_ARGS_B64':base64.b64encode(json.dumps(args).encode()).decode(),'WRAPPER_SHA256':'d'*64}
    result=subprocess.run([sys.executable,'-',str(stub),digest],input=driver,text=True,env=env,capture_output=True,check=True)
    assert json.loads(result.stdout)==args+['--source-sha256',digest,'--wrapper-sha256','d'*64]
    env['B19618_ARGS_B64']=base64.b64encode(json.dumps(['--source-sha256=bad']).encode()).decode()
    result=subprocess.run([sys.executable,'-',str(stub),digest],input=driver,text=True,env=env,capture_output=True)
    assert result.returncode!=0
