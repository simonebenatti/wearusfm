"""Terminal release wiring: actual plans, second-segment receipt, launchers and fail-closed stages."""
import ast
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_continue49046 as L
import scaling_build49046 as B
import scaling_plans49046 as S
import scaling_operations49046 as O
import scaling_terminal_builders as T
import scaling_derive49046 as D
import scaling_dose_interaction as I
import scaling_preserve49046 as A


@pytest.fixture(scope='module')
def actual_context():return L.M.check_source_contract(ROOT/'results/infra/scaling_300m_20261104/training_source_contract.json')


def test_generated_first_segment_is_historical_recipe(actual_context):
    plans=B.build()
    old,cfg,argv=L.validate_plan(plans[0],actual_context)
    assert cfg.max_steps==39237 and old.max_steps==19618
    assert cfg.model.window_ms==100 and cfg.model.tokenizer_activation=='gelu'
    assert cfg.schedule.total_steps==49046 and cfg.loader.patch_ms==25
    assert argv[argv.index('--time-limit-s')+1]=='69300'
    assert set(plans[0]['previous']['files'])==set(json.loads((ROOT/B.BASE/'metadata_snapshot.json').read_text())['candidate_files'])
    assert plans[1]['predecessor_plan_sha256']==hashlib.sha256(B.dump(plans[0])).hexdigest()


@pytest.mark.parametrize('key,value',[
 ('start_step',5000),('end_step',50000),('additional_steps',19618),('minimum_free_bytes',8*2**30),
 ('protected_until_utc','2027-01-01T00:00:00Z'),('protocol','old'),('arm','T100-G-30M'),
 ('execution_source_sha256',{}),('allocation',{}),('memory',{}),
 ('control_checkpoint',{'sha256':'f'*64}),('run_dir','/relative/../bad')])
def test_plan_mutations_stop_before_training(actual_context,key,value):
    plan=B.build()[0];plan[key]=value
    with pytest.raises((ValueError,KeyError)):L.validate_plan(plan,actual_context)


def predecessor(tmp_path,monkeypatch):
    plan=B.build()[1]
    source=tmp_path/'first';source.mkdir()
    inventory={n:{'sha256':'b'*64,'bytes':1} for n in ('checkpoint.pt','checkpoint_predecay.pt',*L.IDENTITY_FILES)}
    receipt={'protocol':L.PROTOCOL,'complete':True,'arm':L.M.ARM,'checkpoint_step':39237,
       'start_step':19618,'additional_steps':19619,'scientific_endpoint':True,'previous_run_unchanged':True,
       'checkpoint':str(source/'checkpoint_predecay.pt'),'checkpoint_sha256':'b'*64,
       'resume_checkpoint':str(source/'checkpoint.pt'),'resume_checkpoint_sha256':'b'*64,
       'slurm_job_id':'12345678','code_sha':'c'*40,'plan_sha256':plan['predecessor_plan_sha256'],
       'control_checkpoint_sha256':L.CONTROL_SHA256,'run_inventory':inventory}
    path=tmp_path/'receipt.json';path.write_text(json.dumps(receipt))
    plan['previous']={'run_dir':str(source),'receipt':{'path':str(path)},'files_from_receipt':True}
    monkeypatch.setenv('SCALING_PREDECESSOR_JOB','12345678')
    monkeypatch.setattr(L.subprocess,'check_output',lambda *a,**k:'12345678|COMPLETED|0:0\n')
    return plan,receipt,path


def test_segment_two_resolves_completed_predecessor_without_overwriting_source(tmp_path,monkeypatch,actual_context):
    plan,receipt,path=predecessor(tmp_path,monkeypatch)
    bound,pin=L.resolve_previous(plan,'c'*40)
    assert bound['previous']['run_dir']==plan['previous']['run_dir']
    assert bound['previous']['files']==receipt['run_inventory']
    assert pin['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    old,cfg,argv=L.validate_plan(bound,actual_context)
    assert (old.max_steps,cfg.max_steps)==(39237,49046)
    assert argv[argv.index('--time-limit-s')+1]=='34200'
    assert plan['previous']['files_from_receipt'] is True  # no mutation of published plan


@pytest.mark.parametrize('key,value',[
 ('complete',False),('checkpoint_step',39236),('start_step',5000),('additional_steps',1),
 ('previous_run_unchanged',False),('protocol','old'),('slurm_job_id','999'),('code_sha','d'*40),
 ('plan_sha256','e'*64),('checkpoint','/wrong/checkpoint_predecay.pt'),
 ('checkpoint_sha256','d'*64),('control_checkpoint_sha256','f'*64)])
def test_bad_predecessor_is_never_resumed(tmp_path,monkeypatch,key,value):
    plan,receipt,path=predecessor(tmp_path,monkeypatch);receipt[key]=value;path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError,KeyError)):L.resolve_previous(plan,'c'*40)


@pytest.mark.parametrize('answer',['12345678|FAILED|1:0\n','12345678|COMPLETED|1:0\n','',
                                    '12345678|COMPLETED|0:0\n12345678|COMPLETED|0:0\n'])
def test_bad_scheduler_predecessor_is_not_accepted(tmp_path,monkeypatch,answer):
    plan,receipt,path=predecessor(tmp_path,monkeypatch)
    monkeypatch.setattr(L.subprocess,'check_output',lambda *a,**k:answer)
    with pytest.raises(ValueError):L.resolve_previous(plan,'c'*40)


def test_explicit_stages_and_complete_resource_accounting():
    assert S.stage_table('train-predecay')[0]['seconds']==71100
    assert S.stage_table('train-final')[0]['seconds']==36000
    assert S.stage_table('archive')[0]['seconds']==3600
    assert {S.DOCUMENTS[n][0] for n in ('training_predecay','training_final','resource_policy','archive_plan')}=={'code'}
    assert set(S.stage_table('train-final')[0]['files'])=={'receipt.json','launch.json','config.json','summary.json','training.log',*(f'rank_{i}.json' for i in range(4))}
    table=S.jobs('extractions')
    assert len(table)==6 and {j['flags'][0] for j in table}==set(S.ARMS)
    assert sum(j['gpus']*j['seconds'] for j in table)==14*3600
    budget=S.budget_table();assert budget['core_hours_with_scheduler_margin']==pytest.approx(347.8)
    assert budget['within_stage_caps'] and budget['within_total_cap'] and O.resource_check(S)
    assert budget['stages']['archive']['jobs']==1 and budget['stages']['checkout']['jobs']==5


def test_training_lines_bind_second_segment_to_real_first_job(monkeypatch):
    monkeypatch.setattr(S,'ledger_job',lambda job,*a:'12345678')
    first=S.stage_lines('train-predecay','c'*40)[0];second=S.stage_lines('train-final','c'*40)[0]
    assert '--time=19:45:00' in first and '--gpu-ore 79' in first
    assert '--time=10:00:00' in second and '--gpu-ore 40' in second
    assert 'SCALING_PREDECESSOR_JOB=12345678' in second
    assert 'training/training_49046.json' in second


def test_extraction_command_uses_metadata_checkout_and_plan_hash(tmp_path,monkeypatch):
    plan={'example':'future evidence built plan'}
    relative=S.PACKAGE+'/'+S.DOCUMENTS['extraction_plan'][1]
    target=tmp_path/relative;target.parent.mkdir(parents=True);target.write_bytes(S.dumps(plan))
    monkeypatch.setattr(S,'document',lambda name,root=None:S.pin(S.checkout('extraction')+'/'+relative,target.read_bytes()))
    argv=S.extraction_sbatch(S.jobs('extractions')[0],'c'*40)
    assert argv[2]==S.checkout('code')+'/scripts/slurm/scaling_evaluation49046.sbatch'
    assert argv[3].startswith(S.checkout('extraction')+'/') and not argv[3].startswith(S.checkout('code')+'/')
    assert 'SCALING_EXTRACTION_PLAN_SHA='+S.digest(target.read_bytes()) in argv[1]


def test_launchers_have_same_actual_protocols_and_all_six_cells():
    extraction=(ROOT/'scripts/slurm/scaling_evaluation49046.sbatch').read_text()
    assert '!= set(arms)' in extraction and '!= {arms[1]}' not in extraction
    assert 'SCALING_EXTRACTION_PLAN_SHA' in extraction and 'T100-G-30M|T100-G-300M)' in extraction
    assert S.E.PROTOCOL in extraction
    readers=(ROOT/'scripts/slurm/scaling_readers49046.sbatch').read_text()
    assert S.POSE_RESOURCE_PROTOCOL in readers and S.C.RESOURCE_PROTOCOL in readers
    assert S.M.INPUT_PROTOCOL in readers
    assert 'dose_summary(' in readers
    for path in ('scripts/scaling_continue49046.py','scripts/train_jepa.py'):
        tree=ast.parse((ROOT/path).read_text());imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        torch_index=next(i for i,n in enumerate(imports) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
        run_index=next((i for i,n in enumerate(imports) if isinstance(n,ast.ImportFrom) and n.module=='wearusfm.training'),999)
        assert torch_index<run_index


@pytest.mark.parametrize('name',D.MODULES)
def test_numerical_modules_are_exact_declared_derivatives(name):
    source=(ROOT/'scripts'/f'{name}19618.py').read_text()
    assert (ROOT/'scripts'/f'{name}49046.py').read_text()==D.derive(source,name)


def test_real_guard_calls_p1_finalization_and_logs_details():
    import inspect
    source=inspect.getsource(S.E.execute)
    assert 'from scaling_finalize49046 import certify_output' in source
    assert "task == 'p1'" in source and 'certify_output(path, lambda q: verify_classification_cache(' in source
    assert 'output_identity_audit' in source


def test_copy_archive_verifies_content_and_keeps_source(tmp_path):
    source=tmp_path/'old';source.write_bytes(b'checkpoint-content')
    sha=hashlib.sha256(source.read_bytes()).hexdigest();before=L.P.file_identity(source)
    result=A.copy_verified(source,tmp_path/'copy',sha,source.stat().st_size)
    assert result['sha256']==sha and L.P.file_identity(source)==before
    with pytest.raises(ValueError):A.copy_verified(source,tmp_path/'bad','f'*64,source.stat().st_size)
    with pytest.raises(FileExistsError):A.copy_verified(source,tmp_path/'copy',sha,source.stat().st_size)


@pytest.mark.parametrize('stage',list(O.PREREQUISITES))
def test_every_stage_has_explicit_prerequisites_and_fails_without_them(tmp_path,stage):
    if O.PREREQUISITES[stage]:
        with pytest.raises((OSError,ValueError)):O.dependencies(S,stage,tmp_path,tmp_path/'ledger')
    else:assert O.dependencies(S,stage,tmp_path,tmp_path/'ledger')=={}


def test_secondary_is_descriptive_even_when_sufficient_statistics_match():
    old=json.loads((ROOT/T.OLD/'evidence/compare/report.json').read_text())
    new=copy.deepcopy(old);new['checkpoint_step']=49046
    result=I.analyse(old,new)
    assert result['comparable'] is False and result['inferential_interaction_computed'] is False
    assert result['epn']['descriptive_difference_pp']==0 and result['pose']['descriptive_difference_E_dyn']==0
    assert 'ci95' not in json.dumps(result)
    new['runtime']={};new['inputs']={};new['pose']['candidate_E_dyn']-=.01;new['pose']['delta_E_dyn']-=.01
    assert I.analyse(old,new)['comparable'] is False


def test_secondary_bad_reference_hash_stops_before_write(tmp_path):
    old=ROOT/T.OLD/'evidence/compare/report.json';new=tmp_path/'new';out=tmp_path/'out'
    new.write_bytes(old.read_bytes())
    with pytest.raises(ValueError):I.execute(old,'f'*64,new,out)
    assert not out.exists()


def test_gate_failure_emits_no_lines_and_unknown_stage_stops(tmp_path):
    with pytest.raises(ValueError):S.admitted_lines('train-final',root=tmp_path,ledger=tmp_path/'ledger')
    with pytest.raises(ValueError):S.stage_table('another-training')


def fake_accounting(job,slurm='12345678',state='COMPLETED'):
    header='JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|MaxRSS|Timelimit|AllocTRES'
    tres=f"cpu={job['cpus']},node=1"+(f",gres/gpu={job['gpus']}" if job['gpus'] else '')
    return header+f"\n{slurm}|{state}|0:0|1|{job['cpus']}|1M|{S.clock(job['seconds'])}|{tres}\n"


def collect_synthetic_stage(tmp_path,stage,contents,slurm='12345678'):
    import base64
    table=S.stage_table(stage,tmp_path)
    files=[]
    for job in table:
        for name,path in job['files'].items():
            data=S.dumps(contents.get(name,{}))
            files.append({'path':path,'canonical':True,'sha256':S.digest(data),'bytes':len(data),
                          'content':base64.b64encode(data).decode()})
        files.append({'path':S.log_path(job,slurm),'missing':True})
    reading={'time_utc':'2026-11-19T10:00:00Z','files':files,
             'sacct':{slurm:{'exit':0,'stdout':fake_accounting(table[0],slurm)}},
             'listings':{job['folder']:sorted(job['files']) for job in table}}
    return S.collect(stage,table,{j['id']:slurm for j in table},reading,S.evidence_dir(stage,tmp_path))


def copy_training_documents(root):
    for name in ('training_39237.json','training_49046.json'):
        canonical=root/S.PACKAGE/'training'/name;canonical.parent.mkdir(parents=True,exist_ok=True)
        canonical.write_bytes((ROOT/S.PACKAGE/'training'/name).read_bytes())
        source=ROOT/B.BASE/name;dest=root/B.BASE/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    source=ROOT/T.OLD/'evaluation/extraction_plan.json';dest=root/T.OLD/'evaluation/extraction_plan.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    relative=T.LAUNCH+'/control_line2_evidence/continuation/49046/receipt.json'
    dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())


def test_collect_training_then_build_actual_extraction_plan_and_dual_arm_guard(tmp_path,monkeypatch):
    copy_training_documents(tmp_path)
    plan=json.loads((tmp_path/B.BASE/'training_49046.json').read_text())
    receipt={'complete':True,'slurm_job_id':'12345678','checkpoint_step':49046,'start_step':39237,
             'plan_sha256':S.digest((tmp_path/B.BASE/'training_49046.json').read_bytes()),
             'checkpoint':plan['run_dir']+'/checkpoint_postdecay.pt','checkpoint_sha256':'c'*64,
             'protocol':L.PROTOCOL,'scientific_endpoint':True,'previous_run_unchanged':True,
             'additional_steps':9809,'code_sha':'c'*40,
             'run_inventory':{'checkpoint_postdecay.pt':{'sha256':'c'*64,'bytes':4673783756}}}
    collect_synthetic_stage(tmp_path,'train-final',{'receipt.json':receipt,'summary.json':{'stopped':'max_steps','steps':49046}})
    monkeypatch.setattr(S,'ledger_job',lambda *a:'12345678')
    monkeypatch.setattr(S,'checkout_commit',lambda *a:'c'*40)
    extraction=T.extraction_plan_from_receipts(S,tmp_path)
    assert set(extraction['checkpoints'])==set(S.ARMS) and extraction['step']==49046
    assert extraction['checkpoints'][S.ARMS[1]]['path']==receipt['checkpoint']
    assert extraction['checkpoints'][S.ARMS[1]]['bytes']==4673783756  # actual receipt, not inferred19k size
    assert extraction['checkpoints'][S.ARMS[0]]['sha256']==L.CONTROL_SHA256
    assert extraction['training_source_contract']==S.shared('training_source_contract')
    # Actual collected evidence is checked for content corruption before constructing any future plan.
    path=S.evidence_dir('train-final',tmp_path)/'receipt.json'
    path.write_text('{}')
    with pytest.raises(ValueError):T.extraction_plan_from_receipts(S,tmp_path)


def test_completed_prerequisite_accounting_is_rechecked_and_binds_to_ledger(tmp_path,monkeypatch):
    copy_training_documents(tmp_path)
    collect_synthetic_stage(tmp_path,'train-predecay',{})
    monkeypatch.setattr(S,'ledger_job',lambda *a:'12345678')
    assert O.dependencies(S,'train-final',tmp_path,tmp_path/'ledger')=={'SCALING-49046-train-39237-20261118':'12345678'}
    raw=S.evidence_dir('train-predecay',tmp_path)/'sacct.txt'
    raw.write_text(raw.read_text().replace('COMPLETED','FAILED'))
    with pytest.raises(ValueError):O.dependencies(S,'train-final',tmp_path,tmp_path/'ledger')


def test_initial_checkout_has_real_owned_documents_and_passes_gate(tmp_path,monkeypatch):
    import shutil
    destination=tmp_path/S.PACKAGE/'training';destination.mkdir(parents=True)
    for path in (ROOT/S.PACKAGE/'training').iterdir():shutil.copyfile(path,destination/path.name)
    monkeypatch.setattr(S,'release_check',lambda *a,**k:0)
    def blob(commit,relative,root=None):
        p=tmp_path/relative
        if not p.exists():p=ROOT/relative
        return S.digest(p.read_bytes()) if p.exists() else None
    monkeypatch.setattr(S,'_blob_sha256',blob)
    quota=f'Disk quotas for prj 20147512 (pid 20147512):\n{S.WORK}\n1000 1073741824 0\n'
    reading={'commands':{'squeue':{'exit':0,'stdout':''},'quota':{'exit':0,'stdout':quota}},
             'exists':{S.checkout('code'):False},'sacct':{}}
    checks,facts=S.checkout_gate('code','c'*40,'c'*40,reading,None,tmp_path,tmp_path/'ledger')
    assert all(ok for _,ok in checks),checks
    assert facts['tranche']==['SCALING-49046-code-checkout-20261118']


@pytest.mark.parametrize('bad',[None,False,True,1,255])
def test_unreadable_queue_and_quota_cannot_admit_jobs(bad):
    reading={'commands':{'squeue':{'exit':bad,'stdout':''},'quota':{'exit':0,'stdout':''}}}
    with pytest.raises(ValueError):S.lane(reading)
    reading['commands']['squeue']['exit']=0;reading['commands']['quota']['exit']=bad
    with pytest.raises(ValueError):S.lane(reading)


def test_original_fit_gate_is_retained_on_real_accepted_shards(monkeypatch):
    import test_scaling_gates19618 as G
    monkeypatch.setattr(G,'S',S)
    result=G.run_fit_state()
    assert result['fits']==420 and result['all_converged_by_criterion'] is True
    changed=G.run_fit_state(lambda resources,shards:shards[G.FIRST]['report']['audit_summary'].update(
        converged_by_criterion=False,convergence_warnings=1,unresolved_fits=[{}]))
    assert changed['all_converged_by_criterion'] is False


def test_gate_to_lines_training_real_documents_and_checkout_receipt(tmp_path,monkeypatch):
    import calendar,time
    import shutil
    copy_training_documents(tmp_path)
    for relative in (S.PACKAGE+'/training',):
        shutil.copytree(ROOT/relative,tmp_path/relative,dirs_exist_ok=True)
    ledger=tmp_path/'ledger';name=S.checkout_job('code')['id']
    entry=ledger/name;entry.mkdir(parents=True)
    state={'agente':'gpt','campagna':S.CAMPAIGN,'stato':'sottomesso','job_slurm':'12345678',
        'sbatch':['--export=ALL,CAMPAIGN='+S.CHECKOUTS['code']+',CAMPAIGN_SHA='+('c'*40)+',EXPECTED_SHA='+('c'*40)]}
    (entry/'state.json').write_text(json.dumps(state))
    collect_synthetic_stage(tmp_path,'checkout-code',{})
    stage='train-predecay';table=S.stage_table(stage,tmp_path)
    documents={n:S.document(n,tmp_path) for n in S.STAGE_DOCUMENTS[stage]};commits={'code':'c'*40}
    quota=f'Disk quotas for prj 20147512 (pid 20147512):\n{S.WORK}\n1000 1073741824 0\n'
    reading={'time_utc':'2026-11-19T10:00:00Z','commands':{'squeue':{'exit':0,'stdout':''},'quota':{'exit':0,'stdout':quota},
         'head:code':{'exit':0,'stdout':'c'*40},'status:code':{'exit':0,'stdout':''}},
         'files':[{'canonical':True,**p} for p in documents.values()],
         'exists':{j['out']:False for j in table},'sacct':{}}
    monkeypatch.setattr(S,'_blob_sha256',lambda commit,relative,root=None:S.digest((ROOT/relative).read_bytes()))
    checks,facts=S.gate(reading,commits,documents,table,{},tmp_path,ledger)
    assert all(ok for _,ok in checks),checks
    operator={'commit':'c'*40,'files':{'test':'a'*64}}
    monkeypatch.setattr(O,'attestation',lambda *a:operator)
    context={'commits':commits,'documents':documents,'submitted':{},'operator':operator,
        'operational_inputs':O.context(S,stage,tmp_path,ledger)}
    relative,data=S.gate_record(stage,checks,facts,reading,context)
    S._write(relative,data,tmp_path)
    now=calendar.timegm(time.strptime(reading['time_utc'],'%Y-%m-%dT%H:%M:%SZ'))
    lines=S.admitted_lines(stage,root=tmp_path,ledger=ledger,now=now)
    assert len(lines)==1 and '--gpu-ore 79' in lines[0]
    # A changed document never passes from an older successful gate to a sender.
    path=tmp_path/S.PACKAGE/S.DOCUMENTS['training_predecay'][1]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='document changed|training plan mirror differs'):S.admitted_lines(stage,root=tmp_path,ledger=ledger,now=now)


def test_gate_cannot_advance_when_live_predecessor_failed(tmp_path,monkeypatch):
    copy_training_documents(tmp_path)
    job=S.stage_table('train-final',tmp_path)[0]
    monkeypatch.setattr(O,'dependencies',lambda *a:{'previous':'12345678'})
    monkeypatch.setattr(S,'_collected_checkout',lambda *a:True)
    quota=f'Disk quotas for prj 20147512 (pid 20147512):\n{S.WORK}\n1000 1073741824 0\n'
    reading={'commands':{'squeue':{'exit':0,'stdout':''},'quota':{'exit':0,'stdout':quota}},'files':[],
             'exists':{job['out']:False},'sacct':{'12345678':{'exit':0,'stdout':fake_accounting(job,state='FAILED')}}}
    checks,facts=S.gate(reading,{}, {},[job],{},tmp_path,tmp_path/'ledger')
    assert ('cross-stage predecessors COMPLETED 0:0',False) in checks
    assert not all(ok for _,ok in checks)


def test_finalization_logs_are_in_the_collection_contract():
    for job in S.jobs('extractions'):
        if job['flags'][2]=='p1':
            assert 'output_identity_events.jsonl' in job['files'] and 'p1.log' in job['files']


def test_strict_completed_requires_one_actual_successful_parent():
    job=S.stage_table('train-final')[0];raw=fake_accounting(job)
    assert S._completed({'exit':0,'stdout':raw},'12345678') is True
    assert S._completed({'exit':False,'stdout':raw},'12345678') is False
    assert S._completed({'exit':0,'stdout':raw+raw.splitlines()[1]+'\n'},'12345678') is False
