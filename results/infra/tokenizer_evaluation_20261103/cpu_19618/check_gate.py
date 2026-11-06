"""Offline gate from collected, byte-identical reports; no network or submission."""
from pathlib import Path
import argparse, hashlib, json, math, csv, io

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--validate-job',required=True)
p.add_argument('--benchmark-job',required=True)
p.add_argument('--code-sha',required=True)
a=p.parse_args()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
base=a.root/'results/infra/tokenizer_evaluation_20261103'
plan_path=base/'pose_readout_plan_19618.json'
policy_path=base/'pose_readout_resource_policy_19618.json'
plan=json.loads(plan_path.read_text());policy=json.loads(policy_path.read_text())
assert policy['plan_sha256']==sha(plan_path)
assert all(sha(a.root/k)==v for k,v in plan['sources'].items())
hashes={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (a.directory/'sha256sum_reports.txt').read_text().splitlines()}
rows={row['JobID']:row for row in csv.DictReader(io.StringIO((a.directory/'sacct_pre_ridge.txt').read_text()),delimiter='|')}
reports={};jobs={}
for stage,job,cpu,status in [('validate',a.validate_job,2,'validated'),('benchmark',a.benchmark_job,8,'train_only_timing')]:
    row=rows[job]
    assert row['State']=='COMPLETED' and row['ExitCode']=='0:0'
    assert int(row['AllocCPUS'])==cpu and 'gres/gpu' not in row['AllocTRES']
    path=a.directory/(stage+'_report.json'); report=json.loads(path.read_text())
    remote=next(j['output']+'/report.json' for j in policy['jobs'] if j['stage']==stage)
    assert sha(path)==hashes[remote]
    assert report['complete'] is True and report['status']==status
    assert report['protocol']==plan['protocol'] and report['step']==19618
    assert report['slurm_job_id']==job and report['plan_sha256']==sha(plan_path)
    assert report['recipe']==plan['recipe'] and report['input_pins']==plan['caches'] and report['reference_pins']==plan['references']
    assert report['runtime']['code_sha']==a.code_sha and report['runtime']['sources']==plan['sources']
    assert report['runtime']['torch_threads']==cpu
    assert set(report['runtime']['threads'].values())=={str(cpu)}
    assert all(item['num_threads']==cpu for item in report['runtime']['threadpools'])
    reports[stage]=report
    jobs[stage]={'job_id':int(job),'state':row['State'],'exit_code':row['ExitCode'],'elapsed_seconds':int(row['ElapsedRaw']),'allocated_core_hours':int(row['ElapsedRaw'])*cpu/3600,'report_sha256':sha(path)}

v,b=reports['validate'],reports['benchmark']
assert v['scientific_metrics_computed'] is False
assert set(v['inspections'])=={arm+'/'+split for arm in plan['caches'] for split in ('train','val')}
assert set(b['inspections'])=={'T25-L/train'} and b['arm']=='T25-L'
assert b['inspections']['T25-L/train']==v['inspections']['T25-L/train']
train=v['inspections']['T25-L/train'];val=v['inspections']['T25-L/val']
assert len(train['users'])==40 and len(val['users'])==15 and set(train['users']).isdisjoint(val['users'])
assert train['users']==sorted(sum(plan['fold_users'],[]))
for arm in plan['caches']:
    for split,ref in [('train',train),('val',val)]:
        x=v['inspections'][arm+'/'+split]
        assert x['users']==ref['users'] and x['targets']==ref['targets']
        assert x['cache_sha256']==plan['caches'][arm][split]['sha256']
        assert x['checkpoint_sha256']==plan['caches'][arm][split]['checkpoint_sha256']
assert b['timings']['n_features']==384 and b['timings']['n_users']==40
V=jobs['validate']['elapsed_seconds'];S=b['timings']['statistics_seconds'];E=b['timings']['three_eigensystems_seconds'];R=10
assert V>0 and math.isfinite(S) and math.isfinite(E) and S>0 and E>0
raw=1.5*(2*V+3*((1+R)*S+13*E))+120
rounded=math.ceil(raw/60)*60
gate={'protocol':'tokenizer-pose-cpu19618-gate-20261104-v1','gate_pass':rounded<=1800,'plan_sha256':sha(plan_path),'runtime_code_sha':a.code_sha,'resource_policy_sha256':sha(policy_path),'preceding_jobs':jobs,'timing':{'formula':policy['ridge_timing_gate']['definition'],'V':V,'S':S,'E':E,'R':R,'raw_required_seconds':raw,'rounded_required_seconds':rounded,'walltime_ceiling_seconds':1800},'verified':{'original_report_sha256':True,'complete_reports':True,'same_source_plan_dose_recipe':True,'cache_reference_pins':True,'common_targets_metadata_support':True,'train_val_users_disjoint':True,'benchmark_only_T25_L_train':True,'benchmark_threads_environment_torch_and_pools_8':True,'validation_metrics_not_computed':True},'ridge':next(j for j in policy['jobs'] if j['stage']=='ridge'),'after_submission':'No background monitoring reactivated.'}
(a.directory/'pre_ridge_gate.json').write_text(json.dumps(gate,indent=2,allow_nan=False)+'\n')
print(json.dumps(gate,indent=2,allow_nan=False))
