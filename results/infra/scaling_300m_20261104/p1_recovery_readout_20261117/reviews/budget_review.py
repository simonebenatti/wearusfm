"""Independent local accounting audit. No SSH, scores, writes to the repo or jobs."""
from pathlib import Path
from fractions import Fraction
import datetime
import hashlib
import json
import math

ROOT=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-p1-recovery')
BASE=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_readout_20261117/evidence'
BUDGET=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_release_20261116/budget.json'
OUT=Path('/private/tmp/p1-r2-budget-review-20261117.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
checks=[]
def check(name,value):
    checks.append({'name':name,'passed':bool(value)})
    assert value,name
def close(name,x,y):check(name,math.isclose(float(x),float(y),rel_tol=0,abs_tol=1e-9))

snapshot=load(BASE/'snapshot.json');budget=load(BUDGET);receipt=load(BASE/'p1/receipt.json')
cmd=snapshot['commands']['sacct']
check('successful fresh accounting command',type(cmd['exit']) is int and cmd['exit']==0)
check('textual sacct exactly retained',cmd['stdout']==(BASE/'sacct.txt').read_text())
expected={
 '64571999':('COMPLETED','0:0',7330,8,1),
 '64572000':('FAILED','1:0',1882,8,1),
 '64572001':('COMPLETED','0:0',5967,8,1),
 '64596915':('FAILED','1:0',1793,8,1),
 '64610004':('COMPLETED','0:0',41,2,0),
 '64610181':('COMPLETED','0:0',977,8,1),
 '64620427':('COMPLETED','0:0',34,2,0),
 '64620434':('COMPLETED','0:0',1772,8,1),
}
jobs={};ignored=[]
for line in cmd['stdout'].splitlines():
    fields=line.split('|')
    if not fields[0].isdigit():
        ignored.append(fields[0]);continue
    job,state,exit_code,elapsed,cpus,tres=fields[:6]
    counts=dict(value.split('=',1) for value in tres.split(','))
    data=(state,exit_code,int(elapsed),int(cpus),int(counts.get('gres/gpu',0)))
    check('one parent record '+job,job not in jobs)
    check('CPU allocation matches TRES '+job,int(counts['cpu'])==data[3])
    start,end=map(datetime.datetime.fromisoformat,(fields[8],fields[9]))
    check('elapsed agrees with start/end '+job,(end-start).total_seconds()==data[2])
    jobs[job]=data
check('exact eight parent allocations',jobs==expected)
check('steps excluded rather than double-counted',len(ignored)==16 and all('.batch' in x or '.extern' in x for x in ignored))

def gpu(ids):return sum((Fraction(jobs[j][2]*jobs[j][4],3600) for j in ids),Fraction())
def core(ids):return sum((Fraction(jobs[j][2]*jobs[j][3],3600) for j in ids),Fraction())
previous=('64571999','64572000','64572001','64596915','64610181')
stage=previous+('64620434',)
p1=('64572000','64596915','64620434')
checkouts=('64610004','64620427')
previous_gpu=gpu(previous);new_gpu=gpu(('64620434',));new_core=core(('64620434',))
stage_gpu=gpu(stage);stage_core=core(stage);p1_gpu=gpu(p1);p1_core=core(p1)
checkout_core=core(checkouts)
close('previous baseline reconciles release budget',previous_gpu,budget['stage_gpu_hours_consumed'])
check('P1 actual inside approved requested duration',jobs['64620434'][2]<=budget['p1_r2']['seconds'])
check('P1 actual GPU and attached core within maximum',float(new_gpu)<=budget['p1_r2']['max_gpu_hours'] and float(new_core)<=budget['p1_r2']['max_core_hours'])
check('one recovery checkout within requested duration',jobs['64620427'][2]<=budget['one_remaining_checkout']['seconds'])
check('one recovery checkout core within maximum',float(core(('64620427',)))<=budget['one_remaining_checkout']['max_core_hours'])
check('stage GPU below release ceiling and approved cap',float(stage_gpu)<=budget['stage_gpu_hours_max']<=budget['stage_gpu_hours_cap'])
check('stage attached CPU below release ceiling and approved cap',float(stage_core)<=budget['stage_attached_core_hours_max']<=budget['stage_attached_core_hours_cap'])
check('P1 cumulative GPU below ceiling and subcap',float(p1_gpu)<=budget['p1_cumulative_gpu_hours_max']<=budget['p1_cumulative_gpu_hours_cap'])
check('P1 cumulative attached CPU below ceiling and subcap',float(p1_core)<=budget['p1_cumulative_core_hours_max']<=budget['p1_cumulative_core_hours_cap'])
check('exactly three total P1 attempts',len(p1)==3 and sum(jobs[j][0]=='FAILED' for j in p1)==2 and jobs[p1[-1]][0]=='COMPLETED')
cpu_envelope=Fraction(str(budget['cpu_only_prior_envelope']))+checkout_core
check('CPU envelope with two actual checkouts within release ceiling',float(cpu_envelope)<=budget['cpu_only_envelope_max'])
check('CPU future-reader envelope remains within 351',float(cpu_envelope)<=budget['cpu_only_cap']==351)
close('diagnostic checkout reconciles previous accounting',core(('64610004',)),budget['completed_diagnostic_checkout_core_hours'])

pin=snapshot['files']['p1/receipt.json']
check('receipt byte count and SHA match collected evidence',(BASE/'p1/receipt.json').stat().st_size==pin['bytes'] and sha(BASE/'p1/receipt.json')==pin['sha256'])
check('receipt belongs to successful recovery allocation',receipt['complete'] is True and receipt['slurm_job_id']=='64620434' and receipt['logical_job_id']=='SCALING-300M-eval19618-p1-r2-20261116')
check('receipt timing bounded by billed parent allocation',0<receipt['elapsed_seconds']<=jobs['64620434'][2])
check('null diagnostic retained without causal assertion',receipt['diagnostic_outcome']=='NOT_REPRODUCED' and receipt['release_criterion']=='scaling-p1-single-recovery-after-clean-diagnostic-v2')
check('failed history preserved',receipt['failed_attempts_preserved']==[{'job_id':'64572000','state':'FAILED','exit_code':'1:0'},{'job_id':'64596915','state':'FAILED','exit_code':'1:0'}])
check('receipt code equals collected checkout HEAD',snapshot['commands']['code_head']['exit']==0 and receipt['code_sha']==snapshot['commands']['code_head']['stdout'].strip())
check('old and new checkout clean',all(snapshot['commands'][k]['exit']==0 and snapshot['commands'][k]['stdout']=='' for k in ('code_status','old_code_status')))

costs={
 'p1_r2_gpu_hours':float(new_gpu),'p1_r2_attached_core_hours':float(new_core),
 'p1_cumulative_gpu_hours':float(p1_gpu),'p1_cumulative_attached_core_hours':float(p1_core),
 'stage_gpu_hours_including_diagnostic':float(stage_gpu),'stage_attached_core_hours_including_diagnostic':float(stage_core),
 'diagnostic_checkout_core_hours':float(core(('64610004',))),
 'recovery_checkout_core_hours':float(core(('64620427',))),
 'two_checkouts_core_hours':float(checkout_core),
 'stage_attached_plus_two_checkouts_core_hours':float(stage_core+checkout_core),
 'cpu_only_prior_spent_plus_future_envelope':budget['cpu_only_prior_envelope'],
 'cpu_only_spent_plus_future_envelope_with_actual_checkouts':float(cpu_envelope),
 'cpu_only_approved_cap':351,
}
artifact={
 'verdict':'CONFORME','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'snapshot_at':snapshot['at'],'scope':'Read-only local accounting review of completed P1-r2, prior allocations and CPU-only envelope',
 'parent_allocations':{k:dict(zip(('state','exit_code','elapsed_seconds','cpus','gpus'),v)) for k,v in jobs.items()},
 'ignored_accounting_steps':ignored,'computed':costs,'checks':checks,'passed':len(checks),'failed':0,
 'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (BASE/'sacct.txt',BASE/'snapshot.json',BASE/'p1/receipt.json',BUDGET)},
 'interpretation':['P1-r2 has completed and consumed 1772 parent-allocation seconds, not its smaller wrapper-only timing.',
 'All GPU-associated CPU and both CPU-only checkouts remain within their approved separate envelopes.',
 'The CPU-only value includes the existing future-reader envelope; it is not measured expenditure of all readers.',
 'The three-attempt P1 scope is exhausted; unused numerical budget does not authorize another attempt.',
 'No new submissions, reader plans, numerical scores or future launches were reviewed or authorized.'],
}
OUT.write_text(json.dumps(artifact,indent=2)+'\n')
print(json.dumps({'audit':str(OUT),'verdict':artifact['verdict'],'passed':len(checks),'computed':costs}))
