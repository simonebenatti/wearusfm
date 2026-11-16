import hashlib,json,math,re,datetime,xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-p1-recovery')
BASE=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_release_20261116'
OLD=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_20261116'
P=BASE/'evidence/preflight'
OUT=Path('/private/tmp/p1-release-v2-budget-review-20261116.json')
checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,value):
    checks.append({'name':name,'passed':bool(value)})
    assert value,name
def equal(name,a,b):check(name,math.isclose(a,b,rel_tol=0,abs_tol=1e-9))
def load(p):return json.loads(Path(p).read_text())
b=load(BASE/'budget.json');a=load(BASE/'criterion_amendment.json');r=load(P/'remote.json')
account=load(ROOT/b['accounting']['path'])
check('accounting pinned',sha(ROOT/b['accounting']['path'])==b['accounting']['sha256'])
check('amendment pinned',sha(BASE/'criterion_amendment.json')==b['criterion_amendment_sha256'])
check('explicit single extraction amendment',a['approved'] is True and a['approved_by']=='Simone' and a['max_new_extractions']==1 and a['retry_authorized'] is False and a['cause_established'] is False)
check('actual diagnostic outcome retained',a['diagnostic_outcome']=='NOT_REPRODUCED' and a['diagnostic_readers_status']=='READERS_CLEAR_IN_TEST')
for name,item in r['commands'].items():
    check('command succeeded '+name,type(item['exit']) is int and item['exit']==0)
    check('raw stdout retained '+name,(P/(name+'.txt')).read_text()==item['stdout'])
check('empty queue is successful observation',r['commands']['squeue']['stdout']=='')
check('old checkout unchanged',r['commands']['old_code_head']['stdout'].strip()=='df2de35c05381e7a5a69edde409171b210a3b5d8' and r['commands']['old_code_status']['stdout']=='')
check('new ref absent',r['commands']['deploy_ref']['stdout']=='')
check('scheduler overtime allowance',re.search(r'^OverTimeLimit\s*=\s*3 min$',r['commands']['config']['stdout'],re.M) is not None)
check('WORK quota room',re.search(r'933\.1G\s+1T\s+',r['commands']['quota']['stdout']) is not None)
for path,state in r['new_paths'].items():
    check('new path absent '+path,state['exists'] is False and state['is_symlink'] is False and state['canonical']==path)
for name,files in r['failed_directories'].items():
    check('no failed receipt adoption '+name,'receipt.json' not in files)
expected={'64571999':('COMPLETED','0:0',7330,8,1),'64572000':('FAILED','1:0',1882,8,1),'64572001':('COMPLETED','0:0',5967,8,1),'64596915':('FAILED','1:0',1793,8,1),'64610004':('COMPLETED','0:0',41,2,0),'64610181':('COMPLETED','0:0',977,8,1)}
jobs={}
for line in r['commands']['sacct']['stdout'].splitlines():
    fields=line.split('|')
    if not fields[0].isdigit():continue
    j,state,exit_code,elapsed,cpus,tres=fields[:6]
    counts=dict(v.split('=',1) for v in tres.split(','))
    data=(state,exit_code,int(elapsed),int(cpus),int(counts.get('gres/gpu',0)))
    check('unique accounting parent '+j,j not in jobs)
    jobs[j]=data
check('six exact historical allocation records',jobs==expected)
gpu=sum(v[2]*v[4]/3600 for v in jobs.values())
core=sum(v[2]*v[3]/3600 for v in jobs.values() if v[4])
checkout=jobs['64610004'][2]*jobs['64610004'][3]/3600
equal('stage actual GPU',gpu,b['stage_gpu_hours_consumed'])
equal('previous accounting GPU',gpu,account['extractions_plus_diagnostic_gpu_hours'])
equal('previous accounting attached core',core,account['extractions_plus_diagnostic_attached_core_hours'])
new=b['p1_r2'];next_gpu=(new['seconds']+new['margin_seconds'])*new['gpus']/3600
next_core=(new['seconds']+new['margin_seconds'])*new['cpus']/3600
equal('new P1 ceiling',next_gpu,2.05);equal('new P1 attached core ceiling',next_core,16.4)
equal('stage ceiling',gpu+next_gpu,b['stage_gpu_hours_max'])
equal('stage attached core ceiling',core+next_core,b['stage_attached_core_hours_max'])
p1=sum(jobs[j][2]/3600 for j in ('64572000','64596915'))+next_gpu
equal('P1 cumulative ceiling',p1,b['p1_cumulative_gpu_hours_max'])
equal('P1 attached core ceiling',p1*8,b['p1_cumulative_core_hours_max'])
check('GPU and attached CPU caps',b['stage_gpu_hours_max']<=7.3 and b['stage_attached_core_hours_max']<=58.4 and p1<=3.1 and p1*8<=24.8)
c=b['one_remaining_checkout'];next_checkout=(c['seconds']+c['margin_seconds'])*c['cpus']/3600
equal('one remaining checkout ceiling',next_checkout,c['max_core_hours'])
equal('actual diagnostic checkout cost',checkout,b['completed_diagnostic_checkout_core_hours'])
cpu=b['cpu_only_prior_envelope']+checkout+next_checkout
equal('CPU-only envelope',cpu,b['cpu_only_envelope_max']);check('CPU cap',cpu<=351)
prep=load(OLD/'evidence/preparation/remote_metadata.json')
for path,value in r['files'].items():
    if 'text' not in value:continue
    raw=value['text'].encode()
    check('collected file bytes and SHA '+path,len(raw)==value['bytes'] and hashlib.sha256(raw).hexdigest()==value['sha256'])
    if path.endswith('/pose/receipt.json') or path.endswith('/p2/receipt.json'):
        prior=prep['files'][path]
        check('preserved accepted receipt '+path,prior['sha256']==value['sha256'])
    if path.endswith('/diag/scaling_p1_identity_20261116_v1/report.json'):
        check('diagnostic exact archived pin',value['sha256']==a['diagnostic_report_sha256']==sha(OLD/'evidence/run_final/report.json'))
ledger=Path.home()/'wearusfm_local/jobs'
ids=['SCALING-P1-recovery-checkout-20261116','SCALING-300M-eval19618-p1-r2-20261116']
for logical in ids:
    check('snapshot unused ID '+logical,r['local_ledger'][logical].get('unused') is True)
    check('current local ID directory absent '+logical,not (ledger/logical).exists())
    check('current local ledger lacks ID '+logical,logical not in (ledger/'ledger.tsv').read_text())
t=load(BASE/'cpu_tests.json')
check('CPU complete test receipt',t['complete'] is True and t['passed'] is True and t['counts']['passed']==163 and t['failed']==0 and t['exit']==0)
check('CPU log SHA',sha(BASE/'cpu_tests.log')==t['log_sha256'])
check('CPU junit SHA',sha(BASE/'cpu_tests.junit.xml')==t['junit_sha256'])
suite=ET.parse(BASE/'cpu_tests.junit.xml').getroot().find('testsuite')
check('CPU junit counts',int(suite.attrib['tests'])==163 and all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped')))
for path,pin in t['source_sha256'].items():check('CPU source pin '+path,sha(ROOT/path)==pin)
result={'verdict':'APPROVATO','scope':'Independent local read-only accounting and operational preflight review; no SSH or submissions','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_at':r['at'],'checks':checks,'passed':len(checks),'failed':0,'computed':{'stage_gpu_actual':gpu,'stage_gpu_max':gpu+next_gpu,'stage_attached_core_max':core+next_core,'p1_gpu_max':p1,'p1_core_max':p1*8,'cpu_only_max':cpu},'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (BASE/'budget.json',BASE/'criterion_amendment.json',BASE/'cpu_tests.json',P/'remote.json',P/'sacct.txt')},'limitations':['Snapshot establishes state only at collection; final submitter reservation remains authoritative.','Checkpoint file stat recorded in preflight; original wrapper must perform its full runtime checkpoint SHA/identity guard.','No review of submission plan or manifest in this accounting/preflight artifact.']}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'verdict':result['verdict'],'passed':len(checks),'computed':result['computed'],'audit':str(OUT)}))
