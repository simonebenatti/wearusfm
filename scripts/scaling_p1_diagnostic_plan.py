"""Build diagnostic inputs and amended costs from published evidence; Mac only."""
import hashlib
import json
from pathlib import Path
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_20261116'
OLD='df2de35c05381e7a5a69edde409171b210a3b5d8'
REMOTE='/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/'
OUTPUT=REMOTE+'scaling_300m_20261104/evaluation19618_v1/T100-G-300M/'
CONTROL=REMOTE+'tokenizer_evaluation_20261103/19618/T100-G/p1/'

def digest(data):return hashlib.sha256(data).hexdigest()
def check(ok,message):
    if not ok:raise ValueError(message)

def build(evidence,cpu_pin,probe_inventory,extraction_plan):
    cmds=evidence['commands'];files=evidence['files']
    check(all(type(v['exit']) is int and v['exit']==0 for v in cmds.values()),'failed remote collection')
    check(cmds['code_head']['stdout'].strip()==OLD and not cmds['code_status']['stdout'],'frozen source checkout changed')
    check('OverTimeLimit           = 3 min' in cmds['config']['stdout'],'scheduler grace not confirmed')
    jobs={}
    for line in cmds['sacct']['stdout'].splitlines():
        row=line.split('|')
        if row[0].isdigit():jobs[row[0]]=row
    expected={'64571999':('COMPLETED',7330),'64572000':('FAILED',1882),'64572001':('COMPLETED',5967),'64596915':('FAILED',1793)}
    for job,(state,elapsed) in expected.items():
        r=jobs[job]
        check((r[1],r[2],int(r[3]),int(r[4]))==(state,'0:0' if state=='COMPLETED' else '1:0',elapsed,8),'historical accounting changed')
        check('gres/gpu=1' in r[5],'historical GPU allocation changed')
    spec=[('p1_db6_failed',OUTPUT+'p1/ninapro_db6.npz',None),
          ('p1r1_db3_failed',OUTPUT+'p1-r1/ninapro_db3.npz',None),
          ('p1r1_db2_completed',OUTPUT+'p1-r1/ninapro_db2.npz','sidecar'),
          ('pose_val',OUTPUT+'pose/val.npz','val'),
          ('p2_epn',OUTPUT+'p2/epn612.npz','epn612'),
          ('control30m_db3',CONTROL+'ninapro_db3.npz','ninapro_db3')]
    inputs=[]
    for ident,path,key in spec:
        r=files[path];check(r['canonical']==path and not r['symlink'],'noncanonical cache')
        pin=None;kind='diagnostic';extra={}
        if key:
            provpath=path+'.provenance.json';prov=files[provpath]
            check(digest(prov['text'].encode())==prov['sha256'],'sidecar collected hash differs')
            p=json.loads(prov['text']);check(p['cache']==path and p['cache_bytes']==r['bytes'],'sidecar size/path differ')
            pin=p['cache_sha256'];extra={'provenance_path':provpath,'provenance_sha256':prov['sha256']}
            if key!='sidecar':
                kind='accepted';receiptpath=str(Path(path).parent/'receipt.json');receipt=files[receiptpath]
                check(digest(receipt['text'].encode())==receipt['sha256'],'receipt collected hash differs')
                q=json.loads(receipt['text']);cache=q['outputs'][key]
                check(q['complete'] is True and q['checkpoint_unchanged'] is True and q['checkpoint_step']==19618,'unaccepted receipt')
                check(cache['path']==path and cache['sha256']==pin and cache['bytes']==r['bytes'] and cache['provenance_sha256']==prov['sha256'],'receipt/sidecar mismatch')
                if ident=='control30m_db3':
                    old=probe_inventory['caches']['T100-G']['ninapro_db3']
                    check(all(old[k]==v for k,v in {'path':path,'sha256':pin,'receipt_sha256':receipt['sha256'],'provenance_sha256':prov['sha256']}.items()),'historical control pin differs')
                else:
                    check(q['code_sha']==OLD and q['checkpoint_sha256']==extraction_plan['checkpoints']['T100-G-300M']['sha256'],'accepted 300M lineage differs')
                extra.update(receipt_path=receiptpath,receipt_sha256=receipt['sha256'])
        inputs.append({'id':ident,'path':path,'bytes':r['bytes'],'sha256':pin,'pin_kind':kind,**extra})
    manifest={'protocol':'SCALING-P1-IDENTITY-DIAG-v1','frozen_code_dir':'/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_evaluation19618_20261106_v1',
              'frozen_code_sha':OLD,'frozen_sources':evidence['sources'],'inputs':inputs,'candidate_tests':cpu_pin,
              'approval_commit':'81f1f76','validation_order':'legacy SHA; 2s pause; synthetic NPZ validation; descriptor SHA',
              'recovery_output_barrier':'none','no_failed_cache_adoption':True}
    seconds=sum(v[1] for v in expected.values());p1seconds=1882+1793
    estimate=Fraction(seconds+1800+7380,3600);p1=Fraction(p1seconds+7380,3600)
    budget={'protocol':'scaling-p1-budget-amendment-20261116-v1','approved':True,'approval_commit':'81f1f76',
      'approval_quote':'approvo tutto, correggi e prepara tutto e lancia tu se non hai altre cose da correggere',
      'accounting_source':'evidence/preparation/sacct.txt','accounting_sha256':digest(cmds['sacct']['stdout'].encode()),
      'historical_gpu_hours':seconds/3600,'historical_attached_core_hours':seconds*8/3600,
      'diagnostic':{'requested_seconds':1620,'scheduler_margin_seconds':180,'gpus':1,'cpus':8,'memory_gib':32,'gpu_hours_cap':.5,'core_hours_cap':4},
      'conditional_p1':{'requested_seconds':7200,'scheduler_margin_seconds':180,'gpus':1,'cpus':8,'memory_gib':100,'gpu_hours_cap':2.05,'core_hours_cap':16.4},
      'computed_stage_gpu_hours_max':float(estimate),'computed_stage_attached_core_hours_max':float(estimate*8),
      'stage_gpu_hours_cap':7.30,'stage_attached_core_hours_cap':58.40,
      'computed_p1_gpu_hours_max':float(p1),'p1_gpu_hours_cap':3.10,'p1_attached_core_hours_cap':24.80,'p1_max_attempts':3,
      'cpu_only_prior_consumed_plus_future_max':349.917778,'extra_checkouts_max':2,
      'checkout_each':{'cpus':2,'requested_seconds':300,'scheduler_margin_seconds':180,'core_hours_max':float(Fraction(960,3600))},
      'cpu_only_consumed_plus_future_max':349.917778+float(Fraction(1920,3600)),'cpu_only_cap':351,
      'retry_authorized':False,'source_freeze':'results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json'}
    check(estimate<=Fraction(730,100) and p1<=Fraction(310,100) and budget['cpu_only_consumed_plus_future_max']<=351,'budget exceeds approval')
    return manifest,budget

def main():
    evidence=json.loads((BASE/'evidence/preparation/remote_metadata.json').read_text())
    cpu=BASE/'cpu_tests.json';pin={'path':str(cpu.relative_to(ROOT)),'sha256':digest(cpu.read_bytes()),'passed':True}
    inventory=json.loads((ROOT/'results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json').read_text())
    plan=json.loads((ROOT/'results/infra/scaling_300m_20261104/continuation19618_20261106/evaluation/extraction_plan.json').read_text())
    manifest,budget=build(evidence,pin,inventory,plan)
    freeze=ROOT/budget['source_freeze']
    budget['source_freeze_sha256']=digest(freeze.read_bytes())
    budget['source_freeze_bytes']=freeze.stat().st_size
    amendment=ROOT/'results/infra/scaling_300m_20261104/continuation19618_20261111_checkout_cpus/budget_amendment.json'
    budget['prior_checkout_amendment']={'path':str(amendment.relative_to(ROOT)),
                                      'sha256':digest(amendment.read_bytes()),'bytes':amendment.stat().st_size}
    for name,obj in [('diagnostic_manifest.json',manifest),('budget_amendment.json',budget)]:
        (BASE/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
