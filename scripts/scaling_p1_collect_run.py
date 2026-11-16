"""Read one scheduler/output snapshot of the already submitted P1 diagnostic."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_20261116'
NAMES=('SCALING-P1-identity-checkout-20261116','SCALING-P1-identity-diag-20261116')
REMOTE=r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
jobs=JOBS
out=Path('/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/diag/scaling_p1_identity_20261116_v1')
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':{},'files':{}}
for name,args in [('sacct',['sacct','-n','-P','-j',','.join(jobs),'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS,NodeList,Start,End,Timelimit']),('squeue',['squeue','-h','-j',','.join(jobs),'-o','%i|%j|%T|%R'])]:
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)
 result['commands'][name]={'argv':args,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
paths=[out/'report.json',out/'events.jsonl',Path('/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_'+jobs[0]+'.out'),Path('/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_p1_identity_diag_'+jobs[1]+'.out')]
for path in paths:
 if path.exists():
  if path.stat().st_size>16000000: raise ValueError('unexpected large evidence')
  data=path.read_bytes();result['files'][path.name]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()}
print(json.dumps(result))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('new snapshot directory required')
    states=[json.loads((Path.home()/'wearusfm_local/jobs'/name/'state.json').read_text()) for name in NAMES]
    if not all(s.get('stato')=='sottomesso' for s in states):raise SystemExit('both logical jobs must be submitted')
    ids=[str(s['job_slurm']) for s in states]
    code=REMOTE.replace('JOBS',repr(ids))
    r=subprocess.run(['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','python3 -c '+shlex.quote(code)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
    if r.returncode:raise SystemExit(r.stderr)
    evidence=json.loads(r.stdout);out.mkdir(parents=True)
    (out/'snapshot.json').write_text(json.dumps(evidence,indent=2)+'\n');(out/'ssh.stderr.txt').write_text(r.stderr)
    for key,item in evidence['commands'].items():(out/(key+'.txt')).write_text(item['stdout'])
    for key,item in evidence['files'].items():(out/key).write_text(item['text'])
    print(evidence['commands']['sacct']['stdout']);print(evidence['commands']['squeue']['stdout'])
    if 'report.json' in evidence['files']:
        report=json.loads(evidence['files']['report.json']['text'])
        print(json.dumps({k:report.get(k) for k in ('complete','outcome','readers_status','elapsed_seconds','metadata_only_cases','errors')}))
    elif 'events.jsonl' in evidence['files']:
        events=evidence['files']['events.jsonl']['text'].splitlines()
        for line in events[-2:]:
            item=json.loads(line);print(json.dumps({'kind':item['kind'],'elapsed_seconds':item['elapsed_seconds']}))

if __name__=='__main__':main()
