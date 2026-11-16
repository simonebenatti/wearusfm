"""Read-only initial evidence collection after the two unique submissions."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parent
OUT = BASE / 'evidence/submitted'
plan = json.loads((BASE/'submission_plan.json').read_text())
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT/'snapshot.json').exists(), 'initial snapshot already collected'
receipts = {}
for key in ('checkout','p1'):
    item = plan[key]
    raw = (Path.home()/'wearusfm_local/jobs'/item['id']/'state.json').read_bytes()
    value = json.loads(raw)
    assert value['stato']=='sottomesso' and value['uscita']==0
    assert value['script_stdin_sha256']==item['stdin_sha256']
    args = item['argv'][item['argv'].index('--')+1:]
    expected = ['sbatch','--parsable','--job-name='+item['id'],'--comment=wearusfm:'+item['id']]
    if key=='p1':
        expected += ['--dependency=afterok:'+str(receipts['checkout']['job_slurm'])]
        assert value['codice_sha']==plan['code_sha']
    assert value['sbatch']==expected+args
    receipts[key]=value
    target = OUT/(key+'_receipt.json')
    assert not target.exists() or target.read_bytes()==raw
    target.write_bytes(raw)
assert receipts['checkout']['job_slurm']==64620427
assert receipts['p1']['job_slurm']==64620434
remote = r'''
from pathlib import Path
import json,hashlib,subprocess,datetime
code=Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_p1_recovery_20261116_v1')
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 return dict(argv=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands={},files={})
for name,args in [
 ('sacct',['sacct','-n','-P','-j','64620427,64620434','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,Start,End,Timelimit']),
 ('squeue',['squeue','-u','sbenatti','-h','-o','%i|%j|%T|%q|%R']),
 ('p1_scontrol',['scontrol','show','job','-o','64620434'])]:
 r['commands'][name]=command(args)
for name,args in [('code_head',['git','-C',str(code),'rev-parse','HEAD']),('code_status',['git','-C',str(code),'status','--porcelain','--untracked-files=all'])]:
 if code.exists():r['commands'][name]=command(args)
for name,path in [
 ('checkout_log','/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_64620427.out'),
 ('p1_slurm_log','/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_p1_recovery_64620434.out'),
 ('p1_log','/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/T100-G-300M/p1-r2/p1.log'),
 ('manifest',str(code/'results/infra/scaling_300m_20261104/p1_recovery_release_20261116/recovery_manifest.json')),
 ('helper',str(code/'scripts/scaling_output_identity.py'))]:
 p=Path(path);f=dict(path=path,exists=p.exists());r['files'][name]=f
 if p.exists():
  assert p.stat().st_size<2**20
  b=p.read_bytes();f.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),text=b.decode('utf8'))
print(json.dumps(r))
'''
p=subprocess.run(['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','python3 -'],input=remote,text=True,capture_output=True,timeout=90)
err = OUT/'ssh.stderr.txt'
if err.exists():
    err.rename(OUT/'ssh.initial_network_error.txt')
err.write_text(p.stderr)
assert p.returncode==0,p.stderr
snapshot=json.loads(p.stdout)
(OUT/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
for key,rec in snapshot['commands'].items():
    (OUT/(key+'.txt')).write_text(rec['stdout'])
for key,rec in snapshot['files'].items():
    if rec.get('exists') and key.endswith('log'):
        (OUT/(key+'.txt')).write_text(rec['text'])
assert snapshot['commands']['sacct']['exit']==0
assert snapshot['commands']['squeue']['exit']==0
if 'code_head' in snapshot['commands']:
    assert snapshot['commands']['code_head']['exit']==0
    assert snapshot['commands']['code_head']['stdout'].strip()==plan['code_sha']
    assert snapshot['commands']['code_status']['exit']==0
    assert not snapshot['commands']['code_status']['stdout'].strip()
if snapshot['files']['manifest'].get('exists'):
    assert snapshot['files']['manifest']['sha256']==plan['manifest_sha256']
summary=dict(at=snapshot['at'],code_sha=plan['code_sha'],receipts_match_plan=True,
             jobs={k:v['job_slurm'] for k,v in receipts.items()},
             sacct=snapshot['commands']['sacct']['stdout'],squeue=snapshot['commands']['squeue']['stdout'],
             final_acceptance=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
