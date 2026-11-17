"""Bounded read-only CPU-reader preparation snapshot. No jobs or remote mutations."""
from pathlib import Path
import json,shlex,subprocess

BASE=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import json,subprocess,datetime,hashlib
root=Path('/leonardo_work/IscrB_WearUsFM')
code=root/'wearusfm_campaigns/scaling_300m_evaluation19618_20261106_v1'
r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands={},paths={},files={})
commands={
 'squeue_all':['squeue','-u','sbenatti','-h','-o','%i|%j|%T|%q|%P|%R'],
 'squeue_serial':['squeue','-u','sbenatti','-p','lrd_all_serial','-h','-o','%i|%j|%T|%q|%R'],
 'quota':['lfs','quota','-p','20147512',str(root)],
 'scheduler_config':['scontrol','show','config'],
 'serial_partition':['scontrol','show','partition','lrd_all_serial','-o'],
 'login08':['scontrol','show','node','login08','-o'],
 'code_head':['git','-C',str(code),'rev-parse','HEAD'],
 'code_status':['git','-C',str(code),'status','--porcelain','--untracked-files=all'],
 'sacct':['sacct','-n','-P','-j','64194074,64571999,64572001,64620434','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,NodeList,Partition,Start,End,Timelimit,AllocTRES'],
 'deploy_refs':['git','-C',str(root/'wearusfm'),'for-each-ref','--format=%(refname)|%(objectname)','refs/heads/deploy/scaling_300m_plans19618_20261106_v1','refs/heads/deploy/scaling_300m_budgets19618_20261106_v1','refs/heads/deploy/scaling_300m_inputs19618_20261106_v1'],
 'repo_config':['git','-C',str(root/'wearusfm'),'config','--local','--list']}
for k,args in commands.items():
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
 r['commands'][k]=dict(argv=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
 if p.returncode:raise RuntimeError(k+': '+p.stderr)
for name in ['plans','budgets','inputs']:
 p=root/('wearusfm_campaigns/scaling_300m_'+name+'19618_20261106_v1');r['paths'][str(p)]=dict(exists=p.exists(),symlink=p.is_symlink())
p=root/'wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/cpu';r['paths'][str(p)]=dict(exists=p.exists(),listing=sorted(q.name for q in p.iterdir()) if p.exists() else [])
for name in ['pre-receive','post-receive','update','post-update']:
 p=root/'wearusfm/.git/hooks'/name
 r['paths'][str(p)]=dict(exists=p.exists(),executable=p.exists() and bool(p.stat().st_mode&0o111))
for name in ['scripts/slurm/scaling_readers19618.sbatch','scripts/scaling_pose_readout19618.py','scripts/scaling_probe19618.py','scripts/scaling_compare19618.py','scripts/scaling_reading19618.py','scripts/submit_job.py']:
 p=code/name;b=p.read_bytes();r['files'][name]=dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
print(json.dumps(r))
'''
def main():
    out=BASE/'preflight'
    if out.exists():raise SystemExit('new preflight directory required')
    p=subprocess.run(['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','python3 -c '+shlex.quote(REMOTE)],capture_output=True,text=True,timeout=180)
    if p.returncode:raise SystemExit(p.stderr)
    result=json.loads(p.stdout)
    ledger=Path.home()/'wearusfm_local/jobs'
    result['ledger']={q.parent.name:json.loads(q.read_text()) for q in sorted(ledger.glob('SCALING-19618-*/state.json'))}
    out.mkdir(parents=True)
    (out/'snapshot.json').write_text(json.dumps(result,indent=2)+'\n');(out/'ssh.stderr.txt').write_text(p.stderr)
    for k,v in result['commands'].items():(out/(k+'.txt')).write_text(v['stdout'])
    print(json.dumps(dict(at=result['at'],queue=result['commands']['squeue_all']['stdout'],quota=result['commands']['quota']['stdout'],paths=result['paths'],ledger=list(result['ledger'])),indent=2))
if __name__=='__main__':main()
