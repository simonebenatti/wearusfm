"""Read-only Leonardo collection; never submits, cancels or writes remotely."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'results/infra/pose_temporal_20261102/execution'
REGISTRY = Path('/Users/simonebenatti/wearusfm_local/jobs')
records = []
for stage in ('checkout', 'validate', 'benchmark', 'run'):
    path = REGISTRY / f'POSE-TEMP-{stage}-20261102/state.json'
    if path.exists():
        state = json.loads(path.read_text())
        if state.get('job_slurm'):
            records.append((stage, state))
ids = [str(s['job_slurm']) for _, s in records]
assert ids and all(x.isdecimal() for x in ids)
base = '/leonardo_work/IscrB_WearUsFM/wearusfm_runs'
paths = []
for stage, state in records:
    jid = state['job_slurm']
    paths.append(f'{base}/logs/' + (f'campaign_checkout_{jid}.out' if stage == 'checkout' else f'pose_temporal_{jid}.out'))
    if stage != 'checkout':
        paths.append(f'{base}/results/pose_temporal_20261102/{stage}.json')
program = '''import json,subprocess,hashlib
from pathlib import Path
ids=IDS
paths=PATHS
commands={'sacct':['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,JobName%60,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,Start,End'],
          'squeue':['squeue','-h','-j',','.join(ids),'-o','%i|%j|%T|%R']}
out={'commands':{},'files':{}}
for key,command in commands.items():
    p=subprocess.run(command,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out['commands'][key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for name in paths:
    p=Path(name)
    if p.is_file():
        raw=p.read_bytes()
        out['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode('utf-8')}
print(json.dumps(out))
'''.replace('IDS',repr(ids)).replace('PATHS',repr(paths))
remote = 'python3 -c ' + shlex.quote(program)
proc = subprocess.run(['ssh','leonardo',remote],capture_output=True,text=True,timeout=90)
if proc.returncode:
    raise SystemExit(proc.stderr)
payload = json.loads(proc.stdout)
assert payload['commands']['sacct']['returncode'] == 0, payload['commands']['sacct']
DEST.mkdir(parents=True,exist_ok=True)
timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest=[]
for remote_path,item in payload['files'].items():
    target=DEST/Path(remote_path).name
    raw=item['text'].encode('utf-8')
    assert hashlib.sha256(raw).hexdigest()==item['sha256']
    target.write_bytes(raw)
    manifest.append({'remote_path':remote_path,'local_name':target.name,'sha256':item['sha256'],'bytes':len(raw)})
for name,item in payload['commands'].items():
    (DEST/(name+'.txt')).write_text(item['stdout'])
    if item['stderr']: (DEST/(name+'.stderr.txt')).write_text(item['stderr'])
(DEST/'submissions.json').write_text(json.dumps([{'stage':stage,**state} for stage,state in records],indent=2)+'\n')
(DEST/'collection.json').write_text(json.dumps({'collected_at_utc':timestamp,'files':manifest,'ids':ids,'read_only_remote':True,'squeue_returncode':payload['commands']['squeue']['returncode']},indent=2)+'\n')
print(json.dumps({'collected_at_utc':timestamp,'ids':ids,'files':[x['local_name'] for x in manifest],'sacct':payload['commands']['sacct']['stdout'],'squeue':payload['commands']['squeue']['stdout']},indent=2))
