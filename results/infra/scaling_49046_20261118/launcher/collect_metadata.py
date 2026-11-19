from pathlib import Path
import ast,json,subprocess,shlex
HERE=Path(__file__).resolve().parent
source=ast.parse((HERE/'collect_inputs.py').read_text())
remote=next(n.value.value for n in ast.walk(source) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets)).replace('2000000','16000000').replace('capture_output=True,text=True','stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True')
manifest=json.loads((HERE/'control_line2_evidence/cache_manifest.json').read_text())
files=[v['receipt']['path'] for kind in ('classification','pose') for v in manifest[kind]['49046'].values()]
files+=['/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105/fits/pose/49046/report.json']
q={'candidate':'/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/scaling_300m_20261104/T100-G-300M-19618-from5000-v3','files':files}
command=['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','python3 -c '+shlex.quote(remote)]
r=subprocess.run(command,input=json.dumps(q),capture_output=True,text=True,timeout=180)
assert r.returncode==0,r.stderr
out=HERE/'metadata_snapshot.json';out.write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n');print(out)
