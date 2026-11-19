"""Run or verify reproducible local CPU validation of the target amendment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contrastive_prepare import ROOT, source_digest

DEST=ROOT/'results/infra/contrastive_targets_20261118'

def pin(path):
    path=path.resolve();data=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)), 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}

def historical_logs():
    return [pin(p) for p in sorted((DEST/'full_suite').glob('*.log'))]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEST/'amendment_20261119/validation')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--verify',type=Path)
    a=parser.parse_args()
    if a.verify:
        receipt=json.loads(a.verify.read_text())
        if receipt['source_sha256']!=source_digest(): raise ValueError('validation source changed')
        for item in [*receipt['historical_logs'],*[x['log'] for x in receipt['runs']]]:
            if pin(ROOT/item['path'])!=item: raise ValueError('validation log changed: '+item['path'])
        for item in receipt['test_files']:
            if pin(ROOT/item['path'])!=item: raise ValueError('validation test changed: '+item['path'])
        print(json.dumps({'verified':True,'all_passed':all(r['returncode']==0 for r in receipt['runs'])}));return
    a.out=a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=False)
    files=sorted((ROOT/'tests/cpu').glob('test_contrastive*.py'))
    files += [ROOT/'tests/cpu/test_target_windows_fix.py',ROOT/'tests/cpu/test_rms_target_window.py',ROOT/'tests/cpu/test_ms_fast_target_window.py']
    files=sorted(set(p for p in files if p.is_file()))
    env=os.environ.copy();environment={'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'),
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    env.update(environment)
    command=[a.python,'-m','pytest','-q','-p','no:cacheprovider','-o','tmp_path_retention_policy=none',
             *[str(p.relative_to(ROOT)) for p in files]]
    source_before=source_digest();tests_before=[pin(p) for p in files]
    start=time.monotonic();log=a.out/'pytest.log'
    with log.open('wb') as stream: proc=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    historical_changed=subprocess.check_output(['git','diff','--name-only','438c8d65','--','results/infra/contrastive_comparison_20261116'],cwd=ROOT,text=True).splitlines()
    if source_digest()!=source_before or [pin(p) for p in files]!=tests_before:
        raise ValueError('sources or tests changed during local validation; do not use this log as a package receipt')
    receipt={'schema':'contrastive-local-validation-v2','source_sha256':source_before,
        'historical_artifacts_changed_from_base':historical_changed,
        'base_commit':'438c8d65','date':'2026-11-19','cwd':str(ROOT),'environment':environment,
        'producer':'scripts/contrastive_local_validation.py','cuda_tested':False,
        'runs':[{'command':command,'returncode':proc.returncode,'elapsed_s':time.monotonic()-start,'log':pin(log)}],
        'test_files':tests_before,'historical_logs':historical_logs(),
        'historical_scope':'Existing 18/11 logs retained and pinned, not rerun by this command. Undocumented old121-test selection superseded by explicit current full contrastive test list.'}
    destination=a.out/'local_validation.json';destination.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'receipt':str(destination),'returncode':proc.returncode,'log':str(log)}))
    raise SystemExit(proc.returncode)

if __name__=='__main__': main()
