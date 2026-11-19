"""Reproduce local checks for the revision of Claude review 527f8da; no cluster calls."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from pose_temponet import provenance as P

def pin(path):
    path=path.resolve();data=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    dest=args.out.resolve();dest.mkdir(parents=True,exist_ok=False)
    source=P.source_identity();runtime=sorted(ROOT.glob('tests/cpu/test_pose_temponet*.py'))
    historical=[ROOT/'tests/cpu'/('test_'+n+'.py') for n in ('pose_dynamics','tokenizer_pose_readout','consolidation_long_pose','consolidation_long_pose_runtime','consolidation_long_common')]
    tests_before=[pin(p) for p in runtime+historical]
    env=os.environ.copy();settings={'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'};env.update(settings)
    runs=[]
    for name,files in [('runtime',runtime),('historical',historical)]:
        command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--tb=short','-o','tmp_path_retention_policy=none',*[str(p.relative_to(ROOT)) for p in files]]
        log=dest/(name+'.log');start=time.monotonic()
        with log.open('wb') as stream: result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        text=log.read_text();failures=re.findall(r'^FAILED (.+?)(?: - |$)',text,re.M)
        expected=['tests/cpu/test_consolidation_long_common.py::test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned']
        accepted=result.returncode==0 if name=='runtime' else result.returncode in (0,1) and failures in ([],expected)
        runs.append({'name':name,'command':command,'returncode':result.returncode,'elapsed_s':time.monotonic()-start,'log':pin(log),'failures':failures,'expected_baseline_failure_only':name=='historical' and failures==expected,'accepted':accepted})
    if source!=P.source_identity() or tests_before!=[pin(p) for p in runtime+historical]:
        raise RuntimeError('source or tests changed during validation')
    data={'base_commit':'2f982114719dc2abd723bdbf0147d7fef34a731c','review_commit':'527f8dae716398203f8f2d7413d93494f020e185','source_sha256':source,'cwd':str(ROOT),'environment':settings,'test_files':tests_before,'runs':runs,'cuda_tested':False,'no_new_regression':all(r['accepted'] for r in runs)}
    (dest/'local_validation.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'receipt':str(dest/'local_validation.json'),'no_new_regression':data['no_new_regression']}))
    raise SystemExit(0 if data['no_new_regression'] else 1)
if __name__=='__main__':main()
