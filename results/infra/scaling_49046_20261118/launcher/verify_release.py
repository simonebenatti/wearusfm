"""One CPU regression run, captured with source pins. No network or job submission."""
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[4]
# Location is results/infra/<campaign>/launcher: parents[4] is the repository.
FILES=sorted(set(ROOT.glob('tests/cpu/test_scaling*49046*.py'))|{ROOT/'tests/cpu/test_scaling_output_identity.py'})
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1'}

def sources():
    paths=sorted(set(ROOT.glob('scripts/*49046.py'))|set(ROOT.glob('scripts/slurm/*49046.sbatch'))|
                 {ROOT/'scripts/scaling_terminal_builders.py',ROOT/'scripts/scaling_dose_interaction.py'}|set(FILES))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    initial=sources();start=time.monotonic()
    command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*[str(p.relative_to(ROOT)) for p in FILES]]
    log=Path(__file__).parent/'verification_stdout.txt'
    with log.open('w') as stream:
        result=subprocess.run(command,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=600)
    output=log.read_text()
    print(output,flush=True)
    unchanged=sources()==initial
    record={'utc':str(datetime.datetime.now(datetime.timezone.utc)),'command':command,'environment':{k:ENV[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH')},
            'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,'sources_unchanged':unchanged,
            'source_sha256':initial,'test_files':len(FILES),'complete':result.returncode==0 and unchanged}
    (Path(__file__).parent/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    return 0 if record['complete'] else 1
if __name__=='__main__':sys.exit(main())
