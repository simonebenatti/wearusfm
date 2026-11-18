"""Reproduce the explicit old/new source and exact evidence manifest; no jobs."""
from pathlib import Path
import hashlib,json,subprocess,gzip
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
OLD_COMMIT='f67acb06ae8846acb6d967dd29c6b8d7ff09bf07'
OLD='ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b'
def sha(b): return hashlib.sha256(b).hexdigest()
def oldbytes(path): return subprocess.check_output(['git','show',OLD_COMMIT+':'+path],cwd=ROOT)
def selection(paths):
    paths=list(paths)
    return (sorted(p for p in paths if p.startswith('src/wearusfm/') and p.endswith('.py'))+
            sorted(p for p in paths if p.startswith('scripts/contrastive_') and '/' not in p[8:] and p.endswith('.py'))+
            sorted(p for p in paths if p.startswith('scripts/slurm/contrastive') and p.endswith('.sbatch'))+
            ['scripts/train_jepa.py','scripts/slurm/lib_code_dir.sh'])
oldpaths=selection(subprocess.check_output(['git','ls-tree','-r','--name-only',OLD_COMMIT],cwd=ROOT,text=True).splitlines())
newpaths=selection(str(p.relative_to(ROOT)) for base in ('src/wearusfm','scripts') for p in (ROOT/base).rglob('*') if p.is_file())
def rows(paths,read): return [{'path':p,'sha256':sha(read(p))} for p in paths]
def digest(rows): return sha(b''.join(r['path'].encode()+b'\0'+bytes.fromhex(r['sha256']) for r in rows))
oldrows=rows(oldpaths,oldbytes);newrows=rows(newpaths,lambda p:(ROOT/p).read_bytes())
assert digest(oldrows)==OLD
base=ROOT/'results/infra/contrastive_comparison_20261116'
receipt=json.loads((base/'calibration_completed_20261117/calibration_report.json').read_text())
audit=gzip.decompress((base/'audit_completed_20261117/audit.json.gz').read_bytes())
plan='results/infra/contrastive_comparison_20261116/plan.json'
assert (ROOT/plan).read_bytes()==oldbytes(plan)
manifest={'protocol':'contrastive-source-compatibility-20261118-v1','historical_source_sha256':OLD,'current_source_sha256':digest(newrows),'historical_sources':oldrows,'current_sources':newrows,'frozen_documents':{plan:{'sha256':sha((ROOT/plan).read_bytes())}},'artifacts':{'audit':{'sha256':sha(audit)},'calibration_run':{name:{'sha256':sha((base/'calibration_completed_20261117/run'/name).read_bytes())} for name in ('config.json','summary.json','metrics.jsonl')},'calibration_receipt':{'canonical_sha256':sha(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode())}}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'current_source_sha256':digest(newrows),'old_count':len(oldrows),'current_count':len(newrows),'manifest_sha256':sha((OUT/'manifest.json').read_bytes())}))
