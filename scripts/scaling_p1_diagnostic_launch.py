"""Fail closed on the diagnostic allocation and new, contained paths; stdlib only."""
import argparse
import hashlib
from pathlib import Path
import re

WORK = Path('/leonardo_work/IscrB_WearUsFM')
DIAG = Path('/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/diag')

def validate(code, manifest, manifest_sha256, out, job, job_info, work=WORK, diag=DIAG):
    code, out = Path(code), Path(out)
    if not code.is_absolute() or code.resolve(strict=True) != code or not code.is_relative_to(work/'wearusfm_campaigns'):
        raise ValueError('canonical frozen campaign path required')
    rel = Path(manifest)
    path = (code/rel).resolve(strict=True)
    if rel.is_absolute() or not path.is_relative_to(code) or not path.is_file():
        raise ValueError('manifest must be contained in frozen code')
    if re.fullmatch('[0-9a-f]{64}',manifest_sha256) is None or hashlib.sha256(path.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError('manifest pin mismatch')
    if not out.is_absolute() or out.resolve() != out or not out.is_relative_to(diag) or out == diag or out.exists() or out.is_symlink():
        raise ValueError('new canonical diagnostic output required')
    fields = dict(part.split('=',1) for part in job_info.split() if '=' in part)
    expected={'JobId':str(job),'JobState':'RUNNING','NumNodes':'1','NumCPUs':'8','NumTasks':'1',
              'Partition':'boost_usr_prod','QOS':'boost_qos_dbg','TimeLimit':'00:27:00'}
    if not str(job).isdigit() or int(job)<=0 or any(fields.get(k)!=v for k,v in expected.items()):
        raise ValueError('actual diagnostic job/allocation mismatch')
    tres=dict(part.split('=',1) for part in fields.get('AllocTRES','').split(',') if '=' in part)
    if any(tres.get(k)!=v for k,v in {'cpu':'8','node':'1','gres/gpu':'1'}.items()) or tres.get('mem') not in ('32G','32768M'):
        raise ValueError('actual diagnostic resources differ from 1GPU/8CPU/32GiB')
    return True

def main():
    p=argparse.ArgumentParser()
    for name in ('code','manifest','manifest-sha256','out','job','job-info'):p.add_argument('--'+name,required=True)
    validate(**vars(p.parse_args()))

if __name__=='__main__':main()
