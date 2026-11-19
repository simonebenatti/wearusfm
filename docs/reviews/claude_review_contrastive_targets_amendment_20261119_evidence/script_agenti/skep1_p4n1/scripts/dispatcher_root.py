"""Reproduce: dispatcher accepts any fresh out below $WORK/wearusfm_runs, incl. the OLD cohort root."""
import hashlib, json, os, sys, tempfile
from pathlib import Path
work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
os.environ['WORK'] = str(work)
import contrastive_job as J
from contrastive_prepare import source_digest
dummy = work / 'inputs'; dummy.mkdir(exist_ok=True)
def pin(name, text='x'):
    p = dummy / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
job = {'status': 'APPROVED_FOR_PREFLIGHT', 'mode': 'readout', 'source_sha256': source_digest(),
       'review': pin('review.json'), 'cohort': pin('cohort.json'), 'cuda_report': pin('cuda.json'),
       'audit': pin('audit.json'), 'calibration_report': pin('calibration_report.json'),
       'calibration_summary': pin('cal/summary.json'),
       'full': {a: pin(f'full/{a}/summary.json') for a in ('C0','CN','CR')},
       'resumed': {a: pin(f'res/{a}/summary.json') for a in ('CN','CR')}}
for label, out in (('fresh_root dichiarata', work/'wearusfm_runs/results/contrastive_targets_20261118/readout.json'),
                   ('radice VECCHIA 20261116_v2', work/'wearusfm_runs/results/contrastive_preflight_20261116_v2/readout_new.json'),
                   ('fuori da wearusfm_runs', work/'elsewhere/readout.json')):
    job['out'] = str(out)
    try:
        J.build_command(job, 'cpu', 3600); print(label, '-> ACCETTATO dal dispatcher')
    except ValueError as e:
        print(label, '-> RIFIUTATO:', e)
# existing output is rejected
old = work/'wearusfm_runs/results/contrastive_targets_20261118/calibration'; old.mkdir(parents=True, exist_ok=True)
job['out'] = str(old)
try:
    J.build_command(job, 'cpu', 3600); print('output esistente -> ACCETTATO')
except ValueError as e:
    print('output esistente sotto fresh_root -> RIFIUTATO:', e)
