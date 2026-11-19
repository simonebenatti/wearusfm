"""Probe: does build_command accept an OLD-cohort review pin for a NEW audit job?"""
import hashlib, json, os, sys
from pathlib import Path
CLONE = Path(sys.argv[1]); WORK = Path(sys.argv[2])
os.environ['WORK'] = str(WORK)
sys.path.insert(0, str(CLONE / 'scripts'))
import contrastive_job as J
from contrastive_prepare import source_digest
def pin(rel):
    p = CLONE / rel
    return {'path': rel, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
old_review = 'docs/reviews/contrastive_preflight_release_20261117.md'
job = {'mode': 'audit', 'status': 'APPROVED_FOR_PREFLIGHT', 'source_sha256': source_digest(),
       'review': pin(old_review),
       'out': str(WORK / 'wearusfm_runs' / 'probe' / 'audit.json'),
       'config': pin('results/infra/contrastive_targets_20261118/review_configs/calibration.config.json'),
       'inputs': {k: {'path': '/leonardo_work/IscrB_WearUsFM/nonexistent_'+k, 'sha256': '0'*64}
                  for k in ('manifest', 'scales', 'window_seconds', 'pose_stats')},
       'roots': []}
stages = []
orig_checked = J.checked_file
def traced(spec):
    r = orig_checked(spec); stages.append(('pin_ok', spec['path'])); return r
J.checked_file = traced
try:
    J.build_command(job, 'cpu', 7200)
    print('ACCEPTED (unexpected on Mac)')
except Exception as exc:
    print('stopped at:', type(exc).__name__, str(exc)[:120])
print('pins accepted before stop:', stages)
print('old review status line:', [l for l in (CLONE/old_review).read_text().splitlines() if l.strip()][:2])
