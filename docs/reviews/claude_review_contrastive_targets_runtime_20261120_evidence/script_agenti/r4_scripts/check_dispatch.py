"""Exercise contrastive_job.build_command's allocation/cell guards against the REAL preflight plan."""
import hashlib, json, os, sys, tempfile
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root/'scripts'))
os.environ['WORK'] = '/leonardo_work/IscrB_WearUsFM'
import contrastive_job as J, contrastive_release as R
J.source_digest = lambda: 'a'*64
R.validate_review_binding = lambda b: None
R.validate_audit_accounting = lambda *a, **k: None
tmp = Path(tempfile.mkdtemp(dir=sys.argv[2]))
def pin(name):
    p = tmp/name; p.write_text('{}'); return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
plan = json.loads(R.PREFLIGHT.read_text())
def job_for(cell):
    rv = pin('review.json')
    j = {'id': cell['id'], 'mode': cell['mode'], 'arm': cell['arm'], 'status': 'APPROVED_FOR_PREFLIGHT', 'source_sha256': 'a'*64,
         'package_review': {'review': rv}, 'review': rv, 'audit': pin('audit.json'), 'audit_sacct': pin('acct.json'),
         'out': '/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/probe_'+cell['id']}
    if cell['repeat'] is not None: j['repeat'] = cell['repeat']
    return j
def outcome(job, kind, secs):
    try: J.build_command(job, kind, secs); return 'ACCEPTED'
    except (ValueError, KeyError, FileNotFoundError) as e: return type(e).__name__+': '+str(e)[:70]
rows = []
for cell in plan['cells']:
    kind = 'gpu' if cell['mode'] in J.GPU_MODES else 'cpu'
    for secs in sorted({cell['seconds'], cell['seconds']-1, cell['seconds']+1, 1800, 3600, 7200}):
        r = outcome(job_for(cell), kind, secs)
        passed_alloc = not ('allocation' in r or 'fixed preflight cell' in r)
        rows.append((cell['id'], secs, passed_alloc, r))
        if passed_alloc != (secs == cell['seconds']): print('UNEXPECTED', cell['id'], secs, r)
    # wrong kind
    other = 'cpu' if kind == 'gpu' else 'gpu'
    r = outcome(job_for(cell), other, cell['seconds'])
    if 'allocation' not in r: print('UNEXPECTED kind', cell['id'], r)
print('checked', len(rows), 'allocation cases; accepted-by-guard only at plan seconds:', all(p == (s == c) for (i, s, p, r), c in zip(rows, [next(x['seconds'] for x in plan['cells'] if x['id']==row[0]) for row in rows])))
# cell-level guard independent of ALLOCATED_SECONDS: mutate the constant for 'full'
J.ALLOCATED_SECONDS['full'] = 1801
cell = next(c for c in plan['cells'] if c['mode']=='full')
print('mutated ALLOCATED_SECONDS full=1801 ->', outcome(job_for(cell), 'gpu', 1801))
J.ALLOCATED_SECONDS['full'] = 1800
# arm/repeat swaps
cell = next(c for c in plan['cells'] if c['id'].startswith('CONTRAST-RELOAD-CN-A'))
j = job_for(cell); j['repeat'] = 'b'; print('CN-A with repeat b ->', outcome(j, 'gpu', 1800))
j = job_for(cell); j['arm'] = 'CR'; print('CN-A with arm CR ->', outcome(j, 'gpu', 1800))
j = job_for(cell); j['mode'] = 'full'; print('reload id with mode full ->', outcome(j, 'gpu', 1800))
j = job_for(cell); j['id'] = 'CONTRAST-RELOAD-CN-C-targets-v3-20261118'; print('unknown id ->', outcome(j, 'gpu', 1800))
for c in plan['cells'][:3]: print(c['id'], 'at plan seconds ->', outcome(job_for(c), 'gpu' if c['mode'] in J.GPU_MODES else 'cpu', c['seconds']))
