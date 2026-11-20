"""Spread between 4 concurrent ranks at the out_dir check, for the NEW pre-check work (Mac, indicative only)."""
import json, os, subprocess, sys, time
from pathlib import Path
clone = Path(sys.argv[1]); trials = int(sys.argv[2])
if os.environ.get('V2_CHILD'):
    start = float(os.environ['V2_START'])
    while time.time() < start: time.sleep(0.0005)
    t0 = time.time()
    sys.path[:0] = [str(clone / 'src'), str(clone / 'scripts')]
    import torch  # trainer import order, as contrastive_run
    import contrastive_run as runner
    import contrastive_release as rel
    from contrastive_prepare import source_digest
    t_import = time.time()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=clone, text=True).strip()
    digest = source_digest()
    rel.verify_reviewed_sources(commit, digest)          # 125 git show + byte comparison
    for p in (rel.PLAN, rel.PREFLIGHT):                    # the two plan git show of validate_review_binding
        rel.git_bytes(commit, str(Path(p).relative_to(rel.ROOT)))
    for _ in range(4): source_digest()                     # validate_release/_review_binding/_audit/_cohort
    t_end = time.time()
    print(json.dumps({'rank': int(os.environ['RANK']), 't0': t0, 'import': t_import, 'end': t_end}))
    sys.exit(0)
rows = []
for trial in range(trials):
    start = time.time() + 1.5
    procs = [subprocess.Popen([sys.executable, '-B', __file__, str(clone), '0'],
             env=dict(os.environ, V2_CHILD='1', V2_START=str(start), RANK=str(r)), stdout=subprocess.PIPE, text=True)
             for r in range(4)]
    res = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
    ends = sorted(r['end'] for r in res); imps = sorted(r['import'] for r in res)
    dur = [r['end'] - r['import'] for r in res]
    rows.append({'trial': trial, 'import_spread_s': round(imps[-1] - imps[0], 3), 'end_spread_s': round(ends[-1] - ends[0], 3),
                 'validation_s': [round(d, 2) for d in dur]})
    print(json.dumps(rows[-1]), flush=True)
print('max end spread', max(r['end_spread_s'] for r in rows))
