#!/usr/bin/env python3
"""Independent count-based recomputation and deliberate pairing/hash failures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
report_path = HERE / 'paired_report.json'
r = json.loads(report_path.read_text())
rng = np.random.default_rng(r['seed'])
points, draws = {}, {}
checks = []
for ds, block in sorted(r['datasets'].items()):
    users = sorted(block['paired_counts']['T25-L']['subjects'])
    take = rng.integers(0,len(users),(r['n_bootstrap'],len(users)))
    values, pp = [], []
    for arm in ('T25-L','B19618'):
        c = block['paired_counts'][arm]
        n = np.array([c['subjects'][s]['n'] for s in users],dtype=float)
        h = np.array([c['subjects'][s]['correct'] for s in users],dtype=float)
        den, hit = n[take].sum(1), h[take].sum(1)
        active = den > 0
        v = np.divide(hit,den,out=np.zeros_like(den),where=active).sum(1)/active.sum(1)
        point = np.mean(h.sum(0)/n.sum(0))
        np.testing.assert_allclose(point,block['cells'][arm]['balanced_accuracy'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(100*np.quantile(v,[.025,.975]),block['cells'][arm]['ci95_percent'],rtol=0,atol=1e-12)
        values.append(v);pp.append(point)
    points[ds],draws[ds] = np.array(pp),np.array(values)
    np.testing.assert_allclose(100*(pp[0]-pp[1]),block['contrast']['delta_pp'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(100*np.quantile(values[0]-values[1],[.025,.975]),block['contrast']['ci95_pp'],rtol=0,atol=1e-12)
    checks.append(f'{ds}: independent direct-draw count bootstrap agrees')
for task, weights in r['weights'].items():
    p = sum(w*points[d] for d,w in weights.items())
    v = sum(w*draws[d] for d,w in weights.items())
    np.testing.assert_allclose(100*(p[0]-p[1]),r['tasks'][task]['contrast']['delta_pp'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(100*np.quantile(v[0]-v[1],[.025,.975]),r['tasks'][task]['contrast']['ci95_pp'],rtol=0,atol=1e-12)
    checks.append(f'{task}: fixed-weight paired contrast agrees')
spec=importlib.util.spec_from_file_location('repeatability',ROOT/'scripts/tokenizer_b_repeatability.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a={'row':np.array([0,1]),'y':np.array(['noGesture','fist']), 'pred':np.array(['fist','fist']),
   's':np.array(['u1','u2']),'fold':np.array([0,1])}
for key in ('row','y','s','fold'):
    b={k:v.copy() for k,v in a.items()}
    b[key]=b[key][::-1]
    try: m.paired(a,b)
    except ValueError: checks.append(f'guard rejects changed {key}')
    else: raise AssertionError(f'guard accepted changed {key}')
try: m.pinned(report_path,'0'*64)
except ValueError: checks.append('guard rejects wrong file hash')
else: raise AssertionError('guard accepted wrong file hash')
m.normalize({k:v.copy() for k,v in a.items()})
checks.append('native text EPN-like labels accepted without numeric isfinite')
out={'complete':True,'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
     'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'method':'independent direct subject-draw indexing and summation, versus multinomial-count matrix products in implementation',
     'checks':checks}
(HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
