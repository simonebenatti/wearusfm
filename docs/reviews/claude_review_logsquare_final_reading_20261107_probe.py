"""Classification: independent recomputation of the four-arm paired comparison from the per-subject counts in compare.json.

usage: python -I claude_review_logsquare_final_reading_20261107_probe.py <root> <scripts/tokenizer_probe.py at 61f0021>
where <root>/probe/compare.json is a copy of the output of job 63775886; the second argument supplies ARMS, TASKS, WEIGHTS, SEED and N_BOOT.
"""
import ast, json, math, re, sys
import numpy as np
src = open(sys.argv[2]).read()
def const(name):
    m = re.search(r'^' + name + r'\s*=\s*(.+)$', src, re.M)
    return eval(m.group(1), {"__builtins__": {}}) if m else None
ARMS_H, TASKS, WEIGHTS = const("ARMS"), const("TASKS"), const("WEIGHTS")
_, SEED, N_BOOT = eval(re.search(r"^MAX_ITER, SEED, N_BOOT = (.+)$", src, re.M).group(1), {"__builtins__": {}})
print('ARMS storici:', ARMS_H, '| TASKS:', TASKS, '| WEIGHTS:', WEIGHTS, '| SEED:', SEED, '| N_BOOT:', N_BOOT)
ARMS = (*ARMS_H, 'T100-LQ')
rep = json.load(open(f'{sys.argv[1]}/probe/compare.json'))
rng = np.random.default_rng(SEED)
points, samples, problems = {}, {}, []
def arrays(stats):
    subjects = sorted(stats['subjects'])
    n = np.asarray([stats['subjects'][s]['n'] for s in subjects], dtype=float)
    hits = np.asarray([stats['subjects'][s]['correct'] for s in subjects], dtype=float)
    return subjects, n, hits
for ds in sorted(rep['datasets']):
    blk = rep['datasets'][ds]
    stats = [blk['paired_counts'][a] for a in ARMS]
    users, den_u, _ = arrays(stats[0])
    counts = [arrays(s)[2] for s in stats]
    for s in stats:
        u, n, _ = arrays(s)
        if u != users or not np.array_equal(n, den_u) or s['classes'] != stats[0]['classes']: problems.append(f'{ds}: appaiamento')
    draw = rng.integers(0, len(users), size=(N_BOOT, len(users)))
    den = den_u[draw].sum(1); present = den > 0; cp = present.sum(1)
    vals = np.column_stack([np.divide(h[draw].sum(1), den, out=np.zeros_like(den), where=present).sum(1)/cp for h in counts])
    total = den_u.sum(0)
    point = np.asarray([np.mean(h.sum(0)[total > 0]/total[total > 0]) for h in counts])
    points[ds], samples[ds] = point, vals
    for i, a in enumerate(ARMS):
        c = blk['cells'][a]
        ci = 100*np.quantile(vals[:, i], [.025, .975])
        if not (math.isclose(point[i], c['balanced_accuracy'], rel_tol=1e-12) and np.allclose(ci, c['ci95_percent'], rtol=1e-12)): problems.append(f'{ds}/{a}: cella')
    co = {'T100-LQ_minus_T100-L': (0., -1., 0., 1.), 'T100-LQ_minus_T25-L': (-1., 0., 0., 1.), 'T100-LQ_minus_T100-G': (0., 0., -1., 1.)}
    for name, k in co.items():
        k = np.asarray(k); ci = 100*np.quantile(vals @ k, [.025, .975]); d = 100*(point @ k); c = blk['contrasts'][name]
        if not (math.isclose(d, c['delta_pp'], rel_tol=1e-9, abs_tol=1e-12) and np.allclose(ci, c['ci95_pp'], rtol=1e-9)): problems.append(f'{ds}/{name}: contrasto')
    # context: control-vs-control contrasts, same draws (mine)
    for name, k in {'T25-L_minus_T100-L': (1., -1., 0., 0.), 'T100-G_minus_T100-L': (0., -1., 1., 0.)}.items():
        k = np.asarray(k); ci = 100*np.quantile(vals @ k, [.025, .975]); d = 100*(point @ k)
        print(f'  contesto {ds:12s} {name:20s} delta={d:+.3f} pp IC95=[{ci[0]:+.3f}, {ci[1]:+.3f}]')
    # per-subject sign count of the primary contrast (pooled per-subject BAcc of LQ vs T100-L)
    def subj_bacc(h, n): 
        pres = n > 0; return np.array([np.mean(h[j][pres[j]]/n[j][pres[j]]) for j in range(len(n))])
    b_lq, b_l = subj_bacc(counts[3], den_u), subj_bacc(counts[1], den_u)
    print(f'  {ds:12s} per soggetto LQ vs T100-L: meglio {int((b_lq > b_l).sum())}, uguale {int((b_lq == b_l).sum())}, peggio {int((b_lq < b_l).sum())} su {len(users)}')
for task, dss in TASKS.items():
    w = WEIGHTS[task]
    p = np.sum([wi*points[d] for wi, d in zip(w, dss)], axis=0); s = np.sum([wi*samples[d] for wi, d in zip(w, dss)], axis=0)
    blk = rep['tasks'][task]
    for i, a in enumerate(ARMS):
        c = blk['cells'][a]; ci = 100*np.quantile(s[:, i], [.025, .975])
        if not (math.isclose(p[i], c['balanced_accuracy'], rel_tol=1e-12) and np.allclose(ci, c['ci95_percent'], rtol=1e-12)): problems.append(f'{task}/{a}: cella')
    for name, k in {'T100-LQ_minus_T100-L': (0., -1., 0., 1.), 'T100-LQ_minus_T25-L': (-1., 0., 0., 1.), 'T100-LQ_minus_T100-G': (0., 0., -1., 1.)}.items():
        k = np.asarray(k); ci = 100*np.quantile(s @ k, [.025, .975]); d = 100*(p @ k); c = blk['contrasts'][name]
        ni = 'supported' if ci[0] > -1 else 'loss_beyond_margin' if ci[1] < -1 else 'unresolved'
        if not (math.isclose(d, c['delta_pp'], rel_tol=1e-9, abs_tol=1e-12) and np.allclose(ci, c['ci95_pp'], rtol=1e-9) and ni == c['noninferiority_1pp']): problems.append(f'{task}/{name}: contrasto/NI')
    for name, k in {'T25-L_minus_T100-L': (1., -1., 0., 0.), 'T100-G_minus_T100-L': (0., -1., 1., 0.)}.items():
        k = np.asarray(k); ci = 100*np.quantile(s @ k, [.025, .975]); d = 100*(p @ k)
        print(f'  contesto {task:12s} {name:20s} delta={d:+.3f} pp IC95=[{ci[0]:+.3f}, {ci[1]:+.3f}] NI1pp={"supported" if ci[0] > -1 else "loss_beyond_margin" if ci[1] < -1 else "unresolved"}')
print('PROBLEMI nel ricalcolo:', problems if problems else 'nessuno: celle, contrasti ed etichette NI identici')
