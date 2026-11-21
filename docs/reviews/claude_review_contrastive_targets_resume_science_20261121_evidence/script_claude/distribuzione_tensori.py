import json, sys, statistics as st
d = json.load(open(sys.argv[1]))
def q(xs, p):
    xs = sorted(xs); 
    if not xs: return float('nan')
    i = min(len(xs)-1, max(0, int(round(p*(len(xs)-1))))); return xs[i]
for arm in ('CN', 'CR'):
    p = d['arms'][arm]['resume']['pairs']['continuous__reload_a']
    ft = p['floating_tensor_relative_to_left']
    groups = {'exp_avg': {}, 'exp_avg_sq': {}, 'queue': {}, 'other': {}}
    for k, v in ft.items():
        if k.endswith('.exp_avg'): groups['exp_avg'][k] = v
        elif k.endswith('.exp_avg_sq'): groups['exp_avg_sq'][k] = v
        elif k.startswith('contrastive_state.queue'): groups['queue'][k] = v
        else: groups['other'][k] = v
    print('=====', arm, {g: len(v) for g, v in groups.items()})
    print('   other keys:', list(groups['other'].items())[:20])
    for g in ('exp_avg', 'exp_avg_sq', 'queue'):
        xs = list(groups[g].values())
        nz = [x for x in xs if x != 0]
        print(f'  {g}: n={len(xs)} zero={len(xs)-len(nz)} min={min(xs):.3e} q25={q(xs,.25):.3e} med={q(xs,.5):.3e} q75={q(xs,.75):.3e} q95={q(xs,.95):.3e} max={max(xs):.3e}')
        for thr in (0.001, 0.002, 0.005, 0.0075, 0.009, 0.01):
            print(f'     > {thr}: {sum(1 for x in xs if x > thr)}', end='')
        print()
    top = sorted(groups['exp_avg'].items(), key=lambda kv: -kv[1])[:25]
    print('  top exp_avg:'); [print(f'     {k}: {v:.6f}') for k, v in top]
    top = sorted(groups['exp_avg_sq'].items(), key=lambda kv: -kv[1])[:8]
    print('  top exp_avg_sq:'); [print(f'     {k}: {v:.6f}') for k, v in top]
    if groups['queue']:
        qs = groups['queue']
        idx = sorted((int(k.split('.')[3]), v) for k, v in qs.items())
        nzidx = [i for i, v in idx if v != 0]
        print('  queue: nonzero entries', len(nzidx), 'first nz idx', nzidx[:5], 'last nz idx', nzidx[-5:], 'max', max(v for _, v in idx))
