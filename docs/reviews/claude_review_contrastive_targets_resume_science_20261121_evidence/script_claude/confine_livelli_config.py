import json, sys, statistics as st, math
R, J = sys.argv[1], sys.argv[2]
def load(p): return {r['step']: r for r in (json.loads(l) for l in open(p))}
runs = {k: load(f'{R}/{d}/{k}/metrics.jsonl') for k, d in (('CN','benchmark_execution_20261121'),('CR','benchmark_execution_20261121'),
        ('CN_A','reloads_execution_20261121'),('CN_B','reloads_execution_20261121'),('CR_A','reloads_execution_20261121'),('CR_B','reloads_execution_20261121'))}
for a, b in (('CN','CN_A'),('CR','CR_A')):
    for s in (545, 546, 547):
        x, y = runs[a][s], runs[b][s]
        print(a, b, s, 'grad_norm', repr(x['grad_norm']), repr(y['grad_norm']), 'abs', abs(x['grad_norm']-y['grad_norm']), 'rel', abs(x['grad_norm']-y['grad_norm'])/x['grad_norm'],
              '| clip', x['grad_clip_factor'], '| lr', x['lr'])
def rel(a, b): return 0.0 if a == b else abs(a-b)/max(abs(a), 1e-300)
for a, b, f2 in (('CN','CN_A','contrast_global_grad_nce'),('CR','CR_A','contrast_global_grad_rnc')):
    for f in ('grad_norm', 'contrast_global_grad_base', f2, 'contrast_projector_grad_norm'):
        xs = [rel(runs[a][s][f], runs[b][s][f]) for s in range(548, 577)]
        e, l = xs[:14], xs[14:]
        gm = lambda v: math.exp(sum(math.log(max(x, 1e-300)) for x in v)/len(v)) if all(x > 0 for x in v) else float('nan')
        print(f'{a}/{b} {f:30s} 548-576 mediana {st.median(xs):.2e} max {max(xs):.2e} | media geom. 548-561 {gm(e):.2e} 562-576 {gm(l):.2e}')
# clip sempre attivo?
for a in ('CN', 'CR'):
    cf = [runs[a][s]['grad_clip_factor'] for s in range(1, 577)]
    print(a, 'clip factor: min', min(cf), 'max', max(cf), 'quanti < 1:', sum(1 for c in cf if c < 1), 'su', len(cf), '| ultimi 32: min', min(cf[-32:]), 'max', max(cf[-32:]))
    w = [runs[a][s].get('contrast_nce_weight') for s in (1, 128, 256, 384, 512, 513, 544, 576)]
    w2 = [runs[a][s].get('contrast_rnc_weight') for s in (1, 128, 256, 384, 512, 513, 544, 576)]
    print(a, 'peso nce a 1,128,256,384,512,513,544,576:', w, '| rnc:', w2)
d = json.load(open(J))
for arm in ('CN', 'CR'):
    ft = d['arms'][arm]['resume']['pairs']['continuous__reload_a']['floating_tensor_relative_to_left']
    q = [v for k, v in ft.items() if k.startswith('contrastive_state.queue') and v != 0]
    print(arm, 'chiavi diverse:', len(q), 'minimo non nullo', min(q), 'mediana', st.median(q), 'max', max(q))
    for i in (266, 267, 268, 269):
        print('   ', i, 'exp_avg', ft[f'optimizer.state.{i}.exp_avg'], 'exp_avg_sq', ft[f'optimizer.state.{i}.exp_avg_sq'])
cfg = json.load(open(f'{R}/benchmark_execution_20261121/CN/config.json'))
def find(o, pre=''):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)): find(v, pre + k + '.')
            elif any(t in k.lower() for t in ('beta', 'clip', 'weight_decay', 'lr', 'eps', 'amp', 'accum', 'micro', 'temperature', 'tau', 'queue', 'ema', 'momentum', 'batch')): print('   cfg', pre + k, '=', v)
find(cfg)
