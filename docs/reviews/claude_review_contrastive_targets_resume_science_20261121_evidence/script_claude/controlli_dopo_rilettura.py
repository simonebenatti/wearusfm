import json, sys, statistics as st, math, collections
R, J, NAMES = sys.argv[1], sys.argv[2], sys.argv[3]
names = json.load(open(NAMES))
names = names if isinstance(names, list) else names.get('names', names)
flat = [n if isinstance(n, str) else n[1] if isinstance(n, (list, tuple)) else n.get('name') for n in names]
cnt = collections.Counter(n.split('.')[0] for n in flat)
shared = ('tokenizer', 'identity', 'local', 'pool', 'backbone')
print('tensori per prefisso:', dict(cnt), '| totale', len(flat), '| condivisi', sum(cnt[p] for p in shared), '| fuori', len(flat) - sum(cnt[p] for p in shared))
d = json.load(open(J))
ft = {a: d['arms'][a]['resume']['pairs']['continuous__reload_a']['floating_tensor_relative_to_left'] for a in ('CN', 'CR')}
cn = [ft['CN'][f'optimizer.state.{i}.exp_avg'] for i in range(266)]
cr = [ft['CR'][f'optimizer.state.{i}.exp_avg'] for i in range(266)]
print('primi momenti 0-265: mediana CN %.4f%% CR %.4f%% rapporto %.3f | media CN %.4f%% CR %.4f%% rapporto %.3f | mediana dei rapporti %.3f | media geometrica dei rapporti %.3f' % (
    st.median(cn)*100, st.median(cr)*100, st.median(cn)/st.median(cr), st.mean(cn)*100, st.mean(cr)*100, st.mean(cn)/st.mean(cr),
    st.median([a/b for a, b in zip(cn, cr)]), math.exp(st.mean([math.log(a/b) for a, b in zip(cn, cr)]))))
allcn = sorted(((ft['CN'][f'optimizer.state.{i}.exp_avg'], i) for i in range(270)), reverse=True)
print('CN primi momenti, primi 6:', [(i, round(v*100, 3)) for v, i in allcn[:6]], '| fra 0,75% e 1%:', sum(1 for v, _ in allcn if 0.0075 < v < 0.01))
sq = sorted(((ft['CN'][f'optimizer.state.{i}.exp_avg_sq'], i) for i in range(270)), reverse=True)
print('CN secondi momenti, primi 6:', [(i, round(v*100, 4)) for v, i in sq[:6]])
qcn = [v for k, v in ft['CN'].items() if k.startswith('contrastive_state.queue')]
qcr = [v for k, v in ft['CR'].items() if k.startswith('contrastive_state.queue')]
for name, q in (('CN', qcn), ('CR', qcr)):
    post = q[4979:]; nz = [v for v in q if v]
    print(name, 'chiavi: dopo il fork', len(post), 'mediana su tutte %.4f%%' % (st.median(post)*100), '| diverse', len(nz), 'mediana %.4f%% minimo %.4f%% massimo %.4f%%' % (st.median(nz)*100, min(nz)*100, max(nz)*100),
          '| non nulle prima di 4979:', sum(1 for v in q[:4979] if v))
def load(p): return {r['step']: r for r in (json.loads(l) for l in open(p))}
runs = {k: load(f'{R}/{dd}/{k}/metrics.jsonl') for k, dd in (('CN','benchmark_execution_20261121'),('CR','benchmark_execution_20261121'),
        ('CN_A','reloads_execution_20261121'),('CN_B','reloads_execution_20261121'),('CR_A','reloads_execution_20261121'),('CR_B','reloads_execution_20261121'))}
for a, b in (('CN_A', 'CN_B'), ('CR_A', 'CR_B')):
    c = collections.Counter(); mx = {}
    for s in range(545, 577):
        for k, v in runs[a][s].items():
            w = runs[b][s].get(k)
            if isinstance(v, float) and isinstance(w, float) and v != w and not k.startswith('t_'):
                c[k] += 1; mx[k] = max(mx.get(k, 0), abs(v-w)/max(abs(v), 1e-300))
    print(a, 'vs', b, 'campi diversi (update su 32, scarto massimo):', {k: (n, float('%.1e' % mx[k])) for k, n in sorted(c.items())})
for s in (545, 546):
    x, y = runs['CN'][s], runs['CN_A'][s]
    diff = {k: (v, y[k]) for k, v in x.items() if isinstance(v, float) and k in y and v != y[k] and not k.startswith('t_')}
    print('CN vs CN_A update', s, 'campi diversi:', {k: float('%.2e' % (abs(a-b)/max(abs(a), 1e-300))) for k, (a, b) in diff.items()})
xs = {f: [abs(runs['CN'][s][f]-runs['CN_A'][s][f])/abs(runs['CN'][s][f]) for s in range(548, 577)] for f in ('grad_norm', 'contrast_global_grad_base', 'contrast_global_grad_nce', 'contrast_projector_grad_norm')}
ys = {f: [abs(runs['CR'][s][f]-runs['CR_A'][s][f])/abs(runs['CR'][s][f]) for s in range(548, 577)] for f in ('grad_norm', 'contrast_global_grad_base', 'contrast_global_grad_rnc')}
print('CN mediane 548-576:', {k: float('%.2e' % st.median(v)) for k, v in xs.items()}); print('CR mediane 548-576:', {k: float('%.2e' % st.median(v)) for k, v in ys.items()})
def slope(v, s0=548):
    pts = [(s0+i, math.log10(x)) for i, x in enumerate(v) if x > 0]
    n = len(pts); mx_ = sum(p[0] for p in pts)/n; my = sum(p[1] for p in pts)/n
    return sum((p[0]-mx_)*(p[1]-my) for p in pts)/sum((p[0]-mx_)**2 for p in pts)
print('pendenza log10 grad_norm 548-576: CN %.3f CR %.3f decadi/update' % (slope(xs['grad_norm']), slope(ys['grad_norm'])))
