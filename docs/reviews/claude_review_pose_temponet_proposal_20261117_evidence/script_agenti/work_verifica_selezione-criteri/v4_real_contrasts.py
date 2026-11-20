# Deviazione bootstrap e fra utenti di contrasti gia' pubblicati del lungo (dinamiche), codice mio.
import json, numpy as np
A = json.load(open('rec/LONG-FIT-G49046-pose.json'))['cells']; B = json.load(open('rec/LONG-FIT-G19618-pose.json'))['cells']
def pu(c): return c['analysis']['groups']['dynamic']['centered']['per_user']
pairs = {'49046 head - 49046 ridge': (pu(A['trained_head']), pu(A['post_ridge'])),
         '49046 ridge - 19618 ridge': (pu(A['post_ridge']), pu(B['post_ridge'])),
         '49046 head - 19618 head': (pu(A['trained_head']), pu(B['trained_head'])),
         '19618 head - 19618 ridge': (pu(B['trained_head']), pu(B['post_ridge'])),
         '49046 post - 49046 pre': (pu(A['post_ridge']), pu(A['pre_ridge']))}
d = np.random.default_rng(20261102).integers(0, 15, (2000, 15))
for name, (c, r) in pairs.items():
    users = sorted(c); assert users == sorted(r)
    Cc = np.array([c[u]['sums']['centered'] for u in users]); Cr = np.array([r[u]['sums']['centered'] for u in users])
    V = np.array([c[u]['sums']['vary'] for u in users]); assert np.allclose(V, [r[u]['sums']['vary'] for u in users])
    rel = 100 * (1 - Cc.sum() / Cr.sum()); relb = 100 * (1 - Cc[d].sum(1) / Cr[d].sum(1))
    peru = 100 * (1 - Cc / Cr); w = Cr / Cr.sum()
    big = w > 0.01
    print(f'{name}: {rel:.4f}%  sd boot {relb.std(ddof=1):.3f}  IC99 [{np.percentile(relb,0.5):.3f}; {np.percentile(relb,99.5):.3f}]  '
          f'semiampiezza99 {(np.percentile(relb,99.5)-np.percentile(relb,0.5))/2:.3f}  sd fra utenti (peso>1%, n={big.sum()}) {peru[big].std(ddof=1):.3f}')
