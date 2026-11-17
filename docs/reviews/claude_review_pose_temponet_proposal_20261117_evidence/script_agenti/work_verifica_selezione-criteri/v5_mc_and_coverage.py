# (1) limite inferiore IC99 testa-ridge a 49.046 con 2.000 e 20.000 repliche (seme del protocollo); (2) copertura MC mia dell'IC
# percentile 99% con i pesi veri dei 15 utenti (SSE centrata della ridge post), effetti per utente normali o t3.
import json, numpy as np
A = json.load(open('rec/LONG-FIT-G49046-pose.json'))['cells']
pu = lambda c: c['analysis']['groups']['dynamic']['centered']['per_user']
c, r = pu(A['trained_head']), pu(A['post_ridge']); users = sorted(c)
Cc = np.array([c[u]['sums']['centered'] for u in users]); Cr = np.array([r[u]['sums']['centered'] for u in users])
for B in (2000, 20000):
    d = np.random.default_rng(20261102).integers(0, 15, (B, 15))
    rel = 100 * (1 - Cc[d].sum(1) / Cr[d].sum(1))
    print('B', B, 'IC99', np.percentile(rel, [0.5, 99.5]).round(4))
w = Cr / Cr.sum(); nwin = np.array([r[u]['n_windows'] for u in users])
rng = np.random.default_rng(12345)
for tail in ('normal', 't3'):
    for su in (0.7, 1.5, 3.0):
        R, B = 4000, 2000; cover = 0; lo_miss = 0; hi_miss = 0
        d = rng.integers(0, 15, (B, 15))
        for k in range(R):
            zu = rng.standard_normal(15) if tail == 'normal' else rng.standard_t(3, 15) / np.sqrt(3.0)
            ru = 2.0 + su * zu + 10 / np.sqrt(nwin) * rng.standard_normal(15)
            num = w * ru
            est = num.sum() / w.sum(); bs = num[d].sum(1) / w[d].sum(1)
            lo, hi = np.percentile(bs, [0.5, 99.5])
            cover += (lo <= 2.0 <= hi); lo_miss += (lo > 2.0); hi_miss += (hi < 2.0)
        print(f'{tail} sigma_u {su}: copertura IC99 {cover / R:.4f}  (code: {lo_miss / R:.4f} e {hi_miss / R:.4f}); errore MC ~{np.sqrt(0.97 * 0.03 / R):.4f}')
