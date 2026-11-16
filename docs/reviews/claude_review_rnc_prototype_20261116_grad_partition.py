"""Review Claude del prototipo Rank-N-Contrast (16/11/2026): gradiente del calcolo veloce contro la definizione coppia per coppia
del file di test dell'autore (float64), e stesso batch intero contro meta' e quarti ricomposti come fa il contratto delle loss.

Uso: python <questo file> <albero>, dove <albero> ha `src/` di e114a78 col modulo del prototipo e `tests/cpu/test_rank_contrast.py`
(ricetta in claude_review_rnc_prototype_20261116_numbers.txt). Solo dati sintetici.
"""
import sys
sys.path.insert(0, sys.argv[1] + '/src'); sys.path.insert(0, sys.argv[1] + '/tests/cpu')
import torch
from wearusfm.model.rank_contrast import pose_rank_contrast
import test_rank_contrast as T

worst = 0.0
for seed in (0, 1, 2):
    z, pose, tv = T._data(seed=seed)
    for positives in ('cross', 'all'):
        for per_window in (True, False):
            for resolution in (0.0, 0.5):
                za = z.clone().requires_grad_(True)
                loss, *_ = pose_rank_contrast(za, pose, tv, positives=positives, per_window=per_window, resolution=resolution, temperature=0.2, stride=2)
                (ga,) = torch.autograd.grad(loss, za)
                zb = z.clone().requires_grad_(True)
                ref, _ = T.reference(zb, pose[0], pose[1], tv, 0.2, 2, positives, per_window, resolution)
                (gb,) = torch.autograd.grad(ref, zb)
                diff = float((ga - gb).abs().max()); scale = float(gb.abs().max())
                worst = max(worst, diff / scale)
                print(f'seed {seed} {positives:5s} per_window={per_window!s:5s} res={resolution}: loss {float(loss):.12f} ref {float(ref):.12f} | max|dg| {diff:.3e} su max|g| {scale:.3e}')
print('errore relativo massimo del gradiente:', f'{worst:.3e}')
# batch intero contro due meta' e quattro quarti, ricomposti come il contratto (somma dei numeratori / somma dei conteggi)
z, pose, tv = T._data(s=8, p=40, j=20, d=32, seed=5)
mask = pose[1].clone(); mask[[1, 3, 4, 6, 7]] = False   # pose only in windows 0, 2, 5
for positives in ('cross', 'all'):
    for per_window in (True, False):
        kw = dict(temperature=0.1, stride=8, positives=positives, per_window=per_window)
        _, num, cnt, log = pose_rank_contrast(z, (pose[0], mask), tv, **kw)
        parts = [pose_rank_contrast(z[a:b], (pose[0][a:b], mask[a:b]), tv[a:b], **kw) for a, b in ((0, 4), (4, 8))]
        n2, c2 = sum(p[1] for p in parts), sum(int(p[2]) for p in parts)
        q = [pose_rank_contrast(z[a:a + 2], (pose[0][a:a + 2], mask[a:a + 2]), tv[a:a + 2], **kw) for a in (0, 2, 4, 6)]
        n4, c4 = sum(p[1] for p in q), sum(int(p[2]) for p in q)
        f = lambda n, c: float(n) / max(c, 1)
        print(f'{positives:5s} per_window={per_window!s:5s}: intero {f(num, int(cnt)):.6f} (conteggio {int(cnt)}, ancore {int(log["anchors"])}) | due meta {f(n2, c2):.6f} (conteggio {c2}) | quattro quarti {f(n4, c4):.6f} (conteggio {c4})')
