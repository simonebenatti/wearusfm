"""Review Claude del prototipo Rank-N-Contrast (16/11/2026): controllo limitato del modulo corretto da GPT (db5ac41)
contro i difetti dimostrati nel prototipo. Uso: python <questo file> <albero di db5ac41>/src.

Non e' la review del pacchetto: solo (1) valore e gradiente contro una definizione coppia per coppia scritta a parte,
(2) controesempio R1 (pose identiche nella stessa finestra), (3) temperature estreme, (4) conteggio dalle maschere,
(5) dipendenza dalla partizione a livello di modulo (attesa: il modulo la dichiara; la raccolta globale sta nel trainer).
"""
import sys, math, itertools
import torch
sys.path.insert(0, sys.argv[1])
from wearusfm.model import rank_contrast as rc

torch.manual_seed(0)
F = torch.nn.functional


def reference(z, target, mask, time_valid, temperature, stride, per_window, resolution, pair_allowed=None):
    S, P, D = z.shape
    valid = mask & time_valid[..., None]
    anchors = [(s, p) for s in range(S) for p in range(stride // 2, P, stride) if bool(valid[s, p].any())]
    n = len(anchors)
    if n == 0:
        return z.sum() * 0.0, 0, 0, 0
    h = torch.stack([z[s, p] for s, p in anchors]).double()
    u = F.normalize(F.layer_norm(h, h.shape[-1:]), dim=-1)
    sim = (u @ u.T) / temperature
    cand = [[False] * n for _ in range(n)]
    dist = [[0.0] * n for _ in range(n)]
    for i, (si, pi) in enumerate(anchors):
        for k, (sk, pk) in enumerate(anchors):
            if i == k:
                continue
            common = valid[si, pi] & valid[sk, pk]
            if not bool(common.any()):
                continue
            if pair_allowed is not None and not bool(pair_allowed[i, k]):
                continue
            cand[i][k] = True
            diff = (target[si, pi].double() - target[sk, pk].double())[common]
            d = math.sqrt(float((diff * diff).sum()) / int(common.sum()))
            if resolution > 0:
                d = math.floor(d / resolution) * resolution
            dist[i][k] = d
    values, oks, pairs = [], [], 0
    for i in range(n):
        js = [k for k in range(n) if cand[i][k]]
        if len(js) < 2:
            values.append(None); oks.append(False); continue
        num = 0.0; chance = 0.0
        for j in js:
            s_set = [k for k in js if dist[i][k] >= dist[i][j]]
            num = num + (torch.logsumexp(torch.stack([sim[i, k] for k in s_set]), 0) - sim[i, j])
            chance += math.log(len(s_set))
        values.append(num / chance); oks.append(True); pairs += len(js)
    if not any(oks):
        return z.sum() * 0.0, 0, 0, 0
    if per_window:
        tot = 0.0; cnt = 0
        for s in range(S):
            vs = [values[i] for i, (si, _) in enumerate(anchors) if si == s and oks[i]]
            if vs:
                tot = tot + sum(vs) / len(vs); cnt += 1
    else:
        vs = [v for v, o in zip(values, oks) if o]
        tot = sum(vs); cnt = len(vs)
    return tot / cnt, cnt, sum(oks), pairs


def data(S=4, P=24, J=5, D=12, pose_windows=(0, 2, 3), ties=True, partial=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(S, P, D, generator=g, dtype=torch.float64)
    target = torch.randn(S, P, J, generator=g, dtype=torch.float64)
    if ties:
        target[0, 2] = target[0, 6]; target[2, 2] = target[0, 2]; target[3, 10] = target[3, 14]
    mask = torch.zeros(S, P, J, dtype=torch.bool)
    for s in pose_windows:
        mask[s] = True
    if partial:
        mask[2, 6, :2] = False; mask[3, 2] = False; mask[0, 18, 3:] = False
    tv = torch.ones(S, P, dtype=torch.bool); tv[3, 20:] = False
    return z, target, mask, tv


worst_v = worst_g = 0.0
ncase = 0
for seed, per_window, resolution, stride, temp, with_pa in itertools.product(
        (0, 1, 2), (True, False), (0.0, 0.5), (2, 4), (0.1, 0.2), (False, True)):
    z, target, mask, tv = data(seed=seed)
    z.requires_grad_(True)
    pa = None
    if with_pa:
        s_idx, p_idx, v, cand, _, ok = rc.anchor_support(mask & tv[..., None], stride, "all")
        n = s_idx.numel(); gpa = torch.Generator().manual_seed(100 + seed)
        a = torch.rand(n, n, generator=gpa) > 0.15
        pa = (a & a.T) | torch.eye(n, dtype=torch.bool)
    loss, num, count, log = rc.pose_rank_contrast(z, (target, mask), tv, temperature=temp, stride=stride,
                                                  per_window=per_window, resolution=resolution, pair_allowed=pa)
    g1, = torch.autograd.grad(loss, z)
    ref, cnt, n_ok, pairs = reference(z, target, mask, tv, temp, stride, per_window, resolution, pa)
    g2, = torch.autograd.grad(ref, z)
    assert int(count) == cnt and int(log["anchors"]) == n_ok and int(log["pairs"]) == pairs, (int(count), cnt)
    c2 = rc.rank_contrast_count(mask & tv[..., None], stride=stride, per_window=per_window, pair_allowed=pa)
    assert int(c2) == cnt
    worst_v = max(worst_v, abs(float(loss.detach()) - float(ref.detach())) / max(abs(float(ref.detach())), 1e-30))
    worst_g = max(worst_g, float((g1 - g2).abs().max()) / max(float(g2.abs().max()), 1e-30))
    ncase += 1
print(f"(1) definizione coppia per coppia: {ncase} casi; scarto relativo massimo valore {worst_v:.3e}, gradiente {worst_g:.3e}")

# (2) controesempio R1 di GPT: quattro pose identiche, due per finestra, logit uniformi
logits = torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
dist = torch.zeros(4, 4, dtype=torch.float64)
cand = ~torch.eye(4, dtype=torch.bool)
num, chance, n_pos = rc.rank_contrast_pairs(logits, dist, cand)
val = (num / chance).sum()
gl, = torch.autograd.grad(val, logits)
print("(2) R1, pose identiche: derivate sui logit della riga 0 =", [round(float(x), 6) for x in gl[0]])

# (3) temperature estreme contro la definizione
for temp in (1e-5, 5e-5, 1e-4, 1e-3):
    z, target, mask, tv = data(seed=5, ties=False, partial=False)
    z.requires_grad_(True)
    loss, *_ = rc.pose_rank_contrast(z, (target, mask), tv, temperature=temp, stride=2)
    ref, *_ = reference(z, target, mask, tv, temp, 2, True, 0.0)
    g1, = torch.autograd.grad(loss, z); g2, = torch.autograd.grad(ref, z)
    print(f"(3) tau {temp:g}: modulo {float(loss):.9f} definizione {float(ref):.9f} "
          f"scarto rel valore {abs(float(loss)-float(ref))/abs(float(ref)):.2e} gradiente {float((g1-g2).abs().max())/float(g2.abs().max()):.2e} "
          f"finito {bool(torch.isfinite(g1).all())}")

# (4) cross rifiutato
try:
    rc.pose_rank_contrast(z, (target, mask), tv, temperature=0.1, stride=2, positives="cross")
    print("(4) cross ACCETTATO")
except ValueError as e:
    print("(4) cross rifiutato:", str(e)[:70])

# (5) partizione a livello di modulo: stesse 8 finestre, intero contro meta' ricomposte col contratto
z, target, mask, tv = data(S=8, P=160, J=20, D=16, pose_windows=(0, 2, 5), ties=False, partial=False, seed=7)
tv[:] = True
def run(idx):
    l, n, c, _ = rc.pose_rank_contrast(z[idx], (target[idx], mask[idx]), tv[idx], temperature=0.1, stride=8)
    return float(n), int(c)
whole = run(list(range(8)))
for parts in ([[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]]):
    nums, cnts = zip(*[run(p) for p in parts])
    print(f"(5) {len(parts)} parti: somma numeratori/somma conteggi = {sum(nums)/max(sum(cnts),1):.6f} (conteggio {sum(cnts)}) contro intero {whole[0]/whole[1]:.6f} (conteggio {whole[1]})")
