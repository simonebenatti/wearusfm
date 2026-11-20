# Costruzione indipendente delle cinque teste dal testo del protocollo (righe 29-49) e controlli strutturali/numerici.
import sys, itertools
import numpy as np
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9/src"
sys.path.insert(0, SNAP)
import torch
from torch import nn
from wearusfm.model.pose_head import PoseHead

class PerPatch(nn.Module):
    """(B,T,C) -> LN per patch -> corpo su (B,C,T) -> (B,T,20)"""
    def __init__(self, body, dim=384):
        super().__init__(); self.norm = nn.LayerNorm(dim, eps=1e-5); self.body = body
    def forward(self, x):
        return self.body(self.norm(x).transpose(1, 2)).transpose(1, 2)
def zero_last(m):
    last = [mm for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear))][-1]
    with torch.no_grad(): last.weight.zero_(); last.bias.zero_()
    return m
def tcn():
    return zero_last(PerPatch(nn.Sequential(
        nn.Conv1d(384, 64, 1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=1, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(), nn.Dropout(0.1),
        nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1),
        nn.Conv1d(32, 20, 1))))
def mlp():
    w = [384, 96, 96, 96, 64, 32, 20]; layers = []
    for i in range(6):
        layers.append(nn.Conv1d(w[i], w[i + 1], 1))
        if i < 5: layers.append(nn.ReLU())
        if i in (3, 4): layers.append(nn.Dropout(0.1))   # dopo i hidden da 64 e 32
    return zero_last(PerPatch(nn.Sequential(*layers)))
def l13():
    return zero_last(PerPatch(nn.Sequential(nn.Conv1d(384, 20, 13, padding=6))))
def l0():
    return PoseHead(384)
def count(m):
    tot = sum(p.numel() for p in m.parameters())
    wts = sum(mm.weight.numel() for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear)))
    bias = sum(mm.bias.numel() for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear)))
    ln = sum(p.numel() for mm in m.modules() if isinstance(mm, nn.LayerNorm) for p in mm.parameters())
    ntr = sum(1 for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear)))
    nrelu = sum(1 for mm in m.modules() if isinstance(mm, nn.ReLU))
    ndrop = sum(1 for mm in m.modules() if isinstance(mm, nn.Dropout))
    return tot, wts, bias, ln, ntr, nrelu, ndrop
torch.manual_seed(0)
heads = {"L0": l0(), "MLP": mlp(), "L13": l13(), "TCN": tcn()}
print("testa: totale, pesi, bias, LN, trasformazioni, ReLU, Dropout")
for k, m in heads.items():
    print(" ", k, count(m))
# MAC per patch = pesi (ogni peso usato una volta per patch, stride 1, nessun pooling); per finestra x160
for k, m in heads.items():
    print("  MAC/finestra", k, count(m)[1] * 160)
# RF simbolico per la TCN e L13 e per l'ipotetica 2/4/8
def offsets(dils, k=3):
    cnt = {}
    for comb in itertools.product(*[[-d, 0, d] for d in dils]):
        s = sum(comb); cnt[s] = cnt.get(s, 0) + 1
    return cnt
c = offsets((1, 2, 3)); span = range(-6, 7)
print("TCN offsets 1/2/3:", [c.get(o, 0) for o in span], "somma", sum(c.values()), "buchi", sum(1 for o in span if o not in c))
c = offsets((2, 4, 8)); span = range(-14, 15)
print("ipotesi k3 d=2/4/8:", "raggiunti", len(c), "su", len(span), "buchi", sum(1 for o in span if o not in c))
c = offsets((2, 2, 4, 4, 8, 8)); span = range(-28, 29)
print("ipotesi sei k3 d=2,2,4,4,8,8 (coppie come TEMPONet):", "raggiunti", len(c), "su", len(span), "buchi", sum(1 for o in span if o not in c))
# dipendenza dal padding: perturbo SOLO i valori di padding (ingresso costruito con 6 patch fittizie per lato rimpiazzate)
torch.manual_seed(1)
for k in ("TCN", "L13", "MLP", "L0"):
    m = heads[k]
    # ultimo strato reso casuale per vedere le dipendenze
    last = [mm for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear))][-1]
    with torch.no_grad(): last.weight.normal_(0, 0.1); last.bias.normal_(0, 0.1)
    m.eval()
    x = torch.randn(3, 160, 384, dtype=torch.float32)
    # Jacobiano rispetto all'ingresso per tutti i centri (somma assoluta): quali offset contano
    xg = x.clone().requires_grad_(True)
    J = []
    for t0 in (6, 80, 153):
        g = torch.autograd.grad(m(xg)[:, t0].sum(), xg)[0].abs().sum((0, 2))
        J.append(sorted(set((np.nonzero(g.numpy() > 0)[0] - t0).tolist())))
    print(" ", k, "offset con gradiente per centri 6/80/153:", [ (j[0], j[-1], len(j)) for j in J])
# zero-init e primo passo AdamW (ricetta del par. 4) su dati finti
def recipe_step(m, x, y, mask, lr):
    decay = [p for n, p in m.named_parameters() if p.ndim > 1]
    nodecay = [p for n, p in m.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.999), eps=1e-8)
    res = []
    for step in range(2):
        opt.zero_grad()
        pred = m(x)
        err = ((pred - y) ** 2) * mask
        loss = (err.sum((1, 2)) / mask.sum((1, 2))).mean()
        loss.backward()
        lastw = [mm for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear))][-1].weight
        inner = [p for p in m.parameters() if p is not lastw and p is not [mm for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear))][-1].bias]
        nz = sum(int((p.grad != 0).sum()) for p in inner); tot = sum(p.numel() for p in inner)
        fin = all(torch.isfinite(p.grad).all() for p in m.parameters())
        res.append((step, round(float(loss), 6), nz, tot, fin))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return res
torch.manual_seed(2)
x = torch.randn(32, 160, 384); y = torch.randn(32, 160, 20)
mask = torch.zeros(32, 160, 20); mask[:, 6:154] = (torch.rand(32, 148, 20) > 0.15).float()
for k, f in (("L0", l0), ("MLP", mlp), ("L13", l13), ("TCN", tcn)):
    torch.manual_seed(0); m = f(); m.train()
    print(" ", k, "lr1e-3 (passo, loss, grad interni non nulli, totale interni, finiti):", recipe_step(m, x, y, mask, 1e-3))
# semi: L0 e L13 hanno init identica fra semi?
for k, f in (("L0", l0), ("L13", l13), ("MLP", mlp), ("TCN", tcn)):
    sd = []
    for s in (0, 1, 2):
        torch.manual_seed(s); sd.append(torch.cat([p.detach().flatten() for p in f().parameters()]))
    print(" ", k, "init identica fra semi 0/1/2:", bool(torch.equal(sd[0], sd[1]) and torch.equal(sd[0], sd[2])))
