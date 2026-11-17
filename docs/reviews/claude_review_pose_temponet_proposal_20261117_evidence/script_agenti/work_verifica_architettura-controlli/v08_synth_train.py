# Generatore sintetico DIVERSO da quello del revisore: stato AR(1) a 8 dimensioni, feature con forte componente comune,
# bersaglio con parte lineare istantanea, derivata (s_{t+2}-s_{t-2}) e quadratica. Ricetta del par. 4. Solo un indizio.
import sys, time, math
import torch
from torch import nn
torch.set_num_threads(1)
T, D, J, NTR, NHO = 40, 384, 20, 960, 320
def gen(n, g, A, Bm, c, W1, W2, W3):
    s = torch.zeros(n, T + 4, 8)
    s[:, 0] = torch.randn(n, 8, generator=g)
    for t in range(1, T + 4):
        s[:, t] = 0.95 * s[:, t - 1] + math.sqrt(1 - 0.95 ** 2) * torch.randn(n, 8, generator=g)
    st = s[:, 2:T + 2]
    x = c + st @ A + torch.tanh(st) @ Bm + 0.5 * torch.randn(n, T, D, generator=g)
    y = st @ W1 + (s[:, 4:T + 4] - s[:, 0:T]) @ W2 + (st ** 2 - 1) @ W3 + 0.3 * torch.randn(n, T, J, generator=g)
    return x, y
g = torch.Generator().manual_seed(77)
A, Bm = torch.randn(8, D, generator=g), torch.randn(8, D, generator=g)
c = 4.0 * torch.randn(1, 1, D, generator=g)
W1, W2, W3 = [torch.randn(8, J, generator=g) / math.sqrt(8) for _ in range(3)]
xtr, ytr = gen(NTR, g, A, Bm, c, W1, W2, W3); xho, yho = gen(NHO, g, A, Bm, c, W1, W2, W3)
mu, sd = ytr.mean((0, 1)), ytr.std((0, 1)); ytr, yho = (ytr - mu) / sd, (yho - mu) / sd
mask = torch.zeros(T, dtype=torch.bool); mask[6:T - 6] = True
class PerPatch(nn.Module):
    def __init__(self, body):
        super().__init__(); self.norm = nn.LayerNorm(D); self.body = body
    def forward(self, x): return self.body(self.norm(x).transpose(1, 2)).transpose(1, 2)
def zl(m):
    last = [mm for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear))][-1]
    with torch.no_grad(): last.weight.zero_(); last.bias.zero_()
    return m
def tcn(): return zl(PerPatch(nn.Sequential(nn.Conv1d(D, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(), nn.Dropout(0.1),
        nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(32, J, 1))))
def mlp():
    w = [D, 96, 96, 96, 64, 32, J]; L = []
    for i in range(6):
        L.append(nn.Conv1d(w[i], w[i + 1], 1))
        if i < 5: L.append(nn.ReLU())
        if i in (3, 4): L.append(nn.Dropout(0.1))
    return zl(PerPatch(nn.Sequential(*L)))
def l0(): return zl(PerPatch(nn.Sequential(nn.Conv1d(D, J, 1))))
def loss_fn(p, y): return ((p - y) ** 2)[:, mask].mean((1, 2)).mean()
def run(make, lr, seed=0):
    torch.manual_seed(seed); m = make()
    dec = [p for p in m.parameters() if p.ndim > 1]; nod = [p for p in m.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.01}, {"params": nod, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    gs = torch.Generator().manual_seed(1000 + seed); hist = []
    for ep in range(1, 51):
        m.train(); perm = torch.randperm(NTR, generator=gs)
        for i in range(0, NTR, 32):
            b = perm[i:i + 32]; opt.zero_grad(); l = loss_fn(m(xtr[b]), ytr[b]); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad(): hist.append(float(loss_fn(m(xho), yho)))
    e = min(range(50), key=lambda i: (hist[i], i))
    return e + 1, hist[e], hist[9], hist[24], hist[49]
t0 = time.time()
for name, mk in (("L0", l0), ("MLP", mlp), ("TCN", tcn)):
    for lr in (1e-4, 1e-3):
        ep, best, h10, h25, h50 = run(mk, lr)
        print("%-4s lr %g: minimo all'epoca %2d (%.4f); epoca 10 %.4f, 25 %.4f, 50 %.4f   [%.0f s]" % (name, lr, ep, best, h10, h25, h50, time.time() - t0), flush=True)
