# Mondo lineare sintetico (mio): la TCN della specifica (senza ramo lineare) contro L0 e contro la stessa TCN con un ramo lineare
# aggiunto (L0 + TCN, entrambi con uscita a zero). Ricetta del par. 4. Solo per mostrare che i bracci non sono annidati.
import math, time
import torch
from torch import nn
torch.set_num_threads(1)
T, D, J, NTR, NHO = 28, 384, 20, 640, 320
g = torch.Generator().manual_seed(11)
A = torch.randn(8, D, generator=g); c = 4.0 * torch.randn(1, 1, D, generator=g); W = torch.randn(8, J, generator=g) / math.sqrt(8)
def gen(n):
    s = torch.zeros(n, T, 8); s[:, 0] = torch.randn(n, 8, generator=g)
    for t in range(1, T): s[:, t] = 0.95 * s[:, t - 1] + math.sqrt(1 - 0.95 ** 2) * torch.randn(n, 8, generator=g)
    x = c + s @ A + 1.0 * torch.randn(n, T, D, generator=g)
    y = s @ W + 0.5 * torch.randn(n, T, J, generator=g)
    return x, y
xtr, ytr = gen(NTR); xho, yho = gen(NHO)
mu, sd = ytr.mean((0, 1)), ytr.std((0, 1)); ytr, yho = (ytr - mu) / sd, (yho - mu) / sd
mask = torch.zeros(T, dtype=torch.bool); mask[6:T - 6] = True
def body():
    return nn.Sequential(nn.Conv1d(D, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(), nn.Dropout(0.1),
        nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(32, J, 1))
class Head(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.norm = nn.LayerNorm(D); self.kind = kind
        self.tcn = body() if kind in ("TCN", "TCN+lin") else None
        self.lin = nn.Conv1d(D, J, 1) if kind in ("L0", "TCN+lin") else None
        for m in (self.tcn[-1] if self.tcn is not None else None, self.lin):
            if m is not None:
                with torch.no_grad(): m.weight.zero_(); m.bias.zero_()
    def forward(self, x):
        h = self.norm(x).transpose(1, 2); out = 0
        if self.tcn is not None: out = out + self.tcn(h)
        if self.lin is not None: out = out + self.lin(h)
        return out.transpose(1, 2)
def loss_fn(p, y): return ((p - y) ** 2)[:, mask].mean((1, 2)).mean()
def run(kind, lr, seed=0):
    torch.manual_seed(seed); m = Head(kind)
    dec = [p for p in m.parameters() if p.ndim > 1]; nod = [p for p in m.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.01}, {"params": nod, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    gs = torch.Generator().manual_seed(1000 + seed); hist = []
    for ep in range(50):
        m.train(); perm = torch.randperm(NTR, generator=gs)
        for i in range(0, NTR, 32):
            b = perm[i:i + 32]; opt.zero_grad(); l = loss_fn(m(xtr[b]), ytr[b]); l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad(): hist.append(float(loss_fn(m(xho), yho)))
    e = min(range(50), key=lambda i: (hist[i], i))
    return e + 1, hist[e], hist[49]
t0 = time.time()
for kind in ("L0", "TCN", "TCN+lin"):
    for lr in (1e-4, 1e-3):
        ep, best, last = run(kind, lr)
        print("%-8s lr %g: minimo %.4f all'epoca %2d; epoca 50 %.4f   [%.0f s]" % (kind, lr, best, ep, last, time.time() - t0), flush=True)
