"""Runner del verificatore: ricetta del protocollo (PROT:29-35, 61-69) con codice mio. Un fit: testa, LR, fold.
Allena sugli utenti del fold k (ID ordinati alternati), valuta alla fine di ogni epoca sul fold 1-k in eval/no_grad."""
import sys, json, time, argparse
import numpy as np, torch
from torch import nn
ap = argparse.ArgumentParser()
ap.add_argument('--dir'); ap.add_argument('--arm'); ap.add_argument('--lr', type=float); ap.add_argument('--fold', type=int)
ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--over', default='{}'); ap.add_argument('--out')
ap.add_argument('--init', default='torch')
a = ap.parse_args()
sys.path.insert(0, a.dir)
import gen_v as G
torch.set_num_threads(1)
T, D, J, C = G.T, G.D, G.J, G.CROP
SL = slice(C, T - C)
p = dict(G.P0); p.update(json.loads(a.over))
t0 = time.time()
d = G.make(p, n_val_users=1, wpu=48)
ids = np.arange(40)
fu = [ids[::2], ids[1::2]]
tr = np.isin(d['utr'], fu[a.fold]); he = np.isin(d['utr'], fu[1 - a.fold])
xtr, ytr = d['xtr'][tr], d['ytr'][tr]
xhe, yhe, dhe, ychy = d['xtr'][he], d['ytr'][he], d['dtr'][he], d['yctr'][he]
del d

def zero(m):
    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias); return m
class L0(nn.Module):
    def __init__(s): super().__init__(); s.n = nn.LayerNorm(D, eps=1e-5); s.o = zero(nn.Linear(D, J))
    def forward(s, x): return s.o(s.n(x))
class L13(nn.Module):
    def __init__(s): super().__init__(); s.n = nn.LayerNorm(D, eps=1e-5); s.o = zero(nn.Conv1d(D, J, 13, padding=6))
    def forward(s, x): return s.o(s.n(x).transpose(1, 2)).transpose(1, 2)
class TCN(nn.Module):
    def __init__(s):
        super().__init__(); s.n = nn.LayerNorm(D, eps=1e-5)
        s.b = nn.Sequential(nn.Conv1d(D, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                            nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=3, dilation=3),
                            nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1))
        s.o = zero(nn.Conv1d(32, J, 1))
    def forward(s, x): return s.o(s.b(s.n(x).transpose(1, 2))).transpose(1, 2)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.n = nn.LayerNorm(D, eps=1e-5)
        s.b = nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(),
                            nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1))
        s.o = zero(nn.Linear(32, J))
    def forward(s, x): return s.o(s.b(s.n(x)))
torch.manual_seed(0)
m = dict(L0=L0, L13=L13, TCN=TCN, MLP=MLP)[a.arm]()
if a.init == 'kaiming':  # variante: inizializzazione che conserva il segnale negli strati nascosti, uscita sempre a zero
    for mod in m.b.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(mod.weight, nonlinearity='relu'); nn.init.zeros_(mod.bias)
npar = sum(q.numel() for q in m.parameters())
dec = [q for q in m.parameters() if q.ndim >= 2]; nod = [q for q in m.parameters() if q.ndim < 2]
opt = torch.optim.AdamW([{'params': dec, 'weight_decay': 0.01}, {'params': nod, 'weight_decay': 0.0}], lr=a.lr,
                        betas=(0.9, 0.999), eps=1e-8)
g = torch.Generator().manual_seed(777 + a.fold)

@torch.no_grad()
def evaluate(x, y, dyn, yclean=None):
    m.eval()
    acc = dict(mse=0.0, n=0, cs=0.0, vs=0.0, csa=0.0, vsa=0.0, lev=0.0, tot=0.0)
    for i in range(0, len(x), 64):
        pr = m(torch.from_numpy(x[i:i+64].astype(np.float32)))[:, SL].double()
        t = torch.from_numpy(y[i:i+64, SL]).double(); dd = torch.from_numpy(dyn[i:i+64])
        e2 = (pr - t) ** 2
        acc['mse'] += float(e2.mean(dim=(1, 2)).sum()); acc['n'] += len(pr); acc['tot'] += float(e2.sum())
        cp, ct = pr - pr.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
        cs, vs = ((cp - ct) ** 2).sum(dim=(1, 2)), (ct ** 2).sum(dim=(1, 2))
        acc['cs'] += float(cs[dd].sum()); acc['vs'] += float(vs[dd].sum()); acc['csa'] += float(cs.sum()); acc['vsa'] += float(vs.sum())
        acc['lev'] += float(((pr.mean(1) - t.mean(1)) ** 2).sum() * (T - 2 * C))
    return dict(mse=acc['mse'] / acc['n'], E_dyn=acc['cs'] / acc['vs'], E_all=acc['csa'] / acc['vsa'], level_share=acc['lev'] / acc['tot'])

curve, trl = [], []
n = len(xtr)
for ep in range(1, a.epochs + 1):
    m.train()
    order = torch.randperm(n, generator=g).numpy()
    tot, st = 0.0, 0
    for i in range(0, n, 32):
        ix = np.sort(order[i:i+32])
        xb = torch.from_numpy(xtr[ix].astype(np.float32)); yb = torch.from_numpy(ytr[ix][:, SL])
        loss = ((m(xb)[:, SL] - yb) ** 2).mean(dim=(1, 2)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        tot += loss.item(); st += 1
    trl.append(tot / st)
    curve.append(evaluate(xhe, yhe, dhe))
    print(f'{a.arm} lr {a.lr:g} fold {a.fold} ep {ep} steps {st} train {trl[-1]:.4f} held mse {curve[-1]["mse"]:.4f} E_dyn {curve[-1]["E_dyn"]:.4f}', flush=True)
json.dump(dict(arm=a.arm, lr=a.lr, fold=a.fold, params=npar, steps=st, train=trl, held=curve, over=p, seconds=time.time() - t0),
          open(a.out, 'w'), indent=0)
print('fatto', time.time() - t0, flush=True)
