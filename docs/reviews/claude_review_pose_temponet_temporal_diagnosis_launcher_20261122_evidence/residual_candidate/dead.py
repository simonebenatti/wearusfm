"""Il ramo profondo e' morto? Giocattolo, fold 0. Rigioca i primi passi con la stessa ricetta e conta le unita' ReLU attive.
Uso: dead.py <variante cur|res|warm> <seme> <epoche>"""
import sys, json
import numpy as np, torch
from torch import nn
import toy_res as TR
from pose_temponet import heads, metrics
torch.set_num_threads(1)
variant, seed, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = np.load(TR.WORK+"data_f0.npz"); x, fit = d["x"], d["fit"]
y = d["y"].astype(np.float64)
seeds = heads.trajectory_seeds(seed, 0)
model = heads.make_head("TCN_post", seed=seed)
target = y[fit][:, 6:-6]
if variant == "res":
    for p in list(model.norm.parameters())+list(model.affine.parameters()): p.requires_grad_(False)
    target = target-d["ridge_fit"].astype(np.float64)
groups = [{"params": [p for p in g["params"] if p.requires_grad], "weight_decay": g["weight_decay"]} for g in heads.optimizer_groups(model, 0.01)]
opt = torch.optim.AdamW(groups, lr=0.003, betas=(0.9, 0.999), eps=1e-8)
torch.manual_seed(seeds["dropout"]); shuffle = torch.Generator().manual_seed(seeds["shuffle"])
xt = torch.from_numpy(x[fit]); yt = torch.from_numpy(target.astype(np.float32)); sup = torch.ones((32, 148, 20), dtype=torch.bool)
probe = xt[::15][:64].float()

def census(tag):
    model.eval(); out = []
    with torch.no_grad():
        h = model.norm(probe).transpose(1, 2)
        for m in model.deep:
            h = m(h)
            if isinstance(m, nn.ReLU):
                alive = int((h > 0).any(dim=2).any(dim=0).sum())
                out.append("%d/%d vive, attive %.3f" % (alive, h.shape[1], float((h > 0).float().mean())))
        std_t = float(h.std(dim=2).mean())
    model.train()
    last = [m for m in model.deep if isinstance(m, nn.Conv1d)][-1]
    print("%s %s s%d | %s | uscita profonda: dev. std nel tempo %.5f | |W ultimo| %.5f" % (tag, variant, seed, " ; ".join(out), std_t, float(last.weight.abs().mean())), flush=True)

census("passo 0")
step = 0
for ep in range(1, epochs+1):
    perm = torch.randperm(len(fit), generator=shuffle)
    for s in range(0, len(fit), 32):
        b = perm[s:s+32]
        opt.zero_grad(set_to_none=True)
        loss = metrics.per_window_mse(model(xt[b].float())[:, 6:-6], yt[b], sup[:len(b)])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
        if step in (1, 2, 3, 5, 10, 20, 29, 58, 87):
            census("passo %d (norma grad prima del clip %.2f)" % (step, float(gn)))
census("fine epoca %d" % epochs)
