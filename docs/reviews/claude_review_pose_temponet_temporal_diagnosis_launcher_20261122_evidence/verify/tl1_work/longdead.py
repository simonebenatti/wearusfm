"""SCETTICO TL-1 - GIOCATTOLO (latenti sintetici). La morte e' irreversibile sull'orizzonte reale (150 epoche)? Il gradiente e' ESATTAMENTE zero?
Uso: longdead.py <seed> <lr> <variante> <passi_finali>   (ricetta vera train.Trajectory; sonde in modo eval E in modo train)"""
import sys, json, time
import numpy as np, torch
from torch import nn
import toy_tl as K
from pose_temponet import metrics
torch.backends.nnpack.set_flags(False)
seed, lr, variant, last = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
traj, (x, fit, held, ystar, y, refs, a, thr) = K.build(variant, seed, lr)
m = traj.model
relu_idx = [i for i, mod in enumerate(m.deep) if isinstance(mod, nn.ReLU)]
xfit = x[fit]; yfit = y[fit][:, K.CROP:-K.CROP]

def probe_eval():
    mode = m.training; m.eval(); cnt = None; tot = 0
    with torch.no_grad():
        for s in range(0, len(fit), 64):
            acts = []
            hooks = [m.deep[i].register_forward_hook(lambda mod, i_, o: acts.append(o)) for i in relu_idx]
            m(torch.from_numpy(xfit[s:s+64].astype(np.float32)))
            for hk in hooks: hk.remove()
            c = [(o > 0).sum(dim=(0, 2)).double() for o in acts]
            cnt = c if cnt is None else [u+v for u, v in zip(cnt, c)]
            tot += acts[0].shape[0]*acts[0].shape[2]
    m.train(mode)
    return {"eval_active_fraction_all_fit": [round(float(c.sum()/(tot*len(c))), 5) for c in cnt],
            "eval_units_never_active_all_fit": ["%d/%d" % (int((c == 0).sum()), len(c)) for c in cnt]}

def probe_train_grad():
    """Gradiente vero (modo train, dropout attivo, RNG globale: NON tocca lo stream della traiettoria) su tutto il fitting."""
    mode = m.training; m.train()
    perm = torch.randperm(len(fit), generator=torch.Generator().manual_seed(777)).numpy()   # batch mescolati fissi (come in training: utenti misti)
    state = torch.get_rng_state(); torch.manual_seed(987654)
    sq = {"hidden": 0.0, "out": 0.0, "affine": 0.0}; nz = 0; cnt_last = torch.zeros(32, dtype=torch.double); batches = 0
    for s in range(0, len(fit), 32):
        acts = []
        hk = m.deep[relu_idx[-1]].register_forward_hook(lambda mod, i_, o: acts.append(o.detach()))
        ix = np.sort(perm[s:s+32])
        xb = torch.from_numpy(xfit[ix].astype(np.float32)); yb = torch.from_numpy(yfit[ix].astype(np.float32))
        for p in m.parameters():
            p.grad = None
        pred = m(xb)[:, K.CROP:-K.CROP]
        loss = metrics.per_window_mse(pred, yb, torch.ones_like(yb, dtype=torch.bool))
        loss.backward(); hk.remove()
        cnt_last += (acts[0] > 0).sum(dim=(0, 2)).double()
        hb = 0.0
        for name, p in m.named_parameters():
            if p.grad is None: continue
            v = float(p.grad.detach().pow(2).sum())
            if name.startswith("affine"): sq["affine"] += v
            elif name.startswith("deep.12"): sq["out"] += v
            elif name.startswith("deep."): sq["hidden"] += v; hb += v
        nz += int(hb > 0.0); batches += 1
    for p in m.parameters():
        p.grad = None
    torch.set_rng_state(state); m.train(mode)
    return {"train_hidden_grad_rms_per_batch": (sq["hidden"]/batches)**0.5, "train_out_grad_rms_per_batch": (sq["out"]/batches)**0.5,
            "train_affine_grad_rms_per_batch": (sq["affine"]/batches)**0.5, "batches_with_nonzero_hidden_grad": "%d/%d" % (nz, batches),
            "train_last_hidden_units_never_active": "%d/32" % int((cnt_last == 0).sum()),
            "train_last_hidden_active_fraction": float(cnt_last.sum()/(32*len(fit)*K.T_LEN))}

steps = sorted(set([0, 29, 58, 290, 580, 870, 1160, 1740, 2320, 2900, 3480, 4350]) | {last})
t0 = time.time()
for s in [s for s in steps if s <= last]:
    traj.train_to_step(s)
    row = {"seed": seed, "lr": lr, "variant": variant, "step": s, "epoch": round(s/traj.steps_per_epoch, 2), "t": round(time.time()-t0, 1)}
    row.update(probe_eval()); row.update(probe_train_grad())
    row["bias_last_hidden_min_max"] = [round(float(m.deep[relu_idx[-1]-1].bias.min()), 4), round(float(m.deep[relu_idx[-1]-1].bias.max()), 4)]
    print(json.dumps(row), flush=True)
