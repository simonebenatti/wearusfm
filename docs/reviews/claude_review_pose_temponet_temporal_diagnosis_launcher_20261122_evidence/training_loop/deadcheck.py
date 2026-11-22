"""GIOCATTOLO: stato delle ReLU del ramo profondo nei primi passi (ricetta vera, train.Trajectory). Uso: deadcheck.py <seed> [lr] [variante]"""
import sys, json, math
import numpy as np, torch
from torch import nn
import toy_tl as K
torch.backends.nnpack.set_flags(False)
seed = int(sys.argv[1]); lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003
variant = sys.argv[3] if len(sys.argv) > 3 else "base"
traj, (x, fit, held, ystar, y, refs, a, thr) = K.build(variant, seed, lr)
m = traj.model
xb = torch.from_numpy(x[fit[::8]].astype(np.float32))      # 115 finestre fisse del fitting

def probe():
    mode = m.training; m.eval(); acts = []
    hooks = [mod.register_forward_hook(lambda mod, i, o: acts.append(o)) for mod in m.deep if isinstance(mod, nn.ReLU)]
    with torch.no_grad():
        h = m.norm(xb)
        deep = m.deep(h.transpose(1, 2)).transpose(1, 2); aff = m.affine(h)
    for hk in hooks: hk.remove()
    m.train(mode)
    out = {"deep_out_rms": float(deep.pow(2).mean().sqrt()), "affine_out_rms": float(aff.pow(2).mean().sqrt()),
           "active_fraction": [round(float((o > 0).float().mean()), 3) for o in acts],
           "dead_units": ["%d/%d" % (int(((o > 0).float().mean(dim=(0, 2)) < 0.001).sum()), o.shape[1]) for o in acts],
           "w_out_rms": float(m.deep[12].weight.pow(2).mean().sqrt())}
    return out

steps = [0, 1, 2, 3, 5, 10, 20, 29, 58, 116, 290]
import os
W = int(os.environ.get("TOY_WARMUP", "0"))
for s in steps:
    if W:      # warmup lineare (a tratti) del LR sui primi W passi: NON e' la ricetta congelata
        for group in traj.optimizer.param_groups:
            group["lr"] = lr*min(1.0, s/W)
    traj.train_to_step(s)
    print(json.dumps({"seed": seed, "lr": lr, "variant": variant, "step": s, **probe()}), flush=True)
