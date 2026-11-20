"""SINTETICO, mondo lineare di r2 (stessi parametri): la MLP della proposta e' sotto-allenata a 50 epoche?
Un solo fold (allena fold 0, valuta fold 1), stesse funzioni di synth_selection.py senza modifiche.
 a) L0 lr 1e-4, 50 epoche (riferimento sullo stesso fold);
 b) MLP lr 1e-3, 150 epoche (stessa ricetta, solo piu' epoche);
 c) MLP con inizializzazione che conserva il segnale (kaiming_normal_ fan_in per ReLU, bias nascosti a zero; uscita sempre a zero),
    lr 1e-3 e 1e-4, 50 epoche.
Nessun dato del progetto."""
import json, sys, time
import numpy as np
import torch
from torch import nn
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione")
import synth_selection as S

class MLPk(S.MLP):
    def __init__(self):
        super().__init__()
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
S.ARMS["MLPk"] = MLPk
torch.set_num_threads(1)
p = dict(sig_g=0.5, sig_n=0.6, sig_o=0.6, sig_l=0.7, sig_eps=0.5, sig_eta=0.6, s_c=0.25, s_f=0.2, t_c=0.5, t_f=0.35, a_static=0.25, temporal=False)
world = S.make_world(1001)
xtr, ytr, utr = S.generate(world, 2001, 40, 48, p)
xva, yva, uva = S.generate(world, 3001, 15, 48, p)
ytr, yva = S.standardize(ytr, yva)
thr = float(np.median(S.pose_var(yva)))
dtr = S.pose_var(ytr) >= thr
ids = np.arange(40); fm = [np.isin(utr, f) for f in (ids[::2], ids[1::2])]
f0, f1 = (xtr[fm[0]], ytr[fm[0]], dtr[fm[0]]), (xtr[fm[1]], ytr[fm[1]], dtr[fm[1]])
out = {}
for name, arm, lr, ep in (("L0_1e-4_50", "L0", 1e-4, 50), ("MLP_1e-3_150", "MLP", 1e-3, 150), ("MLPk_1e-3_50", "MLPk", 1e-3, 50), ("MLPk_1e-4_50", "MLPk", 1e-4, 50)):
    t = time.time()
    r = S.fit(arm, lr, f0[0], f0[1], {"held": f1}, seed=0, shuffle_seed=10_000, epochs=ep)
    c = r["curves"]["held"]
    mse = [e["mse_sum"] / e["n"] for e in c]; ed = [e["csse_dyn"] / e["css_dyn"] for e in c]
    out[name] = {"mse": mse, "E_dyn": ed, "train_loss": r["train_loss"]}
    pts = [e for e in (1, 10, 25, 50, 75, 100, 150) if e <= ep]
    print(f"{name}: MSE held min {min(mse):.4f} ep {int(np.argmin(mse))+1}; E_dyn min {min(ed):.4f} ep {int(np.argmin(ed))+1}; "
          + " ".join(f"ep{e}: {mse[e-1]:.4f}/{ed[e-1]:.4f}" for e in pts) + f" ({time.time()-t:.0f}s)", flush=True)
json.dump(out, open("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione/budget_MLP.json", "w"))
