"""SINTETICO, mondo temporale (stessi parametri di synth_temporal.py): la TCN della proposta con inizializzazione che conserva il segnale
(kaiming_normal_ fan_in per ReLU, bias nascosti a zero, uscita sempre a zero) impara la dinamica che la TCN standard non impara?
Un solo fold (allena fold 0, legge fold 1), 50 epoche, stessa ricetta e stesse funzioni di synth_selection.py. Nessun dato del progetto."""
import json, sys, time
import numpy as np
import torch
from torch import nn
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione")
import synth_selection as S

class TCNk(S.TCN):
    def __init__(self):
        super().__init__()
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
S.ARMS["TCNk"] = TCNk
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
p = dict(sig_g=0.5, sig_n=0.6, sig_o=0.6, sig_l=0.7, sig_eps=0.5, sig_eta=0.6, s_c=0.25, s_f=0.2, t_c=0.5, t_f=0.35, a_static=0.25, temporal=True)
world = S.make_world(1001)
xtr, ytr, utr = S.generate(world, 2001, 40, 48, p)
xva, yva, uva = S.generate(world, 3001, 15, 48, p)
ytr, yva = S.standardize(ytr, yva)
thr = float(np.median(S.pose_var(yva)))
dtr = S.pose_var(ytr) >= thr
ids = np.arange(40); fm = [np.isin(utr, f) for f in (ids[::2], ids[1::2])]
f0, f1 = (xtr[fm[0]], ytr[fm[0]], dtr[fm[0]]), (xtr[fm[1]], ytr[fm[1]], dtr[fm[1]])
out = {}
for name, arm, lr in (("L0_1e-4", "L0", 1e-4), ("TCN_1e-3", "TCN", 1e-3), ("TCNk_1e-3", "TCNk", 1e-3), ("TCNk_1e-4", "TCNk", 1e-4)):
    t = time.time()
    r = S.fit(arm, lr, f0[0], f0[1], {"held": f1}, seed=0, shuffle_seed=10_000, epochs=50)
    c = r["curves"]["held"]
    mse = [e["mse_sum"] / e["n"] for e in c]; ed = [e["csse_dyn"] / e["css_dyn"] for e in c]
    out[name] = {"mse": mse, "E_dyn": ed, "train_loss": r["train_loss"]}
    print(f"{name}: MSE held min {min(mse):.4f} ep {int(np.argmin(mse))+1}; E_dyn min {min(ed):.4f} ep {int(np.argmin(ed))+1}; "
          f"E_dyn all'epoca del minimo MSE {ed[int(np.argmin(mse))]:.4f}; ep10/25/50 E_dyn {ed[9]:.4f}/{ed[24]:.4f}/{ed[49]:.4f}; "
          f"loss train ep50 {r['train_loss'][-1]:.4f} ({time.time()-t:.0f}s)", flush=True)
    json.dump(out, open("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione/tcn_init_temporal.json", "w"))
