"""SINTETICO, mondo lineare di r2: L13 della proposta a lr 1e-4, allena il fold 0 e legge il fold 1, 25 epoche (rilancio ridotto per
controllare il risultato del processo L13 del tentativo precedente: minimo della MSE all'epoca 1, minimo di E_dyn molto piu' tardi).
Stesse funzioni di synth_selection.py senza modifiche; L0 sullo stesso fold e' in budget_MLP.json. Nessun dato del progetto."""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione")
import synth_selection as S
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
p = dict(sig_g=0.5, sig_n=0.6, sig_o=0.6, sig_l=0.7, sig_eps=0.5, sig_eta=0.6, s_c=0.25, s_f=0.2, t_c=0.5, t_f=0.35, a_static=0.25, temporal=False)
world = S.make_world(1001)
xtr, ytr, utr = S.generate(world, 2001, 40, 48, p)
xva, yva, uva = S.generate(world, 3001, 15, 48, p)
ytr, yva = S.standardize(ytr, yva)
thr = float(np.median(S.pose_var(yva))); dtr = S.pose_var(ytr) >= thr
ids = np.arange(40); fm = [np.isin(utr, f) for f in (ids[::2], ids[1::2])]
f0, f1 = (xtr[fm[0]], ytr[fm[0]], dtr[fm[0]]), (xtr[fm[1]], ytr[fm[1]], dtr[fm[1]])
t = time.time()
r = S.fit("L13", 1e-4, f0[0], f0[1], {"held": f1}, seed=0, shuffle_seed=10_000, epochs=25,
          log=lambda s: print(s, flush=True) if (" epoca  1 " in s or " epoca  2 " in s or " epoca  5 " in s or " epoca 10 " in s or " epoca 15 " in s or " epoca 20 " in s or " epoca 25 " in s) else None)
c = r["curves"]["held"]
mse = [e["mse_sum"] / e["n"] for e in c]; ed = [e["csse_dyn"] / e["css_dyn"] for e in c]
json.dump({"mse": mse, "E_dyn": ed, "train_loss": r["train_loss"]}, open("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione/l13_fold.json", "w"))
print(f"L13 1e-4 fold0->fold1: MSE held min {min(mse):.4f} ep {int(np.argmin(mse))+1} (E_dyn li' {ed[int(np.argmin(mse))]:.4f}); E_dyn min {min(ed):.4f} ep {int(np.argmin(ed))+1}; ({time.time()-t:.0f}s)", flush=True)
