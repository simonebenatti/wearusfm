"""RC-3, prova diretta nel giocattolo: il ramo profondo allenato sul residuo corregge l'errore della base sugli esclusi?
Base = ridge alpha CV (1,0) congelata, LN fissa, ramo affine a zero congelato (candidato di GPT); bersaglio = y - base sul fitting.
Sugli esclusi: e_b = base - clean_ls (errore di stima della base, lineare in z); d_T = y* - clean_ls (resto non lineare da imparare).
c_b = <g,e_b>/<e_b,e_b> (-1 = errore della base corretto del tutto; 0 = invisibile); c_T = <g,d_T>/<d_T,d_T> (1 = resto imparato).
Uso: rc3_deep.py <max_secondi>"""
import sys, time, json
import numpy as np, torch
torch.set_num_threads(1)
sys.argv, cap = sys.argv[:1], float(sys.argv[1])
import importlib.util
spec = importlib.util.spec_from_file_location("toy", "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc3_work/rc3_toy_lib.py")
toy = importlib.util.module_from_spec(spec); spec.loader.exec_module(toy)
from pose_temponet import linear, metrics, heads
x, fit, held, c, F = toy.x, toy.fit, toy.held, toy.c, toy.F
A, ST, CROP, J, TL = toy.A, toy.ST, toy.CROP, toy.J, toy.TL
mix_clean = {"L": 1., "T": A}; mix_y = {**mix_clean, "epsilon": ST}
ystar = c["L"]+A*c["T"]; y = ystar+ST*c["epsilon"]
base_fit = F.predict_mixture(fit, mix_y, 1.0)[0]; base_held = F.predict_mixture(held, mix_y, 1.0)[0]
clean_held = F.predict_mixture(held, mix_clean, 0.)[0]
yh, ysh = y[held][:, CROP:-CROP], ystar[held][:, CROP:-CROP]
e_b, d_T = base_held-clean_held, ysh-clean_held
thr = float(np.median(toy.pose_var(ystar[fit]))); dyn = toy.pose_var(ystar[held]) >= thr
E_clean, _ = toy.stats(clean_held, yh, dyn); E_or, _ = toy.stats(ysh, yh, dyn); margin = E_clean-E_or
E_base, mse_base = toy.stats(base_held, yh, dyn)
print("base ridge: R %.4f | ||e_b||^2 medio %.5f | ||d_T||^2 medio %.5f | cos(e_b,d_T) %.4f" % (
    (E_clean-E_base)/margin, (e_b**2).mean(), (d_T**2).mean(), (e_b*d_T).sum()/np.sqrt((e_b**2).sum()*(d_T**2).sum())), flush=True)
model = heads.make_head("TCN_post", seed=0); seeds = heads.trajectory_seeds(0, 0)
for p in list(model.norm.parameters())+list(model.affine.parameters()): p.requires_grad_(False)
groups = [{"params": [p for p in g["params"] if p.requires_grad], "weight_decay": g["weight_decay"]} for g in heads.optimizer_groups(model, 0.01)]
opt = torch.optim.AdamW(groups, lr=0.003, betas=(0.9, 0.999), eps=1e-8)
torch.manual_seed(seeds["dropout"]); gen = torch.Generator().manual_seed(seeds["shuffle"])
xt, xh = torch.from_numpy(x[fit]), torch.from_numpy(x[held]).float()
yt = torch.from_numpy((y[fit][:, CROP:-CROP]-base_fit).astype(np.float32)); sup = torch.ones((32, TL-2*CROP, J), dtype=torch.bool)
t0 = time.time(); rows = []
for ep in range(1, 200):
    model.train(); perm = torch.randperm(len(fit), generator=gen)
    for s in range(0, len(fit), 32):
        b = perm[s:s+32]; opt.zero_grad(set_to_none=True)
        loss = metrics.per_window_mse(model(xt[b].float())[:, CROP:-CROP], yt[b], sup[:len(b)])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        g = np.concatenate([model(xh[s:s+128])[:, CROP:-CROP].numpy().astype(np.float64) for s in range(0, len(held), 128)])
    E, mse = toy.stats(base_held+g, yh, dyn)
    row = {"epoch": ep, "c_b": float((g*e_b).sum()/(e_b**2).sum()), "c_T": float((g*d_T).sum()/(d_T**2).sum()),
           "cos_b": float((g*e_b).sum()/np.sqrt((g**2).sum()*(e_b**2).sum()+1e-300)), "R": (E_clean-E)/margin, "mse": mse, "g_ms": float((g**2).mean())}
    rows.append(row)
    print("ep %d c_b %+.4f cos(g,e_b) %+.4f c_T %+.4f R %.4f mse %.5f ||g||^2 %.5f t %.0fs" % (ep, row["c_b"], row["cos_b"], row["c_T"], row["R"], mse, row["g_ms"], time.time()-t0), flush=True)
    json.dump({"R_base": (E_clean-E_base)/margin, "rows": rows}, open("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc3_work/rc3_deep.json", "w"))
    if time.time()-t0 > cap: break
best = min(rows, key=lambda r: (r["mse"], r["epoch"]))
print("SCELTA (min MSE esclusi): ep %d c_b %+.4f c_T %+.4f R %.4f | base da sola R %.4f" % (best["epoch"], best["c_b"], best["c_T"], best["R"], (E_clean-E_base)/margin))
