"""Scettico RC-2. Giocattolo rifatto con la classe CONGELATA train.Trajectory (ciclo, ottimizzatore, clip, semi, dropout del
codice vero), non con il ciclo scritto a mano dall'altro revisore. Dati: gli stessi latenti sintetici (toy_res.gen, fold 0).
Uso: rc2_run.py <cur|res> <seme 0..2> <lr> <epoche_obiettivo> <max_secondi>
Ripresa esatta fra esecuzioni: ck_<nome>.pt (Trajectory.state()/restore())."""
import sys, json, time, os
import numpy as np, torch
torch.set_num_threads(1)
from torch import nn
from pose_temponet import heads, train, metrics
W = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc2_work/"
CROP = 6
variant, seed, lr, target_ep, max_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
assert variant in ("cur", "res")
FOLD = int(os.environ.get("RC2_FOLD", "0"))      # 0 = stesso fold dell'altro revisore; 1 = fold scambiato (aggiunta mia)
name = ("f%d_" % FOLD if FOLD else "")+"%s_s%d_lr%g" % (variant, seed, lr)
t0 = time.time()
d = np.load(W+"data_f%d.npz" % FOLD); refs = json.load(open(W+"refs_f%d.json" % FOLD))
x, fit, held = d["x"], d["fit"], d["held"]
y = d["y"].astype(np.float64); ystar = d["ystar"].astype(np.float64)
tgt = y[fit][:, CROP:-CROP]
base_held = None
if variant == "res":
    tgt = tgt-d["ridge_fit"].astype(np.float64)
    base_held = d["ridge_held"].astype(np.float64)
data = train.TensorTrainingSet(features=torch.from_numpy(x[fit]), target=torch.from_numpy(tgt.astype(np.float32)),
                               support=torch.ones((len(fit), 160-2*CROP, 20), dtype=torch.bool),
                               window_ids=["w%05d" % i for i in range(len(fit))])

def factory(seed_, fold_):
    m = heads.make_head("TCN_post", seed=seed_)
    if variant == "res":      # LN fissa (identita' affine iniziale) e ramo affine fermo a zero: uscita = solo ramo profondo
        for p in list(m.norm.parameters())+list(m.affine.parameters()):
            p.requires_grad_(False)
    return m

traj = train.Trajectory(factory, data, lr=lr, seed=seed, fold=FOLD, device="cpu")
spe = traj.steps_per_epoch
ck, lj = W+"ck_%s.pt" % name, W+"run_%s.json" % name
if os.path.exists(ck):
    traj.restore(torch.load(ck, weights_only=False)); log = json.load(open(lj))
else:
    log = {"name": name, "variant": variant, "seed": seed, "lr": lr, "impl": "train.Trajectory congelata", "steps_per_epoch": spe,
           "trainable": sum(p.numel() for p in traj.model.parameters() if p.requires_grad), "points": []}
# scala del cancello ricalcolata in modo indipendente dove possibile
mask_fit = np.ones(ystar[fit].shape, dtype=bool)
thr = metrics.training_threshold(ystar[fit], mask_fit)
# ystar e' salvato in float32: la soglia ricalcolata differisce di ~1e-9 relativo da quella in refs (float64); uso quella di refs
assert abs(thr-refs["threshold"]) < 1e-7*max(1, abs(thr)), (thr, refs["threshold"])
thr = refs["threshold"]
dyn = metrics.dynamic_variance(ystar[held], np.ones(ystar[held].shape, dtype=bool)) >= thr
assert int(dyn.sum()) == refs["held"]["n_dyn"], (int(dyn.sum()), refs["held"]["n_dyn"])
yh, ysh = y[held][:, CROP:-CROP], ystar[held][:, CROP:-CROP]

def sse_dyn(pred, target):
    e = pred-target
    e = e-e.mean(axis=1, keepdims=True)
    return float(np.square(e[dyn]).sum())

sse_or = sse_dyn(ysh, yh)
assert abs(sse_or-refs["held"]["sse_oracle"]) < 1e-6*sse_or, (sse_or, refs["held"]["sse_oracle"])
sse_cl = refs["held"]["sse_cleanLS"]; margin = sse_cl-sse_or
xh = torch.from_numpy(x[held])

def score():
    m = traj.model; was = m.training; m.eval(); tot, aff, alive = [], [], None
    with torch.no_grad():
        for s in range(0, len(held), 128):
            xb = xh[s:s+128].float()
            tot.append(m(xb)[:, CROP:-CROP].double().numpy())
            aff.append(m.affine(m.norm(xb))[:, CROP:-CROP].double().numpy())
            if s == 0:
                h = m.norm(xb[:64]).transpose(1, 2); counts = []
                for layer in m.deep:
                    h = layer(h)
                    if isinstance(layer, nn.ReLU):
                        counts.append([int((h > 0).any(dim=2).any(dim=0).sum()), int(h.shape[1]), float((h > 0).float().mean())])
                alive = counts
    m.train(was)
    p, pa = np.concatenate(tot), np.concatenate(aff)
    if base_held is not None:
        p, pa = p+base_held, base_held
    mse = float(np.square(p-yh).mean(axis=(1, 2)).mean())
    return {"mse": mse, "rec": (sse_cl-sse_dyn(p, yh))/margin, "rec_linear_part": (sse_cl-sse_dyn(pa, yh))/margin,
            "clean_rec": 1-sse_dyn(p, ysh)/margin, "relu": alive}

done = traj.updates//spe
if done == 0 and not log["points"]:
    log["points"].append({"epoch": 0, "held": score()})
print(name, "ripresa da epoca", done, "soglia", thr, "n_dyn", int(dyn.sum()), "preparazione %.0fs" % (time.time()-t0), flush=True)
for ep in range(done+1, target_ep+1):
    traj.train_to_step(ep*spe)
    h = score(); rec = traj.epoch_losses[-1]
    log["points"].append({"epoch": ep, "held": h, "train_loss": rec["loss_sum"]/rec["window_count"], "t": time.time()-t0})
    print("%s ep %d train %.5f held mse %.5f rec %+.4f lin %+.4f penult %d/%d t %.0fs" % (
        name, ep, rec["loss_sum"]/rec["window_count"], h["mse"], h["rec"], h["rec_linear_part"], h["relu"][-1][0], h["relu"][-1][1], time.time()-t0), flush=True)
    json.dump(log, open(lj, "w")); torch.save(traj.state(), ck)
    if time.time()-t0 > max_s:
        print(name, "STOP tempo all'epoca", ep, flush=True); break
