"""Toy analog of the temporal positive control on SYNTHETIC latents (no real data).
Purpose: where does the minimum of the noisy-MSE selection curve fall when the SAME recipe
(TCN_post, AdamW lr 0.003, wd 0.01, batch 32, dropout 0.1, clip 1, 150 epochs) is trained on
the clean target y* instead of y?  Uses the frozen heads/train/controls code of the checkout.
Arguments: target(clean|noisy) epochs user_shift out_json
"""
import sys, json, time
import numpy as np, torch
torch.set_num_threads(1)
from pose_temponet import heads, train, controls, metrics

target_kind, epochs, shift, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(12345)
U, WPU, T, D, K, J = 40, 46, 160, 384, 48, 20
W = rng.standard_normal((D, K)).astype(np.float32) / np.sqrt(K)
rho = np.linspace(0.5, 0.98, K).astype(np.float32)
feats, meta = [], []
for u in range(U):
    Wu = W + shift * rng.standard_normal((D, K)).astype(np.float32) / np.sqrt(K)
    bu = 0.5 * rng.standard_normal(D).astype(np.float32)
    cu = np.exp(0.3 * rng.standard_normal(K)).astype(np.float32)
    f = np.empty((WPU, T, K), dtype=np.float32)
    f[:, 0] = rng.standard_normal((WPU, K))
    innov = rng.standard_normal((WPU, T, K)).astype(np.float32) * np.sqrt(1 - rho**2)
    for t in range(1, T):
        f[:, t] = rho * f[:, t-1] + innov[:, t]
    g = np.exp(0.5 * rng.standard_normal((WPU, 1, 1))).astype(np.float32)
    x = (f * cu * g) @ Wu.T + bu + 0.3 * rng.standard_normal((WPU, T, D)).astype(np.float32)
    feats.append(x.astype(np.float32))
    meta += [{"user": "u%02d" % u, "w": i} for i in range(WPU)]
X = np.concatenate(feats)
users = np.array([m["user"] for m in meta])
names = sorted(set(users))
fit = np.flatnonzero(np.isin(users, names[::2])); held = np.flatnonzero(np.isin(users, names[1::2]))
support = np.ones((len(X), T - 12, J), dtype=bool)
comp = controls.generate_components(X, meta, support, fit, held, names, outer_fold=0)
print("hidden active fraction fitting", comp["hidden_active_fraction_fitting"], flush=True)
SIGMA = 1.3
Zf = None
def ln(x):
    return torch.nn.functional.layer_norm(torch.as_tensor(x), (D,), eps=1e-5).numpy().astype(np.float64)
def edyn_stats(pred, tgt, dyn):
    p, y = pred[dyn], tgt[dyn]
    pc, yc = p - p.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    return float(((pc - yc) ** 2).sum()), float((yc ** 2).sum())
def pwmse(pred, tgt):
    return float(((pred - tgt) ** 2).mean(axis=(1, 2)).mean())
# closed-form references and choice of a (sigma fixed 1.3): oracle margin fraction closest to 0.08 on held
Zfit = ln(X[fit])[:, 6:-6].reshape(-1, D); Zheld = ln(X[held])[:, 6:-6].reshape(-1, D)
A = np.concatenate([Zfit, np.ones((len(Zfit), 1))], 1); G = A.T @ A
Ah = np.concatenate([Zheld, np.ones((len(Zheld), 1))], 1)
best = None
for a in [0.2 * k for k in range(1, 16)]:
    clean, obs = controls.synthetic_target(comp, a, SIGMA)
    pv = clean.var(axis=1).mean(axis=1); thr = float(np.median(pv[fit])); dynh = pv[held] >= thr
    beta = np.linalg.lstsq(G, A.T @ clean[fit][:, 6:-6].reshape(-1, J), rcond=1e-12)[0]
    pred_h = (Ah @ beta).reshape(len(held), T - 12, J)
    yh, ysh = obs[held][:, 6:-6], clean[held][:, 6:-6]
    s_ls, ss = edyn_stats(pred_h, yh, dynh); s_or, _ = edyn_stats(ysh, yh, dynh)
    frac = (s_ls - s_or) / s_ls
    row = (abs(frac - 0.08), a, s_ls / ss, s_or / ss, frac)
    if best is None or row < best: best = row
_, a, E_ls, E_or, frac = best
print("chosen a=%.1f sigma=%.2f  held E_cleanLS=%.6f E_oracle=%.6f margin_frac=%.4f" % (a, SIGMA, E_ls, E_or, frac), flush=True)
clean, obs = controls.synthetic_target(comp, a, SIGMA)
pv = clean.var(axis=1).mean(axis=1); thr = float(np.median(pv[fit])); dynh = pv[held] >= thr; dynf = pv[fit] >= thr
beta = np.linalg.lstsq(G, A.T @ clean[fit][:, 6:-6].reshape(-1, J), rcond=1e-12)[0]
ls_h = (Ah @ beta).reshape(len(held), T - 12, J); ls_f = (A @ beta).reshape(len(fit), T - 12, J)
yh, ysh = obs[held][:, 6:-6], clean[held][:, 6:-6]; yf, ysf = obs[fit][:, 6:-6], clean[fit][:, 6:-6]
ref = {"a": a, "sigma": SIGMA,
       "held": {"E_cleanLS_noisy": E_ls, "E_oracle_noisy": E_or,
                "E_cleanLS_clean": (lambda s: s[0] / s[1])(edyn_stats(ls_h, ysh, dynh)),
                "mse_cleanLS_clean": pwmse(ls_h, ysh), "mse_cleanLS_noisy": pwmse(ls_h, yh), "mse_oracle_noisy": pwmse(ysh, yh)},
       "fit": {"E_cleanLS_noisy": (lambda s: s[0] / s[1])(edyn_stats(ls_f, yf, dynf)),
               "E_oracle_noisy": (lambda s: s[0] / s[1])(edyn_stats(ysf, yf, dynf)),
               "E_cleanLS_clean": (lambda s: s[0] / s[1])(edyn_stats(ls_f, ysf, dynf)),
               "mse_cleanLS_clean": pwmse(ls_f, ysf)}}
print("refs", json.dumps(ref), flush=True)
del A, Ah, Zfit, Zheld
train_target = clean if target_kind == "clean" else obs
ids = ["w%05d" % i for i in range(len(fit))]
data = train.TensorTrainingSet(torch.as_tensor(X[fit]), torch.as_tensor(train_target[fit].astype(np.float32)),
                               torch.as_tensor(support[fit]), ids)
traj = train.Trajectory(lambda seed, fold: heads.make_head("TCN_post", seed), data, lr=0.003, seed=0, fold=0, device="cpu")
Xh, Xf = torch.as_tensor(X[held]), torch.as_tensor(X[fit])
def predict(model, x):
    model.eval(); outs = []
    with torch.no_grad():
        for s in range(0, len(x), 64):
            outs.append(model(x[s:s+64])[:, 6:-6].double().numpy())
    model.train(); return np.concatenate(outs)
rows = []; t0 = time.time(); spe = traj.steps_per_epoch
print("steps per epoch", spe, "fit windows", len(fit), "held windows", len(held), flush=True)
for ep in range(1, epochs + 1):
    traj.train_to_step(ep * spe)
    ph = predict(traj.model, Xh)
    sn, ssn = edyn_stats(ph, yh, dynh); sc, ssc = edyn_stats(ph, ysh, dynh)
    row = {"epoch": ep, "held_mse_noisy": pwmse(ph, yh), "held_mse_clean": pwmse(ph, ysh),
           "held_edyn_noisy": sn / ssn, "held_edyn_clean": sc / ssc,
           "train_loss": traj.epoch_losses[-1]["loss_sum"] / traj.epoch_losses[-1]["window_count"]}
    if ep % 5 == 0 or ep <= 3 or ep == epochs:
        pf = predict(traj.model, Xf)
        fn, fsn = edyn_stats(pf, yf, dynf); fc, fsc = edyn_stats(pf, ysf, dynf)
        row.update({"fit_mse_clean": pwmse(pf, ysf), "fit_edyn_noisy": fn / fsn, "fit_edyn_clean": fc / fsc})
    rows.append(row)
    if ep % 10 == 0 or ep <= 2:
        print(json.dumps(row), "elapsed %.0fs" % (time.time() - t0), flush=True)
    json.dump({"refs": ref, "rows": rows, "target": target_kind, "shift": shift}, open(out, "w"))
print("done", flush=True)
