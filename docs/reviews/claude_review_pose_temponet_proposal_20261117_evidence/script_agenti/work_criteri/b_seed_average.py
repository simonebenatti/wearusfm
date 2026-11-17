"""Punto (b): che cosa stima la media degli accumulatori per utente fra tre semi.
Dati SINTETICI (non i dati del progetto). Usa le funzioni vere di scripts/pose_dynamics.py dello snapshot c15aac9
(window_joint_moments, accumulate, pooled_summary) per calcolare E_dyn come nella valutazione lunga."""
import importlib.util, sys
import numpy as np
SNAP = sys.argv[1]
spec = importlib.util.spec_from_file_location("pose_dynamics", f"{SNAP}/scripts/pose_dynamics.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

rng = np.random.default_rng(7)
n_users, T, J = 15, 148, 20
wins_per_user = rng.integers(5, 60, n_users); wins_per_user[3] = 2        # utenti con numero diverso di finestre
users = np.repeat([f"u{i:02d}" for i in range(n_users)], wins_per_user)
W_ = len(users)
# bersaglio: livello per finestra/giunto + dinamica lenta; maschera con buchi e qualche cella con meno di 2 punti
t = np.arange(T)[None, :, None]
level = rng.normal(0, 1, (W_, 1, J))
dyn = rng.normal(0, 0.7, (W_, 1, J))*np.sin(2*np.pi*t/rng.uniform(40, 200, (W_, 1, J)) + rng.uniform(0, 6, (W_, 1, J)))
y = (level + dyn).astype(np.float32)
mask = rng.random((W_, T, J)) > 0.15
mask[::7, :, 5] = False                      # giunto senza supporto in alcune finestre
mask[1::11, 1:, 9] = False                   # un solo punto valido: escluso dalle statistiche centrate (n>=2)
user_gain = np.repeat(rng.uniform(0.2, 0.6, n_users), wins_per_user)[:, None, None]
def head(seed, gain_shift):
    r = np.random.default_rng(100+seed)
    return ((user_gain+gain_shift)*dyn + 0.6*level + r.normal(0, 0.35, y.shape) + r.normal(0, 0.2, (W_, 1, J))).astype(np.float32)
arms = {"TCN": [head(s, 0.10) for s in range(3)], "L0": [head(10+s, 0.0) for s in range(3)]}
sel = np.ones(W_, bool)

def per_user_acc(pred):
    acc = {}
    D.accumulate(acc, D.window_joint_moments(pred, y, mask), users, sel, 2, D.CENTERED_FIELDS)
    return acc
def edyn(acc):
    return D.pooled_summary(acc, D.CENTERED_FIELDS)["centered_mse_ratio"]
def mean_acc(accs):
    out = {}
    for u in accs[0]:
        assert all(a[u]["n_valid_elements"] == accs[0][u]["n_valid_elements"] for a in accs)
        out[u] = {k: accs[0][u][k] for k in ("n_windows", "n_window_joints", "n_valid_elements")}
        out[u]["sums"] = {k: float(np.mean([a[u]["sums"][k] for a in accs])) for k in D.CENTERED_FIELDS}
    return out
res = {}
for arm, preds in arms.items():
    accs = [per_user_acc(p) for p in preds]
    e_seed = [edyn(a) for a in accs]
    e_acc = edyn(mean_acc(accs))                       # regola del protocollo
    e_ens = edyn(per_user_acc(np.mean(preds, axis=0)))  # ensemble di predizioni (NON previsto)
    den_equal = all(accs[0][u]["sums"]["vary"] == a[u]["sums"]["vary"] for a in accs for u in a)
    corr_seed = [D.pooled_summary(a, D.CENTERED_FIELDS)["pooled_centered_correlation"] for a in accs]
    corr_acc = D.pooled_summary(mean_acc(accs), D.CENTERED_FIELDS)["pooled_centered_correlation"]
    vr_seed = [D.pooled_summary(a, D.CENTERED_FIELDS)["variance_ratio"] for a in accs]
    vr_acc = D.pooled_summary(mean_acc(accs), D.CENTERED_FIELDS)["variance_ratio"]
    res[arm] = dict(accs=accs, e_seed=e_seed, e_acc=e_acc)
    print(f"[{arm}] E_dyn per seme = {[round(v, 6) for v in e_seed]}")
    print(f"   media aritmetica dei tre E_dyn        = {np.mean(e_seed):.15f}")
    print(f"   E_dyn dagli accumulatori mediati      = {e_acc:.15f}   differenza = {e_acc-np.mean(e_seed):.2e}   denominatori identici fra semi: {den_equal}")
    print(f"   E_dyn dell'ensemble (media predizioni) = {e_ens:.15f}   (piu' basso: {100*(1-e_ens/np.mean(e_seed)):.2f}% sotto la media delle tre teste)")
    print(f"   correlazione: per seme {[round(v, 5) for v in corr_seed]} media {np.mean(corr_seed):.6f}; dai momenti mediati {corr_acc:.6f}; differenza {corr_acc-np.mean(corr_seed):+.2e}")
    print(f"   rapporto varianze: media dei semi {np.mean(vr_seed):.12f}; dai momenti mediati {vr_acc:.12f}; differenza {vr_acc-np.mean(vr_seed):+.2e}")
# contrasto: rapporto delle medie (protocollo) contro media dei rapporti per seme
rel_acc = 100*(1-res["TCN"]["e_acc"]/res["L0"]["e_acc"])
rel_pairs = [100*(1-a/b) for a, b in zip(res["TCN"]["e_seed"], res["L0"]["e_seed"])]
rel_all9 = [100*(1-a/b) for a in res["TCN"]["e_seed"] for b in res["L0"]["e_seed"]]
print(f"miglioramento relativo TCN vs L0: dagli accumulatori mediati {rel_acc:.6f}%; per seme appaiato {[round(v, 4) for v in rel_pairs]} (media {np.mean(rel_pairs):.6f}%); media delle 9 coppie {np.mean(rel_all9):.6f}%")
# bootstrap: in ogni replica la statistica dagli accumulatori mediati coincide con la media delle statistiche dei semi
us = sorted(res["TCN"]["accs"][0])
draws = np.random.default_rng(20261102).integers(0, len(us), (2000, len(us)))
def arrs(acc, k): return np.array([acc[u]["sums"][k] for u in us])
den = arrs(res["TCN"]["accs"][0], "vary")
num_seed = np.stack([arrs(a, "centered") for a in res["TCN"]["accs"]])
boot_seed = num_seed[:, draws].sum(2)/den[draws].sum(1)           # (3, 2000)
boot_acc = num_seed.mean(0)[draws].sum(1)/den[draws].sum(1)
print(f"bootstrap (2000 repliche, seme 20261102): max |E*_mediato - media_s E*_s| = {np.abs(boot_acc-boot_seed.mean(0)).max():.2e}")
print(f"   sd bootstrap di un seme = {boot_seed.std(axis=1, ddof=1).round(5).tolist()}; sd della media dei tre = {boot_acc.std(ddof=1):.5f}")
print(f"   dispersione fra i tre semi (sd campionaria di E_dyn) = {np.std(res['TCN']['e_seed'], ddof=1):.5f}: NON entra nel bootstrap sugli utenti")
