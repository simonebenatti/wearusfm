"""Estensione della prova di rappresentabilita' (solo CPU, feature sintetiche, nessun fit, nessun update).
(0) ricostruisce l'incorporamento di GPT con le stesse formule e ne ricontrolla l'errore, anche senza crop;
(a) errore della stessa rete con dropout attivo (model.train()), media su maschere;
(b) gradiente della loss produttiva (metrics.per_window_mse) sul bersaglio pulito y* in eval e in train.
"""
import json, sys
import numpy as np
import torch
from pose_temponet import controls as C, heads, metrics

COEFF = 0.4
def build(fold, n_windows=4, feature="iid", seed_offset=0):
    rng = np.random.default_rng(20261122 + fold + seed_offset)
    if feature == "iid":
        x = rng.standard_normal((n_windows, 160, 384))
    else:  # AR(1) nel tempo, rho 0.9: latenti lisci
        e = rng.standard_normal((n_windows, 160, 384)); x = np.empty_like(e); x[:, 0] = e[:, 0]
        for t in range(1, 160): x[:, t] = 0.9*x[:, t-1] + np.sqrt(1-0.81)*e[:, t]
    x = x.astype(np.float16)
    meta = [{'user': f'user{i}', 'window': i} for i in range(n_windows)]
    support = np.ones((n_windows, 148, 20), dtype=bool)
    half = n_windows//2
    a = C.generate_components(x, meta, support, np.arange(half), np.arange(half, n_windows),
                              [f'user{i}' for i in range(n_windows)], outer_fold=fold)
    pupil = heads.make_head('TCN_post', 0)
    layers = [m for m in pupil.deep if isinstance(m, torch.nn.Conv1d)]
    teacher = a['generator_states']['T']; linear = a['generator_states']['L']
    wkeys = [k for k in teacher if k.endswith('.weight')]
    assert len(layers) == len(wkeys) == 6
    with torch.no_grad():
        for layer, key in zip(layers, wkeys):
            layer.weight.zero_(); layer.bias.zero_()
            w = teacher[key]; b = teacher[key[:-6]+'bias']
            assert layer.kernel_size == (w.shape[2],), (layer, w.shape)
            layer.weight[:w.shape[0], :w.shape[1]].copy_(w); layer.bias[:len(b)].copy_(b)
        lstd, lmean, tstd, tmean = (torch.tensor(a['normalization'][k][s], dtype=torch.float32)
                                    for k in ('L', 'T') for s in ('std', 'mean'))
        pupil.affine.weight.copy_(linear['weight']/lstd[:, None]); pupil.affine.bias.copy_((linear['bias']-lmean)/lstd)
        layers[-1].weight.mul_((COEFF/tstd)[:, None, None]); layers[-1].bias.copy_((layers[-1].bias-tmean)*COEFF/tstd)
    return x, a, pupil, layers

def rel(err, ref): return float(np.sqrt(np.mean(err**2)/np.mean(ref**2)))
def centered_ratio(err, ref):  # analogo a E_dyn su tutte le finestre: SS centrata nel tempo dell'errore / del bersaglio
    e = err-err.mean(axis=1, keepdims=True); r = ref-ref.mean(axis=1, keepdims=True)
    return float((e**2).sum()/(r**2).sum())
def gnorm(model):
    tot, parts = 0.0, {}
    for n, p in model.named_parameters():
        g = 0.0 if p.grad is None else float(p.grad.detach().double().pow(2).sum())
        tot += g; key = n.split('.')[0]; parts[key] = parts.get(key, 0.0)+g
    return tot**.5, {k: v**.5 for k, v in parts.items()}

out = []
for feature, n_windows in (("iid", 4), ("iid", 16), ("ar0.9", 16)):
    for fold in (0, 1):
        x, a, pupil, layers = build(fold, n_windows, feature)
        xt = torch.from_numpy(x).float()
        ystar = a['L']+COEFF*a['T']; tpart = COEFF*a['T']
        row = {"feature": feature, "windows": n_windows, "fold": fold,
               "ln_weight_is_one": bool((pupil.norm.weight == 1).all()), "ln_bias_is_zero": bool((pupil.norm.bias == 0).all()),
               "ln_eps_head": pupil.norm.eps, "ln_eps_generator": a['layernorm_eps'],
               "dropout_p": [m.p for m in pupil.deep if isinstance(m, torch.nn.Dropout)],
               "nonzero_hidden_channels": [int((l.weight.abs().sum(dim=(1, 2)) > 0).sum()) for l in layers],
               "param_norm_embedded": float(sum(p.detach().double().pow(2).sum() for p in pupil.parameters())**.5),
               "hidden_active_fraction_fitting": a['hidden_active_fraction_fitting'],
               "T_norm_std_min_max": [float(np.min(a['normalization']['T']['std'])), float(np.max(a['normalization']['T']['std']))]}
        pupil.eval()
        with torch.no_grad(): y = pupil(xt).numpy().astype(np.float64)
        e = y-ystar
        row["eval"] = {"max_abs_crop6": float(np.abs(e[:, 6:-6]).max()), "rel_rmse_crop6": rel(e[:, 6:-6], ystar[:, 6:-6]),
                       "max_abs_full160": float(np.abs(e).max()), "rel_rmse_full160": rel(e, ystar),
                       "share_T_in_ystar_power_crop6": float(np.mean(tpart[:, 6:-6]**2)/np.mean(ystar[:, 6:-6]**2))}
        # (a) dropout attivo
        pupil.train(); K = 200; torch.manual_seed(1234+fold)
        acc = np.zeros_like(ystar); sq = 0.0; cr = 0.0; mse_loss = 0.0
        yt = torch.from_numpy(ystar[:, 6:-6]).float(); sup = torch.ones(yt.shape, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(K):
                p_ = pupil(xt)
                mse_loss += float(metrics.per_window_mse(p_[:, 6:-6], yt, sup))
                yk = p_.numpy().astype(np.float64); acc += yk
                ek = (yk-ystar)[:, 6:-6]; sq += np.mean(ek**2); cr += centered_ratio(ek, ystar[:, 6:-6])
        mean_err = (acc/K-ystar)[:, 6:-6]
        row["train_dropout"] = {"masks": K,
            "rel_rmse_vs_ystar": float(np.sqrt(sq/K/np.mean(ystar[:, 6:-6]**2))),
            "rel_rmse_vs_0.4T": float(np.sqrt(sq/K/np.mean(tpart[:, 6:-6]**2))),
            "mean_centered_error_ratio_vs_ystar": cr/K,
            "expected_train_mode_clean_loss_per_window_mse": mse_loss/K,
            "bias_rel_rmse_vs_0.4T (errore dell'uscita media sulle maschere)": rel(mean_err, tpart[:, 6:-6]),
            "var_ystar_crop6": float(np.mean(ystar[:, 6:-6]**2))}
        # (b) gradienti sul bersaglio pulito
        g = {}
        for mode in ("eval", "train"):
            pupil.train(mode == "train"); torch.manual_seed(99+fold)
            reps = 1 if mode == "eval" else 50; norms = []; parts_acc = {}; losses = []
            for _ in range(reps):
                pupil.zero_grad(set_to_none=True)
                loss = metrics.per_window_mse(pupil(xt)[:, 6:-6], yt, sup); loss.backward()
                n, parts = gnorm(pupil); norms.append(n); losses.append(float(loss))
                for k, v in parts.items(): parts_acc.setdefault(k, []).append(v)
            g[mode] = {"loss_mean": float(np.mean(losses)), "grad_norm_mean": float(np.mean(norms)), "grad_norm_min": float(np.min(norms)),
                       "grad_norm_max": float(np.max(norms)), "by_block_mean": {k: float(np.mean(v)) for k, v in parts_acc.items()}, "reps": reps}
        # riferimento di scala: gradiente della stessa loss nello stato iniziale congelato della testa (uscita zero)
        init = heads.make_head('TCN_post', 0).eval(); init.zero_grad(set_to_none=True)
        l0 = metrics.per_window_mse(init(xt)[:, 6:-6], yt, sup); l0.backward(); n0, p0 = gnorm(init)
        g["frozen_init_eval"] = {"loss": float(l0), "grad_norm": n0, "by_block": p0,
                                 "param_norm_init": float(sum(p.detach().double().pow(2).sum() for p in init.parameters())**.5)}
        row["clean_loss_gradient"] = g
        out.append(row)
json.dump({"scope": "CPU, feature sintetiche FP16, nessun fit ne' update; pesi del teacher incorporati come nello script di GPT",
           "clip_norm_recipe": 1.0, "rows": out}, sys.stdout, indent=1)
