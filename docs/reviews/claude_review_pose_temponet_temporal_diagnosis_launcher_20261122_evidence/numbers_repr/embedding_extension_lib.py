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

