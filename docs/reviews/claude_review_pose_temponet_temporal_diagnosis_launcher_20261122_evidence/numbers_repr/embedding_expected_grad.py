"""Norma del gradiente MEDIO sulle maschere di dropout (gradiente atteso della loss pulita in modalita' train)
nei pesi del teacher incorporato. Feature sintetiche, nessun update."""
import json, sys, importlib.util
import numpy as np, torch
from pose_temponet import heads, metrics
spec = importlib.util.spec_from_file_location("ext", "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/numbers_repr/embedding_extension_lib.py")
ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
rows = []
for fold in (0, 1):
    x, a, pupil, layers = ext.build(fold, 8, "iid")
    xt = torch.from_numpy(x).float(); ystar = a['L']+0.4*a['T']
    yt = torch.from_numpy(ystar[:, 6:-6]).float(); sup = torch.ones(yt.shape, dtype=torch.bool)
    params = [p for p in pupil.parameters()]
    pupil.train(); torch.manual_seed(7+fold); K = 64
    acc = [torch.zeros_like(p, dtype=torch.float64) for p in params]; norms = []
    for _ in range(K):
        pupil.zero_grad(set_to_none=True)
        metrics.per_window_mse(pupil(xt)[:, 6:-6], yt, sup).backward()
        norms.append(float(sum(p.grad.double().pow(2).sum() for p in params))**.5)
        for s, p in zip(acc, params): s += p.grad.double()
    mean_norm = float(sum((s/K).pow(2).sum() for s in acc))**.5
    # quota del gradiente atteso sui canali "spenti" (pesi a zero): la ricetta li riaccende?
    named = dict(zip([n for n, _ in pupil.named_parameters()], [s/K for s in acc]))
    w1 = named["deep.0.weight"]   # (64,384,1)
    rows.append({"fold": fold, "windows": 8, "masks": K, "single_mask_grad_norm_mean": float(np.mean(norms)),
                 "expected_grad_norm": mean_norm, "noise_floor_if_zero_mean": float(np.mean(norms))/K**.5,
                 "expected_grad_first_layer_active8": float(w1[:8].pow(2).sum())**.5,
                 "expected_grad_first_layer_inactive56": float(w1[8:].pow(2).sum())**.5,
                 "expected_grad_output_layer": float(named["deep.9.weight"].pow(2).sum())**.5 if "deep.9.weight" in named else None,
                 "param_names": [n for n in named][:20]})
json.dump(rows, sys.stdout, indent=1)
