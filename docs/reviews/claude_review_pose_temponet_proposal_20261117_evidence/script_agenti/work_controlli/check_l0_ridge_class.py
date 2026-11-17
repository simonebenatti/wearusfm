"""Controllo ESEGUITO (seconda stesura): la PoseHead (LN affine -> Linear) dello snapshot c15aac9 e la ridge storica
(LN senza parametri, eps 1e-5 -> mappa affine; scripts/pose_temporal_readout.py:normalized_inputs) hanno la stessa classe di funzioni:
ogni PoseHead si riscrive esattamente come LN senza parametri seguita da una mappa affine, e viceversa (gamma=1, beta=0).
Quindi L0 rifittata e ridge storica differiscono solo per la ricetta di fit (obiettivo, regolarizzazione, ottimizzatore)."""
import sys
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9"
sys.path.insert(0, SNAP + "/src")
sys.path.insert(0, SNAP + "/scripts")
import numpy as np
import torch
from wearusfm.model.pose_head import PoseHead
import pose_temporal_readout as T

torch.manual_seed(0)
h = PoseHead(384).double()
with torch.no_grad():
    for p in h.parameters():
        p.normal_(std=0.7)
x = np.random.default_rng(0).normal(size=(3, 160, 384)).astype(np.float32) * 3 + 1
lengths = np.array([160, 160, 160])
with torch.no_grad():
    y_head = h(torch.from_numpy(x).double()).numpy()
xn = T.normalized_inputs(x, lengths).astype(np.float64)          # LN senza parametri della ridge (FP32, eps 1e-5)
g, b_ln = h.norm.weight.detach().numpy(), h.norm.bias.detach().numpy()
W, b = h.out.weight.detach().numpy(), h.out.bias.detach().numpy()
W_eq = (W * g[None, :]).T                                          # (384, 20)
b_eq = W @ b_ln + b
y_ridge_form = xn @ W_eq + b_eq
print("eps della LN: PoseHead", h.norm.eps, "| ridge", T.EPS)
print("PoseHead contro LN senza parametri + mappa affine equivalente: scarto massimo", float(np.abs(y_head - y_ridge_form).max()),
      "| ampiezza tipica", float(np.abs(y_head).mean()))
