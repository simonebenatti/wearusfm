"""Controllo ESEGUITO (dati sintetici): riduzione della loss della PoseHead nel pretraining (jepa.pose_terms, per_window=True)
contro l'obiettivo della ridge storica (pose_e3_b.sufficient / prepare_solver / score). Codice dello snapshot c15aac9, non modificato."""
import sys
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9"
sys.path.insert(0, SNAP + "/src")
sys.path.insert(0, SNAP + "/scripts")
import numpy as np
import torch
from wearusfm.training.jepa import pose_terms
import pose_e3_b as E

rng = np.random.default_rng(0)
S, P, J, D = 6, 30, 4, 8
pred = torch.tensor(rng.normal(size=(S, P, J)), dtype=torch.float32)
target = torch.tensor(rng.normal(size=(S, P, J)), dtype=torch.float32)
mask = torch.zeros(S, P, J, dtype=torch.bool)
valid_patches = [30, 30, 3, 12, 0, 25]  # finestre con supporti diversi; una senza elementi validi
for s, n in enumerate(valid_patches):
    mask[s, :n] = True
time_valid = torch.ones(S, P, dtype=torch.bool)

loss_w, num_w, den_w, _ = pose_terms(pred, (target, mask), time_valid, per_window=True)
loss_e, num_e, den_e, _ = pose_terms(pred, (target, mask), time_valid, per_window=False)
err = ((pred - target) ** 2 * mask).numpy().astype(np.float64)
per_win = [err[s].sum() / mask[s].sum().item() for s in range(S) if mask[s].any()]
print("pose_terms per_window=True :", float(loss_w), "| media delle medie per finestra:", float(np.mean(per_win)), "| finestre contate:", int(den_w))
print("pose_terms per_window=False:", float(loss_e), "| SSE / elementi validi      :", err.sum() / mask.sum().item(), "| elementi contati:", int(den_e))
w_elem = np.array([mask[s].sum().item() for s in range(S)], dtype=float)
print("peso relativo di ciascuna finestra, per elemento:", np.round(w_elem / w_elem.sum(), 4).tolist())
print("peso relativo di ciascuna finestra, per finestra:", np.round((w_elem > 0) / (w_elem > 0).sum(), 4).tolist())

# ridge storica: minimo di SSE aggregata sugli elementi validi + alpha ||w||^2 (intercetta libera), per giunto
x = rng.normal(size=(S * P, D))
y = x @ rng.normal(size=(D, J)) + 0.5 * rng.normal(size=(S * P, J))
m = mask.numpy().reshape(S * P, J).copy()
m[:, 1] &= rng.random(S * P) > 0.3  # un giunto con maschera diversa
stats = E.merge_stats([E.sufficient(x, y, m)], J)
alpha = 3.0
w, b = E.prepare_solver(stats, D, J)(alpha)
grad_w = np.zeros((D, J)); grad_b = np.zeros(J)
for j in range(J):
    sel = m[:, j]
    r = y[sel, j] - x[sel] @ w[:, j] - b[j]
    grad_w[:, j] = -2 * x[sel].T @ r + 2 * alpha * w[:, j]   # gradiente di SSE aggregata + alpha ||w||^2
    grad_b[j] = -2 * r.sum()
print("gradiente dell'obiettivo «SSE aggregata sugli elementi validi + alpha||w||^2» nella soluzione del solver: max |grad_w| =",
      float(np.abs(grad_w).max()), ", max |grad_b| =", float(np.abs(grad_b).max()))
# lo stesso punto NON e' stazionario per l'obiettivo pesato per finestra
win = np.repeat(np.arange(S), P)
g2 = np.zeros((D, J))
for j in range(J):
    for s in range(S):
        sel = m[:, j] & (win == s)
        if sel.any():
            r = y[sel, j] - x[sel] @ w[:, j] - b[j]
            g2[:, j] += -2 * x[sel].T @ r / sel.sum()
print("gradiente (solo termine dati) dell'obiettivo «media per finestra» nello stesso punto: max |grad_w| =", float(np.abs(g2).max()))
sse, count = E.score(stats, w, b)
direct = sum(((y[m[:, j], j] - x[m[:, j]] @ w[:, j] - b[j]) ** 2).sum() for j in range(J))
print("E.score: SSE", sse, "elementi", count, "| SSE diretta sugli elementi validi", float(direct), "| elementi validi", int(m.sum()))
