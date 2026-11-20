"""Secondo tentativo: completa la prova sintetica interrotta (h_synthetic_trainability.py / h_synthetic_he_init.py).
Stessi dati finti, stessa ricetta (AdamW 0.9/0.999, eps 1e-8, wd 0.01 su matrici/kernel, clip 1, batch 32, LR costante, 50 epoche,
ultimo strato a zero, seme 0). Confronta inizializzazione Torch di default (quella del protocollo) con He (kaiming_normal_, bias 0)
per MLP e TCN, piu' L0 come riferimento. Dati finti, NON del progetto: misura solo la velocita' di addestramento della ricetta."""
import sys, time, json
W = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura"
sys.path.insert(0, W)
sys.argv = [sys.argv[0], "50"]
import torch
import h_synthetic_trainability as H
import h_synthetic_he_init as HE
from heads import L0, MLP, TCN

print("torch", torch.__version__, "thread", torch.get_num_threads(), "epoche", H.EPOCHS, flush=True)
out = {}
for regime, offset_scale in (("senza componente comune", 0.0), ("componente comune dominante", 3.0)):
    gen = torch.Generator().manual_seed(20261117)
    A = torch.randn(H.D, H.R, generator=gen) / H.R ** 0.5
    m_vec = offset_scale * torch.randn(H.D, generator=gen)
    Ws = [torch.randn(H.J, H.R, generator=gen) for _ in range(4)]
    xtr, ytr, _ = H.make(H.NTR, gen, A, m_vec, Ws)
    xhe, yhe, parts = H.make(H.NHE, gen, A, m_vec, Ws)
    c = slice(H.CROP, H.T - H.CROP)
    mu, sd = ytr[:, c].mean(dim=(0, 1)), ytr[:, c].std(dim=(0, 1))
    ytr, yhe = (ytr - mu) / sd, (yhe - mu) / sd
    print(f"\n===== regime: {regime} =====", flush=True)
    out[regime] = {}
    for name, cls in (("L0", L0), ("MLP-default", MLP), ("MLP-He", HE.MLPHe), ("TCN-default", TCN), ("TCN-He", HE.TCNHe)):
        for lr in (1e-4, 1e-3):
            t0 = time.perf_counter()
            curve, _ = H.fit(cls, lr, xtr, ytr, xhe, yhe)
            best = min(range(len(curve)), key=lambda i: curve[i])
            out[regime][f"{name}@{lr}"] = curve
            print(f"  {name:12s} LR {lr:g}: MSE tenuta fuori epoca 1 = {curve[0]:.4f}, 10 = {curve[9]:.4f}, 25 = {curve[24]:.4f}, 50 = {curve[-1]:.4f}; minimo {curve[best]:.4f} all'epoca {best + 1}  [{time.perf_counter() - t0:.0f} s]", flush=True)
json.dump(out, open(W + "/j_synthetic_init_completion.json", "w"))
print("FINE", flush=True)
