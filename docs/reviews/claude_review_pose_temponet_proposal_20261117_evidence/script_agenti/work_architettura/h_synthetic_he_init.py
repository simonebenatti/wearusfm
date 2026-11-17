"""Stessa prova sintetica di h_synthetic_trainability.py, ma con gli strati nascosti di MLP e TCN inizializzati alla He
(kaiming_normal_, fan_in, ReLU) e bias a zero; ultimo strato a zero come nel protocollo. Serve solo a misurare quanto pesa la scelta
«inizializzazione Torch standard» sulla velocita' di addestramento. Dati finti, non del progetto."""
import sys, time, json
W = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura"
sys.path.insert(0, W)
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "50"]
import torch
from torch import nn
import h_synthetic_trainability as H
from heads import MLP, TCN, D, J, last_layer


def he_(m):
    last = last_layer(m)
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)) and mod is not last:
            nn.init.kaiming_normal_(mod.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(mod.bias)
    return m


class MLPHe(MLP):
    def __init__(self):
        super().__init__(); he_(self)


class TCNHe(TCN):
    def __init__(self):
        super().__init__(); he_(self)


def main():
    out = {}
    for regime, offset_scale in (("senza componente comune", 0.0), ("componente comune dominante", 3.0)):
        gen = torch.Generator().manual_seed(20261117)
        A = torch.randn(D, H.R, generator=gen) / H.R ** 0.5
        m_vec = offset_scale * torch.randn(D, generator=gen)
        Ws = [torch.randn(J, H.R, generator=gen) for _ in range(4)]
        xtr, ytr, _ = H.make(H.NTR, gen, A, m_vec, Ws)
        xhe, yhe, parts = H.make(H.NHE, gen, A, m_vec, Ws)
        c = slice(H.CROP, H.T - H.CROP)
        mu, sd = ytr[:, c].mean(dim=(0, 1)), ytr[:, c].std(dim=(0, 1))
        ytr, yhe = (ytr - mu) / sd, (yhe - mu) / sd
        print(f"\n===== regime: {regime} (inizializzazione He, bias zero) =====", flush=True)
        for name, cls in (("TCN", TCNHe), ("MLP", MLPHe)):
            torch.manual_seed(0)
            st = H.init_stats(cls(), xtr)
            print(f"  attivazioni all'inizializzazione, {name} (seme 0, 64 finestre):")
            for i, s in enumerate(st, 1):
                print(f"    hidden {i} (larghezza {s['larghezza']}): accese {s['accese']}, unita' mai accese {s['mai_accese']}, sempre accese {s['sempre_accese']}, quota variabile fra posizioni {s['quota_variabile']}, rms {s['rms']:.4f}")
        out[regime] = {}
        for name, cls in (("MLP", MLPHe), ("TCN", TCNHe)):
            for lr in (1e-4, 1e-3):
                t0 = time.perf_counter()
                curve, _ = H.fit(cls, lr, xtr, ytr, xhe, yhe)
                best = min(range(len(curve)), key=lambda i: curve[i])
                out[regime][f"{name}@{lr}"] = curve
                print(f"  {name:4s} LR {lr:g}: MSE tenuta fuori epoca 1 = {curve[0]:.4f}, 10 = {curve[min(9, len(curve) - 1)]:.4f}, 25 = {curve[min(24, len(curve) - 1)]:.4f}, ultima ({len(curve)}) = {curve[-1]:.4f}; minimo {curve[best]:.4f} all'epoca {best + 1}  [{time.perf_counter() - t0:.0f} s]", flush=True)
    json.dump(out, open(W + "/h_synthetic_he_init.json", "w"))


if __name__ == "__main__":
    main()
