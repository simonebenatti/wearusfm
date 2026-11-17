"""Segnale dipendente dall'ingresso negli strati nascosti all'inizializzazione Torch standard (PROT:35), per MLP e TCN della proposta.
Ingresso: LayerNorm affine (gamma=1, beta=0) di feature gaussiane correlate nel tempo (AR(1) per canale, rho=0.9) - dopo la LN ogni
vettore ha media 0 e varianza 1 sui 384 canali, come le cache reali dopo la LN. Si misura, per ogni strato dopo la ReLU:
 - var_input: varianza fra finestre e istanti di ogni unita' (parte che dipende dall'ingresso), media sulle unita';
 - mean_sq: quadrato della media di ogni unita' (parte costante, dominata dai bias), media sulle unita';
 - var_time: varianza nel tempo dentro la finestra, media su finestre e unita' (la parte che serve alla dinamica).
Nessun dato del progetto, nessun fit. Seme 0 come la proposta; si ripete con semi 1 e 2."""
import sys
import numpy as np
import torch
from torch import nn

sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_selezione")
D, T, J, B = 384, 160, 20, 64
torch.set_num_threads(1)

def mlp():
    return nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(),
                         nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())

def tcn():
    return nn.Sequential(nn.Conv1d(D, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1, dilation=1), nn.ReLU(),
                         nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=3, dilation=3), nn.ReLU(),
                         nn.Conv1d(64, 32, 1), nn.ReLU())

rng = np.random.default_rng(123)
x = np.empty((B, T, D))
x[:, 0] = rng.normal(0, 1, (B, D))
for t in range(1, T):
    x[:, t] = 0.9 * x[:, t - 1] + np.sqrt(1 - 0.81) * rng.normal(0, 1, (B, D))
x = torch.nn.functional.layer_norm(torch.from_numpy(x).float(), (D,))
print("ingresso dopo LN: var fra finestre/istanti per canale %.3f, var nel tempo %.3f" % (x.reshape(-1, D).var(0).mean(), x.var(1).mean()))
for seed in (0, 1, 2):
    for name, make in (("MLP", mlp), ("TCN", tcn)):
        torch.manual_seed(seed)
        net = make()
        h = x if name == "MLP" else x.transpose(1, 2)
        rows = []
        with torch.no_grad():
            for layer in net:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    a = h if name == "MLP" else h.transpose(1, 2)  # (B,T,C)
                    flat = a.reshape(-1, a.shape[-1])
                    var_in = float(flat.var(0).mean()); msq = float((flat.mean(0) ** 2).mean()); vt = float(a.var(1).mean())
                    dead = float((flat.max(0).values <= 0).float().mean())
                    rows.append((a.shape[-1], var_in, msq, vt, dead))
        print(f"seme {seed} {name}: " + " | ".join(f"C{c}: var_in {v:.2e} mean^2 {m:.2e} rapporto {v/m if m>0 else float('inf'):.3f} var_tempo {vt:.2e} morte {d:.2f}" for c, v, m, vt, d in rows))
