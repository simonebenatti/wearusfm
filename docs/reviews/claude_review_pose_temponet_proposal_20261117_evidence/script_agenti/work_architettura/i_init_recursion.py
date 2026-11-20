"""Conto a mano della propagazione del segnale all'inizializzazione (PyTorch di default) nelle teste MLP e TCN, e confronto con la misura.
Default di nn.Linear / nn.Conv1d: kaiming_uniform_(a=sqrt(5)) -> pesi U(-1/sqrt(fan_in), 1/sqrt(fan_in)), varianza 1/(3 fan_in); bias uguale.
Pre-attivazione: Var(z) = q/3 + 1/(3 fan_in), con q = media dei quadrati dell'ingresso; dopo la ReLU (z simmetrica): q' = Var(z)/2."""
import sys, math
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura")
import torch
from torch import nn
from heads import MLP, TCN, D, P

# verifica del default di torch
torch.manual_seed(0)
lin = nn.Linear(384, 64); conv = nn.Conv1d(64, 64, 3)
print("nn.Linear(384,64): max|w| = %.5f (1/sqrt(384) = %.5f), var(w)*3*fan_in = %.3f; max|b| = %.5f" % (float(lin.weight.abs().max()), 1 / math.sqrt(384), float(lin.weight.var()) * 3 * 384, float(lin.bias.abs().max())))
print("nn.Conv1d(64,64,3): max|w| = %.5f (1/sqrt(192) = %.5f), var(w)*3*fan_in = %.3f; max|b| = %.5f" % (float(conv.weight.abs().max()), 1 / math.sqrt(192), float(conv.weight.var()) * 3 * 192, float(conv.bias.abs().max())))

def recursion(fan_ins):
    q, rows = 1.0, []
    for f in fan_ins:
        sig, bias = q / 3, 1 / (3 * f)
        q = (sig + bias) / 2
        rows.append((f, math.sqrt(q), bias / (sig + bias)))
    return rows

for name, fans in (("TCN", (384, 192, 192, 192, 64)), ("MLP", (384, 96, 96, 96, 64))):
    print(f"\n{name}: previsione del conto (fan_in, rms dopo la ReLU, quota della varianza della pre-attivazione dovuta al solo bias)")
    for i, (f, rms, share) in enumerate(recursion(fans), 1):
        print(f"  hidden {i}: fan_in {f:3d}, rms previsto {rms:.4f}, quota del bias {share:.2f}")

# misura su ingresso gaussiano bianco (dopo la LN ogni patch ha media 0 e varianza 1 fra le 384 feature, qualunque sia il dato)
print("\nmisura (media su 20 semi, ingresso gaussiano 8x160x384, eval):")
for name, cls in (("TCN", TCN), ("MLP", MLP)):
    stats = None
    for seed in range(20):
        torch.manual_seed(seed)
        m = cls().eval()
        acts = []
        hooks = [mod.register_forward_hook(lambda mod, i, o: acts.append(o)) for mod in m.modules() if isinstance(mod, nn.ReLU)]
        with torch.no_grad():
            m(torch.randn(8, P, D))
        for h in hooks:
            h.remove()
        row = []
        for a in acts:
            a = a.transpose(1, 2) if a.shape[1] != P else a  # -> (B,T,C)
            on = (a > 0).float().mean(dim=(0, 1))
            row.append((float(a.pow(2).mean().sqrt()), float((on == 0).float().mean()), float((on == 1).float().mean())))
        stats = row if stats is None else [tuple(x + y for x, y in zip(r, s)) for r, s in zip(row, stats)]
    for i, r in enumerate(stats, 1):
        print(f"  {name} hidden {i}: rms {r[0] / 20:.4f}, unita' mai accese {r[1] / 20:.3f}, sempre accese {r[2] / 20:.3f}")

# la LN per patch toglie media e scala della patch: x e 3x danno la stessa uscita
ln = nn.LayerNorm(D, eps=1e-5)
x = torch.randn(2, 5, D)
print("\nLN per patch: max |LN(3x) - LN(x)| = %.2e ; max |LN(x + 2) - LN(x)| = %.2e ; media dei quadrati per patch dopo la LN: min %.6f max %.6f" % (
    float((ln(3 * x) - ln(x)).abs().max()), float((ln(x + 2) - ln(x)).abs().max()), float(ln(x).pow(2).mean(-1).min()), float(ln(x).pow(2).mean(-1).max())))
