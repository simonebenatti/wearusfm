import sys, math
import numpy as np
import torch
from torch import nn
print("torch", torch.__version__)
# (a) Linear e Conv1d k1 dallo stesso seme; LayerNorm e generatore
torch.manual_seed(5); a = nn.Linear(384, 64)
torch.manual_seed(5); b = nn.Conv1d(384, 64, 1)
print("Linear vs Conv1d(k1) stesso seme: pesi uguali", torch.equal(a.weight, b.weight[:, :, 0]), "bias uguali", torch.equal(a.bias, b.bias))
torch.manual_seed(5); r1 = torch.rand(1).item()
torch.manual_seed(5); nn.LayerNorm(384); r2 = torch.rand(1).item()
print("LayerNorm non consuma il generatore:", r1 == r2)
# (b) default di init: limite e varianza
torch.manual_seed(0); l = nn.Linear(384, 64)
print("max|w| %.5f (1/sqrt(384)=%.5f), var*3*fan_in=%.3f, max|b| %.5f" % (l.weight.abs().max(), 1 / math.sqrt(384), l.weight.var().item() * 3 * 384, l.bias.abs().max()))
# (c) ricorsione dei momenti: Var(z) = q * fan_in * Var(w) + Var(b) = q/3 + 1/(3 fan_in); q' = Var(z)/2
def rec(fans):
    q = 1.0; out = []
    for f in fans:
        vz = q / 3 + 1 / (3 * f); out.append((f, vz, (1 / (3 * f)) / vz)); q = vz / 2
    return out
print("TCN (fan_in per strato 384,192,192,192,64):")
for f, vz, fr in rec([384, 192, 192, 192, 64]): print("   fan_in %3d Var(z)=%.6f quota bias=%.3f" % (f, vz, fr))
print("MLP (fan_in 384,96,96,96,64):")
for f, vz, fr in rec([384, 96, 96, 96, 64]): print("   fan_in %3d Var(z)=%.6f quota bias=%.3f" % (f, vz, fr))
# (d) Monte Carlo: unita' del quinto hidden mai/sempre accese, due distribuzioni d'ingresso
def tcn_body():
    return nn.Sequential(nn.Conv1d(384, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                         nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(),
                         nn.Conv1d(64, 32, 1), nn.ReLU())
def mlp_body():
    w = [384, 96, 96, 96, 64, 32]; L = []
    for i in range(5): L += [nn.Conv1d(w[i], w[i + 1], 1), nn.ReLU()]
    return nn.Sequential(*L)
def inputs(kind, g):
    if kind == "iid":
        x = torch.randn(32, 160, 384, generator=g)
    else:  # componente comune forte + struttura a basso rango lenta nel tempo (piu' simile a latenti)
        base = torch.randn(1, 1, 384, generator=g) * 3.0
        f = torch.randn(32, 160, 8, generator=g).cumsum(1) / 10.0
        A = torch.randn(8, 384, generator=g)
        x = base + f @ A + 0.3 * torch.randn(32, 160, 384, generator=g)
    return nn.functional.layer_norm(x, (384,)).transpose(1, 2)
for kind in ("iid", "comune+basso_rango"):
    for name, mk in (("TCN", tcn_body), ("MLP", mlp_body)):
        never = always = tot = 0
        for seed in range(20):
            torch.manual_seed(seed); m = mk().eval(); g = torch.Generator().manual_seed(1000 + seed)
            with torch.no_grad():
                h = m[:-1](inputs(kind, g))   # pre-attivazione del quinto hidden
            on = (h > 0).reshape(32, -1)  # (unita', ...) -> riordino
            on = (h > 0).permute(1, 0, 2).reshape(32, -1)
            never += int((~on.any(1)).sum()); always += int(on.all(1).sum()); tot += 32
        print("  %-20s %s: quinto hidden mai acceso %.1f%%, sempre acceso %.1f%%" % (kind, name, 100 * never / tot, 100 * always / tot))
