# Metodo diverso dai revisori: propagazione simbolica degli insiemi di dipendenza (nessun peso), poi controprova con autograd.
import sys
import numpy as np
WT = "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/src"
sys.path.insert(0, WT)
import torch
from wearusfm.model import temponet as TN

def pad(deps, left, right):
    return [set() for _ in range(left)] + deps + [set() for _ in range(right)]
def conv(deps, k, d=1, s=1, p=0):
    L = len(deps); x = [set() for _ in range(p)] + deps + [set() for _ in range(p)]
    Lp = len(x); out = []
    o = 0
    while o * s + (k - 1) * d < Lp:
        st = set()
        for j in range(k):
            st |= x[o * s + j * d]
        out.append(st); o += 1
    return out
def pool(deps, k=2):
    return [deps[2 * o] | deps[2 * o + 1] for o in range(len(deps) // 2)]
def temponet_deps(T, causal, dil=(2, 4, 8), strides=(1, 2, 4)):
    deps = [{t} for t in range(T)]
    for d, s in zip(dil, strides):
        for _ in range(2):
            tot = 2 * d
            l, r = (tot, 0) if causal else (tot // 2, tot - tot // 2)
            deps = conv(pad(deps, l, r), 3, d)
        deps = conv(deps, 5, 1, s, 2)
        deps = pool(deps)
    return deps

print("output_length simbolico vs TN.output_length:")
for T in (37, 38, 101, 102, 160, 256):
    print("  T=%d  simbolico=%d  TN=%d" % (T, len(temponet_deps(T, True)), TN.output_length(T)))
print("receptive_field() =", TN.receptive_field())
# elemento lontano dai bordi, T grande
T = 4096
for causal in (True, False):
    deps = temponet_deps(T, causal)
    mid = len(deps) // 2
    a, b = deps[mid], deps[mid + 1]
    def contig(s): return max(s) - min(s) + 1 == len(s)
    print("T=4096 causal=%s: elemento %d: %d passi contigui=%s [%d..%d]; successivo spostato di %d; unione %d" % (
        causal, mid, len(a), contig(a), min(a), max(a), min(b) - min(a), len(a | b)))
for causal in (True, False):
    deps = temponet_deps(160, causal)
    u = set().union(*deps)
    print("T=160 causal=%s: %d elementi; dip. elemento0 [%d..%d] (%d), elemento1 [%d..%d] (%d); uscita: %d passi [%d..%d]" % (
        causal, len(deps), min(deps[0]), max(deps[0]), len(deps[0]), min(deps[1]), max(deps[1]), len(deps[1]), len(u), min(u), max(u)))

# controprova con autograd (eval: BN fissa), 5 semi, ingressi casuali, 384 canali
for causal in (True, False):
    hits = np.zeros(160, dtype=int)
    for seed in range(5):
        torch.manual_seed(seed)
        m = TN.TEMPONet(in_channels=384, window=160, out_dim=20, causal=causal).eval()
        x = torch.randn(1, 384, 160, requires_grad=True)
        y = m(x)
        g = torch.autograd.grad(y.abs().sum(), x)[0][0].abs().sum(0)
        hits += (g > 0).numpy()
    nz = np.nonzero(hits)[0]
    print("autograd causal=%s: passi con gradiente non nullo in almeno un seme: %d, primo %d, ultimo %d; in tutti e 5: %d" % (
        causal, len(nz), nz.min(), nz.max(), int((hits == 5).sum())))
torch.manual_seed(0)
m = TN.TEMPONet(in_channels=384, window=160, out_dim=20)
print("parametri TEMPONet(384,160,20):", sum(p.numel() for p in m.parameters()), "out_length", m.out_length)
for shp in ((2, 384, 159), (2, 160, 384), (2, 384, 161)):
    try:
        m(torch.randn(*shp)); print(shp, "accettato")
    except ValueError as e:
        print(shp, "rifiutato:", str(e)[:50])
