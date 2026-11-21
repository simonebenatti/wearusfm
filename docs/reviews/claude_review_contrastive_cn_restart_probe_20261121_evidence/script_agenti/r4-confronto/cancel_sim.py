"""SIMULATION: fp32 allreduce of 4 pre-divided addends in two ring orders; relative_l2 as in compare_vector."""
import torch
torch.manual_seed(0)
N = 200_000
def ring(x, start):  # sequential fp32 accumulation starting at rank `start` (NCCL ring chunk owner rotation)
    order = [(start + k) % 4 for k in range(4)]
    acc = x[order[0]].clone()
    for r in order[1:]: acc = acc + x[r]
    return acc
for ratio in [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
    mu = torch.randn(N)
    g = (mu + ratio * torch.randn(4, N)).float() / 4  # per-rank local grads, noise/mean = ratio, then div_(4) (exact)
    a, b = ring(g, 0), ring(g, 1)  # e.g. different chunk boundaries -> different starting rank for the element
    x, y = a.double(), b.double()
    rel = float((x - y).norm() / max(x.norm(), y.norm()))
    canc = float(g.double().abs().sum(0).norm() / x.norm())
    frac = float((a != b).float().mean())
    print(f"noise/mean={ratio:7.1f}  sum|g|/|sum| (L2)={canc:9.2f}  diff elems={frac:.3f}  relative_l2={rel:.3e}  {'>1e-6' if rel>1e-6 else ''}")
