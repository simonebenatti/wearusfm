"""SIMULATION: anti-correlated per-rank gradients (sum much smaller than each addend), fp32, two ring orders."""
import torch
torch.manual_seed(0)
N = 200_000
def ring(x, start):
    acc = x[start].clone()
    for k in range(1, 4): acc = acc + x[(start + k) % 4]
    return acc
for K in [1, 3, 10, 20, 30, 100, 1000]:
    v = torch.randn(N); s = torch.tensor([1., -1., 1., -1.]).unsqueeze(1)
    g = ((s * v * K + torch.randn(4, N)) / 4).float()
    a, b = ring(g, 0), ring(g, 1)
    x, y = a.double(), b.double()
    rel = float((x - y).norm() / max(x.norm(), y.norm()))
    canc = float(g.double().abs().sum(0).norm() / x.norm())
    print(f"K={K:5d} sum|g|/|sum|={canc:8.1f} relative_l2={rel:.3e} {'EXCEEDS 1e-6' if rel>1e-6 else ''}")
