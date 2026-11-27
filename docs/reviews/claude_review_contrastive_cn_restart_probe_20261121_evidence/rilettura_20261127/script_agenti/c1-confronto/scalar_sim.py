"""fp32 simulation: 4-addend DDP mean (div by 4, SUM) in different orders, ratio vs new scalar denominator."""
import itertools, math
import numpy as np
import torch
import contrastive_cn_restart_compare as c

rng = np.random.default_rng(1)
f = np.float32
u = 2.0 ** -24
print("bound per ordering gamma_3*sum|x| ~ 3u; two orderings differ <= 6u =", 6 * u)

orders = list(itertools.permutations(range(4)))
def sums(x):
    out = set()
    for p in orders:
        a = x[p[0]]
        for i in p[1:]:
            a = f(a + x[i])  # sequential (ring-like)
        out.add(float(a))
    for p in orders:  # pairwise (tree-like)
        out.add(float(f(f(x[p[0]] + x[p[1]]) + f(x[p[2]] + x[p[3]]))))
    return sorted(out)

worst_new = worst_old = 0.
exceed_old = exceed_new = 0
N = 200000
for t in range(N):
    scale = 10 ** rng.uniform(-8, 0)
    g = rng.standard_normal(4) * scale
    if t % 2:  # force near cancellation
        g[3] = -(g[0] + g[1] + g[2]) * (1 + rng.uniform(-1e-5, 1e-5))
    g = g.astype(f)
    x = (g / f(4)).astype(f)
    assert np.all(x * f(4) == g)  # division by 4 exact (no subnormals here)
    s = sums(x)
    d = s[-1] - s[0]
    mean_abs = float(np.mean(np.abs(g.astype(np.float64))))
    old = d / max(abs(s[-1]), abs(s[0]), 1e-12)
    new = d / max(abs(s[-1]), abs(s[0]), 1e-12, mean_abs)
    worst_new, worst_old = max(worst_new, new), max(worst_old, old)
    exceed_old += old > 1e-6
    exceed_new += new > 1e-6
print(f"N={N}: worst old ratio {worst_old:.3e}, worst new ratio {worst_new:.3e}; "
      f"old>1e-6: {exceed_old}, new>1e-6: {exceed_new}")

# Through the real code path: compare_vector + screen_reduced_gradients with scales from the four locals
g = np.array([0.3, -0.2999, 0.12, -0.1201], dtype=f)
x = (g / f(4)).astype(f)
s = sums(x)
ordred, warmred = torch.tensor([s[0]], dtype=torch.float32), torch.tensor([s[-1]], dtype=torch.float32)
scale = sum(float(torch.linalg.vector_norm(torch.tensor([v], dtype=torch.float32), dtype=torch.float64)) / 4 for v in g)
for pid, name in c.SCALAR_ADDEND_PARAMETERS.items():
    row = {"id": pid, "name": name, "reduced": c.compare_vector(ordred, warmred)}
    exc = c.screen_reduced_gradients([row], {253: scale, 257: scale})
    print(f"rounding-only {name}: raw {row['reduced']['relative_l2']:.3e} screen {row['reduced']['screen_relative_l2']:.3e} excessive={exc}")
    # true excess of 1e-3 relative to scale
    bumped = torch.tensor([s[0] + 1e-3 * scale], dtype=torch.float32)
    row = {"id": pid, "name": name, "reduced": c.compare_vector(ordred, bumped)}
    exc = c.screen_reduced_gradients([row], {253: scale, 257: scale})
    print(f"true 1e-3 excess {name}: screen {row['reduced']['screen_relative_l2']:.3e} excessive={exc}")
    bumped = torch.tensor([s[0] + 5e-6 * scale], dtype=torch.float32)
    row = {"id": pid, "name": name, "reduced": c.compare_vector(ordred, bumped)}
    exc = c.screen_reduced_gradients([row], {253: scale, 257: scale})
    print(f"excess 5e-6 {name}: screen {row['reduced']['screen_relative_l2']:.3e} excessive={exc}")
# 268 others unchanged: screen denominator equals the original
row = {"id": 252, "name": "anchor_heads.rms.weight", "reduced": c.compare_vector(torch.randn(384), torch.randn(384))}
c.screen_reduced_gradients([row], {253: 1e9, 257: 1e9})
print("other tensor: screen == raw:", row["reduced"]["screen_relative_l2"] == row["reduced"]["relative_l2"])
