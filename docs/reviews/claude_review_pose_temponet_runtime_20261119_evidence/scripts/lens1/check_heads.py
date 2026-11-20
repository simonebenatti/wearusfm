"""Lens 1 independent checks: architecture, init, RF/crop, decay groups, loss."""
import math
import numpy as np
import torch
from torch import nn

from pose_temponet import heads, metrics, data as D, train as T

torch.set_num_threads(1)
print("== parameter counts / weights / biases / MACs ==")
for name in heads.HEADS:
    m = heads.make_head(name)
    total = sum(p.numel() for p in m.parameters())
    weights = sum(p.numel() for n, p in m.named_parameters() if p.ndim >= 2)
    nonln_bias = sum(p.numel() for n, p in m.named_parameters() if p.ndim == 1 and not n.startswith("norm."))
    print(name, total, "spec", heads.PARAMETERS[name], "matrix weights", weights, "biases(non-LN)", nonln_bias,
          "MAC/160", weights * 160)

print("== layer sequence ==")
for name in ("MLP", "TCN_post"):
    m = heads.make_head(name)
    print(name, [type(l).__name__ + (str(tuple(l.weight.shape)) if hasattr(l, "weight") else "")
                 + (f"d{l.dilation[0]}p{l.padding[0]}" if isinstance(l, nn.Conv1d) else "")
                 + (f"p{l.p}" if isinstance(l, nn.Dropout) else "") for l in m.deep])

print("== initialization ==")
for name in heads.HEADS:
    m = heads.make_head(name, seed=1)
    print(name, "LN gamma all 1:", bool(torch.equal(m.norm.weight, torch.ones(384))),
          "LN beta all 0:", bool(torch.equal(m.norm.bias, torch.zeros(384))), "eps", m.norm.eps,
          "affine LN:", m.norm.elementwise_affine)
    if name in ("L0", "L13"):
        print("   out zero:", bool(m.out.weight.abs().sum() == 0 and m.out.bias.abs().sum() == 0))
        continue
    tr = [l for l in m.deep if isinstance(l, (nn.Linear, nn.Conv1d))]
    for l in tr[:-1]:
        fan_in = l.weight[0].numel()
        print("   hidden", tuple(l.weight.shape), "std/expected %.4f" % (l.weight.std().item() / math.sqrt(2 / fan_in)),
              "mean %.4f" % l.weight.mean().item(), "bias zero", bool(l.bias.abs().sum() == 0))
    print("   last g zero:", bool(tr[-1].weight.abs().sum() == 0 and tr[-1].bias.abs().sum() == 0),
          "affine zero:", bool(m.affine.weight.abs().sum() == 0 and m.affine.bias.abs().sum() == 0))

print("== same init across TCN_post/TCN_pre and seeds ==")
print("TCN_post s0 == TCN_pre s0:", heads.state_sha256(heads.make_head("TCN_post", 0)) == heads.state_sha256(heads.make_head("TCN_pre", 0)))
print("MLP s0 != s1:", heads.state_sha256(heads.make_head("MLP", 0)) != heads.state_sha256(heads.make_head("MLP", 1)))
print("seeds f0 s0", heads.trajectory_seeds(0, 0), "f1 s0", heads.trajectory_seeds(0, 1), "refit s2", heads.trajectory_seeds(2, 2))

print("== receptive field and crop6 independence from zero padding (T=160) ==")
g = torch.Generator().manual_seed(7)
for name in ("L13", "TCN_post", "TCN_pre", "L0", "MLP"):
    m = heads.make_head(name).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 1:
                p.add_(torch.randn(p.shape, generator=g) * 0.05)
    x = torch.randn(1, 160, 384, generator=g)
    # embed inside longer real context: if crop6 outputs never touch padding they are identical
    left, right = torch.randn(1, 10, 384, generator=g), torch.randn(1, 10, 384, generator=g)
    big = torch.cat([left, x, right], 1)
    with torch.no_grad():
        a = m(x)[:, 6:154]
        b = m(big)[:, 16:164]
        edge_a = m(x)[:, 5]
        edge_b = m(big)[:, 15]
    rf = set()
    xx = x.clone().requires_grad_(True)
    out = m(xx)[0, 6, 0]
    grad = torch.autograd.grad(out, xx)[0].abs().sum(-1)[0]
    rf6 = torch.nonzero(grad > 0).flatten().tolist()
    out = m(xx)[0, 153, 0]
    grad = torch.autograd.grad(out, xx)[0].abs().sum(-1)[0]
    rf153 = torch.nonzero(grad > 0).flatten().tolist()
    print(name, "crop6 outputs: %d" % a.shape[1], "max|diff| crop6 vs embedded %.3e" % (a - b).abs().max().item(),
          "t=5 differs (padding reached):", bool((edge_a - edge_b).abs().max() > 0),
          "RF(t=6)=[%d..%d] n=%d" % (rf6[0], rf6[-1], len(rf6)), "RF(t=153)=[%d..%d]" % (rf153[0], rf153[-1]))

print("== weight-decay groups ==")
for name in heads.HEADS:
    m = heads.make_head(name)
    groups = heads.optimizer_groups(m)
    ids = {id(p): n for n, p in m.named_parameters()}
    print(name, "decay:", sorted(ids[id(p)] for p in groups[0]["params"]), "wd", groups[0]["weight_decay"])
    print("   no-decay:", sorted(ids[id(p)] for p in groups[1]["params"]), "wd", groups[1]["weight_decay"])

print("== central support rule ==")
mask = np.ones((3, 160, 20), dtype=bool)
lengths = np.array([160, 150, 13])
mask[1, 150:] = False
mask[2, 13:] = False
kv = np.arange(160)[None] < lengths[:, None]
cs = D.central_support(mask, lengths, kv)
for i, n in enumerate(lengths):
    t = np.flatnonzero(cs[i, :, 0]) + 6
    print("N=%d" % n, "centers", (t.min(), t.max()) if len(t) else None, "count", len(t), "expected max", n - 7)

print("== per-window MSE vs manual (partial masks) ==")
rng = np.random.default_rng(0)
p = torch.tensor(rng.normal(size=(4, 148, 20)), dtype=torch.float32)
y = torch.tensor(rng.normal(size=(4, 148, 20)), dtype=torch.float32)
s = torch.tensor(rng.random((4, 148, 20)) < 0.6)
s[2] = False
y[~s] = float("nan")
loss = metrics.per_window_mse(p, y, s)
manual = []
for i in range(4):
    if s[i].sum() == 0:
        continue
    manual.append(((p[i][s[i]] - y[i][s[i]]) ** 2).sum().item() / s[i].sum().item())
print("loss", float(loss), "manual", float(np.mean(manual)), "n windows", len(manual))
pp = p.clone().requires_grad_(True)
metrics.per_window_mse(pp, y, s).backward()
print("grad finite with NaN targets outside support:", bool(torch.isfinite(pp.grad).all()),
      "grad zero outside support:", bool((pp.grad[~s] == 0).all()))
