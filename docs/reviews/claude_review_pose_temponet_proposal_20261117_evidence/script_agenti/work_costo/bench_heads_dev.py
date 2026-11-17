"""Seconda misura SINTETICA: stesse teste di bench_heads.py, dispositivo a scelta (cpu oppure mps).

mps = GPU integrata dell'Apple M3 Pro: NON e' una A100 e NON e' CUDA. Serve solo a vedere
quanto pesa il costo fisso per passo quando le convoluzioni non sono il collo di bottiglia.
Uso: python bench_heads_dev.py <device> <threads> <eval_batch> <out.json> [n_windows]
"""
import json
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn

DEV = sys.argv[1]
THREADS = int(sys.argv[2])
EVAL_BATCH = int(sys.argv[3])
OUT = sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 1907
P, D, J, B, CROP = 160, 384, 20, 32, 6
torch.set_num_threads(THREADS)
dev = torch.device(DEV)
rng = np.random.default_rng(0)


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def seq(*mods):
    return nn.Sequential(*mods)


class Head(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.ln = nn.LayerNorm(D, eps=1e-5)
        if kind == "L0":
            self.net = seq(nn.Linear(D, J))
        elif kind == "MLP":
            self.net = seq(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(),
                           nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                           nn.Linear(32, J))
        elif kind == "L13":
            self.net = seq(nn.Conv1d(D, J, 13, padding=6))
        elif kind == "TCN":
            self.net = seq(nn.Conv1d(D, 64, 1), nn.ReLU(),
                           nn.Conv1d(64, 64, 3, dilation=1, padding=1), nn.ReLU(),
                           nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(),
                           nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(), nn.Dropout(0.1),
                           nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1),
                           nn.Conv1d(32, J, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x = self.ln(x)
        if self.kind in ("L13", "TCN"):
            return self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.net(x)


def optimizer(m, lr=1e-3):
    decay = [p for p in m.parameters() if p.ndim >= 2]
    rest = [p for p in m.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.999), eps=1e-8)


def window_loss(pred, y, mask):
    se = (pred - y) ** 2 * mask
    return se.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)


def epoch_train(m, opt, x16, y, mask, order):
    m.train()
    sync()
    t0 = time.perf_counter()
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        xb = torch.from_numpy(x16[idx].astype(np.float32)).to(dev)
        yb, mb = y[idx].to(dev), mask[idx].to(dev)
        loss = window_loss(m(xb), yb, mb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    sync()
    return time.perf_counter() - t0


@torch.inference_mode()
def epoch_eval(m, x16, y, mask):
    m.eval()
    sync()
    t0, total = time.perf_counter(), 0.0
    for s in range(0, len(x16), EVAL_BATCH):
        xb = torch.from_numpy(x16[s:s + EVAL_BATCH].astype(np.float32)).to(dev)
        total += float(window_loss(m(xb), y[s:s + EVAL_BATCH].to(dev), mask[s:s + EVAL_BATCH].to(dev)).sum())
    sync()
    return time.perf_counter() - t0


x16 = rng.standard_normal((N, P, D), dtype=np.float32).astype(np.float16)
y = torch.from_numpy(rng.standard_normal((N, P, J), dtype=np.float32))
mask_np = rng.random((N, P, J)) < 0.9
mask_np[:, :CROP] = False
mask_np[:, P - CROP:] = False
mask = torch.from_numpy(mask_np.astype(np.float32))

result = {"synthetic": True, "device": DEV, "torch": torch.__version__, "threads": torch.get_num_threads(),
          "n_windows": N, "batch": B, "eval_batch": EVAL_BATCH, "heads": {}}
for kind in ("L0", "MLP", "L13", "TCN"):
    torch.manual_seed(0)
    m = Head(kind).to(dev)
    opt = optimizer(m)
    gen = torch.Generator().manual_seed(0)
    times = [epoch_train(m, opt, x16, y, mask, torch.randperm(N, generator=gen).numpy()) for _ in range(4)]
    ev = [epoch_eval(m, x16, y, mask) for _ in range(3)]
    result["heads"][kind] = {"params": sum(p.numel() for p in m.parameters()), "steps_per_epoch": -(-N // B),
                             "train_epoch_seconds": times, "eval_epoch_seconds": ev}
    print(kind, result["heads"][kind]["params"], "train", [round(t, 3) for t in times], "eval", [round(t, 3) for t in ev], flush=True)
result["peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("peak RSS bytes", result["peak_rss_bytes_macos"])
with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
