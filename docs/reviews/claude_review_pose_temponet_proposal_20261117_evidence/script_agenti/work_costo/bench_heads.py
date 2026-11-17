"""Micro-benchmark SINTETICO su CPU (Mac) delle cinque teste della proposta c15aac9.

Non e' il codice di GPT (che non esiste ancora): e' una ricostruzione dalle specifiche
del protocollo (par. 3 e 4), con dati casuali della stessa forma delle cache.
Serve solo a dare l'ordine di grandezza del costo di una epoca.

Dati: N finestre x 160 patch x 384 feature in FP16 (come la cache), cast a FP32 per batch.
Ricetta: AdamW lr 1e-3, wd 0,01 sui soli pesi, clipping norma 1, batch 32, ultimo batch
conservato, shuffle per epoca, loss MSE mascherata sul crop6 media per finestra.
"""
import json
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1907
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = sys.argv[3] if len(sys.argv) > 3 else None
P, D, J, B, CROP = 160, 384, 20, 32, 6
torch.set_num_threads(THREADS)
torch.manual_seed(0)
rng = np.random.default_rng(0)


class L0(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln, self.out = nn.LayerNorm(D, eps=1e-5), nn.Linear(D, J)

    def forward(self, x):
        return self.out(self.ln(x))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        w = [D, 96, 96, 96, 64, 32, J]
        self.ln = nn.LayerNorm(D, eps=1e-5)
        layers = []
        for i in range(6):
            layers.append(nn.Linear(w[i], w[i + 1]))
            if i < 5:
                layers.append(nn.ReLU())
            if i in (3, 4):  # dopo i hidden da 64 e 32
                layers.append(nn.Dropout(0.1))
        self.net = nn.Sequential(*layers)
        self.out = self.net[-1]

    def forward(self, x):
        return self.net(self.ln(x))


class L13(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln, self.out = nn.LayerNorm(D, eps=1e-5), nn.Conv1d(D, J, 13, padding=6)

    def forward(self, x):
        return self.out(self.ln(x).transpose(1, 2)).transpose(1, 2)


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D, eps=1e-5)
        self.net = nn.Sequential(
            nn.Conv1d(D, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, dilation=1, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(32, J, 1))
        self.out = self.net[-1]

    def forward(self, x):
        return self.net(self.ln(x).transpose(1, 2)).transpose(1, 2)


def make(cls):
    m = cls()
    nn.init.zeros_(m.out.weight)
    nn.init.zeros_(m.out.bias)
    return m


def optimizer(m, lr=1e-3):
    decay = [p for n, p in m.named_parameters() if p.ndim >= 2]
    rest = [p for n, p in m.named_parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.999), eps=1e-8)


def loss_fn(pred, y, mask):
    se = (pred - y) ** 2 * mask
    per_window = se.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)
    return per_window.mean()


def epoch_train(m, opt, x16, y, mask, order):
    m.train()
    steps, t0 = 0, time.perf_counter()
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        xb = torch.from_numpy(x16[idx].astype(np.float32))
        loss = loss_fn(m(xb), y[idx], mask[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        steps += 1
    return time.perf_counter() - t0, steps, float(loss)


@torch.inference_mode()
def epoch_eval(m, x16, y, mask, batch=256):
    m.eval()
    t0, total = time.perf_counter(), 0.0
    for s in range(0, len(x16), batch):
        xb = torch.from_numpy(x16[s:s + batch].astype(np.float32))
        pred = m(xb)
        se = (pred - y[s:s + batch]) ** 2 * mask[s:s + batch]
        total += float((se.sum(dim=(1, 2)) / mask[s:s + batch].sum(dim=(1, 2)).clamp_min(1)).sum())
    return time.perf_counter() - t0


t_gen = time.perf_counter()
x16 = rng.standard_normal((N, P, D), dtype=np.float32).astype(np.float16)
y = torch.from_numpy(rng.standard_normal((N, P, J), dtype=np.float32))
mask_np = rng.random((N, P, J)) < 0.9
mask_np[:, :CROP] = False
mask_np[:, P - CROP:] = False
mask = torch.from_numpy(mask_np.astype(np.float32))
t_gen = time.perf_counter() - t_gen

result = {"synthetic": True, "device": "cpu", "torch": torch.__version__, "threads": torch.get_num_threads(),
          "n_windows": N, "patches": P, "dim": D, "batch": B, "x_bytes_fp16": int(x16.nbytes), "gen_seconds": t_gen, "heads": {}}
for name, cls in (("L0", L0), ("MLP", MLP), ("L13", L13), ("TCN", TCN)):
    torch.manual_seed(0)
    m = make(cls)
    n_params = sum(p.numel() for p in m.parameters())
    opt = optimizer(m)
    gen = torch.Generator().manual_seed(0)
    times = []
    for ep in range(3):
        order = torch.randperm(N, generator=gen).numpy()
        sec, steps, last = epoch_train(m, opt, x16, y, mask, order)
        times.append(sec)
    ev = [epoch_eval(m, x16, y, mask) for _ in range(2)]
    result["heads"][name] = {"params": n_params, "steps_per_epoch": steps, "train_epoch_seconds": times,
                             "eval_epoch_seconds": ev, "last_loss": last}
    print(name, n_params, "train epoch s", [round(t, 3) for t in times], "eval epoch s", [round(t, 3) for t in ev], flush=True)

result["peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("peak RSS (bytes, macOS ru_maxrss)", result["peak_rss_bytes_macos"])
if OUT:
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
