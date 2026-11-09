"""Profilo del passo di training di TEMPONet su GPU (replica DB8): ms per passo con impostazioni cuDNN diverse e kernel piu' costosi.

Il collaudo 2 (63986820) ha misurato 18,5 ms per passo in eager e 14,6 col CUDA graph, con `cudnn.deterministic=True`, `benchmark=False` e TF32
spento: il tempo e' nei kernel, non nei lanci. Qui, sugli stessi dati (S1 A1 della cache, ingresso in microvolt, batch 64), per ciascuna
combinazione (deterministic, benchmark) si misurano eager e graph su 50 passi dopo il riscaldamento, e per la combinazione del collaudo si
elencano i 15 kernel CUDA con il tempo totale maggiore (torch.profiler, 10 passi eager). Solo misure: nessun modello salvato, nessun risultato.

Uso: python3 scripts/temponet_db8_profile.py --cache <cache> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import temponet_db8 as run
from wearusfm.harness import db8_regression as db8

SETTINGS = ((True, False), (False, False), (False, True))  # (cudnn.deterministic, cudnn.benchmark)


def measure(sig, cfg, device, use_graph: bool, steps: int = 50, warmup: int = 10) -> float:
    import torch

    model, opt, loss_fn = run.build(cfg, 777, device)
    model.train()
    st = run.Stepper(model, opt, loss_fn, sig, cfg.batch, use_graph, device)
    gen = torch.Generator().manual_seed(1)
    batches = sig.ends[torch.randperm(len(sig.ends), generator=gen)[: (steps + warmup) * cfg.batch].to(device)].view(-1, cfg.batch)
    for i in range(warmup):
        st.step(batches[i])
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(warmup, warmup + steps):
        st.step(batches[i])
    torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / steps


def top_kernels(sig, cfg, device, n: int = 15) -> list[dict]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    model, opt, loss_fn = run.build(cfg, 777, device)
    model.train()
    st = run.Stepper(model, opt, loss_fn, sig, cfg.batch, False, device)
    batches = sig.ends[: 20 * cfg.batch].view(20, cfg.batch)
    for i in range(10):
        st.step(batches[i])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(10, 20):
            st.step(batches[i])
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        t = getattr(ev, "self_cuda_time_total", 0) or getattr(ev, "self_device_time_total", 0)
        if t:
            rows.append({"kernel": ev.key[:160], "calls": ev.count, "total_ms": t / 1000.0})
    rows.sort(key=lambda r: -r["total_ms"])
    return rows[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args(argv)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cfg = argparse.Namespace(window=db8.WINDOW, symmetric_padding=False, lr=1e-4, batch=64)
    rec = db8.load_cached(args.cache, args.subject, 1)
    sig = run.Signal([rec], device, run.EMG_UNIT_SCALE["uV"], db8.WINDOW, db8.STEP)
    out = {"subject": args.subject, "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0), "settings": []}
    for det, bench in SETTINGS:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = det, bench
        row = {"deterministic": det, "benchmark": bench,
               "eager_ms_per_step": measure(sig, cfg, device, False), "graph_ms_per_step": measure(sig, cfg, device, True)}
        print(row, flush=True)
        out["settings"].append(row)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    out["top_kernels_deterministic"] = top_kernels(sig, cfg, device)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    out["top_kernels_benchmark"] = top_kernels(sig, cfg, device)
    for key in ("top_kernels_deterministic", "top_kernels_benchmark"):
        print(key)
        for r in out[key]:
            print(f"  {r['total_ms']:8.2f} ms  {r['calls']:5d}  {r['kernel']}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
