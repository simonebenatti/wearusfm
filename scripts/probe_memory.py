#!/usr/bin/env python3
"""Sonda della memoria per GPU dei rung della ladder (collaudo dell'infrastruttura, `docs/collaudo_infra_gpu_1.md`): il caso peggiore del
dataloader (finestre tutte da `--window-s`, montaggio intero senza montaggi virtuali, di default Hyser: 4 griglie, 256 canali a 2048 Hz),
`--per-gpu` finestre per GPU, regole della finestra 1, bf16 e activation checkpointing come nel training. Per ogni (rung, finestre per GPU):
`--steps` aggiornamenti completi sullo STESSO batch (forward, backward, clip, AdamW, EMA: dal primo aggiornamento esistono gli stati di
AdamW), poi memoria massima allocata e riservata da torch e secondi per passo; un OOM si registra e si passa al caso successivo.
DDP aggiunge i buffer dei gradienti (~4 byte per parametro): stimati a parte, `ddp_buckets_gib`.

  python3 scripts/probe_memory.py --manifest M --root R1 --root R2 --scales S --window-seconds W --out probe_memory.json \\
      [--rungs 30m,100m,300m] [--per-gpu 4,8] [--dataset hyser] [--window-s 4.0] [--steps 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.jepa import ema_update, jepa_losses, make_teacher  # noqa: E402


def probe(rung: str, per_gpu: int, index, scales: dict, window_s: dict, args, device: str) -> dict:
    cfg = R.with_window1_rules(R.ladder_config(rung))
    cfg = replace(cfg, loader=replace(cfg.loader, min_window_s=args.window_s, max_window_s=args.window_s, virtual=None))
    loader = L.PretrainLoader(index, cfg.loader, dict(scales), window_s)
    batch = loader.batch(per_gpu, np.random.default_rng(args.seed))
    out = {"rung": rung, "per_gpu": per_gpu, "channels": list(batch.counts), "n_patches": list(batch.n_patches),
           "datasets": sorted({r["dataset"] for r in batch.rows})}
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    student = teacher = opt = None
    try:
        student = WearUsFM(cfg.model).to(device)
        teacher = make_teacher(student).to(device)
        opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = sum(p.numel() for p in student.parameters())
        out.update(parameters=n, ddp_buckets_gib=4 * n / 2**30)
        inp, visible, _ = R._to_device(*L.to_model_inputs(batch), device)
        kinds = torch.as_tensor(batch.kind).to(device)
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda and cfg.amp_bf16)
        times = []
        for _ in range(args.steps):
            t0 = time.time()
            student.train()
            opt.zero_grad(set_to_none=True)
            with amp:
                losses = jepa_losses(student, teacher, inp, visible, cfg.jepa, anchor_targets=batch.anchor_targets, kinds=kinds)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            opt.step()
            ema_update(teacher, student, cfg.jepa.ema_momentum)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        out.update(oom=False, step_s=times, total_loss=float(losses["total"].detach()), adam_states=len(opt.state) > 0)
    except torch.cuda.OutOfMemoryError as e:
        out.update(oom=True, error=str(e)[:300])
    if cuda:
        out.update(max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    del student, teacher, opt
    if cuda:
        torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--scales", type=Path, default=None)
    ap.add_argument("--window-seconds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rungs", default="30m,100m,300m")
    ap.add_argument("--per-gpu", default="4,8")
    ap.add_argument("--dataset", default="hyser")
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    index = R.load_index(args.manifest, args.root, (args.dataset,))
    scales = json.loads(args.scales.read_text())["scales"] if args.scales else {}
    window_s = json.loads(args.window_seconds.read_text())["window_s"]
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
    report = {"device": args.device, "gpu": gpu, "dataset": args.dataset, "window_s": args.window_s, "steps": args.steps, "cases": []}
    for rung in args.rungs.split(","):
        for b in (int(x) for x in args.per_gpu.split(",")):
            r = probe(rung, b, index, scales, window_s, args, args.device)
            report["cases"].append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("channels", "n_patches")}), flush=True)
            args.out.write_text(json.dumps(report, indent=1))  # a ogni caso: un timeout non perde i casi gia' misurati
    return 0


if __name__ == "__main__":
    sys.exit(main())
