#!/usr/bin/env python3
"""Prova (A) del protocollo `w1_gpu_dati_v1` (`docs/protocollo_gpu_dati_finestra1.md`; Simone, 14/10/2026: «sì, via al protocollo GPU»):
equivalenza numerica su GPU del trainer col contratto delle loss distribuite. Gli STESSI batch globali (il DDP da' a ogni processo il suo stream:
due training non si confrontano passo per passo) calcolati in quattro modi, per 8 aggiornamenti consecutivi dallo stesso stato iniziale:
  ref       un processo, batch intero, il percorso di sempre (`jepa_losses`), come il trainer senza DDP ne' accumulo;
  acc       un processo, 4 micro-batch (`_accumulate_terms`);
  ddp       DDP su tutti i processi (NCCL), una parte del batch per processo;
  ddp_acc   DDP con 2 micro-batch per processo.
In float32 (TF32 spento) e in bf16 (autocast del trainer); in bf16 anche una ripetizione del riferimento (rumore, descrittivo). Dropout del
muscolo a 0 (come il test CPU): con il dropout i numeri casuali differiscono fra processi e il confronto non avrebbe senso.

Confronti per aggiornamento, contro il riferimento della stessa precisione: loss totale (differenza relativa), gradiente dopo la riduzione
(||g - g_ref|| / ||g_ref||), pesi e teacher dopo AdamW ed EMA (distanza dal riferimento / spostamento del riferimento dallo stato iniziale).
Criteri (congelati nel protocollo prima della misura):
  float32: loss e gradiente <= 1e-5 relativo; pesi e teacher <= 1e-2 dello spostamento (controllo del cablaggio di optimizer ed EMA: AdamW
           amplifica le differenze minime sui gradienti quasi nulli);
  bf16:    loss <= 1e-2 relativo; gradiente, pesi e teacher: distanza dal riferimento bf16 <= distanza fra riferimento bf16 e float32.

  torchrun --standalone --nproc_per_node=4 scripts/ddp_equivalence.py --manifest M --scales S --window-seconds W --root R... --out REPORT.json
"""
import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.jepa import ema_update, jepa_losses, make_teacher  # noqa: E402
from wearusfm.training.pairing import batch_sha256  # noqa: E402

PROTOCOL = "w1_gpu_dati_v1/A"
BATCH_SEED = 6161
CRITERIA = {"fp32": {"loss": 1e-5, "grad": 1e-5, "params": 1e-2}, "bf16": {"loss": 1e-2}}


def equivalence_config(preset: str = "sanity") -> R.RunConfig:
    """Il preset firmato con le regole della finestra 1, tutti i dataset, dropout del muscolo a 0."""
    cfg = R.with_window1_rules(R.sanity_config() if preset == "sanity" else R.ladder_config(preset))
    return replace(cfg, datasets=None, model=replace(cfg.model, muscle_dropout=0.0))


def build_batches(cfg, manifest, roots, scales, window_s, n: int = 8, size: int = 32, seed: int = 0) -> list:
    loader = L.PretrainLoader(R.load_index(manifest, roots, cfg.datasets), cfg.loader, dict(scales or {}), window_s)
    rng = np.random.default_rng([seed, BATCH_SEED])
    return [loader.batch(size, rng) for _ in range(n)]


def _flat(tensors) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).double().cpu() for t in tensors])


def _trajectory(cfg, init, batches, *, mode: str, accum: int, ctx, device, dtype, amp):
    """8 aggiornamenti come nel trainer (learning rate di `_lr_at`, clipping, AdamW, EMA). Rende, per aggiornamento: metriche, gradiente
    ridotto, pesi e teacher dopo il passo (vettori piatti sulla CPU)."""
    student = WearUsFM(cfg.model)
    student.load_state_dict(init)
    student = student.to(device=device, dtype=dtype) if dtype is not None else student.to(device)
    teacher = make_teacher(student)
    model = R._ddp_wrap(student, cfg, ctx, device) if ctx.active else None
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out = []
    for t, batch in enumerate(batches):
        for g in opt.param_groups:
            g["lr"] = R._lr_at(t, cfg)
        student.train()
        opt.zero_grad(set_to_none=True)
        if mode == "always":
            m = R._prepare(batch, device, dtype)
            with amp:
                losses = jepa_losses(student, teacher, m.inp, m.visible, cfg.jepa, anchor_targets=m.anchor_targets, kinds=m.kinds)
            losses["total"].backward()
            metrics = {k: float(v.detach()) for k, v in losses.items()}
        else:
            micros = [R._prepare(mb, device, dtype) for mb in L.split_batch(batch, accum)]
            failure, metrics = R._accumulate_terms(model, student, teacher, micros, cfg, ctx, device, amp)
            if failure is not None:
                raise RuntimeError(f"{mode} x {accum}: {failure}")
        grad = _flat(p.grad if p.grad is not None else torch.zeros_like(p) for p in student.parameters())
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        opt.step()
        ema_update(teacher, student, cfg.jepa.ema_momentum)
        out.append({"metrics": metrics, "grad": grad, "params": _flat(student.parameters()), "teacher": _flat(teacher.parameters())})
    return out


def _rel(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> float:
    s = float(scale.norm())
    return float((a - b).norm()) / s if s > 0 else float((a - b).norm())


def _compare(var, ref, p0) -> list:
    """Per aggiornamento: loss totale (relativa), gradiente (relativo alla norma del riferimento), pesi e teacher (distanza dal riferimento,
    assoluta e relativa allo spostamento del riferimento dallo stato iniziale p0: lo stato iniziale del teacher e' quello dello studente)."""
    rows = []
    for v, r in zip(var, ref):
        loss = abs(v["metrics"]["total"] - r["metrics"]["total"]) / max(abs(r["metrics"]["total"]), 1e-12)
        rows.append({"loss": loss, "grad": _rel(v["grad"], r["grad"], r["grad"]),
                     "params": _rel(v["params"], r["params"], r["params"] - p0), "teacher": _rel(v["teacher"], r["teacher"], r["teacher"] - p0),
                     "params_abs": float((v["params"] - r["params"]).norm()), "teacher_abs": float((v["teacher"] - r["teacher"]).norm()),
                     "total": v["metrics"]["total"]})
    return rows


def run_equivalence(cfg, batches, init, *, rank: int, world: int, device, dtype=None, precisions=("fp32", "bf16"), accum=(4, 2)) -> dict | None:
    """Tutte le traiettorie e i confronti; il rapporto sul processo 0 (None sugli altri). I processi diversi da 0 aspettano mentre il processo
    0 calcola le varianti a un processo."""
    dist = torch.distributed
    one = R.DistContext()
    ctx = R.DistContext(rank, world, rank) if world > 1 else one
    p0 = _flat(init[k] for k, _ in WearUsFM(cfg.model).named_parameters())
    report = {"protocol": PROTOCOL, "world_size": world, "updates": len(batches), "windows_per_batch": len(batches[0].signals),
              "criteria": CRITERIA, "precisions": {}}
    refs = {}
    for prec in precisions:
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if prec == "bf16" else nullcontext()
        rows = {}
        if rank == 0:
            refs[prec] = _trajectory(cfg, init, batches, mode="always", accum=1, ctx=one, device=device, dtype=dtype, amp=amp)
            rows["acc"] = _compare(_trajectory(cfg, init, batches, mode="contract", accum=accum[0], ctx=one, device=device, dtype=dtype,
                                               amp=amp), refs[prec], p0)
            if prec == "bf16":
                rows["ref_repeat"] = _compare(_trajectory(cfg, init, batches, mode="always", accum=1, ctx=one, device=device, dtype=dtype,
                                                          amp=amp), refs[prec], p0)
        if world > 1:
            dist.barrier()
            mine = [L.split_batch(b, world)[rank] for b in batches]
            for name, k in (("ddp", 1), ("ddp_acc", accum[1])):
                traj = _trajectory(cfg, init, mine, mode="contract", accum=k, ctx=ctx, device=device, dtype=dtype, amp=amp)
                if rank == 0:
                    rows[name] = _compare(traj, refs[prec], p0)
                dist.barrier()
        if rank == 0:
            report["precisions"][prec] = {"reference_total": [r["metrics"]["total"] for r in refs[prec]], "variants": rows}
    if rank != 0:
        return None
    report["passed"] = _verdict(report, refs)
    return report


def _verdict(report: dict, refs: dict) -> dict:
    """I criteri congelati. In bf16 la soglia di gradiente, pesi e teacher e' la distanza fra i riferimenti bf16 e float32 allo stesso
    aggiornamento (il DDP e l'accumulo non devono aggiungere piu' errore del bf16 stesso); la ripetizione del riferimento e' descrittiva."""
    out = {}
    fp32 = report["precisions"].get("fp32")
    if fp32:
        c = CRITERIA["fp32"]
        out["fp32"] = all(row["loss"] <= c["loss"] and row["grad"] <= c["grad"] and row["params"] <= c["params"] and row["teacher"] <= c["params"]
                          for rows in fp32["variants"].values() for row in rows)
    bf = report["precisions"].get("bf16")
    if bf:
        if "fp32" not in refs:
            raise ValueError("bf16: serve anche il riferimento float32 per la soglia")
        noise = [{"grad": _rel(rb["grad"], rf["grad"], rf["grad"]), "params_abs": float((rb["params"] - rf["params"]).norm()),
                  "teacher_abs": float((rb["teacher"] - rf["teacher"]).norm())} for rb, rf in zip(refs["bf16"], refs["fp32"])]
        bf["bf16_vs_fp32"] = noise
        out["bf16"] = all(row["loss"] <= CRITERIA["bf16"]["loss"] and row["grad"] <= n["grad"] and row["params_abs"] <= n["params_abs"]
                          and row["teacher_abs"] <= n["teacher_abs"]
                          for name, rows in bf["variants"].items() if name != "ref_repeat" for row, n in zip(rows, noise))
    out["all"] = bool(out) and all(out.values())
    return out


def main(argv=None) -> int:
    import os

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--window-seconds", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--preset", default="sanity", choices=["sanity", "100m", "300m"])
    ap.add_argument("--updates", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ctx = R.dist_from_env()
    torch.distributed.init_process_group("nccl")
    device = f"cuda:{ctx.local_rank}"
    torch.cuda.set_device(ctx.local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if ctx.is_main and args.out.exists():
        raise SystemExit(f"{args.out} esiste: rapporto nuovo")
    cfg = equivalence_config(args.preset)
    scales = json.loads(args.scales.read_text())["scales"]
    window_s = json.loads(args.window_seconds.read_text())["window_s"]
    batches = build_batches(cfg, args.manifest, args.root, scales, window_s, n=args.updates)
    hashes = [batch_sha256(b) for b in batches]
    every = [None] * ctx.world_size
    torch.distributed.all_gather_object(every, hashes)
    if any(h != hashes for h in every):
        raise SystemExit("batch diversi fra i processi: il confronto non vale")
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).state_dict()
    report = run_equivalence(cfg, batches, init, rank=ctx.rank, world=ctx.world_size, device=device)
    if ctx.is_main:
        report.update(preset=args.preset, batch_sha256=hashes, code={"dir": os.environ.get("WEARUSFM_CODE_DIR"),
                                                                     "sha": os.environ.get("WEARUSFM_CODE_SHA")})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, allow_nan=False))
        print(json.dumps({"passed": report["passed"]}), flush=True)
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
