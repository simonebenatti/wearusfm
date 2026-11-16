#!/usr/bin/env python3
"""Pinned technical CUDA/NCCL checks of the contrastive trainer; never submits jobs.

Every rank computes its full-batch references BEFORE initializing NCCL: contrastive
gather helpers inspect the actual process group, not the trainer's DistContext.
The two updates use active CN/CR weights, unequal window lengths, and no pose on
rank 0. Two additional real-trainer probes exercise pre-update coverage STOP and
a nonfinite representation on only rank 0, including the C0 zero-weight path.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from contrastive_prepare import _read, config_from_dict, source_digest
from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime, attach_contrastive_head
from wearusfm.training.jepa import make_teacher, ema_update
from wearusfm.training.pairing import batch_sha256, state_sha256

PROTOCOL = "contrastive-cuda-technical-v1"
CRITERIA = {"fp32_loss_relative": 1e-5, "fp32_gradient_relative": 1e-5,
            "fp32_displacement_relative": .01, "bf16_loss_relative": .01,
            "bf16_noise_floor": 1e-5}
TECHNICAL_OVERRIDES = {"global_batch_size": 8, "updates": 2, "datasets": ["emg2pose"],
                       "muscle_dropout": 0., "window_seconds": [1., 4.], "fill_steps": 0,
                       "ramp_steps": 1, "min_negatives": 1, "rnc_stride": 8,
                       "nce_weight": 1., "rnc_weight": 1.,
                       "coverage_block": 1000, "coverage_quotas": None, "workers": 0,
                       "first_rank_pose": "removed", "gradient_diagnostics": "every update"}


def technical_config(cfg, arm):
    cc = ContrastiveConfig(**cfg.contrastive)
    cc = replace(cc, arm=arm, nce_weight=1., rnc_weight=1., fill_steps=0, ramp_steps=1, min_negatives=1, rnc_stride=8,
                 grad_every=1, grad_steps=(), grad_dense_start=0, grad_dense_end=0,
                 calibration_only=False, coverage_block=1000, coverage_quotas=None,
                 coverage_exempt_datasets=(), coverage_global_batch_size=8, enforce_coverage=True)
    return replace(cfg, contrastive=asdict(cc), batch_size=8, accum_steps=1, max_steps=2,
                   datasets=("emg2pose",), model=replace(cfg.model, muscle_dropout=0.),
                   loader=replace(cfg.loader, min_window_s=1., max_window_s=4.),
                   fixed_eval_every=0, pose_panel_every=0, pose_grad_every=0,
                   prefetch=False, prefetch_audit=False, log_batch_origin=False, ckpt_every=1)


def build_batches(cfg, manifest, roots, scales, window_s):
    loader = L.PretrainLoader(R.load_index(manifest, roots, cfg.datasets, cfg.nested), cfg.loader, dict(scales), window_s)
    rng = np.random.default_rng([cfg.seed, 20261116])
    batches = []
    for _ in range(64):
        batch = loader.batch(8, rng)
        if len({s.shape[-1] for s in batch.signals}) < 2 or batch.pose is None:
            continue
        if not all(p is not None and bool(np.asarray(p[1]).any()) for p in batch.pose):
            continue
        # Rank 0 receives the first two physical windows and no pose labels.
        batch.pose[:2] = [None, None]
        batches.append(batch)
        if len(batches) == 2:
            return batches
    raise ValueError("cannot construct two unequal-length batches with valid pose on ranks 1-3")


def _flat(tensors):
    return torch.cat([tensor.detach().reshape(-1).double().cpu() for tensor in tensors])


def _norm(tensor):
    return float(torch.linalg.vector_norm(tensor, dtype=torch.float64))


def _relative_difference(value, reference, scale=None):
    return _norm(value - reference) / max(_norm(reference if scale is None else scale), 1e-30)


def trajectory(cfg, initial, batches, ctx, device, precision):
    """Same objective contract, clipping, optimizer, EMA, and queue commit as R.train."""
    if not ctx.active and torch.distributed.is_initialized():
        raise RuntimeError("full-batch reference must precede process-group initialization")
    student = attach_contrastive_head(WearUsFM(cfg.model)).to(device)
    student.load_state_dict(initial)
    teacher = make_teacher(student)
    engine = ContrastiveRuntime(cfg.contrastive)
    model = R._ddp_wrap(student, cfg, ctx, device) if ctx.active else None
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    result = []
    for step, batch in enumerate(batches):
        for group in optimizer.param_groups:
            group["lr"] = R._lr_at(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        student.train()
        engine.prepare(batch, step)
        micro = R._prepare(batch, device)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
        failure, metrics = R._accumulate_terms(model, student, teacher, [micro], cfg, ctx, device, amp, contrastive=engine)
        if failure or engine.before_update():
            raise RuntimeError(failure or "unexpected coverage stop in equivalence trajectory")
        gradient = _flat(p.grad if p.grad is not None else torch.zeros_like(p) for p in student.parameters())
        projector = math.sqrt(sum(_norm(p.grad) ** 2 for p in student.contrastive_head.parameters() if p.grad is not None))
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite gradient in CUDA equivalence trajectory")
        optimizer.step()
        ema_update(teacher, student, cfg.jepa.ema_momentum)
        engine.commit()
        result.append({"metrics": metrics, "gradient": gradient, "student": _flat(student.parameters()),
                       "teacher": _flat(teacher.parameters()), "projector_grad_norm": projector,
                       "queue_identity_sha256": state_sha256([e["identity"] for e in engine.queue.state_dict()["entries"]])})
    del optimizer, model, student, teacher, engine
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def compare_trajectory(actual, reference, fp32_reference, initial, precision):
    """A fixed acceptance rule; diagnostic gradients use the same precision rule."""
    rows = []
    for number, (value, expected, fp32) in enumerate(zip(actual, reference, fp32_reference), start=1):
        errors, limits = {}, {}
        for key in ("total", "pose", "contrast_nce", "contrast_rnc", "contrast_global_grad_base",
                    "contrast_global_grad_nce", "contrast_global_grad_rnc"):
            reference_value = expected["metrics"][key]
            errors[key] = abs(value["metrics"][key] - reference_value) / max(abs(reference_value), 1e-12)
            if precision == "fp32":
                limits[key] = CRITERIA["fp32_loss_relative"]
            elif key.startswith("contrast_global_grad_"):
                limits[key] = abs(reference_value - fp32["metrics"][key]) / max(abs(fp32["metrics"][key]), 1e-12) + CRITERIA["bf16_noise_floor"]
            else:
                limits[key] = CRITERIA["bf16_loss_relative"]
        errors["gradient"] = _relative_difference(value["gradient"], expected["gradient"])
        limits["gradient"] = (CRITERIA["fp32_gradient_relative"] if precision == "fp32" else
                              _relative_difference(expected["gradient"], fp32["gradient"]) + CRITERIA["bf16_noise_floor"])
        for key in ("student", "teacher"):
            scale = expected[key] - initial
            errors[key] = _relative_difference(value[key], expected[key], scale)
            limits[key] = (CRITERIA["fp32_displacement_relative"] if precision == "fp32" else
                           _relative_difference(expected[key], fp32[key], scale) + CRITERIA["bf16_noise_floor"])
        finite = all(math.isfinite(x) for x in (*errors.values(), *limits.values(), value["projector_grad_norm"]))
        support = (value["metrics"]["contrast_rnc_windows"] > 0 and value["metrics"]["contrast_nce_eligible"] > 0 and
                   value["queue_identity_sha256"] == expected["queue_identity_sha256"])
        rows.append({"update": number, "errors": errors, "limits": limits,
                     "projector_grad_norm": value["projector_grad_norm"], "metrics": value["metrics"],
                     "passed": finite and support and all(errors[k] <= limits[k] for k in errors)})
    return {"passed": bool(rows) and len(rows) == len(reference) == len(fp32_reference) and all(r["passed"] for r in rows), "updates": rows}


def trainer_stop_checks(cfg, manifest, roots, scales, windows, output, ctx, device):
    """Exercise actual R.train STOP/summary/checkpoint behavior on all four ranks."""
    # A derived production manifest must retain all datasets inside R.train.
    # The equivalence fixture above explicitly filters pose before its manual updates.
    coverage_cfg = replace(cfg, datasets=None, max_steps=2, contrastive={**cfg.contrastive, "arm": "C0", "coverage_block": 2,
                                                        "coverage_min_queries": 1000})
    coverage = R.train(coverage_cfg, manifest, roots, output / "coverage_stop", scales=scales,
                       window_s=windows, device=device, dist=ctx, log=lambda _: None)
    torch.distributed.barrier()
    saved = torch.load(output / "coverage_stop/checkpoint.pt", map_location="cpu", weights_only=False)
    coverage_ok = (coverage["steps"] == 1 and saved["step"] == 1 and coverage["contrastive"]["queue_last_step"] == 0 and
                   coverage["contrastive"]["final_student_sha256"] == state_sha256(saved["student"]) and
                   coverage["contrastive"]["final_teacher_sha256"] == state_sha256(saved["teacher"]) and
                   coverage["contrastive"]["coverage_failed_block"]["updates"] == 2)
    original = ContrastiveRuntime.attach
    def inject(self, terms, student, teacher, micro, enc, enc_t):
        if ctx.rank == 0:
            with torch.no_grad():
                enc.z[0, 0, 0, 0] = float("nan")
        return original(self, terms, student, teacher, micro, enc, enc_t)
    ContrastiveRuntime.attach = inject
    try:
        stop_cfg = replace(cfg, datasets=None, contrastive={**cfg.contrastive, "arm": "C0"})
        numeric = R.train(stop_cfg, manifest, roots, output / "numeric_stop", scales=scales,
                          window_s=windows, device=device, dist=ctx, log=lambda _: None)
    finally:
        ContrastiveRuntime.attach = original
    numeric_ok = numeric["steps"] == 0 and numeric["contrastive"]["queue_last_step"] == -1 and "numerical failure" in numeric["stopped"]
    if ctx.is_main:
        coverage_ok &= len((output / "coverage_stop/metrics.jsonl").read_text().splitlines()) == 1
        numeric_ok &= (output / "numeric_stop/STOP").exists() and (output / "numeric_stop/metrics.jsonl").read_text() == ""
    return {"coverage": bool(coverage_ok), "numeric": bool(numeric_ok), "passed": bool(coverage_ok and numeric_ok)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "scales", "window-seconds", "calibration-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--pose-stats-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--rank-exit-report", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ctx = R.dist_from_env()
    if ctx.world_size != 4 or not torch.cuda.is_available() or torch.distributed.is_initialized():
        raise ValueError("requires torchrun with exactly four fresh CUDA processes")
    from contrastive_rank_exit import validate_receipt
    validate_receipt(args.rank_exit_report, expected_source_sha256=args.expected_source_sha256)
    rank_exit_sha256 = hashlib.sha256(args.rank_exit_report.read_bytes()).hexdigest()
    pins = {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("config", "manifest", "scales", "window_seconds", "calibration_checkpoint")}
    if any(pin != getattr(args, name + "_sha256") for name, pin in pins.items()) or source_digest() != args.expected_source_sha256:
        raise ValueError("technical check source/input pin mismatch")
    base = config_from_dict(_read(args.config))
    if hashlib.sha256((Path(base.loader.pose_root) / "pose_stats.json").read_bytes()).hexdigest() != args.pose_stats_sha256:
        raise ValueError("technical check pose statistics pin mismatch")
    if args.out.exists() or args.out.with_suffix("").exists():
        raise ValueError("technical check requires fresh output/report paths")
    device = f"cuda:{ctx.local_rank}"
    torch.cuda.set_device(ctx.local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scales, windows = _read(args.scales)["scales"], _read(args.window_seconds)["window_s"]
    configs = {arm: technical_config(base, arm) for arm in ("CN", "CR")}
    batches = build_batches(configs["CN"], args.manifest, args.root, scales, windows)
    torch.manual_seed(base.seed)
    initial_model = attach_contrastive_head(WearUsFM(base.model))
    saved = torch.load(args.calibration_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    initial = saved["student"]
    if (saved.get("step") != 512 or saved.get("optimizer", {}).get("state") or
            state_sha256(initial) != saved.get("contrastive_initial_sha256") or state_sha256(saved["teacher"]) != state_sha256(initial)):
        raise ValueError("technical check requires the fixed-weight calibration checkpoint")
    initial_model.load_state_dict(initial)
    initial_vector = _flat(initial_model.parameters())
    initial_sha = state_sha256(initial)
    del initial_model
    references = {arm: {precision: trajectory(cfg, initial, batches, R.DistContext(), device, precision)
                        for precision in ("fp32", "bf16")} for arm, cfg in configs.items()}
    # Initializing earlier would silently gather the supposedly single-rank references.
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        hashes = {"batches": [batch_sha256(batch) for batch in batches], "initial": initial_sha}
        identities = [None] * ctx.world_size
        torch.distributed.all_gather_object(identities, hashes)
        if any(identity != hashes for identity in identities):
            raise ValueError("technical fixtures or initial states differ across ranks")
        report = {"protocol": PROTOCOL, "criteria": CRITERIA, "technical_overrides": TECHNICAL_OVERRIDES,
                  "stop_probe_overrides": {"datasets": "all in the derived manifest", "coverage_block": 2,
                                            "coverage_min_queries": 1000, "nonfinite_rank": 0, "arm": "C0"},
                  "inputs": pins, "pose_stats_sha256": args.pose_stats_sha256, "source_sha256": args.expected_source_sha256,
                  "config_canonical_sha256": hashlib.sha256(json.dumps(_read(args.config), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                  "rank_exit_report_sha256": rank_exit_sha256, "rank_exit_report_path": str(args.rank_exit_report.resolve()),
                  "world_size": 4, "initial_sha256": initial_sha, "batch_sha256": hashes["batches"],
                  "batch_lengths": [[s.shape[-1] for s in b.signals] for b in batches], "arms": {}}
        for arm, cfg in configs.items():
            report["arms"][arm] = {}
            for precision in ("fp32", "bf16"):
                local = [L.split_batch(batch, 4)[ctx.rank] for batch in batches]
                actual = trajectory(cfg, initial, local, ctx, device, precision)
                report["arms"][arm][precision] = compare_trajectory(actual, references[arm][precision], references[arm]["fp32"], initial_vector, precision)
                del actual
        report["stop_checks"] = trainer_stop_checks(configs["CN"], args.manifest, args.root, scales, windows,
                                                     args.out.with_suffix(""), ctx, device)
        own_passed = report["stop_checks"]["passed"] and all(p["passed"] for arm in report["arms"].values() for p in arm.values())
        ranks = [None] * ctx.world_size
        torch.distributed.all_gather_object(ranks, {"rank": ctx.rank, "passed": own_passed, "arms": report["arms"], "stop_checks": report["stop_checks"]})
        report["ranks"] = ranks
        after_pins = {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() for name in pins}
        report["inputs_unchanged"] = after_pins == pins and hashlib.sha256(args.rank_exit_report.read_bytes()).hexdigest() == rank_exit_sha256 and hashlib.sha256(
            (Path(base.loader.pose_root) / "pose_stats.json").read_bytes()).hexdigest() == args.pose_stats_sha256
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                  "device": torch.cuda.get_device_name(), "code_sha": os.environ.get("WEARUSFM_CODE_SHA"),
                                  "gpu_max_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        report["passed"] = all(rank["passed"] for rank in ranks) and report["inputs_unchanged"] and source_digest() == args.expected_source_sha256
        if ctx.is_main:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
            print(json.dumps({"passed": report["passed"], "report": str(args.out)}), flush=True)
        return 0 if report["passed"] else 1
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
