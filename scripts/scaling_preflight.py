"""Sequential, train-only 30M/300M technical screening. No checkpoint or job submission.

Each size sees the same 128 delivered batches: 64 native warmup updates followed
by 64 deliberately accelerated peak-LR stress updates. The latter are technical
only and are never counted as scientific training. Only one model/teacher/AdamW
is resident at a time, so the memory measurement applies to a single size.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from wearusfm.model.fm import WearUsFM
from wearusfm.training import jepa as J
from wearusfm.training import run as R
from wearusfm.training.pairing import batch_sha256, state_sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scaling_tokenizer_preflight", ROOT / "scripts/tokenizer_preflight.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
PROTOCOL = "scaling-300m-preflight-20261104-v1"
ARMS = ("T100-G-30M", "T100-G-300M")
require = P.require


def configurations(base):
    """Only the registered LADDER dimensions/depths differ, including the alarm."""
    require(base.model.dim == 384 and base.model.n_heads == 6 and
            base.model.window_ms == 100. and base.model.tokenizer_activation == "gelu" and
            base.model.k_latents == 64 and base.model.pose_head and base.model.grad_checkpoint,
            "expected frozen T100-G 30M configuration")
    require(base.batch_size == 32 and base.accum_steps == 1 and base.seed == 0 and
            base.lr == 1e-4 and base.grad_clip == 1. and base.warmup_steps == 1962 and
            base.schedule.total_steps == 49046 and base.jepa.pose_weight == 1 and
            base.jepa.anchor_weight == .2 and base.jepa.jepa_weight == 0 and
            base.jepa.masked_loss_radius == 0, "unexpected optimization/objective recipe")
    large = replace(base, model=replace(base.model, **R.LADDER["300m"]))
    # Keep the registered absolute rank alarm. Do not silently scale it with D.
    small_dict, large_dict = R.config_to_dict(base), R.config_to_dict(large)
    for key in R.LADDER["300m"]:
        large_dict["model"][key] = small_dict["model"][key]
    require(small_dict == large_dict, "unregistered difference beyond model preset")
    return dict(zip(ARMS, (base, large)))


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    require(plan.get("protocol") == PROTOCOL and plan.get("arms") == list(ARMS), "wrong scaling protocol/arms")
    require(plan.get("native_warmup_batches") == 64 and plan.get("peak_stress_batches") == 64 and
            plan.get("world_size") == 4 and plan.get("global_batch_size") == 32 and plan.get("num_workers") == 6,
            "wrong fixed benchmark dose/stream")
    require(plan.get("max_memory_fraction") == .9 and plan.get("deadline_seconds") == 1500,
            "benchmark gate policy differs")
    source = P.pinned_file(plan["source_plan"])
    source_plan, configs, states = P.load_plan(source["path"])
    states["source_plan"] = source
    states["scaling_plan"] = P.file_identity(path)
    return plan, source_plan, configurations(configs["T100-G"]), states


def create_arm(cfg, ctx, device):
    torch.manual_seed(cfg.seed)
    student = WearUsFM(cfg.model).to(device)
    teacher = J.make_teacher(student).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if ctx.active:
        torch.manual_seed(cfg.seed * 1_000_003 + ctx.rank + 1)
    wrapped = R._ddp_wrap(student, cfg, ctx, device) if ctx.active else None
    if ctx.active:
        for value in teacher.state_dict().values():
            torch.distributed.broadcast(value, src=0)
    initial = {"state_sha256": state_sha256(student.state_dict()),
               "parameters": sum(p.numel() for p in student.parameters()),
               "frontend_and_gain_sha256": state_sha256({k: v for k, v in student.state_dict().items()
                    if k.startswith("tokenizer.frontend.") or k == "tokenizer.fe_gain"}),
               "fe_gain": float(student.tokenizer.fe_gain)}
    groups = P.selected_parameters(student)
    # The historical "backbone_first" was only norm_t.weight. Inspect actual
    # attention/MLP matrices at both ends of the backbone for this scaling audit.
    groups.pop("backbone_first")
    named = dict(student.named_parameters())
    for label, index in (("first", 0), ("last", len(student.backbone.blocks) - 1)):
        for family, suffix in (("qkv", "time_attn.qkv.weight"), ("mlp", "mlp.0.weight")):
            key = f"backbone.blocks.{index}.{suffix}"
            groups[f"backbone_{label}_{family}"] = {key: named[key]}
    initial["diagnostic_parameter_groups"] = {group: list(parameters) for group, parameters in groups.items()}
    arm = {"cfg": cfg, "student": student, "teacher": teacher, "optimizer": optimizer, "wrapped": wrapped,
           "rng": P.rng_state(device), "initial": initial, "features": P.FeatureMoments(),
           "groups": groups, "observed": [], "records": []}
    original_decode = student.decode

    def capture(enc, ch, qt, offsets):
        arm["observed"].append({"ch": ch.detach(), "time": qt.detach(), "offsets": offsets.detach(),
                                "visible": enc.visible.detach(), "patch_valid": enc.patch_valid.detach()})
        return original_decode(enc, ch, qt, offsets)

    student.decode = capture
    return arm


def update_arm(arm, micro, rows, step, native_steps, ctx, device):
    """Use the audited production update, with explicit temporary stress schedule."""
    cfg = arm["cfg"]
    before = {group: P.norm_of(p.detach() for p in parameters.values())
              for group, parameters in arm["groups"].items()}
    phase = "native_warmup" if step < native_steps else "peak_lr_stress"
    if phase == "peak_lr_stress":
        arm["cfg"] = replace(cfg, warmup_steps=0)
    try:
        record = P.update_arm(arm, micro, rows, step, ctx, device)
    finally:
        arm["cfg"] = cfg
    record.update(phase=phase, scientific_update=False,
                  clip_factor=min(1., cfg.grad_clip / (record["grad_norm_before_clip"] + 1e-6)),
                  parameter_norms_before=before,
                  relative_update_norms={group: record["update_norms"][group] / norm if norm > 0 else None
                                         for group, norm in before.items()})
    return record


def verify_pairing(parts, expected_batches, native_steps=64):
    require(set(parts) == set(ARMS), "missing scaling arm")
    a, b = (parts[name] for name in ARMS)
    require(len(a["delivered"]) == len(b["delivered"]) == expected_batches, "incomplete delivered stream")
    for part in (a, b):
        records = part["records"]
        require(len(records) == expected_batches, "incomplete update records")
        require([row["step"] for row in records] == list(range(1, expected_batches + 1)), "nonconsecutive update records")
        require([row["phase"] for row in records] == ["native_warmup"] * native_steps +
                ["peak_lr_stress"] * (expected_batches - native_steps), "incorrect native/stress phases")
    require(a["delivered"] == b["delivered"], "different delivered data/masks/targets between sizes")
    require(a["initial"]["frontend_and_gain_sha256"] == b["initial"]["frontend_and_gain_sha256"],
            "frontend initialization or gain changed with size")
    for ra, rb in zip(a["records"], b["records"]):
        require(ra["support"] == rb["support"] and ra["lr"] == rb["lr"] and ra["phase"] == rb["phase"],
                "query/target support or optimization schedule differs")
    # Larger tensors/depth consume different dropout RNG. Never demand equal
    # model RNG states or shared weight shapes across the two model sizes.


def summarize(parts, name, native_steps, max_memory_fraction):
    records = [part["arms"][name]["records"] for part in parts]
    require(len({len(rows) for rows in records}) == 1, "rank dose differs")
    times = [sum(max(rows[i][key] for rows in records)
                 for key in ("forward_backward_s", "clip_optimizer_ema_s")) for i in range(len(records[0]))]
    available = min(part["total_device_memory_bytes"] for part in parts)
    allocated = max(row["memory_peak_allocated_bytes"] for rows in records for row in rows)
    reserved = max(row["memory_peak_reserved_bytes"] for rows in records for row in rows)
    require(allocated <= available * max_memory_fraction, "single-model allocated peak exceeds memory gate")
    phase_stats = {}
    for phase, sl in (("native_warmup", slice(8, native_steps)), ("peak_lr_stress", slice(native_steps + 8, None))):
        phase_rows = [row for rows in records for row in rows[sl]]
        require(phase_rows, "empty measured phase")
        phase_stats[phase] = {"normal_update_mean_s": float(np.mean(times[sl])),
            "normal_update_p90_s": float(np.quantile(times[sl], .9)),
            "grad_norm_median": float(np.median([r["grad_norm_before_clip"] for r in phase_rows])),
            "clip_factor_median": float(np.median([r["clip_factor"] for r in phase_rows])),
            "fraction_clipped": float(np.mean([r["clip_factor"] < 1 for r in phase_rows])),
            "relative_update_medians": {g: float(np.median([r["relative_update_norms"][g] for r in phase_rows
                                           if r["relative_update_norms"][g] is not None]))
                                        for g in phase_rows[0]["relative_update_norms"]}}
    return {"initial": parts[0]["arms"][name]["initial"], "phases": phase_stats,
            "normal_update_rank_max_seconds": times, "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved, "device_memory_bytes": available,
            "allocated_fraction": allocated / available, "memory_gate_passed": True}


def run(plan, source, configs, states, out, ctx, device, code_sha):
    cfg = configs[ARMS[0]]
    manifest = Path(states["manifest"]["path"])
    scales = json.loads(Path(states["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(states["windows"]["path"]).read_text())["window_s"]
    R.check_input_files(manifest, Path(states["windows"]["path"]), cfg.nested, states["manifest"]["sha256"])
    index = R.load_index(manifest, [Path(p) for p in source["roots"]], cfg.datasets, cfg.nested)
    if cfg.nested is not None:
        R.check_derived_binding(index.params, cfg, windows)
    require(all(row["split"] == "pretraining" for row in index.rows), "nontraining row in index")
    began, results = time.perf_counter(), {}
    n_batches = plan["native_warmup_batches"] + plan["peak_stress_batches"]
    for name in ARMS:
        cfg = configs[name]
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        arm = create_arm(cfg, ctx, device)
        if ctx.active:
            peers = [None] * ctx.world_size
            torch.distributed.all_gather_object(peers, arm["initial"])
            require(all(peer == arm["initial"] for peer in peers), "model initialization differs by rank")
        guard = P.SourceGuard(index, cfg.loader.pose_root)
        loader = torch.utils.data.DataLoader(R.BatchStream(index, cfg.loader, cfg.batch_size // ctx.world_size, cfg.seed,
                 scales, windows, rank=ctx.rank if ctx.active else None, start=0, segment=cfg.stream_segment),
                 batch_size=None, num_workers=plan["num_workers"], persistent_workers=plan["num_workers"] > 0,
                 generator=torch.Generator().manual_seed(cfg.seed), collate_fn=R._to_shared if plan["num_workers"] > 0 else None)
        iterator, delivered, exposure = iter(loader), [], Counter()
        arm_start = time.perf_counter()
        with (out / f"{name}.rank{ctx.rank}.jsonl").open("x") as stream:
            for step in range(n_batches):
                expired, = R._any_flag(ctx, device, time.perf_counter() - began > plan["deadline_seconds"])
                require(not expired, "preflight internal deadline exceeded")
                batch = R._from_shared(next(iterator)) if plan["num_workers"] > 0 else next(iterator)
                require(batch.index == step and len(batch.signals) == cfg.batch_size // ctx.world_size, "unexpected batch identity")
                guard.observe(batch.rows)
                fingerprint = {"batch_sha256": batch_sha256(batch), "processed_signals_sha256": state_sha256(batch.signals),
                    "input_masks_sha256": state_sha256((batch.visible, batch.kind, batch.qc_valid)),
                    "anchor_targets_sha256": state_sha256(batch.anchor_targets), "pose_targets_sha256": state_sha256(batch.pose),
                    "origin_sha256": state_sha256((batch.rows, batch.windows, batch.presented))}
                micros = R._stage(batch, cfg.accum_steps, device)
                require(len(micros) == 1, "one microbatch required")
                before = R._micro_digest(micros)
                record = update_arm(arm, micros[0], batch.rows, step, plan["native_warmup_batches"], ctx, device)
                require(R._micro_digest(micros) == before and batch_sha256(batch) == fingerprint["batch_sha256"], "input mutation")
                for row, fs in zip(batch.rows, batch.fs):
                    exposure[f"{row['dataset']}/{float(fs):g}"] += 1
                stream.write(json.dumps({"arm": name, "rank": ctx.rank, "fingerprint": fingerprint, **record}, allow_nan=False) + "\n")
                stream.flush()
                delivered.append(fingerprint)
                if ctx.is_main and (step + 1) % 8 == 0:
                    print(json.dumps({"arm": name, "step": step + 1, "phase": record["phase"],
                                      "lr": record["lr"], "update_s": record["normal_update_s"]}), flush=True)
        guard.finish()
        for group in arm["groups"]:
            require(any(r["gradient_norms_before_clip"][group] > 0 for r in arm["records"]), "no gradients: " + group)
            require(any(r["update_norms"][group] > 0 for r in arm["records"]), "no actual updates: " + group)
        results[name] = {"initial": arm["initial"], "delivered": delivered, "records": arm["records"],
                         "exposure_windows": dict(exposure), "wall_seconds": time.perf_counter() - arm_start,
                         "sources_unchanged_after_first_seen": True}
        # Break the capture closure before freeing the one resident model.
        arm["student"].decode = None
        del iterator, loader, arm, micros, batch
        gc.collect()
        if ctx.active:
            torch.distributed.barrier()
    verify_pairing(results, n_batches, plan["native_warmup_batches"])
    P.check_unchanged(states)
    P.code_identity(code_sha)
    P.write_json(out / f"rank{ctx.rank}.json", {"protocol": PROTOCOL, "complete": True, "rank": ctx.rank,
        "arms": results, "wall_seconds": time.perf_counter() - began,
        "total_device_memory_bytes": torch.cuda.get_device_properties(device).total_memory})
    if ctx.active:
        torch.distributed.barrier()
    if ctx.is_main:
        parts = [json.loads((out / f"rank{rank}.json").read_text()) for rank in range(ctx.world_size)]
        require(all(p["complete"] and p["protocol"] == PROTOCOL for p in parts), "incomplete rank")
        for part in parts:
            verify_pairing(part["arms"], n_batches, plan["native_warmup_batches"])
        report = {"protocol": PROTOCOL, "complete": True, "code_sha": code_sha,
            "plan_sha256": states["scaling_plan"]["sha256"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "paired_data_masks_queries_targets": True, "one_model_resident": True,
            "scientific_checkpoints_written": False, "batches_per_arm": n_batches,
            "native_warmup_batches": plan["native_warmup_batches"], "peak_stress_batches": plan["peak_stress_batches"],
            "windows_per_arm": n_batches * cfg.batch_size, "world_size": ctx.world_size,
            "arms": {name: summarize(parts, name, plan["native_warmup_batches"], plan["max_memory_fraction"]) for name in ARMS},
            "inputs": states, "rank_reports": [P.file_identity(out / f"rank{i}.json") for i in range(ctx.world_size)],
            "wall_seconds_max_rank": max(part["wall_seconds"] for part in parts),
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
            "scope": "Technical train-only feasibility and peak-LR stress, not LR optimality or scaling quality. No downstream score.",
            "timing_scope": "Production forward/backward/clip/AdamW/EMA update; rank-max components, diagnostics excluded. Loader overlap and checkpoint overhead absent.",
            "rng_scope": "Stream/masks/targets paired; different model shapes and dropout consume different model RNG.",
            "decision": "Review train-only logs and freeze training/evaluation budget before scientific 5000 run; no automatic submission."}
        P.write_json(out / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    code = P.code_identity(args.expected_code_sha)
    plan, source, configs, states = load_plan(args.plan)
    ctx = R.dist_from_env()
    require(ctx.world_size == 4 and torch.cuda.is_available(), "four CUDA ranks required")
    torch.cuda.set_device(ctx.local_rank)
    device = f"cuda:{ctx.local_rank}"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.distributed.init_process_group("nccl")
    try:
        roots = (ROOT, Path("/leonardo_work/IscrB_WearUsFM/wearusfm"), Path("/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns"))
        require(args.out.is_absolute() and not any(args.out.resolve().is_relative_to(p) for p in roots), "absolute output outside code required")
        bad, = R._any_flag(ctx, device, args.out.exists() or args.out.is_symlink())
        require(not bad, "output exists; no retry or overwrite")
        if ctx.is_main:
            args.out.mkdir(parents=True, exist_ok=False)
        torch.distributed.barrier()
        run(plan, source, configs, states, args.out, ctx, device, code)
    finally:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
