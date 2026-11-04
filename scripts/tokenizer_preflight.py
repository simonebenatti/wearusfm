"""Train-only tokenizer collaudo: three fresh arms, one delivered stream, no checkpoints.

The distributed forward/backward, loss denominators, optimizer, clipping, EMA,
and step-indexed schedule are the existing trainer's. Diagnostics are outside
the timed update. This is a technical run, not a scientific training endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from wearusfm.model.fm import WearUsFM, zero_hidden_samples
from wearusfm.training import jepa as J
from wearusfm.training import run as R
from wearusfm.training.pairing import batch_sha256, state_sha256

PROTOCOL = "tokenizer-preflight-20261103-v1"
ARMS = ("T25-L", "T100-L", "T100-G")
REPO = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_identity(path):
    path = Path(path)
    before = path.stat()
    digest = R.sha256_file(path)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), f"input changed while hashing: {path}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": after.st_size,
            "inode": after.st_ino, "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns}


def pinned_file(pin, root=REPO):
    path = Path(pin["path"])
    if not path.is_absolute():
        path = root / path
    identity = file_identity(path)
    require(identity["sha256"] == pin["sha256"], f"SHA mismatch: {path}")
    return identity


def code_identity(expected):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    require(len(expected) == 40 and actual == expected, "wrong code commit")
    require(subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "scripts"], cwd=REPO).returncode == 0,
            "runtime code differs from frozen commit")
    return actual


def load_plan(path, root=REPO):
    plan = json.loads(Path(path).read_text())
    require(plan.get("protocol") == PROTOCOL, "wrong preflight protocol")
    require(type(plan.get("batches")) is int and plan["batches"] == 64, "this allocation covers exactly 64 batches")
    require(plan.get("global_batch_size") == 32 and plan.get("world_size") == 4 and plan.get("num_workers") == 6,
            "preflight requires historical global batch32, world4 and 6 workers/rank")
    require(set(plan["inputs"]) == {"manifest", "scales", "windows", "pose_stats"}, "incomplete input pins")
    states = {"plan": file_identity(path), "comparison": pinned_file(plan["comparison"], root)}
    states.update({name: pinned_file(pin, root) for name, pin in plan["inputs"].items()})
    comparison = json.loads(Path(states["comparison"]["path"]).read_text())
    require(set(comparison["arm_configurations"]) == set(ARMS), "wrong arm set")
    spec = importlib.util.spec_from_file_location("tokenizer_preflight_trainer_cli", REPO / "scripts/train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    configs = {}
    for name in ARMS:
        entry = comparison["arm_configurations"][name]
        states[name] = pinned_file({"path": entry["config"], "sha256": entry["sha256"]}, root)
        argv = entry["validated_trainer_argv"]
        require(argv[0] == "scripts/train_jepa.py", "unexpected trainer executable")
        args = cli.build_parser().parse_args(argv[1:])
        cfg = cli.build_config(args)
        require(R.config_to_dict(cfg) == R._config_with_defaults(json.loads(Path(states[name]["path"]).read_text())),
                "CLI/config mismatch: " + name)
        for arg, key in (("manifest", "manifest"), ("scales", "scales"), ("window_seconds", "windows")):
            require(str(Path(getattr(args, arg)).resolve()) == states[key]["path"], f"CLI/plan input mismatch: {name}/{key}")
        require([str(p) for p in args.root] == plan["roots"], "CLI/plan roots mismatch")
        require(str((Path(cfg.loader.pose_root) / "pose_stats.json").resolve()) == states["pose_stats"]["path"], "pose statistics mismatch")
        require(cfg.batch_size == 32 and cfg.accum_steps == 1 and cfg.seed == 0 and args.num_workers == 6 and args.distributed,
                "unexpected stream or distributed recipe")
        require(cfg.model.dim == 384 and cfg.model.k_latents == 64 and cfg.model.pose_head and not cfg.model.rvq_codes,
                "not the frozen 30M pose recipe")
        require(cfg.jepa.jepa_weight == 0 and cfg.jepa.anchor_weight == .2 and cfg.jepa.pose_weight == 1 and
                cfg.jepa.masked_loss_radius == 0 and cfg.jepa.loss_per_window and cfg.model.multiscale_anchor,
                "unexpected objective or query support")
        configs[name] = cfg
    baseline = R.config_to_dict(configs[ARMS[0]])
    for name, window, activation in (("T25-L", None, "identity"), ("T100-L", 100., "identity"), ("T100-G", 100., "gelu")):
        actual = R.config_to_dict(configs[name])
        require(actual["model"]["window_ms"] == window and actual["model"]["tokenizer_activation"] == activation,
                "arm identity mismatch: " + name)
        actual["model"]["window_ms"] = baseline["model"]["window_ms"]
        actual["model"]["tokenizer_activation"] = baseline["model"]["tokenizer_activation"]
        require(actual == baseline, "unregistered difference between arms: " + name)
    return plan, configs, states


def check_unchanged(states):
    for key, before in states.items():
        require(file_identity(before["path"]) == before, "input changed: " + key)


class FeatureMoments:
    """Mergeable FP64 moments; no sampled subset and no data-dependent threshold."""
    def __init__(self):
        self.groups = {}

    @torch.no_grad()
    def add(self, key, values, activation):
        x = values.detach().double().reshape(-1, 64)
        require(bool(torch.isfinite(x).all()), "nonfinite real frontend response")
        if not len(x):
            return
        gelu = torch.nn.functional.gelu(values.detach()).double().reshape(-1, 64)  # actual preprojection dtype, then FP64 accumulation
        require(bool(torch.isfinite(gelu).all()), "nonfinite GELU response")
        y = gelu if activation == "gelu" else x
        result = {"n": len(x), "x_sum": x.sum(0), "x_square": x.square().sum(0), "x_cross": x.T @ x,
                  "y_sum": y.sum(0), "y_square": y.square().sum(0), "gelu_minus_half_square": (gelu - .5 * x).square().sum(0)}
        result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        self.merge({key: result})

    def merge(self, groups):
        for key, values in groups.items():
            if key not in self.groups:
                self.groups[key] = {k: np.asarray(v).copy() if k != "n" else int(v) for k, v in values.items()}
            else:
                for field, value in values.items():
                    self.groups[key][field] += value

    def serializable(self):
        return {key: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()}
                for key, values in self.groups.items()}

    def summary(self):
        result = {}
        for key, group in self.groups.items():
            n = group["n"]
            mx, my = group["x_sum"] / n, group["y_sum"] / n
            vx, vy = np.maximum(group["x_square"] / n - mx ** 2, 0), np.maximum(group["y_square"] / n - my ** 2, 0)
            cov = group["x_cross"] / n - np.outer(mx, mx)
            eigen = np.maximum(np.linalg.eigvalsh((cov + cov.T) / 2), 0)[::-1]
            singular = np.sqrt(eigen)
            record = {"n_tokens": n, "scaled_mean": mx.tolist(), "scaled_std": np.sqrt(vx).tolist(),
                      "activated_mean": my.tolist(), "activated_std": np.sqrt(vy).tolist(), "families": {},
                      "gelu_minus_half_x_rms": np.sqrt(group["gelu_minus_half_square"] / n).tolist(),
                      "scaled_centered_singular_values": singular.tolist(),
                      "rank_relative_1e2": int(np.sum(singular > singular[0] * .01)) if singular[0] else 0,
                      "rank_relative_1e3": int(np.sum(singular > singular[0] * .001)) if singular[0] else 0}
            for name, section in (("fourier", slice(0, 24)), ("spline", slice(24, 48)), ("mlp", slice(48, 64))):
                den = float(group["x_square"][section].sum())
                record["families"][name] = {"median_scaled_std": float(np.median(np.sqrt(vx[section]))),
                    "median_activated_std": float(np.median(np.sqrt(vy[section]))),
                    "scaled_mean": float(mx[section].mean()), "activated_mean": float(my[section].mean()),
                    "gelu_minus_half_x_rms_over_x_rms": math.sqrt(float(group["gelu_minus_half_square"][section].sum()) / den)
                    if den > 0 else None}
            result[key] = record
        return result


@torch.no_grad()
def collect_features(student, micro, rows, step, moments):
    """Same student weights, before update; both modes use the same visible/QC support."""
    inp = micro.inp
    masked = zero_hidden_samples(inp.signals, inp.fs, micro.visible, inp.counts, student.tokenizer.patch_s)
    offset = 0
    for row, fs, count, raw, hidden in zip(rows, inp.fs, inp.counts, inp.signals, masked):
        for mode, signal in (("masked_visible", hidden), ("unmasked_same_visible", raw)):
            # Exactly the conversion used before projection in PatchTokenizer.
            x = (student.tokenizer.frontend(signal[None], float(fs))[0] * student.tokenizer.fe_gain).to(student.tokenizer.proj.weight.dtype)
            support = micro.visible[offset:offset + count, :x.shape[1]] & inp.qc_valid[offset:offset + count, None]
            values = x[support]
            for phase in ("trajectory_preupdate", "initial") if step == 0 else ("trajectory_preupdate",):
                moments.add(f"{phase}/{row['dataset']}/{float(fs):g}/{mode}", values, student.cfg.tokenizer_activation)
        offset += count
    return state_sha256(masked)


def support_audit(micro, cfg, rows):
    inp = micro.inp
    lengths = [int(math.floor(x.shape[-1] / fs / (cfg.model.patch_ms / 1000) + 1e-9)) for x, fs in zip(inp.signals, inp.fs)]
    per_ch = torch.repeat_interleave(torch.tensor(lengths, device=micro.visible.device), torch.tensor(inp.counts, device=micro.visible.device))
    real = torch.arange(micro.visible.shape[1], device=micro.visible.device)[None, :] < per_ch[:, None]
    visible = micro.visible & real & inp.qc_valid[:, None]
    hidden = ~micro.visible & real & inp.qc_valid[:, None]
    hidden = J.core_hidden(hidden, visible, J.masked_query_radius(cfg.jepa, cfg.model))
    ch, qt, offsets = J.hidden_queries(hidden, inp.counts)
    targets = J.gather_anchor_targets(micro.anchor_targets, inp.counts, ch, qt)
    masks = {"log_rms": targets["rms_valid"], "log_env": targets["env_valid"],
             "band_shape": targets["spec_valid"][:, None] & targets["band_available"]}
    masks.update({name: targets[name + "_valid"][:, None] & targets[name + "_available"]
                  for name, *_ in J.MS_SCALES if name in targets})
    counts = J.loss_counts(inp, micro.visible, cfg.jepa, micro.anchor_targets, model_cfg=cfg.model,
                          pose_valid=None if micro.pose is None else micro.pose[1])
    pose_support = None if micro.pose is None else micro.pose[1] & J.pose_time_valid(real, inp.counts)[..., None]
    windows = []
    for s, row in enumerate(rows):
        start, end = int(offsets[s]), int(offsets[s + 1])
        kinds = micro.kinds[ch[start:end], qt[start:end]].detach().cpu().tolist()
        windows.append({"dataset": row["dataset"], "fs": float(inp.fs[s]), "n_patches": lengths[s], "queries": end - start,
                        "query_kind_counts": dict(Counter(map(str, kinds))),
                        "anchor_valid_elements": {k: int(v[start:end].sum()) for k, v in masks.items()},
                        "pose_valid_elements": 0 if pose_support is None else int(pose_support[s].sum())})
    expected = {"ch": ch, "time": qt, "offsets": offsets, "visible": visible, "patch_valid": real}
    return {"query_support_sha256": state_sha256(expected), "effective_anchor_targets_sha256": state_sha256(targets),
            "pose_support_sha256": state_sha256(pose_support), "loss_counts": {k: int(v) for k, v in counts.items()},
            "windows": windows}, expected


def selected_parameters(student):
    groups = {name: {} for name in ("fourier", "spline", "mlp", "projection", "pose", "backbone_first")}
    for name, parameter in student.named_parameters():
        group = ("fourier" if name == "tokenizer.frontend.fourier_coef" else
                 "spline" if name == "tokenizer.frontend.spline_coef" else
                 "mlp" if name.startswith("tokenizer.frontend.mlp.") else
                 "projection" if name.startswith("tokenizer.proj.") else
                 "pose" if name.startswith("pose_head.") else None)
        if group is None and name.startswith("backbone.") and not groups["backbone_first"]:
            group = "backbone_first"
        if group is not None:
            groups[group][name] = parameter
    require(all(groups.values()), "missing gradient/update group")
    return groups


def norm_of(tensors):
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))


def rng_state(device):
    return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(device).clone() if str(device).startswith("cuda") else None}


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def create_arms(configs, ctx, device):
    arms = {}
    for name in ARMS:
        cfg = configs[name]
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
        state = student.state_dict()
        initial = {"shared_sha256": state_sha256({k: v for k, v in state.items()
                    if not k.startswith("tokenizer.frontend.") and k != "tokenizer.fe_gain"}),
                   "frontend_and_gain_sha256": state_sha256({k: v for k, v in state.items()
                    if k.startswith("tokenizer.frontend.") or k == "tokenizer.fe_gain"}),
                   "all_sha256": state_sha256(state), "parameters": sum(p.numel() for p in student.parameters()),
                   "fe_gain": float(student.tokenizer.fe_gain)}
        arm = {"cfg": cfg, "student": student, "teacher": teacher, "optimizer": optimizer, "wrapped": wrapped,
               "rng": rng_state(device), "initial": initial, "features": FeatureMoments(), "observed": [],
               "groups": selected_parameters(student), "records": []}
        # Capture references only: no hash/synchronization inside the timed forward.
        original_decode = student.decode

        def capture(enc, ch, qt, offsets, arm=arm, decode=original_decode):
            arm["observed"].append({"ch": ch.detach(), "time": qt.detach(), "offsets": offsets.detach(),
                                    "visible": enc.visible.detach(), "patch_valid": enc.patch_valid.detach()})
            return decode(enc, ch, qt, offsets)

        student.decode = capture
        arms[name] = arm
    require(len({a["initial"]["shared_sha256"] for a in arms.values()}) == 1, "shared initialization differs between arms")
    require(arms["T100-L"]["initial"]["all_sha256"] == arms["T100-G"]["initial"]["all_sha256"], "100ms initial weights/gain differ")
    require(len({state_sha256(a["rng"]) for a in arms.values()}) == 1, "initial RNG differs between arms")
    return arms


def update_arm(arm, micro, rows, step, ctx, device):
    cfg, student, teacher, optimizer = (arm[k] for k in ("cfg", "student", "teacher", "optimizer"))
    restore_rng(arm["rng"], device)
    diagnostic_start = time.perf_counter()
    audit, expected = support_audit(micro, cfg, rows)
    audit["masked_model_input_sha256"] = collect_features(student, micro, rows, step, arm["features"])
    before = {name: p.detach().clone() for group in arm["groups"].values() for name, p in group.items()}
    arm["observed"].clear()
    student.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = R._lr_at(step, cfg)
    sync(device)
    if ctx.active:
        torch.distributed.barrier()  # diagnostic work on slower ranks stays outside the update timer
    diagnostic_before_s = time.perf_counter() - diagnostic_start
    if str(device).startswith("cuda"):
        memory_baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memory_baseline = 0
    start = time.perf_counter()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") and cfg.amp_bf16 else nullcontext()
    failure, losses = R._accumulate_terms(arm["wrapped"], student, teacher, [micro], cfg, ctx, device, amp)
    require(failure is None, f"trainer rejected batch: {failure}")
    sync(device)
    forward_backward_s = time.perf_counter() - start
    diagnostic_start = time.perf_counter()
    gradients = {group: norm_of(p.grad for p in params.values() if p.grad is not None) for group, params in arm["groups"].items()}
    require(all(math.isfinite(v) for v in gradients.values()), "nonfinite family gradient")
    sync(device)
    if ctx.active:
        torch.distributed.barrier()  # exclude other ranks' gradient diagnostics too
    gradient_diagnostic_s = time.perf_counter() - diagnostic_start
    start = time.perf_counter()
    gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
    require(bool(torch.isfinite(gnorm)), "nonfinite full gradient")
    optimizer.step()
    J.ema_update(teacher, student, cfg.jepa.ema_momentum)
    sync(device)
    optimizer_ema_s = time.perf_counter() - start
    diagnostic_start = time.perf_counter()
    arm["rng"] = rng_state(device)
    require(len(arm["observed"]) == 1, "unexpected decoder call count")
    observed = arm["observed"][0]
    require(all(torch.equal(observed[key], value) for key, value in expected.items()), "actual decoder support differs from precount")
    updates = {group: norm_of(p.detach() - before[name] for name, p in params.items()) for group, params in arm["groups"].items()}
    require(all(math.isfinite(v) for v in updates.values()), "nonfinite parameter update")
    # AdamW updates all parameters; check even those outside the diagnostic groups.
    require(all(bool(torch.isfinite(p).all()) for p in student.parameters()), "nonfinite updated parameter")
    audit["actual_query_support_sha256"] = state_sha256(observed)
    record = {"step": step + 1, "lr": R._lr_at(step, cfg), "losses": losses, "grad_norm_before_clip": float(gnorm),
              "gradient_norms_before_clip": gradients, "update_norms": updates, "support": audit,
              "forward_backward_s": forward_backward_s, "clip_optimizer_ema_s": optimizer_ema_s,
              "normal_update_s": forward_backward_s + optimizer_ema_s,
              "memory_baseline_allocated_bytes": memory_baseline,
              "memory_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
              "memory_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else 0,
              "diagnostic_s": diagnostic_before_s + gradient_diagnostic_s + time.perf_counter() - diagnostic_start,
              "rng_after_sha256": state_sha256(arm["rng"])}
    arm["records"].append(record)
    arm["observed"].clear()
    return record


def verify_paired(records):
    require(set(records) == set(ARMS), "missing arm in paired batch")
    reference = records[ARMS[0]]
    for name in ARMS[1:]:
        require(records[name]["support"] == reference["support"], "batch/query/target support parity failure: " + name)
        require(records[name]["rng_after_sha256"] == reference["rng_after_sha256"], "training RNG consumption differs: " + name)


class SourceGuard:
    """Stat guard from first delivered batch; byte identities of delivered data are separate."""
    def __init__(self, index, pose_root):
        self.index, self.pose_root, self.states = index, Path(pose_root), {}

    @staticmethod
    def stat(path):
        s = path.stat()
        return {"bytes": s.st_size, "inode": s.st_ino, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}

    def observe(self, rows):
        from wearusfm.data import pose_targets as PT
        for row in rows:
            require(row["split"] == "pretraining" and row["dataset"] not in ("epn612", "uci", "uci_emg"), "nontraining row delivered")
            session = self.index.path_of(row)
            paths = [session / "metadata.json", session / "data_int16.npy"]
            if row["dataset"] == PT.POSE_DATASET:
                pose = PT.pose_dir(self.pose_root, session)
                paths += [pose / PT.POSE_FILE, pose / PT.VALID_FILE]
                if (pose / PT.META_FILE).exists():
                    paths.append(pose / PT.META_FILE)
            for path in paths:
                state = self.stat(path)
                require(str(path) not in self.states or self.states[str(path)] == state, "source changed: " + str(path))
                self.states[str(path)] = state

    def finish(self):
        for path, state in self.states.items():
            require(self.stat(Path(path)) == state, "source changed: " + path)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(plan, configs, states, out, ctx, device, code_sha):
    cfg = configs[ARMS[0]]
    manifest = Path(states["manifest"]["path"])
    scales = json.loads(Path(states["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(states["windows"]["path"]).read_text())["window_s"]
    R.check_input_files(manifest, Path(states["windows"]["path"]), cfg.nested, states["manifest"]["sha256"])
    index = R.load_index(manifest, [Path(p) for p in plan["roots"]], cfg.datasets, cfg.nested)
    if cfg.nested is not None:
        R.check_derived_binding(index.params, cfg, windows)
    require(all(row["split"] == "pretraining" for row in index.rows), "index includes validation")
    guard = SourceGuard(index, cfg.loader.pose_root)
    began = time.perf_counter()
    arms = create_arms(configs, ctx, device)
    initial = {name: arm["initial"] for name, arm in arms.items()}
    peers = [None] * ctx.world_size
    if ctx.active:
        torch.distributed.all_gather_object(peers, initial)
        require(all(part == initial for part in peers), "initialization differs across ranks")
    loader = torch.utils.data.DataLoader(R.BatchStream(index, cfg.loader, cfg.batch_size // ctx.world_size, cfg.seed,
             scales, windows, rank=ctx.rank if ctx.active else None, start=0, segment=cfg.stream_segment),
             batch_size=None, num_workers=plan["num_workers"], persistent_workers=plan["num_workers"] > 0,
             generator=torch.Generator().manual_seed(cfg.seed), collate_fn=R._to_shared if plan["num_workers"] > 0 else None)
    iterator = iter(loader)
    delivered, exposure = [], Counter()
    with (out / f"rank{ctx.rank}.jsonl").open("x") as stream:
        for step in range(plan["batches"]):
            expired, = R._any_flag(ctx, device, time.perf_counter() - began > 1500)
            require(not expired, "preflight exceeded 25-minute internal deadline")
            start = time.perf_counter()
            batch = R._from_shared(next(iterator)) if plan["num_workers"] > 0 else next(iterator)
            require(batch.index == step and len(batch.signals) == cfg.batch_size // ctx.world_size, "unexpected delivered batch identity")
            data_s = time.perf_counter() - start
            start = time.perf_counter()
            guard.observe(batch.rows)
            baseline = batch_sha256(batch)
            fingerprints = {"batch_sha256": baseline, "processed_signals_sha256": state_sha256(batch.signals),
                "input_masks_sha256": state_sha256((batch.visible, batch.kind, batch.qc_valid)),
                "anchor_targets_sha256": state_sha256(batch.anchor_targets), "pose_targets_sha256": state_sha256(batch.pose),
                "origin_sha256": state_sha256((batch.rows, batch.windows, batch.presented))}
            micros = R._stage(batch, cfg.accum_steps, device)
            require(len(micros) == 1, "preflight recipe requires one microbatch")
            micro = micros[0]
            micro_before = R._micro_digest(micros)
            prepare_s = time.perf_counter() - start
            records = {name: update_arm(arms[name], micro, batch.rows, step, ctx, device) for name in ARMS}
            verify_paired(records)
            require(batch_sha256(batch) == baseline and R._micro_digest(micros) == micro_before, "training mutated delivered inputs")
            for row, fs in zip(batch.rows, batch.fs):
                exposure[f"{row['dataset']}/{float(fs):g}"] += 1
            item = {"step": step + 1, "rank": ctx.rank, "fingerprints": fingerprints, "data_wait_s": data_s,
                    "prepare_and_hash_s": prepare_s, "arms": records}
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            delivered.append(fingerprints)
            if ctx.is_main:
                print(json.dumps({"step": step + 1, "normal_update_s": {n: round(r["normal_update_s"], 3) for n, r in records.items()},
                                  "paired": True}), flush=True)
    # All tensors are deliberately discarded; no scientific checkpoint is written.
    guard.finish()
    check_unchanged(states)
    require(code_identity(code_sha) == code_sha, "code changed during preflight")
    for name, arm in arms.items():
        for group in arm["groups"]:
            require(any(rec["gradient_norms_before_clip"][group] > 0 for rec in arm["records"]), f"zero gradients throughout: {name}/{group}")
            require(any(rec["update_norms"][group] > 0 for rec in arm["records"]), f"no actual updates: {name}/{group}")
    raw = {"rank": ctx.rank, "initial": initial, "exposure_windows": dict(exposure), "delivered": delivered,
           "wall_s": time.perf_counter() - began, "rss_main_process_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
           "sources_stat_first_seen": guard.states, "sources_unchanged_after_first_seen": True,
           "arms": {name: {"records": arm["records"], "feature_moments": arm["features"].serializable()} for name, arm in arms.items()}}
    write_json(out / f"rank{ctx.rank}.json", {"protocol": PROTOCOL, "complete": True, **raw})
    if ctx.active:
        torch.distributed.barrier()
    if ctx.is_main:
        parts = [json.loads((out / f"rank{rank}.json").read_text()) for rank in range(ctx.world_size)]
        merged_exposure = Counter()
        for part in parts:
            require(part["complete"] and len(part["delivered"]) == plan["batches"], "incomplete rank result")
            merged_exposure.update(part["exposure_windows"])
        results = {}
        for name in ARMS:
            moments = FeatureMoments()
            for part in parts:
                moments.merge(part["arms"][name]["feature_moments"])
            times = [sum(max(part["arms"][name]["records"][i][component] for part in parts)
                         for component in ("forward_backward_s", "clip_optimizer_ema_s")) for i in range(plan["batches"])]
            results[name] = {"initial": initial[name], "features": moments.summary(), "normal_update_rank_max_seconds": times,
                "normal_update_median_after_first8_s": float(np.median(times[8:])) if len(times) > 8 else None,
                "normal_update_mean_after_first8_s": float(np.mean(times[8:])) if len(times) > 8 else None,
                "normal_update_p90_after_first8_s": float(np.quantile(times[8:], .9)) if len(times) > 8 else None,
                "max_allocated_bytes_three_arms_resident": max(rec["memory_peak_allocated_bytes"] for part in parts for rec in part["arms"][name]["records"])}
        report = {"protocol": PROTOCOL, "complete": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Train-only technical collaudo; no validation, checkpoint, endpoint score, or generalization verdict.",
            "code_sha": code_sha, "plan_sha256": states["plan"]["sha256"], "inputs": states,
            "batches": plan["batches"], "global_batch_size": cfg.batch_size, "windows_exposed_each_arm": plan["batches"] * cfg.batch_size,
            "paired_stream_query_targets_and_pose_support": True, "exposure_windows": dict(merged_exposure), "arms": results,
            "timing_scope": "Existing trainer update, with DDP barriers outside timers after diagnostics; global time sums rank maxima of forward/backward and clip/optimizer/EMA separately. Input preparation excluded, no loader/GPU overlap. Diagnostics give workers extra time, so observed data_wait_s does not estimate the steady-state loader bottleneck. Three models resident; not a standalone-training memory benchmark.",
            "memory_scope": "Absolute allocator peaks include all three students, EMA teachers and optimizer states; incremental baselines in rank logs. They are not the single-arm peak.",
            "feature_scope": "Scaled FP32 preprojection features, identical visible/QC positions for masked and unmasked student input. Initial means step0 only; trajectory pools preupdate steps0..63. GELU residual is descriptive, never a pass/fail threshold.",
            "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "threads": torch.get_num_threads(),
                "world_size": ctx.world_size, "gpu": torch.cuda.get_device_name(device), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "loaded_modules": os.environ.get("LOADEDMODULES"), "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "packages": sorted(f"{p.metadata['Name']}=={p.version}" for p in importlib.metadata.distributions() if p.metadata.get("Name"))},
            "rank_reports": [file_identity(out / f"rank{i}.json") for i in range(ctx.world_size)],
            "wall_seconds_max_rank": max(part["wall_s"] for part in parts)}
        write_json(out / "report.json", report)
    if ctx.active:
        torch.distributed.barrier()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    code = code_identity(args.expected_code_sha)
    plan, configs, states = load_plan(args.plan)
    ctx = R.dist_from_env()
    require(ctx.world_size == plan["world_size"] and torch.cuda.is_available(), "production preflight requires four CUDA ranks")
    torch.cuda.set_device(ctx.local_rank)
    device = f"cuda:{ctx.local_rank}"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.distributed.init_process_group("nccl")
    try:
        # Every rank checks before rank0 creates the directory; no existing output is reused.
        exists = args.out.exists() or args.out.is_symlink()
        bad, = R._any_flag(ctx, device, exists)
        require(not bad, "output must be new")
        if ctx.is_main:
            args.out.mkdir(parents=True, exist_ok=False)
        torch.distributed.barrier()
        run(plan, configs, states, args.out, ctx, device, code)
    finally:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
