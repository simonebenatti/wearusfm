"""R6-8 skeptic: does observer work land inside the trainer's t_step_s / elapsed_s?

Real CPU trainer (single process, contrastive => _accumulate_terms path).
Run A: no observer. Run B: the six real P._measure_case probes at update U,
called from a patched _accumulate_terms exactly as observe_control does
(measure first, then the live backward). Run C: real trace_resume_updates.
"""
from contextlib import ExitStack
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

import contrastive_target_parity as P
import contrastive_target_resume as C
from contrastive_numeric_profile import numeric_profile
from test_pose_training import _pose_setup, _r3_config
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime

torch.set_num_threads(1)
tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=False)
root, manifest, pose_root = _pose_setup(tmp)
N, U = 12, 6


def cfg_for(arm):
    base = _r3_config(pose_root, total=N, keep=())
    contrast = ContrastiveConfig(arm=arm, nce_weight=.1, rnc_weight=.1, capacity_per_dataset=8, min_negatives=1,
                                 gap_seconds=.001, fill_steps=0, ramp_steps=1, rnc_stride=2)
    return replace(base, max_steps=N, batch_size=4, fixed_eval_every=0, eval_every=1000,
                   model=replace(base.model, muscle_dropout=0.),
                   loader=replace(base.loader, min_window_s=1., max_window_s=1.),
                   contrastive=asdict(contrast), datasets=("emg2pose",))


def records(folder):
    return [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines() if x.strip()]


def run_plain(out, cfg):
    with numeric_profile(require_cuda=False):
        return R.train(cfg, manifest, [root], out, log=lambda _: None)


def run_parity_like(out, cfg):
    prepare, backward = ContrastiveRuntime.prepare, R._accumulate_terms
    captured, timing = {}, {}

    def observed_prepare(runtime, batch, step):
        value = prepare(runtime, batch, step)
        if step + 1 == U:
            captured["batch"] = batch
        return value

    def observed_backward(model, student, teacher, micros, run_cfg, ctx, device, amp, **kw):
        runtime = kw.get("contrastive")
        if runtime is not None and runtime.step + 1 == U:
            began = time.perf_counter()
            batch = captured.pop("batch")
            saved_rng = P.rng_state(device)
            saved_runtime = copy.deepcopy(runtime.state_dict())
            s_state, t_state = P._cpu_state(student), P._cpu_state(teacher)
            before = P._fingerprints(s_state, t_state, saved_runtime["queue"], runtime.coverage, batch, micros, saved_rng)
            cases = [P._measure_case(name, run_cfg, s_state, t_state, saved_runtime, batch, micros, saved_rng, ctx,
                                     device, backward, prepare, grad_diag=kw.get("grad_diag", False), require_cuda=False)
                     for name in P.CASES]
            P.restore_rng(saved_rng, device)
            after = P._fingerprints(student.state_dict(), teacher.state_dict(), runtime.queue.state_dict(),
                                    runtime.coverage, batch, micros, P.rng_state(device))
            timing["observer_s"] = time.perf_counter() - began
            timing["case_backward_s"] = [c["backward_seconds"] for c in cases]
            timing["live_restored"] = before == after
            timing["unique_gradients"] = len({c["gradient_sha256"] for c in cases})
        return backward(model, student, teacher, micros, run_cfg, ctx, device, amp, **kw)

    with numeric_profile(require_cuda=False), ExitStack() as stack:
        stack.enter_context(patch.object(ContrastiveRuntime, "prepare", observed_prepare))
        stack.enter_context(patch.object(R, "_accumulate_terms", observed_backward))
        summary = R.train(cfg, manifest, [root], out, log=lambda _: None)
    return summary, timing


def run_trace(out, cfg):
    out.mkdir(parents=True)
    with numeric_profile(require_cuda=False):
        with C.trace_resume_updates(out / "trace-0.jsonl", "continuous", fork_step=4, end_step=N):
            return R.train(cfg, manifest, [root], out, log=lambda _: None)


cfg = cfg_for("C0")
# Warm-up run so that first-run import/JIT effects do not bias A.
run_plain(tmp / "warm", cfg)
sA = run_plain(tmp / "A", cfg)
sB, timing = run_parity_like(tmp / "B", cfg)
sC = run_trace(tmp / "C", cfg)
A, B, Ct = records(tmp / "A"), records(tmp / "B"), records(tmp / "C")
same = all(a["total"] == b["total"] and a["grad_norm"] == b["grad_norm"] for a, b in zip(A, B))
print("steps", [r["step"] for r in A] == [r["step"] for r in B] == list(range(1, N + 1)))
print("baseline and parity-like runs bit-identical in total/grad_norm:", same)
print("observer timing:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in timing.items()}))
print("t_step_s A :", [round(r["t_step_s"], 3) for r in A])
print("t_step_s B :", [round(r["t_step_s"], 3) for r in B])
print("t_step_s C :", [round(r["t_step_s"], 3) for r in Ct])
medA = sorted(r["t_step_s"] for r in A[2:])[len(A[2:]) // 2]
print(f"B[U] - median(A) = {B[U-1]['t_step_s'] - medA:.3f} s ; observer_s = {timing['observer_s']:.3f} s")
print(f"elapsed A {sA['elapsed_s']:.3f} B {sB['elapsed_s']:.3f} C {sC['elapsed_s']:.3f}; B-A {sB['elapsed_s']-sA['elapsed_s']:.3f}")
trace_rows = [json.loads(x) for x in (tmp / "C" / "trace-0.jsonl").read_text().splitlines()]
print("trace updates", [r["update"] for r in trace_rows], "boundary at", [r["update"] for r in trace_rows if "boundary" in r])
