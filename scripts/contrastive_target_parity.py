"""Prospective zero-weight test on the *live* full-batch control at updates 128/256.

An observer intercepts the trainer immediately before its backward. Six isolated
DDP clones use the real student AND EMA teacher, FIFO, batch, masks and rank RNG.
The live reducer, optimizer, queue and generator stream are never advanced by a
probe. No fixture batch, shifted queue ages or synthetic training state is used.
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import asdict, replace
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from contrastive_numeric_profile import (
    PROFILE, NumericProfileUnavailable, assert_numeric_profile, assert_numeric_flags,
)
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as trainer
from wearusfm.training.contrastive_objectives import ContrastiveRuntime, attach_contrastive_head
from wearusfm.training.jepa import make_teacher
from wearusfm.training.pairing import batch_sha256, state_sha256

UPDATES = (128, 256)
CASES = ("C0a", "CNa", "CRa", "CRb", "CNb", "C0b")
PAIRS = (("C0a", "C0b"), ("CNa", "CNb"), ("CRa", "CRb"),
         ("C0a", "CNa"), ("C0a", "CRa"), ("C0b", "CNb"), ("C0b", "CRb"))
MATCH_FIELDS = ("student_sha256", "teacher_sha256", "queue_sha256", "coverage_sha256",
                "batch_sha256", "micro_sha256", "mask_sha256", "cpu_rng_sha256",
                "cuda_rng_sha256", "python_rng_sha256", "numpy_rng_sha256")


def rng_state(device):
    device = torch.device(device)
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(saved, device):
    torch.set_rng_state(saved["cpu"])
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state(saved["cuda"], torch.device(device))
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])


def _cpu_state(model):
    return {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}


def _gradient_state(model):
    return {name: None if p.grad is None else p.grad.detach().to("cpu", copy=True)
            for name, p in model.named_parameters()}


def _fingerprints(student, teacher, queue, coverage, batch, micros, rng):
    return {"student_sha256": state_sha256(student), "teacher_sha256": state_sha256(teacher),
            "queue_sha256": state_sha256(queue), "coverage_sha256": state_sha256(coverage),
            "batch_sha256": batch_sha256(batch), "micro_sha256": trainer._micro_digest(micros),
            "mask_sha256": state_sha256([m.visible for m in micros]),
            **{name + "_rng_sha256": state_sha256(value) for name, value in rng.items()}}


def _all_gather(record, ctx):
    if not ctx.active:
        return [record]
    records = [None] * ctx.world_size
    torch.distributed.all_gather_object(records, record)
    return records


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device):
    if torch.device(device).type != "cuda":
        return {"cuda_measured": False}
    return {"cuda_measured": True,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "peak_scope": "cumulative process peak, not reset by observer"}


def _measure_case(name, cfg, student_state, teacher_state, saved_runtime, batch, micros,
                  saved_rng, ctx, device, backward, prepare, *, grad_diag=False, require_cuda=True):
    """Fresh reducer per case; all scientific state restored before each backward."""
    arm = name[:2]
    run_cfg = replace(cfg, contrastive={**cfg.contrastive, "arm": arm})
    student = attach_contrastive_head(WearUsFM(cfg.model)).to(device)
    student.load_state_dict(student_state, strict=True)
    teacher = make_teacher(student)
    teacher.load_state_dict(teacher_state, strict=True)
    student.train()
    model = trainer._ddp_wrap(student, run_cfg, ctx, device) if ctx.active else None
    runtime = ContrastiveRuntime(run_cfg.contrastive)
    runtime.expected_datasets = tuple(saved_runtime["expected_datasets"])
    cloned = copy.deepcopy(saved_runtime)
    cloned["config"] = asdict(runtime.cfg)
    step = cloned["queue"]["last_step"] + 1
    runtime.load_state_dict(cloned, completed_steps=step)
    restore_rng(saved_rng, device)
    before = _fingerprints(student.state_dict(), teacher.state_dict(), runtime.queue.state_dict(),
                           runtime.coverage, batch, micros, rng_state(device))
    prepare(runtime, batch, step)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda" else nullcontext()
    with amp:
        flags = assert_numeric_profile(require_cuda=require_cuda,
                                       require_autocast=torch.device(device).type == "cuda")
    _sync(device)
    began = time.perf_counter()
    failure, metrics = backward(model, student, teacher, micros, run_cfg, ctx, device, amp,
                                 contrastive=runtime, grad_diag=grad_diag)
    _sync(device)
    seconds = time.perf_counter() - began
    if failure is not None:
        raise RuntimeError("parity backward failed: " + str(failure))
    gradients = _gradient_state(student)
    if not gradients or not any(v is not None for v in gradients.values()):
        raise ValueError("parity backward produced no gradients")
    if any(v is not None and not bool(torch.isfinite(v).all()) for v in gradients.values()):
        raise ValueError("nonfinite parity gradient")
    norm = math.sqrt(sum(float(torch.linalg.vector_norm(v.double())) ** 2
                         for v in gradients.values() if v is not None))
    after_rng = rng_state(device)
    out = {"case": name, "arm": arm, "before": before,
           "after_rng": {key + "_rng_sha256": state_sha256(value) for key, value in after_rng.items()},
           "gradient_sha256": state_sha256(gradients), "gradient_norm": norm,
           "gradient_tensor_count": sum(v is not None for v in gradients.values()),
           "loss": float(metrics["total"]), "nce_weight": float(metrics["contrast_nce_weight"]),
           "rnc_weight": float(metrics["contrast_rnc_weight"]),
           "nce_eligible": float(metrics["contrast_nce_eligible"]),
           "rnc_windows": float(metrics["contrast_rnc_windows"]),
           "rnc_pairs": float(metrics["contrast_rnc_pairs"]),
           "global_windows": float(metrics["contrast_windows"]),
           "flags": flags, "backward_seconds": seconds, "memory": _memory(device)}
    # Deallocation before next case bounds GPU use to the live model plus one clone.
    del gradients, model, student, teacher, runtime, cloned
    return out


def assess_point(point):
    """Recompute the gate from evidence; never trust the serialized status."""
    errors, missing = [], []
    if point.get("update") not in UPDATES or point.get("state_completed_steps") != point.get("update", 0) - 1:
        errors.append("state is not immediately before the prescribed update")
    rows = point.get("ranks", [])
    if sorted(row.get("rank", -1) for row in rows) != list(range(4)):
        missing.append("four rank records")
    hashes = []
    replicated_inputs = []
    for row in rows:
        if row.get("profile_status") != "AVAILABLE":
            missing.append(f"required profile rank {row.get('rank')}")
            continue
        if row.get("global_batch_size") != 32:
            errors.append("production global batch differs from 32")
        if row.get("live_state_restored") is not True:
            errors.append("live control state was not restored")
        cases = row.get("cases", {})
        if set(cases) != set(CASES):
            missing.append("six backwards per rank")
            continue
        reference = cases["C0a"]
        if any(reference.get("before", {}).get(key) != row.get("live_before", {}).get(key) for key in MATCH_FIELDS):
            errors.append("probe state does not equal the live control state")
        replicated_inputs.append(tuple(reference.get("before", {}).get(key)
                                       for key in ("student_sha256", "teacher_sha256", "queue_sha256", "coverage_sha256")) +
                                 (row.get("global_ids_sha256"),))
        for name in CASES:
            case = cases[name]
            try:
                assert_numeric_flags(case.get("flags", {}), require_autocast=True)
            except NumericProfileUnavailable:
                missing.append("required profile " + name)
            if any(case.get("before", {}).get(key) != reference.get("before", {}).get(key)
                   or not isinstance(case.get("before", {}).get(key), str) for key in MATCH_FIELDS):
                errors.append("initial state/input/RNG differs " + name)
            if (set(case.get("after_rng", {})) != {"cpu_rng_sha256", "cuda_rng_sha256", "python_rng_sha256", "numpy_rng_sha256"}
                    or case.get("after_rng") != reference.get("after_rng")):
                errors.append("backward RNG consumption differs " + name)
            if case.get("nce_weight") != 0.0 or case.get("rnc_weight") != 0.0:
                errors.append("nonzero loss weight " + name)
            for key in ("nce_eligible", "rnc_windows", "rnc_pairs"):
                value = case.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    missing.append("loss support " + name + "/" + key)
            if case.get("global_windows") != 32:
                errors.append("backward global windows differs from 32")
            for key in ("loss", "gradient_norm", "backward_seconds"):
                value = case.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or (key != "loss" and value <= 0):
                    errors.append("invalid measurement " + name + "/" + key)
            digest = case.get("gradient_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append("missing gradient digest " + name)
            hashes.append(digest)
            memory = case.get("memory", {})
            if memory.get("cuda_measured") is not True or memory.get("peak_allocated_bytes", 0) <= 0:
                missing.append("full-batch CUDA memory measurement " + name)
        # Same rank seed, actual before/after; different ranks may intentionally use different RNG.
    if replicated_inputs and len(set(replicated_inputs)) != 1:
        errors.append("model/teacher/queue/global identities differ across ranks")
    if hashes and len(set(hashes)) != 1 and not missing:
        errors.append("reduced gradients are not byte-identical within/between arms/ranks")
    status = "FAILED" if errors else "INCONCLUSIVE" if missing else "PASSED"
    return {"status": status, "errors": sorted(set(errors)), "inconclusive": sorted(set(missing)),
            "within_and_between_byte_identical": bool(hashes) and len(set(hashes)) == 1,
            "unique_gradient_hashes": sorted({h for h in hashes if isinstance(h, str)})}


def validate_parity_report(report):
    points = report.get("points", [])
    if report.get("schema") != 1 or report.get("profile") != PROFILE:
        return {"status": "INCONCLUSIVE", "errors": [], "inconclusive": ["required parity report/profile missing"]}
    if [p.get("update") for p in points] != list(UPDATES):
        return {"status": "INCONCLUSIVE", "errors": [], "inconclusive": ["both prescribed real control states required"]}
    checks = [assess_point(point) for point in points]
    return {"status": "FAILED" if any(c["status"] == "FAILED" for c in checks) else
                      "INCONCLUSIVE" if any(c["status"] == "INCONCLUSIVE" for c in checks) else "PASSED",
            "points": checks, "training_approved": False}


class ControlParityObserver:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.points = []
        self.report = None
        self.captured = {}
        self.active = False
        self.config_sha256 = None

    def measure(self, model, student, teacher, micros, cfg, ctx, device, amp, backward, prepare, **kwargs):
        runtime = kwargs.get("contrastive")
        if runtime is None or runtime.cfg.arm != "C0" or runtime.step + 1 not in UPDATES:
            return
        update = runtime.step + 1
        cfg_digest = hashlib.sha256(json.dumps(trainer.config_to_dict(cfg), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if self.config_sha256 not in (None, cfg_digest):
            raise ValueError("live control configuration changed between prescribed points")
        self.config_sha256 = cfg_digest
        if cfg.batch_size != 32 or cfg.accum_steps != 1 or cfg.amp_bf16 is not True or ctx.world_size != 4:
            raise ValueError("parity requires production C0 batch32/accum1/bf16/four ranks")
        if update in [p["update"] for p in self.points]:
            raise ValueError("duplicate parity control update")
        if id(runtime) not in self.captured:
            raise ValueError("parity observer did not see the consumed live batch")
        batch = self.captured.pop(id(runtime))
        if batch.index != runtime.step:
            raise ValueError("live batch is not immediately before the requested control update")
        if len(runtime.global_ids) != 32:
            raise ValueError("live production batch does not contain 32 windows")
        saved_rng = rng_state(device)
        saved_runtime = copy.deepcopy(runtime.state_dict())
        student_state, teacher_state = _cpu_state(student), _cpu_state(teacher)
        before = _fingerprints(student_state, teacher_state, saved_runtime["queue"], runtime.coverage,
                               batch, micros, saved_rng)
        saved_grads = _gradient_state(student)
        row = {"rank": ctx.rank, "update": update, "profile_status": "AVAILABLE", "cases": {},
               "global_batch_size": len(runtime.global_ids),
               "datasets": sorted({i.dataset for i in runtime.global_ids}),
               "global_ids_sha256": state_sha256(runtime.global_ids),
               "live_before": before}
        started = time.perf_counter()
        try:
            try:
                assert_numeric_profile(require_cuda=True)
            except NumericProfileUnavailable as exc:
                row.update(profile_status="UNAVAILABLE", profile_error=str(exc))
            available = _all_gather(row["profile_status"], ctx)
            if all(value == "AVAILABLE" for value in available):
                for name in CASES:
                    row["cases"][name] = _measure_case(
                        name, cfg, student_state, teacher_state, saved_runtime, batch, micros,
                        saved_rng, ctx, device, backward, prepare, grad_diag=kwargs.get("grad_diag", False))
        finally:
            restore_rng(saved_rng, device)
            # Probe clones never share parameter storage with the live trainer.
            after = _fingerprints(student.state_dict(), teacher.state_dict(), runtime.queue.state_dict(),
                                  runtime.coverage, batch, micros, rng_state(device))
            row["live_state_restored"] = before == after and state_sha256(saved_grads) == state_sha256(_gradient_state(student))
            row["observer_seconds_including_setup"] = time.perf_counter() - started
        point = {"update": update, "state_completed_steps": update - 1,
                 "ranks": _all_gather(row, ctx)}
        point["gate"] = assess_point(point)
        self.points.append(point)
        if not all(r["live_state_restored"] for r in point["ranks"]):
            raise RuntimeError("parity probe modified the live control state")

    def finish(self, ctx):
        from contrastive_prepare import source_digest
        if ctx.active:
            torch.distributed.barrier()  # rank0 has finished writing the trainer summary.
        summary_path = self.out_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        contrastive = summary.get("contrastive", {})
        report = {"schema": 1, "profile": PROFILE,
                  "config_sha256": self.config_sha256,
                  "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest() if summary_path.exists() else None,
                  "initial_state_sha256": contrastive.get("initial_sha256"),
                  "provenance": contrastive.get("provenance"), "required_updates": list(UPDATES),
                  "case_order": list(CASES), "comparison_pairs": [list(p) for p in PAIRS],
                  "source_sha256": source_digest(), "points": self.points,
                  "measurement": "live C0 pre-update states; six isolated reducers; production global batch32",
                  "training_approved": False}
        report["gate"] = validate_parity_report(report)
        report["status"] = report["gate"]["status"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        local = {"schema": 1, "rank": ctx.rank,
                 "points": [{"update": p["update"], "record": next(r for r in p["ranks"] if r["rank"] == ctx.rank)}
                            for p in self.points]}
        with (self.out_dir / f"zero_weight_parity.rank{ctx.rank}.json").open("x") as out:
            json.dump(local, out, indent=2, allow_nan=False)
            out.write("\n")
        if ctx.is_main:
            with (self.out_dir / "zero_weight_parity.json").open("x") as out:
                json.dump(report, out, indent=2, allow_nan=False)
                out.write("\n")
        self.report = report
        return report


@contextmanager
def observe_control(out_dir):
    """Wrap only a new full C0 benchmark; a missing point is explicitly inconclusive."""
    observer = ControlParityObserver(out_dir)
    prepare, backward = ContrastiveRuntime.prepare, trainer._accumulate_terms

    def observed_prepare(runtime, batch, step):
        result = prepare(runtime, batch, step)
        if not observer.active and runtime.cfg.arm == "C0" and step + 1 in UPDATES:
            observer.captured[id(runtime)] = batch
        return result

    def observed_backward(model, student, teacher, micros, cfg, ctx, device, amp, **kwargs):
        if not observer.active:
            observer.active = True
            try:
                observer.measure(model, student, teacher, micros, cfg, ctx, device, amp,
                                 backward, prepare, **kwargs)
            finally:
                observer.active = False
        return backward(model, student, teacher, micros, cfg, ctx, device, amp, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(ContrastiveRuntime, "prepare", observed_prepare))
        stack.enter_context(patch.object(trainer, "_accumulate_terms", observed_backward))
        yield observer
    observer.finish(trainer.dist_from_env())
