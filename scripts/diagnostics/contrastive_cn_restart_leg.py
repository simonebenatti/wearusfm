"""One diagnostic leg: an authenticated CN restart on its original training path."""
import argparse
import copy
import ctypes
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import socket
import sys
from unittest.mock import patch
import uuid

import torch  # Historical torch-first initialization.
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import contrastive_run as production
from contrastive_target_resume import write
from wearusfm.training import run as trainer
from wearusfm.training.pairing import state_sha256

ARCHIVED_JOB = ROOT / "results/infra/contrastive_targets_20261118/releases/reloads_20261121/reload_CN_A.job.json"
FORK_STEP = 544
PARAMETERS = 270


def device_identity(device):
    """The CUDA driver ordinal honors CUDA_VISIBLE_DEVICES, including reordered allocations."""
    props = torch.cuda.get_device_properties(device)
    gpu_uuid = getattr(props, "uuid", None)
    if gpu_uuid is None:
        driver = ctypes.CDLL("libcuda.so.1")
        ordinal = ctypes.c_int()
        raw = (ctypes.c_ubyte * 16)()
        index = torch.device(device).index
        if driver.cuDeviceGet(ctypes.byref(ordinal), index) != 0 or driver.cuDeviceGetUuid(ctypes.byref(raw), ordinal) != 0:
            raise RuntimeError("Unable to identify CUDA device UUID")
        gpu_uuid = str(uuid.UUID(bytes=bytes(raw)))
    return {"host": socket.gethostname(), "device_index": torch.device(device).index,
            "device_name": props.name, "gpu_uuid": str(gpu_uuid)}


def rng_states(device):
    return {"python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "rng_torch_cpu": torch.get_rng_state(),
            "rng_torch_device": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(device, state):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["rng_torch_cpu"])
    if state["rng_torch_device"] is not None:
        torch.cuda.set_rng_state(state["rng_torch_device"], device)


def snapshot(student, teacher, opt, runtime, micros, device):
    serialized = runtime.state_dict()
    persistent = {"cfg", "queue", "expected_datasets", "coverage", "coverage_reports"}
    transient = {k: v for k, v in vars(runtime).items() if k not in persistent}
    grad_none = all(p.grad is None for m in (student, teacher) for p in m.parameters())
    # Include nonpersistent buffers on both models; the original resume trace separately
    # authenticates the checkpointed state, inputs, and torch generators against the fork.
    state = {"student": student.state_dict(), "teacher": teacher.state_dict(),
             "optimizer": opt.state_dict(), "contrastive_state": serialized,
             "prepared_runtime_sha256": transient, "anchor_targets_sha256": [m.anchor_targets for m in micros],
             "student_buffers_sha256": dict(student.named_buffers()),
             "teacher_buffers_sha256": dict(teacher.named_buffers()), **rng_states(device)}
    return {**{key: state_sha256(value) for key, value in state.items()},
            "input_sha256": trainer._micro_digest(micros), "grad_none": grad_none}


def null_backward(backward, student, teacher, opt, runtime, micros, device):
    """Advance the live reducer once, reusing the staged input and an isolated prepared runtime."""
    rng = rng_states(device)
    buffers = [(v, v.detach().clone()) for m in (student, teacher) for v in m.buffers()]
    # The contrastive forward stages pending keys/coverage and may fill runtime scratch.
    # No warm keys, counters, or coverage are visible to the real 545 backward.
    warm_runtime = copy.deepcopy(runtime)
    try:
        failure, _ = backward(warm_runtime)
        if failure:
            raise RuntimeError(f"Null warm-up backward failed: {failure}")
    finally:
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for dest, original in buffers:
                if not torch.equal(dest, original):
                    dest.copy_(original)
        restore_rng(device, rng)


def record_optimizer_step(optimizer, student, folder, rank, expected_count=PARAMETERS):
    """Capture the actual tensors handed to AdamW and the state just after that same step."""
    named = [(name, param) for name, param in student.named_parameters() if param.requires_grad]
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(named) != expected_count or [id(p) for _, p in named] != [id(p) for p in all_params]:
        raise ValueError("Optimizer parameter count or order differs from named student parameters")
    rows = []

    def before(opt, args, kwargs):
        for index, (name, param) in enumerate(named):
            if param.grad is None:
                raise ValueError(f"Missing actual clipped gradient for parameter {index} {name}")
            rows.append({"id": index, "name": name, "post_clip_gradient_sha256": state_sha256(param.grad)})

    def after(opt, args, kwargs):
        for row, (_, param) in zip(rows, named, strict=True):
            state = opt.state[param]
            for key in ("exp_avg", "exp_avg_sq", "step"):
                row[key + "_sha256"] = state_sha256(state[key])
            row["parameter_sha256"] = state_sha256(param)
        write(folder / f"post_update.rank{rank}.json", {"schema": "cn-ddp-optimizer-update-v1",
              "rank": rank, "update": FORK_STEP + 1, "rows": rows})

    return optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)


@contextmanager
def instrument_leg(variant, folder):
    from contrastive_cn_restart_observe import ReductionObservation

    accumulate = trainer._accumulate_terms
    load = torch.optim.AdamW.load_state_dict
    optimizers, step_hooks = [], []
    count = 0

    def remember(opt, state):
        value = load(opt, state)
        optimizers.append(opt)
        return value

    def dispatch(ddp, student, teacher, micros, cfg, ctx, device, amp, **kwargs):
        nonlocal count
        runtime = kwargs["contrastive"]
        if runtime.step != FORK_STEP:
            if not count:
                raise ValueError("First backward did not start at the archived fork")
            while step_hooks:
                step_hooks.pop().remove()
            return accumulate(ddp, student, teacher, micros, cfg, ctx, device, amp, **kwargs)
        if count or len(optimizers) != 1 or len(micros) != 1:
            raise ValueError("Expected one prepared microbatch at the first actual optimizer update")
        count += 1
        opt, = optimizers
        observe = ReductionObservation(ddp, student)
        before = snapshot(student, teacher, opt, runtime, micros, device)
        if not before["grad_none"]:
            raise ValueError("Residual gradients at common fork")

        if variant == "warm":
            def warm_backward(isolated):
                return accumulate(ddp, student, teacher, micros, cfg, ctx, device, amp,
                                  **{**kwargs, "contrastive": isolated})
            null_backward(warm_backward, student, teacher, opt, runtime, micros, device)
        after = snapshot(student, teacher, opt, runtime, micros, device)
        write(folder / f"boundary.rank{ctx.rank}.json", {"schema": "cn-ddp-boundary-v1",
              "variant": variant, "rank": ctx.rank, "world_size": ctx.world_size, "update": FORK_STEP + 1,
              "null_warmups": int(variant == "warm"), "before": before, "after_warmup": after,
              **device_identity(device)})
        if before != after:
            raise ValueError("Null warm-up or boundary observer changed a checked state")
        observe.start()
        result = accumulate(ddp, student, teacher, micros, cfg, ctx, device, amp, **kwargs)
        if result[0]:
            raise RuntimeError(f"First actual backward failed: {result[0]}")
        observe.finish(folder, ctx.rank)
        step_hooks.extend(record_optimizer_step(opt, student, folder, ctx.rank))
        return result

    try:
        with patch.object(trainer, "_accumulate_terms", dispatch), patch.object(torch.optim.AdamW, "load_state_dict", remember):
            yield
    finally:
        while step_hooks:
            step_hooks.pop().remove()
    if count != 1:
        raise RuntimeError("Target backward not observed exactly once")


def validate_argv(argv, plan):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--variant", required=True, choices=("ordinary", "warm"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=("benchmark",))
    parser.add_argument("--time-limit-s", type=int, required=True, choices=(1770,))
    files = ["config", "release", "manifest", "scales", "window-seconds", "audit-report", "cohort",
             "calibration-report", "calibration-run-dir"]
    for name in files:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    job = json.loads(ARCHIVED_JOB.read_text())
    def pinned(spec):
        return (ROOT / spec["path"]).resolve()
    expected = {"config": pinned(job["config"]), "release": pinned(job["release"]),
                "audit-report": pinned(job["audit"]), "cohort": pinned(job["cohort"]),
                "calibration-report": pinned(job["calibration_report"]),
                "calibration-run-dir": pinned(job["calibration_summary"]).parent}
    for name in ("manifest", "scales", "window-seconds"):
        expected[name] = pinned(job["inputs"][name.replace("-", "_")])
    if any(getattr(args, name.replace("-", "_")).resolve() != path for name, path in expected.items()):
        raise ValueError("Diagnostic leg input paths differ from pinned historical CN_A")
    if [p.resolve() for p in args.root] != [Path(p).resolve() for p in job["roots"]]:
        raise ValueError("Diagnostic leg data roots differ from archived CN_A")
    out = Path(plan["output"]) / args.variant
    if args.out_dir != out or out.is_symlink() or out.exists():
        raise ValueError("Diagnostic leg output must be fresh at its exact reviewed path")
    # Reconstruct the parent CLI from parsed, pinned values; no passthrough options.
    original = ["--stage", "benchmark", "--out-dir", str(out), "--time-limit-s", "1770"]
    for name, path in expected.items():
        original.extend(["--" + name, str(path)])
    for path in args.root:
        original.extend(["--root", str(path)])
    return args.variant, out, original


def main():
    from contrastive_cn_restart import require_binding
    variant, folder, args = validate_argv(sys.argv[1:], require_binding())
    if os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("Diagnostic leg requires four distributed CUDA workers")
    with patch.object(sys, "argv", [production.__file__, *args]), instrument_leg(variant, folder):
        production.main()


if __name__ == "__main__":
    main()
