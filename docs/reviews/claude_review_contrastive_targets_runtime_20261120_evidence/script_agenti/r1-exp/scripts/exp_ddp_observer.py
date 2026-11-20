"""R1 experiment: real trainer, 4 gloo ranks, with/without ControlParityObserver.

Modes: baseline (no observer), observer (observe_control wrapping train()).
Prescribed updates patched to (2, 4) because the test schedule has 4 updates.
assert_numeric_profile patched (CPU has no CUDA); everything else is real code.
"""
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp

from wearusfm.training import run as R
import contrastive_target_parity as P
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup

FLAGS = {"profile": P.PROFILE, "matmul_tf32": False, "cudnn_tf32": False, "math_sdp_enabled": True,
         "flash_sdp_enabled": False, "mem_efficient_sdp_enabled": False, "cudnn_sdp_enabled": None,
         "deterministic_algorithms": False, "cudnn_deterministic": False, "cudnn_benchmark": False,
         "float32_matmul_precision": "highest", "autocast_cuda_enabled": True, "autocast_cuda_dtype": "torch.bfloat16"}


def make_cfg(pose, grad_steps=(4,), dropout=.4, prefetch=True):
    cfg = _config(pose)
    return replace(cfg, batch_size=32, amp_bf16=True, prefetch=prefetch,
                   model=replace(cfg.model, muscle_dropout=dropout),
                   contrastive={**cfg.contrastive, "fill_steps": 256, "ramp_steps": 256, "grad_every": 0,
                                "grad_steps": tuple(grad_steps)})


def worker(rank, world, store, mode, root, manifest, pose, out, extra):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=120))
    try:
        P.UPDATES = (2, 4)
        P.assert_numeric_profile = lambda **kw: dict(FLAGS)
        if extra.get("no_restore"):
            # Negative control: probes no longer restore generators.
            P.restore_rng = lambda saved, device: None
        if extra.get("instrument"):
            log = {"live_hook_calls": 0, "probe_ptrs": [], "live_ptrs": None, "overlap": [], "probe_grad_hooks": 0}
            original_measure = P.ControlParityObserver.measure
            original_wrap = P.trainer._ddp_wrap

            def measure(self, model, student, *a, **k):
                handles = []
                if k.get("contrastive") is not None and k["contrastive"].step + 1 in P.UPDATES:
                    log["live_ptrs"] = sorted(p.data_ptr() for p in student.parameters())
                    for p in student.parameters():
                        if p.requires_grad:
                            handles.append(p.register_hook(lambda g: log.__setitem__("live_hook_calls", log["live_hook_calls"] + 1)))
                    live_grad_none_before = all(p.grad is None for p in student.parameters())
                try:
                    return original_measure(self, model, student, *a, **k)
                finally:
                    for h in handles:
                        h.remove()
                    if handles:
                        log["live_grad_none_after"] = all(p.grad is None for p in student.parameters())
                        log["live_grad_none_before"] = live_grad_none_before
                        Path(out, f"instr.rank{rank}.json").write_text(json.dumps(log))

            def wrap(student, cfg_, ctx_, device_):
                ptrs = sorted(p.data_ptr() for p in student.parameters())
                log["probe_ptrs"].append(len(ptrs))
                if log["live_ptrs"] is not None:
                    log["overlap"].append(len(set(ptrs) & set(log["live_ptrs"])))
                return original_wrap(student, cfg_, ctx_, device_)

            P.ControlParityObserver.measure = measure
            P.trainer._ddp_wrap = wrap  # only the observer calls it after the live wrap
        cfg = make_cfg(Path(pose), **extra.get("cfg", {}))
        manager = P.observe_control(Path(out)) if mode == "observer" else nullcontext()
        t0 = time.perf_counter()
        with manager:
            summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), dist=R.DistContext(rank, world, rank),
                              device="cpu", num_workers=0, log=lambda m: None)
        Path(out, f"done.rank{rank}.json").write_text(json.dumps({"steps": summary.get("steps"),
                                                                  "stopped": summary.get("stopped"),
                                                                  "seconds": time.perf_counter() - t0}))
    finally:
        dist.destroy_process_group()


def main():
    base = Path(sys.argv[1])
    mode = sys.argv[2]
    tag = sys.argv[3]
    extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    data = base / "data"
    if not (data / "done").exists():
        data.mkdir(parents=True, exist_ok=True)
        root, manifest, pose = _pose_setup(data)
        (data / "done").write_text(json.dumps([str(root), str(manifest), str(pose)]))
    root, manifest, pose = json.loads((data / "done").read_text())
    out = base / ("run_" + tag)
    store = base / ("store_" + tag)
    if store.exists():
        store.unlink()
    context = mp.spawn(worker, args=(4, str(store), mode, root, manifest, pose, str(out), extra), nprocs=4, join=False)
    deadline = time.monotonic() + 1200
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("exceeded")
    finally:
        for p in context.processes:
            if p.is_alive():
                p.terminate()
    print("OK", tag)


if __name__ == "__main__":
    main()
