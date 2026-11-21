"""R2: whole-chain CPU probe with the real trainer and real diagnostic functions (4 Gloo ranks).

Modes (each a fresh mp.spawn, so each resume is a real process restart):
  fork      0 -> F, writes checkpoint.pt at F
  cont      0 -> N in one process group (continuous)
  plain     resume F -> N, no instrumentation
  ordinary  resume F -> N under leg.instrument_leg("ordinary")
  warm      resume F -> N under leg.instrument_leg("warm")
  hookcont  0 -> N with the reference allreduce_hook registered from the start (hook vs native, no restart)
"""
import json, os, shutil, sys, time
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path

CLONE = Path(os.environ["CLONE"])
sys.path[:0] = [str(CLONE / p) for p in ("src", "scripts", "scripts/diagnostics", "tests/cpu")]

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
F = int(os.environ.get("FORK", 5))
N = int(os.environ.get("STEPS", 9))
CAP_MB = float(os.environ.get("CAP_MB", 0.02))


def make_cfg(pose_root):
    from test_contrastive_runtime import _config
    cfg = _config(pose_root, "CN")
    c = dict(cfg.contrastive)
    c.update(fill_steps=int(os.environ.get("FILL", 2)), ramp_steps=2, grad_every=1)
    if os.environ.get("GRAD_MODE") == "dense":
        c.update(grad_every=0, grad_steps=(), grad_dense_start=F + 1, grad_dense_end=N)
    from wearusfm.training import run as R
    return replace(R.with_wsd(cfg, N, warmup_steps=2, keep_checkpoints=(F,)), batch_size=8, contrastive=c,
                   alarm=R.AlarmRule(patience=10**6))


def worker(rank, mode, out, data, res, nparams):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(res) / f"store_{mode}").as_uri(), rank=rank, world_size=WORLD,
                            timeout=timedelta(seconds=120))
    from wearusfm.training import run as R
    import contrastive_cn_restart_leg as leg
    import contrastive_cn_restart_observe as obs
    from wearusfm.training.pairing import state_sha256
    from contextlib import ExitStack
    from unittest.mock import patch
    root, manifest, pose_root = (Path(p) for p in data)
    cfg = make_cfg(pose_root)
    if mode == "fork":
        cfg = replace(cfg, max_steps=F)
    held, log = {}, []
    wrap = R._ddp_wrap

    def ddp_wrap(student, cfg_, ctx, device):
        from torch.nn.parallel import DistributedDataParallel
        m = DistributedDataParallel(R._Objective(student), device_ids=None, find_unused_parameters=False,
                                    bucket_cap_mb=CAP_MB)
        held["ddp"] = m
        if mode == "hookcont":
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
            m.register_comm_hook(m.process_group, allreduce_hook)
        return m

    clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(params, *a, **k):
        params = list(params)
        log.append({"grad": state_sha256([p.grad for p in params]),
                    "grads": [state_sha256(p.grad) for p in params]})
        return clip(params, *a, **k)

    with ExitStack() as st:
        st.enter_context(patch.object(R, "_ddp_wrap", ddp_wrap))
        st.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", recording_clip))
        if mode in ("ordinary", "warm"):
            st.enter_context(patch.object(leg, "FORK_STEP", F))
            st.enter_context(patch.object(leg, "device_identity", lambda d: {
                "host": "gloo", "device_index": rank, "device_name": "cpu", "gpu_uuid": f"cpu{rank}"}))
            st.enter_context(patch.object(leg, "record_optimizer_step",
                                          partial(leg.record_optimizer_step.__wrapped__ if hasattr(leg.record_optimizer_step, "__wrapped__") else leg.record_optimizer_step,
                                                  expected_count=nparams)))
            class Obs(obs.ReductionObservation):
                def __init__(self, ddp, student):
                    super().__init__(ddp, student, expected_parameters=nparams)
            st.enter_context(patch.object(obs, "ReductionObservation", Obs))
            st.enter_context(leg.instrument_leg(mode, Path(out)))
        t = time.time()
        summary = R.train(cfg, manifest, [root], Path(out), log=lambda m: None, dist=R.DistContext(rank, WORLD, rank))
    logging = held["ddp"]._get_ddp_logging_data()
    keep = {k: logging.get(k) for k in ("bucket_sizes", "rebuilt_bucket_sizes", "rebuilt_per_bucket_param_indices",
                                         "has_rebuilt_buckets", "iteration", "comm_hook")}
    Path(res, f"{mode}.rank{rank}.json").write_text(json.dumps({"summary": summary, "clip_log": log, "ddp": keep,
                                                                 "seconds": time.time() - t}))
    dist.destroy_process_group()


def spawn(mode, out, data, res, nparams):
    t = time.time()
    mp.spawn(worker, args=(mode, str(out), data, str(res), nparams), nprocs=WORLD, join=True)
    return time.time() - t


def main():
    base = Path(sys.argv[1])
    modes = sys.argv[2:] or ["cont", "plain", "ordinary", "warm", "hookcont"]
    from test_contrastive_runtime import _pose_setup
    from wearusfm.model.fm import WearUsFM
    from wearusfm.training.contrastive_objectives import attach_contrastive_head
    base.mkdir(parents=True, exist_ok=True)
    data_dir = base / "data"
    if not data_dir.exists():
        data_dir.mkdir()
    root, manifest, pose_root = _pose_setup(data_dir)
    cfg = make_cfg(pose_root)
    nparams = sum(p.requires_grad for p in attach_contrastive_head(WearUsFM(cfg.model)).parameters())
    res = base / "res"
    res.mkdir(exist_ok=True)
    data = (str(root), str(manifest), str(pose_root))
    times = {}
    for mode in modes:
        out = base / mode
        if mode in ("plain", "ordinary", "warm"):
            if out.exists():
                shutil.rmtree(out)
            out.mkdir()
            shutil.copy2(base / "cont" / f"checkpoint_step{F:07d}.pt", out / "checkpoint.pt")
        times[mode] = spawn(mode, out, data, res, nparams)
        print(mode, f"{times[mode]:.1f}s", flush=True)
    (res / "times.json").write_text(json.dumps(times))
    print("nparams", nparams)


if __name__ == "__main__":
    main()
