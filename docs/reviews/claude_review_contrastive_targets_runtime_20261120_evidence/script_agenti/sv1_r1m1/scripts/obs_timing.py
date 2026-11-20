"""Scettico 1 per R1-M1: il tempo di ControlParityObserver.measure entra in t_step_s e elapsed_s?

Trainer reale (R.train) a fbfde797, 4 rank gloo su CPU, braccio C0, batch globale 32, accum 1, amp_bf16 True
(su CPU l'autocast resta spento), prefetch come la produzione. Unici cambi per la CPU: P.UPDATES=(2,4)
perche' lo schedule di test ha 4 update, P.assert_numeric_profile restituisce i flag del profilo (niente CUDA).
Ogni chiamata a measure e' cronometrata con perf_counter e time.time() e associata all'update (runtime.step+1).
Modi: A = senza osservatore; O = observe_control attorno a R.train (come contrastive_run.py:355-357).
"""
import json, os, sys, time, traceback
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp


def flags():
    import contrastive_target_parity as P
    return {"profile": P.PROFILE, "matmul_tf32": False, "cudnn_tf32": False, "math_sdp_enabled": True,
            "flash_sdp_enabled": False, "mem_efficient_sdp_enabled": False, "cudnn_sdp_enabled": None,
            "deterministic_algorithms": False, "cudnn_deterministic": False, "cudnn_benchmark": False,
            "float32_matmul_precision": "highest", "autocast_cuda_enabled": True, "autocast_cuda_dtype": "torch.bfloat16"}


def make_cfg(pose):
    from test_contrastive_runtime import _config
    cfg = _config(pose)  # C0
    return replace(cfg, batch_size=32, amp_bf16=True, prefetch=True,
                   model=replace(cfg.model, muscle_dropout=.4),
                   contrastive={**cfg.contrastive, "fill_steps": 256, "ramp_steps": 256,
                                "grad_every": 0, "grad_steps": (4,)})


def worker(rank, world, store, mode, root, manifest, pose, out):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=300))
    from wearusfm.training import run as R
    import contrastive_target_parity as P
    log = {"mode": mode, "rank": rank, "measure_calls": []}
    try:
        P.UPDATES = (2, 4)
        fl = flags()
        P.assert_numeric_profile = lambda **kw: dict(fl)
        original = P.ControlParityObserver.measure

        def timed(self, model, student, teacher, micros, cfg, ctx, device, amp, backward, prepare, **kw):
            rt = kw.get("contrastive")
            a, w = time.perf_counter(), time.time()
            n_before = len(self.points)
            try:
                return original(self, model, student, teacher, micros, cfg, ctx, device, amp, backward, prepare, **kw)
            finally:
                log["measure_calls"].append({"update": None if rt is None else rt.step + 1,
                                             "seconds": time.perf_counter() - a, "wall_start": w,
                                             "measured_point": len(self.points) > n_before})
        P.ControlParityObserver.measure = timed
        cfg = make_cfg(Path(pose))
        manager = nullcontext() if mode == "A" else P.observe_control(Path(out))
        t0 = time.perf_counter()
        with manager:
            summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), dist=R.DistContext(rank, world, rank),
                              device="cpu", num_workers=0, log=lambda m: None)
        log.update(steps=summary.get("steps"), stopped=summary.get("stopped"), wall=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        log["exception"] = f"{type(exc).__name__}: {exc}"
        log["trace"] = traceback.format_exc()[-2000:]
    finally:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, f"sv1.rank{rank}.json").write_text(json.dumps(log, indent=1, default=str))
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def main():
    base, mode, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    data = base / "data"
    if not (data / "paths.json").exists():
        from test_pose_training import _pose_setup
        data.mkdir(parents=True, exist_ok=True)
        root, manifest, pose = _pose_setup(data)
        (data / "paths.json").write_text(json.dumps([str(root), str(manifest), str(pose)]))
    root, manifest, pose = json.loads((data / "paths.json").read_text())
    out, store = base / ("run_" + tag), base / ("store_" + tag)
    if out.exists():
        raise SystemExit("output exists: " + str(out))
    if store.exists():
        store.unlink()
    ctx = mp.spawn(worker, args=(4, str(store), mode, root, manifest, pose, str(out)), nprocs=4, join=False)
    deadline = time.monotonic() + 900
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("run exceeded 900 s")
    finally:
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
    print("done", tag)


if __name__ == "__main__":
    main()
