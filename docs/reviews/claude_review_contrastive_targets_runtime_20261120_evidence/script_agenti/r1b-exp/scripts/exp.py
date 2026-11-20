"""R1 (riesecuzione indipendente): trainer reale, 4 rank gloo, con/senza ControlParityObserver.

Modi:
  baseline   nessun osservatore
  observer   observe_control attorno a R.train
  instrument osservatore + strumentazione (hook sui parametri vivi, puntatori, stato del reducer vivo,
             hash del gradiente vivo dopo il backward vivo)
  N1         controllo negativo: restore_rng inerte
  N2         controllo negativo: i probe riusano i parametri del modello vivo in un nuovo DDP
UPDATES patchato a (2, 4) (lo schedule di test ha 4 update); assert_numeric_profile patchato (CPU senza CUDA).
Tutto il resto e' codice reale a 0efa4d45.
"""
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp

from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256
import contrastive_target_parity as P
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup

FLAGS = {"profile": P.PROFILE, "matmul_tf32": False, "cudnn_tf32": False, "math_sdp_enabled": True,
         "flash_sdp_enabled": False, "mem_efficient_sdp_enabled": False, "cudnn_sdp_enabled": None,
         "deterministic_algorithms": False, "cudnn_deterministic": False, "cudnn_benchmark": False,
         "float32_matmul_precision": "highest", "autocast_cuda_enabled": True, "autocast_cuda_dtype": "torch.bfloat16"}
TIMING = ("time", "comm", "compute", "overlap")


def make_cfg(pose):
    cfg = _config(pose)
    return replace(cfg, batch_size=32, amp_bf16=True, prefetch=True,
                   model=replace(cfg.model, muscle_dropout=.4),
                   contrastive={**cfg.contrastive, "fill_steps": 256, "ramp_steps": 256, "grad_every": 0,
                                "grad_steps": (4,)})


def ddp_state(model):
    if model is None:
        return None
    data = model._get_ddp_logging_data()
    return {k: v for k, v in data.items() if not any(t in k for t in TIMING)}


def worker(rank, world, store, mode, root, manifest, pose, out):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=180))
    log = {"mode": mode, "rank": rank}
    try:
        P.UPDATES = (2, 4)
        P.assert_numeric_profile = lambda **kw: dict(FLAGS)
        if mode == "N1":
            P.restore_rng = lambda saved, device: None
        if mode == "N2":
            live = {}
            original_measure_n2 = P.ControlParityObserver.measure

            def measure_n2(self, model, student, *a, **k):
                live["student"] = student
                return original_measure_n2(self, model, student, *a, **k)

            P.ControlParityObserver.measure = measure_n2
            P.WearUsFM = lambda cfg_model: "sentinel"
            P.attach_contrastive_head = lambda sentinel: live["student"]
        if mode == "instrument":
            log.update(live_hook_calls=0, overlap=[], points=[])
            original_measure = P.ControlParityObserver.measure
            original_wrap = P.trainer._ddp_wrap
            state = {"live_ptrs": None}

            def measure(self, model, student, *a, **k):
                rt = k.get("contrastive")
                if rt is None or rt.step + 1 not in P.UPDATES:
                    return original_measure(self, model, student, *a, **k)
                state["live_ptrs"] = {p.data_ptr() for p in student.parameters()}
                handles = [p.register_hook(lambda g: log.__setitem__("live_hook_calls", log["live_hook_calls"] + 1))
                           for p in student.parameters() if p.requires_grad]
                rec = {"update": rt.step + 1, "grad_none_before": all(p.grad is None for p in student.parameters()),
                       "ddp_before": ddp_state(model)}
                try:
                    return original_measure(self, model, student, *a, **k)
                finally:
                    for h in handles:
                        h.remove()
                    rec["grad_none_after"] = all(p.grad is None for p in student.parameters())
                    rec["ddp_after"] = ddp_state(model)
                    rec["ddp_state_unchanged"] = rec["ddp_before"] == rec["ddp_after"]
                    try:
                        model.reducer._check_reducer_finalized()
                        rec["live_reducer_finalized"] = True
                    except Exception as exc:  # noqa: BLE001
                        rec["live_reducer_finalized"] = repr(exc)
                    log["points"].append(rec)

            def wrap(student, cfg_, ctx_, device_):
                if state["live_ptrs"] is not None:
                    log["overlap"].append(len({p.data_ptr() for p in student.parameters()} & state["live_ptrs"]))
                return original_wrap(student, cfg_, ctx_, device_)

            P.ControlParityObserver.measure = measure
            P.trainer._ddp_wrap = wrap
        cfg = make_cfg(Path(pose))
        manager = nullcontext() if mode == "baseline" else P.observe_control(Path(out))
        if mode == "instrument":
            # hash of the LIVE reduced gradient right after the live backward of the observed updates
            inner = P.trainer._accumulate_terms
            live_grads = []
            t0 = time.perf_counter()
            with manager:
                patched = P.trainer._accumulate_terms

                def after(model, student, teacher, micros, cfg_, ctx_, device_, amp, **kw):
                    result = patched(model, student, teacher, micros, cfg_, ctx_, device_, amp, **kw)
                    rt = kw.get("contrastive")
                    if rt is not None and rt.step + 1 in P.UPDATES:
                        live_grads.append({"update": rt.step + 1,
                                           "live_gradient_sha256": state_sha256(P._gradient_state(student))})
                    return result

                P.trainer._accumulate_terms = after
                try:
                    summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), dist=R.DistContext(rank, world, rank),
                                      device="cpu", num_workers=0, log=lambda m: None)
                finally:
                    P.trainer._accumulate_terms = patched
            log["live_grads"] = live_grads
        else:
            t0 = time.perf_counter()
            with manager:
                summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), dist=R.DistContext(rank, world, rank),
                                  device="cpu", num_workers=0, log=lambda m: None)
        log.update(steps=summary.get("steps"), stopped=summary.get("stopped"), seconds=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        log["exception"] = f"{type(exc).__name__}: {exc}"
        log["trace_tail"] = traceback.format_exc()[-1500:]
    finally:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, f"exp.rank{rank}.json").write_text(json.dumps(log, default=str, indent=1))
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def main():
    base, mode, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    data = base / "data"
    if not (data / "done").exists():
        data.mkdir(parents=True, exist_ok=True)
        root, manifest, pose = _pose_setup(data)
        (data / "done").write_text(json.dumps([str(root), str(manifest), str(pose)]))
    root, manifest, pose = json.loads((data / "done").read_text())
    out, store = base / ("run_" + tag), base / ("store_" + tag)
    if store.exists():
        store.unlink()
    context = mp.spawn(worker, args=(4, str(store), mode, root, manifest, pose, str(out)), nprocs=4, join=False)
    deadline = time.monotonic() + 900
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
