"""Scettico n.1 per R1-N1: gradiente ridotto del backward VIVO contro i sei probe allo stesso stato.

Uso: probe_live.py BASE TAG MODE UPDATES_JSON [GC]
  MODE live      DDP vivo invariato (find_unused_parameters=False: bucket ricostruiti all'inizio della 2a iterazione)
  MODE norebuild controfattuale: il reducer del DDP vivo non ricostruisce mai i bucket (proxy: _rebuild_buckets -> False)
  GC = 1         grad_checkpoint=True (come nei config di produzione)
Cattura, per ogni update osservato:
  (a) gradiente vivo subito dopo il backward vivo (prima di clipping e ottimizzatore);
  (b) gradiente di ciascun probe, avvolgendo la funzione `backward` che l'osservatore passa ai probe;
  (c) layout reale dei bucket (nomi dei parametri per bucket) del reducer vivo e di ciascun probe;
  (d) _has_rebuilt_buckets del DDP vivo e i campi del logging DDP (aggiornati prima del rebuild, quindi in ritardo).
Nessun file del repository viene modificato: solo patch in memoria nel processo.
"""
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp

FLAGS = {"profile": "bf16_math_tf32_off", "matmul_tf32": False, "cudnn_tf32": False, "math_sdp_enabled": True,
         "flash_sdp_enabled": False, "mem_efficient_sdp_enabled": False, "cudnn_sdp_enabled": None,
         "deterministic_algorithms": False, "cudnn_deterministic": False, "cudnn_benchmark": False,
         "float32_matmul_precision": "highest", "autocast_cuda_enabled": True,
         "autocast_cuda_dtype": "torch.bfloat16"}


class NoRebuild:
    """Proxy del reducer C++: inoltra tutto tranne _rebuild_buckets, che non ricostruisce mai."""

    def __init__(self, inner):
        self.__dict__["_inner"] = inner

    def __getattr__(self, name):
        return getattr(self.__dict__["_inner"], name)

    def _rebuild_buckets(self):
        return False


def grads_of(module):
    return {n: p.grad.detach().to("cpu", copy=True) for n, p in module.named_parameters() if p.grad is not None}


def digest(grads):
    h = hashlib.sha256()
    for n in sorted(grads):
        t = grads[n].contiguous()
        h.update(n.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def compare(a, b):
    if set(a) != set(b):
        return {"same_keys": False}
    names = sorted(a)
    diff = [n for n in names if not torch.equal(a[n], b[n])]
    mx = max((float((a[n].double() - b[n].double()).abs().max()) for n in names), default=0.0)
    num = sum(float((a[n].double() - b[n].double()).square().sum()) for n in names) ** .5
    den = sum(float(a[n].double().square().sum()) for n in names) ** .5
    return {"same_keys": True, "n": len(names), "n_diff": len(diff), "max_abs": mx,
            "rel_l2": num / den if den else None, "first_diff": diff[:4]}


def layout(ddp):
    names = {p.data_ptr(): n for n, p in ddp.module.named_parameters()}
    reducer = ddp.reducer
    return [[names[p.data_ptr()] for p in b.parameters()] for b in reducer._get_zeros_like_grad_buckets()]


def worker(rank, world, store, root, manifest, pose, out, mode, updates, gc):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=300))
    rec = {"rank": rank, "mode": mode, "updates": updates, "gc": gc, "points": [], "torch": torch.__version__}
    try:
        from wearusfm.training import run as R
        import contrastive_target_parity as P
        from test_contrastive_runtime import _config

        P.UPDATES = tuple(updates)
        P.assert_numeric_profile = lambda **kw: dict(FLAGS)
        state = {"observer": None, "live": None, "probe_grads": [], "probe_layouts": [], "probe_find_unused": []}
        orig_wrap = P.trainer._ddp_wrap

        def wrap(student, cfg_, ctx_, device_):
            ddp = orig_wrap(student, cfg_, ctx_, device_)
            obs = state["observer"]
            if obs is not None and obs.active:
                state["probe_find_unused"].append(bool(ddp.find_unused_parameters))
            else:
                if mode == "norebuild":
                    ddp.reducer = NoRebuild(ddp.reducer)
                state["live"] = ddp
                rec["live_find_unused"] = bool(ddp.find_unused_parameters)
                rec["live_initial_layout"] = layout(ddp)
            return ddp

        P.trainer._ddp_wrap = wrap
        orig_measure = P.ControlParityObserver.measure

        def measure(self, model, student, teacher, micros, cfg_, ctx_, device_, amp, backward, prepare, **kw):
            def probe_backward(m, s, *a, **k):
                result = backward(m, s, *a, **k)
                state["probe_grads"].append(grads_of(s))
                state["probe_layouts"].append(layout(m) if m is not None else None)
                return result
            return orig_measure(self, model, student, teacher, micros, cfg_, ctx_, device_, amp,
                                probe_backward, prepare, **kw)

        P.ControlParityObserver.measure = measure

        cfg = _config(Path(pose))
        model_cfg = replace(cfg.model, muscle_dropout=.4)
        if gc:
            model_cfg = replace(model_cfg, grad_checkpoint=True)
        cfg = replace(cfg, batch_size=32, amp_bf16=True, prefetch=True, model=model_cfg,
                      contrastive={**cfg.contrastive, "fill_steps": 256, "ramp_steps": 256, "grad_every": 0,
                                   "grad_steps": (4,)})
        with P.observe_control(Path(out)) as observer:
            state["observer"] = observer
            inner = P.trainer._accumulate_terms

            def outer(model, student, teacher, micros, cfg_, ctx_, device_, amp, **kw):
                rt = kw.get("contrastive")
                upd = None if rt is None else rt.step + 1
                n0 = len(state["probe_grads"])
                result = inner(model, student, teacher, micros, cfg_, ctx_, device_, amp, **kw)
                if upd in P.UPDATES:
                    live = grads_of(student)
                    probes = state["probe_grads"][n0:]
                    layouts = state["probe_layouts"][n0:]
                    live_layout = layout(model)
                    logging = model._get_ddp_logging_data()
                    rec["points"].append({
                        "update": upd,
                        "n_probes": len(probes),
                        "live_has_rebuilt_buckets_attr": bool(getattr(model, "_has_rebuilt_buckets", None)),
                        "logging_has_rebuilt_buckets": logging.get("has_rebuilt_buckets"),
                        "logging_bucket_sizes": logging.get("bucket_sizes"),
                        "logging_rebuilt_bucket_sizes": logging.get("rebuilt_bucket_sizes"),
                        "live_layout_sizes": [len(b) for b in live_layout],
                        "live_layout_equals_initial": live_layout == rec["live_initial_layout"],
                        "probe_layouts_equal_initial_live": [lay == rec["live_initial_layout"] for lay in layouts],
                        "probe_layouts_equal_live_now": [lay == live_layout for lay in layouts],
                        "live_digest": digest(live),
                        "probe_digests": [digest(p) for p in probes],
                        "live_vs_probe": [compare(live, p) for p in probes],
                        "probe0_vs_probes": [compare(probes[0], p) for p in probes[1:]],
                    })
                return result

            P.trainer._accumulate_terms = outer
            try:
                t0 = time.perf_counter()
                summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), dist=R.DistContext(rank, world, rank),
                                  device="cpu", num_workers=0, log=lambda m: None)
                rec["seconds"] = time.perf_counter() - t0
            finally:
                P.trainer._accumulate_terms = inner
        rec["steps"] = summary.get("steps")
        rec["stopped"] = summary.get("stopped")
        rec["probe_find_unused"] = state["probe_find_unused"]
        report = observer.report or {}
        rec["gate_status"] = report.get("status")
        rec["gate_points"] = [{"status": p.get("status"), "errors": p.get("errors"), "inconclusive": p.get("inconclusive"),
                               "unique_gradient_hashes": len(p.get("unique_gradient_hashes", []))}
                              for p in report.get("gate", {}).get("points", [])]
    except Exception as exc:  # noqa: BLE001
        rec["exception"] = f"{type(exc).__name__}: {exc}"
        rec["trace"] = traceback.format_exc()[-2500:]
    finally:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, f"sk1.rank{rank}.json").write_text(json.dumps(rec, default=str, indent=1))
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def main():
    base, tag, mode, updates = Path(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    gc = len(sys.argv) > 5 and sys.argv[5] == "1"
    assert mode in ("live", "norebuild")
    data = base / "data"
    if not (data / "done.json").exists():
        from test_pose_training import _pose_setup
        data.mkdir(parents=True, exist_ok=True)
        root, manifest, pose = _pose_setup(data)
        (data / "done.json").write_text(json.dumps([str(root), str(manifest), str(pose)]))
    root, manifest, pose = json.loads((data / "done.json").read_text())
    out, store = base / ("run_" + tag), base / ("store_" + tag)
    if out.exists():
        raise SystemExit("output exists: " + str(out))
    if store.exists():
        store.unlink()
    ctx = mp.spawn(worker, args=(4, str(store), root, manifest, pose, str(out), mode, updates, gc),
                   nprocs=4, join=False)
    deadline = time.monotonic() + 540
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline")
    finally:
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
    print("DONE", tag)


if __name__ == "__main__":
    main()
