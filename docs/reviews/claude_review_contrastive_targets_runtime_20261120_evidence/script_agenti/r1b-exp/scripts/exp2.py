"""R1: gradiente vivo contro gradiente del probe allo stesso stato (scarto numerico, non solo hash).

argv: base tag updates(json) [world]
Per ogni update osservato salva il gradiente vivo (dopo il backward vivo) e quello del probe C0a,
e il DDP vivo: ha ricostruito i bucket? Il probe usa un reducer nuovo (bucket iniziali).
"""
import json
import os
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp

from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256
import contrastive_target_parity as P
from test_contrastive_runtime import _config

FLAGS = {"profile": P.PROFILE, "matmul_tf32": False, "cudnn_tf32": False, "math_sdp_enabled": True,
         "flash_sdp_enabled": False, "mem_efficient_sdp_enabled": False, "cudnn_sdp_enabled": None,
         "deterministic_algorithms": False, "cudnn_deterministic": False, "cudnn_benchmark": False,
         "float32_matmul_precision": "highest", "autocast_cuda_enabled": True, "autocast_cuda_dtype": "torch.bfloat16"}


def worker(rank, world, store, root, manifest, pose, out, updates):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=180))
    res = {"rank": rank, "points": []}
    try:
        P.UPDATES = tuple(updates)
        P.assert_numeric_profile = lambda **kw: dict(FLAGS)
        probe_grads = []
        original_grad_state = P._gradient_state

        def grad_state(model):
            g = original_grad_state(model)
            if any(v is not None for v in g.values()):
                probe_grads.append(g)
            return g

        P._gradient_state = grad_state
        cfg = _config(Path(pose))
        cfg = replace(cfg, batch_size=32, amp_bf16=True, prefetch=True, model=replace(cfg.model, muscle_dropout=.4),
                      contrastive={**cfg.contrastive, "fill_steps": 256, "ramp_steps": 256, "grad_every": 0, "grad_steps": (4,)})
        with P.observe_control(Path(out)):
            patched = P.trainer._accumulate_terms

            def after(model, student, teacher, micros, cfg_, ctx_, device_, amp, **kw):
                rt = kw.get("contrastive")
                n_before = len(probe_grads)
                result = patched(model, student, teacher, micros, cfg_, ctx_, device_, amp, **kw)
                if rt is not None and rt.step + 1 in P.UPDATES:
                    live = original_grad_state(student)
                    probes = probe_grads[n_before:]
                    c0a = probes[0]
                    names = [k for k in live if live[k] is not None]
                    diff = max(float((live[k].double() - c0a[k].double()).abs().max()) for k in names)
                    num = sum(float((live[k].double() - c0a[k].double()).square().sum()) for k in names) ** .5
                    den = sum(float(live[k].double().square().sum()) for k in names) ** .5
                    differing = [k for k in names if not torch.equal(live[k], c0a[k])]
                    data = model._get_ddp_logging_data() if model is not None else {}
                    res["points"].append({
                        "update": rt.step + 1, "n_probes": len(probes),
                        "probes_equal_each_other": len({state_sha256(p) for p in probes}) == 1,
                        "live_equal_probe_bytes": state_sha256(live) == state_sha256(c0a),
                        "max_abs_diff": diff, "rel_l2_diff": num / den if den else None,
                        "n_tensors": len(names), "n_tensors_differing": len(differing),
                        "first_differing": differing[:5],
                        "live_has_rebuilt_buckets": data.get("has_rebuilt_buckets"),
                        "live_bucket_sizes": data.get("rebuilt_bucket_sizes", data.get("bucket_sizes")),
                    })
                return result

            P.trainer._accumulate_terms = after
            try:
                summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), dist=R.DistContext(rank, world, rank),
                                  device="cpu", num_workers=0, log=lambda m: None)
            finally:
                P.trainer._accumulate_terms = patched
        res["steps"] = summary.get("steps")
    except Exception as exc:  # noqa: BLE001
        import traceback
        res["exception"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-1200:]
    finally:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, f"exp2.rank{rank}.json").write_text(json.dumps(res, default=str, indent=1))
        dist.destroy_process_group()


def main():
    base, tag, updates = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
    root, manifest, pose = json.loads((base / "data" / "done").read_text())
    out, store = base / ("run_" + tag), base / ("store_" + tag)
    if store.exists():
        store.unlink()
    ctx = mp.spawn(worker, args=(4, str(store), root, manifest, pose, str(out), updates), nprocs=4, join=False)
    deadline = time.monotonic() + 900
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError
    finally:
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
    print("OK", tag)


if __name__ == "__main__":
    main()
