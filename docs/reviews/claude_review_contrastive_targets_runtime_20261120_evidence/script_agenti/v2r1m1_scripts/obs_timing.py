"""Independent check (verifier 2, R1-M1): does the live-C0 parity observer's time land in t_step_s/elapsed_s?

Real trainer (R.train), 4 gloo ranks on CPU, C0 arm, global batch 32, weights zero (fill/ramp 256).
Only in-memory patches: P.UPDATES=(2,4) (4-update schedule), P.assert_numeric_profile -> fake flags so that the six
real probes run on CPU, and a timing wrapper around ControlParityObserver.measure. No file of the clone is modified.
"""
import json, os, sys, time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import torch
import torch.multiprocessing as mp


def worker(rank, world, store, mode, data, out, timing_prefix):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world, timeout=timedelta(seconds=600))
    try:
        from wearusfm.training import run as R
        import contrastive_target_parity as P
        from test_contrastive_runtime import _config
        root, manifest, pose_root = data
        cfg = _config(Path(pose_root), "C0")
        cfg = replace(cfg, batch_size=32,
                      contrastive={**cfg.contrastive, "fill_steps": 256, "ramp_steps": 256, "grad_steps": (4,)})
        P.UPDATES = (2, 4)
        P.assert_numeric_profile = lambda **kw: {"cpu_fake_profile": True}
        calls = []
        original = P.ControlParityObserver.measure

        def timed(self, *a, **k):
            t = time.time()
            try:
                return original(self, *a, **k)
            finally:
                calls.append({"update": k["contrastive"].step + 1, "seconds": time.time() - t})

        P.ControlParityObserver.measure = timed
        ctx = R.DistContext(rank, world, rank)
        kw = dict(log=lambda m: None, dist=ctx, device="cpu")
        if mode == "observer":
            with P.observe_control(Path(out)):
                R.train(cfg, Path(manifest), [Path(root)], Path(out), **kw)
        else:
            R.train(cfg, Path(manifest), [Path(root)], Path(out), **kw)
        Path(f"{timing_prefix}.rank{rank}.json").write_text(json.dumps(
            {"mode": mode, "measure_calls": calls,
             "measure_seconds_by_update": {str(c["update"]): c["seconds"] for c in calls if c["update"] in (2, 4)}}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    base, mode, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    assert mode in ("baseline", "observer")
    if not (base / "data.json").exists():
        from test_pose_training import _pose_setup
        (base / "data").mkdir(parents=True)
        root, manifest, pose_root = _pose_setup(base / "data")
        (base / "data.json").write_text(json.dumps([str(root), str(manifest), str(pose_root)]))
    data = json.loads((base / "data.json").read_text())
    out = base / f"run_{tag}"
    t = time.time()
    mp.spawn(worker, args=(4, str(base / f"store_{tag}"), mode, data, str(out), str(base / f"timing_{tag}")), nprocs=4, join=True)
    print(tag, mode, "wall", round(time.time() - t, 2))
