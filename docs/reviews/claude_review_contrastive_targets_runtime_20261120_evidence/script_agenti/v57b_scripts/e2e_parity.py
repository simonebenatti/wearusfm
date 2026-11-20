"""Probe R5-7 (verifier 2): real observe_control around real R.train on 4 gloo CPU ranks,
then the readout's binding expressions (contrastive_benchmark_readout.py read_preflight)
and validate_parity_report on the file that finish() actually wrote.
Only UPDATES is shrunk (2, 4) so both points fall inside a 4-update run; variant
'profile_stub' also replaces assert_numeric_profile (as the repo's own observer test does)
so the six DDP replay cases are really executed on CPU."""
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
WORK = Path(sys.argv[2]).resolve()
VARIANT = sys.argv[3]
sys.path[:0] = [str(CLONE / "src"), str(CLONE / "scripts"), str(CLONE / "tests/cpu")]

import torch
import torch.multiprocessing as mp

UPDATES = (2, 4)
WORLD = 4


def worker(rank, world, store, cfg, manifest, data, out, provenance, variant):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist
    import contrastive_target_parity as P
    from wearusfm.training import run as R
    P.UPDATES = UPDATES
    if variant.startswith("profile_stub"):
        P.assert_numeric_profile = lambda **kw: {}
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=300))
    try:
        ctx = R.DistContext(rank, world, rank)
        with P.observe_control(out) as observer:
            summary = R.train(cfg, manifest, [data], out, dist=ctx, log=lambda _: None,
                              contrastive_provenance=provenance)
        (out.parent / f"done{rank}.json").write_text(json.dumps(
            {"steps": summary["steps"], "stopped": summary["stopped"], "points": len(observer.points)}))
    finally:
        dist.destroy_process_group()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    from test_pose_training import _pose_setup
    from test_contrastive_runtime import _config
    from contrastive_prepare import source_digest
    import contrastive_benchmark_readout as B
    import contrastive_target_parity as P
    from contrastive_target_parity import validate_parity_report
    P.UPDATES = UPDATES  # the reader uses the same shrunken schedule as the producer
    tmp = WORK / VARIANT
    tmp.mkdir(parents=True, exist_ok=False)
    data, manifest, pose = _pose_setup(tmp)
    base = _config(pose, "C0")
    cfg = replace(base, batch_size=32, max_steps=4, accum_steps=1)
    if VARIANT.endswith("_fill"):  # production-like: both points before the 256-update fill (zero weights)
        cfg = replace(cfg, contrastive={**cfg.contrastive, "fill_steps": 256})
    provenance = {"source_sha256": source_digest(), "calibration_sha256": "c" * 64,
                  "cohort_sha256": "b" * 64, "input_sha256": {"manifest": "m" * 64}}
    out = tmp / "C0"
    store = tmp / "store"
    started = time.monotonic()
    context = mp.spawn(worker, args=(WORLD, str(store), cfg, manifest, data, out, provenance, VARIANT),
                       nprocs=WORLD, join=False)
    deadline = time.monotonic() + 420
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("probe exceeded 420 s")
    finally:
        for p in context.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
    elapsed = time.monotonic() - started
    report = json.loads((out / "zero_weight_parity.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    cfg_file = json.loads((out / "config.json").read_text())
    cohort = {"source_sha256": source_digest(), "initial_state_sha256": summary["contrastive"]["initial_sha256"]}
    # Exact expressions of read_preflight (contrastive_benchmark_readout.py, parity block).
    binding = {
        "source_sha256": report.get("source_sha256") == cohort["source_sha256"],
        "config_sha256": report.get("config_sha256") == canonical(cfg_file),
        "summary_sha256": report.get("summary_sha256") == B.sha(out / "summary.json"),
        "initial_state_sha256": report.get("initial_state_sha256") == cohort["initial_state_sha256"],
        "provenance": report.get("provenance") == provenance,
    }
    assessment = validate_parity_report(report)
    ranks = sorted(p.name for p in out.glob("zero_weight_parity.rank*.json"))
    points = []
    for point in assessment.get("points", []):
        points.append({"status": point["status"], "errors": point["errors"], "inconclusive": point["inconclusive"][:6],
                       "n_inconclusive": len(point["inconclusive"]),
                       "byte_identical": point["within_and_between_byte_identical"],
                       "unique_hashes": len(point["unique_gradient_hashes"])})
    print(json.dumps({"variant": VARIANT, "elapsed_s": round(elapsed, 1),
                      "done": [json.loads((tmp / f"done{r}.json").read_text()) for r in range(WORLD)],
                      "summary_steps": summary["steps"], "stopped": summary["stopped"],
                      "report_status": report.get("status"), "report_updates": [p["update"] for p in report["points"]],
                      "report_config_sha256_is_null": report.get("config_sha256") is None,
                      "profile_status_rows": [[r["profile_status"] for r in p["ranks"]] for p in report["points"]],
                      "live_state_restored": [[r["live_state_restored"] for r in p["ranks"]] for p in report["points"]],
                      "binding_like_readout": binding, "assessment_status": assessment["status"],
                      "assessment_points": points, "rank_files": ranks}, indent=1))
