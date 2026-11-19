"""E2E smoke: drive the 24-job chain through the CLI 'run' entry with REAL stage functions.

Stubs only: SLURM env + scontrol, numeric census, CUDA->CPU device, preflight (needs
cluster caches), benchmarks (cost receipts fixture), cache reader (synthetic caches with
the real reader's output keys), historical prediction reader, and the parity values.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pytest
import torch

WT = Path("/Users/simonebenatti/dev/wearusfm/.claude/worktrees/review-pose-heads-2f98211")
import pose_temponet  # must come from the patched copy
assert "smoke/e2e/pkg" in pose_temponet.__file__, pose_temponet.__file__
from pose_temponet import benchmark, data as D, execution as E, provenance as P, runner as R
from pose_temponet import __main__ as CLI
sys.path.insert(0, str(WT/"tests/cpu"))
from test_pose_temponet_provenance import prepared, release, numeric_fixture  # noqa: F401
from test_pose_temponet_resources import receipts as cost_receipts

OUT = Path(os.environ["E2E_OUT"])
LOG = []


def note(**kw):
    LOG.append(kw)
    print("E2E", json.dumps(kw, default=str), flush=True)


def layer_norm(x):
    x = x.astype(np.float64)
    c = x - x.mean(-1, keepdims=True)
    return c/np.sqrt((c*c).mean(-1, keepdims=True)+1e-5)


def build_cache(users, windows, split, rng, W, static_every=5, short_every=7):
    n = len(users)*windows
    lengths = [150 if (i % short_every == 3) else 160 for i in range(n)]
    x = rng.normal(size=(n, 160, 384)).astype(np.float16)
    pre = (rng.normal(size=(n, 160, 384))*0.5 + x.astype(np.float64)*0.7).astype(np.float16)
    available = np.arange(160)[None, :] < np.asarray(lengths)[:, None]
    x[~available] = 0
    pre[~available] = 0
    z = layer_norm(x)
    signal = z @ W
    target = signal + rng.normal(size=signal.shape)*float(os.environ.get('E2E_NOISE', '1.9'))
    scale = np.ones(n)
    scale[::static_every] = 0.25  # some static windows below the frozen threshold
    target = (target*scale[:, None, None]).astype(np.float32)
    mask = np.repeat(available[..., None], 20, axis=2)
    target[~mask] = 0
    pred = (signal*0.8*scale[:, None, None]).astype(np.float32)
    meta = []
    for i in range(n):
        meta.append({"user": users[i // windows], "recording": i // windows, "window": i % windows,
                     "n_patches": lengths[i]})
    from pose_temponet import metrics as M
    pv = M.dynamic_variance(target, mask).astype(np.float32)
    return {"pre": pre, "lat": x, "key_time_valid": available.copy(), "pred": pred, "target": target,
            "mask": mask, "std": np.full((n, 20), 0.3, np.float32), "pose_var": pv, "meta": meta,
            "config": {"split": split, "which": "teacher", "checkpoint_step": 49046},
            "has_head": True, "dim": 384, "n_recordings": len(users)}


@pytest.fixture
def chain(prepared, monkeypatch, tmp_path):
    repo, plan = prepared
    plan = release(repo, plan, monkeypatch)
    plan["jobs"] = E.job_graph()
    plan["operator_checks"] = {"work_free_bytes": 13*1024**3, "pending_write_bytes": 0}
    original_validate = P.validate_plan
    monkeypatch.setattr(P, "validate_plan", lambda p, **kw: original_validate(p, **{**kw, "source_root": repo}))
    monkeypatch.setattr(R.train, "configure_numerics", lambda: None)  # STUB: env OMP=1 on the Mac
    monkeypatch.setattr(P, "numeric_context", lambda kind: numeric_fixture(kind))
    rng = np.random.default_rng(20261119)
    W = rng.normal(size=(384, 20))/np.sqrt(384)
    caches = {"train": build_cache(plan["producers"]["train"]["users"], int(os.environ.get("E2E_WTRAIN", "7")), "train", rng, W),
              "val": build_cache(plan["producers"]["val"]["users"], 2, "val", rng, W)}
    folds = D.alternating_user_folds(caches["train"])
    monkeypatch.setattr(P, "preflight", lambda p: {"integrity": True, "historical_R1": {"passed": True},
                                                   "train": {"folds": copy.deepcopy(folds)}})
    opened = []
    def load(p, split, *, stage, depth=None):
        assert "cache_"+split in P.stage_inputs(p, stage)
        opened.append((stage, split, depth))
        data = copy.deepcopy(caches[split])
        data["lengths"] = D.lengths_for(data)
        data["central_mask"] = D.central_support(data["mask"], data["lengths"], data["key_time_valid"])
        if depth is not None:
            data["features"] = data["pre" if depth == "pre" else "lat"]
            del data["pre"], data["lat"]
            data["feature_depth"] = depth
        data["input_verification"] = {"fixture": True}
        return data
    monkeypatch.setattr(P, "read_verified_inputs", load)
    cpu, gpu = cost_receipts()
    for measure in (cpu, gpu):
        measure["input_identity"] = R._input_identity(plan)
    gpu["persistent_output_bytes"] = 7*1024**3
    monkeypatch.setattr(benchmark, "benchmark_cpu", lambda cache, **kw: copy.deepcopy(cpu))
    monkeypatch.setattr(benchmark, "benchmark_gpu", lambda cache, **kw: copy.deepcopy(gpu))
    old_post = caches["val"]["pred"][:, 6:-6].astype(np.float64)*0.9
    old_trained = caches["val"]["pred"][:, 6:-6].astype(np.float64)
    def read_predictions(pin):
        if pin == plan["inputs"]["post_ridge_predictions"]:
            return old_post.copy()
        assert pin == plan["inputs"]["trained_head_predictions"], pin
        return old_trained.copy()
    monkeypatch.setattr(R.data, "read_predictions", read_predictions)
    # SLURM allocation stubs, real runner._allocation (B-2 patched in copy)
    current = {}
    original_check_output = R.subprocess.check_output
    def check_output(args, *a, **kw):
        if list(args[:3]) == ["scontrol", "show", "job"]:
            job = current["job"]
            r = job["resources"]
            minutes = -(-r["requested_seconds"]//60)  # Slurm rounds --time up to minutes
            limit = "%02d:%02d:00" % (minutes//60, minutes % 60)
            tres = f"cpu={r['cpus']},mem={r['host_ram_gib']}G,node=1" + (",gres/gpu=1" if r["gpus"] else "")
            return (f"JobId={args[3]} JobName=x TimeLimit={limit} Partition={r['partition']} "
                    f"NumNodes=1 NumCPUs={r['cpus']} AllocTRES={tres}\n")
        return original_check_output(args, *a, **kw)
    monkeypatch.setattr(R.subprocess, "check_output", check_output)
    monkeypatch.setattr(R.torch.cuda, "device_count", lambda: 1)
    def enter(job, number):
        current["job"] = job
        r = job["resources"]
        for k, v in {"SLURM_JOB_ID": str(70000000+number), "SLURM_CPUS_PER_TASK": str(r["cpus"]),
                     "SLURM_GPUS_ON_NODE": str(r["gpus"]), "SLURM_NNODES": "1",
                     "SLURM_MEM_PER_NODE": str(r["host_ram_gib"]*1024), "SLURM_RESTART_COUNT": "0",
                     "TCN_JOB_ID": job["id"]}.items():
            monkeypatch.setenv(k, v)
    return repo, plan, opened, enter, caches, tmp_path


def write_plan(tmp_path, plan, name):
    path = tmp_path/(name+".json")
    pin = P.atomic_json(path, plan, root=tmp_path)
    return path, pin


def test_e2e(chain, monkeypatch):
    repo, plan, opened, enter, caches, tmp_path = chain
    results = []
    plan_path, plan_pin = write_plan(tmp_path, plan, "initial_plan")
    number = 0
    stop_on_fail = os.environ.get("E2E_STOP", "1") == "1"
    measured = None
    for job in plan["jobs"]:
        if job["stage"] not in ("preflight", "cpu_benchmark", "gpu_benchmark"):
            continue
        number += 1
        enter(job, number)
        t0 = time.time()
        try:
            CLI.main(["run", "--plan", str(plan_path), "--plan-sha256", plan_pin["sha256"], "--job-id", job["id"]])
            note(stage=job["stage"], id=job["id"], status="completed", seconds=round(time.time()-t0, 1))
        except Exception as error:
            note(stage=job["stage"], id=job["id"], status="FAILED", error=repr(error), tb=traceback.format_exc())
            raise
    # Operator phase amendment through the CLI resource-plan command
    loaded = P.load_plan(plan_path, require_review=True)
    receipts = {s: P.receipt_path(loaded, E.job_id(s)) for s in ("cpu_benchmark", "gpu_benchmark")}
    measured_path = tmp_path/"measured_plan.json"
    try:
        CLI.main(["resource-plan", "--plan", str(plan_path), "--cpu-receipt", str(receipts["cpu_benchmark"]),
                  "--gpu-receipt", str(receipts["gpu_benchmark"]), "--output", str(measured_path)])
        note(stage="resource-plan", status="completed")
    except Exception as error:
        note(stage="resource-plan", status="FAILED", error=repr(error), tb=traceback.format_exc())
        raise
    measured_pin = P.file_pin(measured_path)
    measured = json.loads(measured_path.read_text())
    seconds = {j["id"]: j["resources"]["requested_seconds"] for j in measured["jobs"]}
    note(stage="measured", requested_seconds=seconds)
    skip = set(os.environ.get("E2E_SKIP_STAGES", "").split(",")) - {""}
    for job in measured["jobs"]:
        if job["stage"] in ("preflight", "cpu_benchmark", "gpu_benchmark"):
            continue
        if job["stage"] in skip:
            continue
        number += 1
        enter(job, number)
        t0 = time.time()
        try:
            CLI.main(["run", "--plan", str(measured_path), "--plan-sha256", measured_pin["sha256"],
                      "--job-id", job["id"]])
            note(stage=job["stage"], id=job["id"], status="completed", seconds=round(time.time()-t0, 1))
        except Exception as error:
            receipt = P.receipt_path(measured, job["id"])
            status = json.loads(Path(receipt).read_text())["status"] if Path(receipt).exists() else None
            note(stage=job["stage"], id=job["id"], status="FAILED", receipt_status=status,
                 error=repr(error), tb=traceback.format_exc(), seconds=round(time.time()-t0, 1))
            if stop_on_fail:
                break
    (OUT/"log.json").write_text(json.dumps(LOG, indent=1, default=str))
    (OUT/"opened.json").write_text(json.dumps(opened))
    root = Path(measured["output_root"])
    if (root/"metrics/report.json").exists():
        rep = json.loads((root/"metrics/report.json").read_text())
        note(report_complete=rep.get("complete"), status=rep.get("status"), reason=rep.get("reason"),
             candidacy=rep.get("candidacy"))
    (OUT/"output_root.txt").write_text(str(root))
