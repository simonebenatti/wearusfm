"""Run the literal scripts/contrastive_run.main() (fbfde797 bytes) on four CPU processes.

Only the evidence validation (with an injected per-rank delay), CUDA calls, the
numeric profile, the trace/parity observers and R.train are stubbed. Lines 309-373
of contrastive_run.py execute as written, with a REAL torch.distributed
init_process_group (env:// rendezvous) and a lazy backend that, like
ProcessGroupNCCL in 2.2.1, communicates nothing in its constructor.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import socket
import sys
import tempfile
import time
from datetime import timedelta

CLONE = sys.argv[1]
WORKDIR = sys.argv[2]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cfg(dict):
    """dict (prepare_reload uses it as the raw config) with the attributes main() reads."""
    def __init__(self, pose_root):
        super().__init__(max_steps=576)
        self.loader = type("L", (), {"pose_root": pose_root})()
        self.nested = None
        self.max_steps = 576


def worker(rank, env, files, delays, eps, case, gate, queue):
    os.environ.update(env)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank))
    sys.path[:0] = [CLONE + "/scripts", CLONE + "/src"]
    import torch
    import torch.distributed as dist
    from torch._C._distributed_c10d import FakeProcessGroup
    import contrastive_run as CR
    import contrastive_numeric_profile as NP
    import contrastive_target_resume as TR

    log = {"rank": rank}
    out = pathlib.Path(files["out"])
    stamp = time.time

    real_exists, real_mkdir = pathlib.Path.exists, pathlib.Path.mkdir

    def exists(self, *a, **k):
        value = real_exists(self, *a, **k)
        if self == out and "check" not in log:
            log["check"], log["out_existed"] = stamp(), value
        return value

    def mkdir(self, *a, **k):
        result = real_mkdir(self, *a, **k)
        if self == out and "mkdir" not in log:
            log["mkdir"] = stamp()
        return result

    pathlib.Path.exists, pathlib.Path.mkdir = exists, mkdir

    def lazy(common_opts, backend_opts):
        return FakeProcessGroup._create_internal(common_opts.group_rank, common_opts.group_size, backend_opts)

    dist.Backend.register_backend("lazy", lazy, extended_api=True, devices=["cpu"])
    real_init = dist.init_process_group

    def init(backend):
        assert backend == "nccl"
        real_init("lazy", init_method="env://", timeout=timedelta(seconds=12))
        log["init_returned"] = stamp()

    def barrier():  # NCCL barrier semantics: every rank waits for all four
        store = dist.distributed_c10d._get_default_store()
        store.add("emulated_barrier", 1)
        deadline = time.time() + 15
        while store.add("emulated_barrier", 0) < 4:
            if time.time() > deadline:
                raise TimeoutError("emulated NCCL barrier: a rank never arrived")
            time.sleep(0.01)

    cfg = Cfg(files["pose_root"])
    cohort = None if case["stage"] == "calibration" else {"initial_state_sha256": "0" * 64}

    def slow_validation(*a, **k):
        time.sleep(delays[rank])
        return cohort

    CR.validate_recipe = lambda *a, **k: cfg
    if case.get("reload"):
        CR.validate_release = lambda *a, **k: cohort
        CR.validate_target_fork = slow_validation
        TR.validate_fork = lambda *a, **k: {}
    else:
        CR.validate_release = slow_validation
    CR.R.check_input_files = lambda *a, **k: None
    CR.R.config_to_dict = lambda c: dict(c)
    CR.R.train = lambda *a, **k: (time.sleep(1.0), {"steps": cfg.max_steps, "stopped": "max_steps"})[1]
    torch.cuda.is_available = lambda: True
    torch.cuda.set_device = lambda i: time.sleep(eps)
    dist.init_process_group, dist.barrier = init, barrier
    torch.distributed.init_process_group, torch.distributed.barrier = init, barrier
    NP.numeric_profile = contextlib.contextmanager(lambda **k: (yield {"profile": "emulated"}))
    TR.trace_resume_updates = contextlib.contextmanager(lambda *a, **k: (yield None))
    import contrastive_target_parity as TP
    TP.observe_control = contextlib.contextmanager(lambda *a, **k: (yield None))

    sys.argv = ["contrastive_run.py", "--config", files["config"], "--stage", case["stage"],
                "--release", files["release"], "--manifest", files["manifest"], "--root", files["root"],
                "--scales", files["scales"], "--window-seconds", files["window"],
                "--out-dir", files["out"], "--time-limit-s", "1770"]
    gate.wait(timeout=120)
    log["start"] = stamp()
    try:
        CR.main()
        log["status"] = "main() completed"
    except BaseException as error:  # noqa: BLE001 - record every outcome
        log["status"] = type(error).__name__ + ": " + str(error).splitlines()[0][:100]
        log["at"] = stamp()
    queue.put(log)


def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value))
    return str(path)


def run(name, case, delays, eps=0.4, agent=False):
    sys.path[:0] = [CLONE + "/scripts", CLONE + "/src"]
    from contrastive_prepare import source_digest
    base = pathlib.Path(tempfile.mkdtemp(dir=WORKDIR))
    (base / "root").mkdir()
    fork = base / "checkpoint_step0000544.pt"
    fork.write_bytes(b"emulated fork bytes")
    release = {"stage": case["stage"], "source_sha256": source_digest(), "inputs": {"manifest": "0" * 64}}
    if case["stage"] == "benchmark":
        release["phase"] = case["phase"]
        release["fork_checkpoint"] = {"path": str(fork), "sha256": hashlib.sha256(fork.read_bytes()).hexdigest()}
    files = {"out": str(base / "out"), "root": str(base / "root"), "pose_root": str(base),
             "config": write(base / "config.json", {"contrastive": {"arm": case.get("arm", "C0")}}),
             "release": write(base / "release.json", release),
             "manifest": write(base / "manifest.json", {}),
             "scales": write(base / "scales.json", {"scales": {}}),
             "window": write(base / "window.json", {"window_s": {}})}
    port = free_port()
    env = {"WORLD_SIZE": "4", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
           "TORCHELASTIC_RESTART_COUNT": "0", "TORCH_DIST_INIT_BARRIER": "0",
           "TORCHELASTIC_USE_AGENT_STORE": "True" if agent else "False"}
    server = None
    if agent:
        from torch.distributed import TCPStore
        server = TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False, timeout=timedelta(seconds=30))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    gate = ctx.Barrier(4)  # all four processes enter main() together, after their imports
    procs = [ctx.Process(target=worker, args=(r, env, files, delays, eps, case, gate, queue)) for r in range(4)]
    for p in procs:
        p.start()
    logs = sorted((queue.get(timeout=90) for _ in procs), key=lambda row: row["rank"])
    for p in procs:
        p.join(timeout=30)
    del server
    origin = min(row["start"] for row in logs if "start" in row)
    for row in logs:
        for key in ("start", "check", "init_returned", "mkdir", "at"):
            if key in row:
                row[key] = round(row[key] - origin, 3)
    print(json.dumps({"scenario": name, "agent_store": agent, "delays_s": delays, "eps_s": eps,
                      "out_contents": sorted(os.listdir(files["out"])) if os.path.isdir(files["out"]) else None,
                      "ranks": logs}), flush=True)


if __name__ == "__main__":
    import torch
    print(json.dumps({"torch": torch.__version__}), flush=True)
    only = set(sys.argv[3:])
    if not only or "H1" in only: run("H1 calibration, rank 2 late 3 s", {"stage": "calibration"}, [0, 0, 3, 0])
    if not only or "H2" in only: run("H2 benchmark full CN, rank 2 late 3 s", {"stage": "benchmark", "phase": "full", "arm": "CN"}, [0, 0, 3, 0])
    if not only or "H3" in only: run("H3 benchmark full CN, rank 0 late 3 s", {"stage": "benchmark", "phase": "full", "arm": "CN"}, [3, 0, 0, 0])
    if not only or "H4" in only: run("H4 benchmark full CN, no lag", {"stage": "benchmark", "phase": "full", "arm": "CN"}, [0, 0, 0, 0])
    if not only or "H7" in only: run("H7 full C0, rank 1 late 3 s", {"stage": "benchmark", "phase": "full", "arm": "C0"}, [0, 3, 0, 0])
    if not only or "H5" in only: run("H5 reload resume_a CN, rank 2 late 3 s", {"stage": "benchmark", "phase": "resume_a", "arm": "CN", "reload": True}, [0, 0, 3, 0])
    if not only or "H6" in only: run("H6 reload resume_a CN, agent store (torch>=2.4), rank 2 late 3 s",
        {"stage": "benchmark", "phase": "resume_a", "arm": "CN", "reload": True}, [0, 0, 3, 0], agent=True)
