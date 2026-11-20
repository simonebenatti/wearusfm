"""Emulate the rank ordering of scripts/contrastive_run.py:315-358 (fbfde797).

Per rank: simulated validation delay -> the literal exists() guard (lines 318/322)
-> simulated set_device (331) -> REAL torch.distributed.init_process_group with
env:// rendezvous (332) and a lazy backend that, like ProcessGroupNCCL in 2.2.1,
does no communication in its constructor -> simulated provenance/profile (334-357)
-> mkdir (352/358), or for the reload path only rank 0 mkdirs (prepare_reload,
contrastive_target_resume.py:100) before a barrier (347).

Store modes:
  rank0   : TORCHELASTIC_USE_AGENT_STORE=False, as torchrun --standalone sets in
            torch 2.2.1 (use_agent_store = backend == "static"): rank 0 hosts the
            TCPStore with world_size=4 and waits for every worker.
  agent   : the agent hosts the store, all ranks are clients (torch >= 2.4 c10d).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

MESSAGE = "new phase requires a new output directory; continuation must be explicit"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, env, out_dir, delays, eps, delta, path, t0, queue):
    os.environ.update(env)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    import torch.distributed as dist
    from torch._C._distributed_c10d import FakeProcessGroup

    def lazy(common_opts, backend_opts):
        return FakeProcessGroup._create_internal(common_opts.group_rank, common_opts.group_size, backend_opts)

    dist.Backend.register_backend("lazy", lazy, extended_api=True, devices=["cpu"])
    log = {"rank": rank}
    out = Path(out_dir)
    try:
        time.sleep(delays[rank])                      # validation, lines 290-317
        log["check"] = round(time.time() - t0, 3)
        if out.exists():                              # lines 318 / 322
            raise ValueError(MESSAGE)
        time.sleep(eps)                               # cuda.is_available + set_device, 329-331
        dist.init_process_group("lazy", init_method="env://", timeout=timedelta(seconds=12))  # 332
        log["init_returned"] = round(time.time() - t0, 3)
        time.sleep(delta)                             # provenance + numeric_profile, 334-340
        if path == "full" or rank == 0:
            out.mkdir(parents=True, exist_ok=(path == "full"))   # 352/358 or resume.py:100
            log["mkdir"] = round(time.time() - t0, 3)
        log["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - record any outcome
        log["status"] = type(error).__name__ + ": " + str(error).splitlines()[0][:110]
        log["at"] = round(time.time() - t0, 3)
    queue.put(log)


def scenario(name, mode, path, delays, eps=0.4, delta=0.2):
    port = free_port()
    env = {"WORLD_SIZE": "4", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
           "TORCH_DIST_INIT_BARRIER": "0", "TORCHELASTIC_RESTART_COUNT": "0"}
    agent = None
    if mode == "agent":
        from torch.distributed import TCPStore
        agent = TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False,
                         timeout=timedelta(seconds=30))
        env["TORCHELASTIC_USE_AGENT_STORE"] = "True"
    else:
        env["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    tmp = Path(tempfile.mkdtemp(dir=sys.argv[1]))
    out_dir = tmp / "out"
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    t0 = time.time() + 2.0  # common start reference after spawn
    procs = [ctx.Process(target=worker, args=(r, env, str(out_dir), delays, eps, delta, path, t0, queue))
             for r in range(4)]
    for p in procs:
        p.start()
    while time.time() < t0:
        time.sleep(0.01)
    logs = [queue.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=30)
    del agent
    logs.sort(key=lambda row: row["rank"])
    print(json.dumps({"scenario": name, "store": mode, "path": path, "delays_s": delays,
                      "eps_s": eps, "delta_s": delta, "ranks": logs}))


if __name__ == "__main__":
    import torch
    print(json.dumps({"torch": torch.__version__}))
    scenario("S1 full, rank 2 late 3 s", "rank0", "full", [0, 0, 3, 0])
    scenario("S2 full, rank 0 late 3 s", "rank0", "full", [3, 0, 0, 0])
    scenario("S3 full, rank 2 late 0.4 s < window 0.6 s", "rank0", "full", [0, 0, 0.4, 0])
    scenario("S4 reload, rank 2 late 3 s", "rank0", "reload", [0, 0, 3, 0])
    scenario("S5 reload, agent store (torch>=2.4), rank 2 late 3 s", "agent", "reload", [0, 0, 3, 0])
