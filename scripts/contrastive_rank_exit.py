"""Bounded four-rank CUDA/NCCL exception probe; no training or submission.

The parent requires all four initialized workers, the intentional rank-0 error,
nonzero torchrun completion, and disappearance of every observed child process.
PID plus process start time prevents confusing a reused PID with our worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

PROTOCOL = "contrastive-rank-exit-v1"
SENTINEL = "WEARUSFM_INTENTIONAL_RANK_ZERO_EXCEPTION_20261116"
WORLD_SIZE = 4
COMPLETION_SECONDS = 60
CLEANUP_SECONDS = 20


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_table():
    """Read identities, including zombies; missing/broken ps is a hard failure."""
    try:
        result = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid=,stat=,lstart="],
                                text=True, capture_output=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot attest child-process cleanup: ps unavailable/failed") from exc
    table = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5 or not all(x.isdigit() for x in fields[:3]) or not fields[3] or not fields[4]:
            raise ValueError("invalid ps process-identity output")
        pid, ppid, pgid = map(int, fields[:3])
        table[pid] = {"pid": pid, "ppid": ppid, "pgid": pgid,
                      "stat": fields[3], "start_token": fields[4]}
    if not table:
        raise ValueError("empty ps process-identity output")
    return table


def _same_process(identity, table):
    return table.get(identity["pid"], {}).get("start_token") == identity["start_token"]


def _observe_children(table, launcher_pid, tracked):
    parents = {p for p, identity in tracked.items() if _same_process(identity, table)}
    if launcher_pid not in tracked and launcher_pid in table:
        tracked[launcher_pid] = table[launcher_pid]
        parents.add(launcher_pid)
    changed = True
    while changed:
        changed = False
        for pid, row in table.items():
            if pid in parents or row["ppid"] in parents:
                if pid not in parents:
                    parents.add(pid)
                    changed = True
                tracked.setdefault(pid, row)


def _validate_workers(workers):
    if (len(workers) != WORLD_SIZE or {w.get("rank") for w in workers} != set(range(WORLD_SIZE)) or
            len({w.get("pid") for w in workers}) != WORLD_SIZE):
        raise ValueError("four distinct initialized worker identities required")
    for w in workers:
        if (type(w.get("rank")) is not int or w.get("world_size") != WORLD_SIZE or
                w.get("cuda_initialized") is not True or w.get("nccl_initialized") is not True or
                any(type(w.get(k)) is not int or w[k] <= 0 for k in ("pid", "pgid")) or
                not isinstance(w.get("start_token"), str) or not w["start_token"]):
            raise ValueError("invalid CUDA/NCCL worker initialization evidence")
    return sorted(workers, key=lambda w: w["rank"])


def worker_evidence(folder):
    paths = [Path(folder)/f"worker_{rank}.json" for rank in range(WORLD_SIZE)]
    if not all(p.is_file() for p in paths):
        raise ValueError("torchrun failed before all four workers initialized")
    return _validate_workers([json.loads(p.read_text()) for p in paths])


def _terminate_observed(child, tracked):
    """Bound cleanup; never signal a PID whose original identity has disappeared."""
    deadline = time.monotonic()+CLEANUP_SECONDS
    started = time.monotonic()
    last = []
    while time.monotonic() < deadline:
        table = process_table()
        _observe_children(table, child.pid, tracked)
        last = [row for row in tracked.values() if _same_process(row, table)]
        sig = signal.SIGTERM if time.monotonic()-started < 2 else signal.SIGKILL
        for row in reversed(last):
            try:
                os.kill(row["pid"], sig)
            except ProcessLookupError:
                pass
        # Reap the launcher; an unreaped zombie is not accepted as 'gone'.
        try:
            child.wait(timeout=.1)
        except subprocess.TimeoutExpired:
            pass
        if not last:
            return []
        time.sleep(.1)
    return last


def _launch_command(folder):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            str(Path(__file__).resolve()), "--worker", "--evidence-dir", str(folder)]


def run_probe(folder, *, expected_source_sha256, timeout_s=COMPLETION_SECONDS):
    if timeout_s != COMPLETION_SECONDS:
        raise ValueError("the frozen rank-exit completion deadline is 60 seconds")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    # Verify ps before allocating any children. Failure is explicit and launches nothing.
    process_table()
    started, tracked, workers, remaining = time.monotonic(), {}, [], []
    child, failure, timed_out = None, None, False
    with (folder/"stdout.log").open("w") as stdout, (folder/"stderr.log").open("w") as stderr:
        try:
            child = subprocess.Popen(_launch_command(folder), stdout=stdout, stderr=stderr, start_new_session=True)
            deadline = started+timeout_s
            while True:
                table = process_table()
                _observe_children(table, child.pid, tracked)
                code = child.poll()
                if code is not None:
                    workers = worker_evidence(folder)
                    for w in workers:
                        tracked.setdefault(w["pid"], w)
                    remaining = [row for row in tracked.values() if _same_process(row, table)]
                    if not remaining:
                        break
                if time.monotonic() >= deadline:
                    timed_out = True
                    raise TimeoutError("four-rank exception/cleanup exceeded 60 seconds")
                time.sleep(.1)
            if child.returncode == 0:
                raise ValueError("torchrun unexpectedly succeeded; intentional worker failure not observed")
            if time.monotonic()-started > timeout_s:
                timed_out = True
                raise TimeoutError("processes did not disappear before the 60-second deadline")
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if child is not None:
                try:
                    remaining = _terminate_observed(child, tracked)
                except Exception as cleanup:
                    failure += f"; cleanup not attestable: {type(cleanup).__name__}: {cleanup}"
                    # Stop our known launcher. SLURM cgroup remains outer cleanup;
                    # unverified cleanup can never permit the subsequent harness.
                    if child.poll() is None:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=2)
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            pass
                    remaining = [{"cleanup_unverified": True}]
    elapsed = time.monotonic()-started
    logs = (folder/"stdout.log").read_text()+(folder/"stderr.log").read_text()
    if SENTINEL not in logs:
        failure = (failure+"; " if failure else "")+"intentional rank-zero exception marker missing"
    passed = (failure is None and not timed_out and elapsed <= timeout_s and not remaining and
              child is not None and child.returncode is not None and child.returncode != 0 and len(workers) == 4)
    report = {"protocol": PROTOCOL, "passed": passed, "source_sha256": expected_source_sha256,
              "host": socket.gethostname(),
              "world_size": WORLD_SIZE, "all_workers_initialized": len(workers) == 4,
              "workers": workers, "tracked_processes": list(tracked.values()),
              "launcher_pid": child.pid if child else None, "launcher_returncode": child.returncode if child else None,
              "sentinel": SENTINEL, "timed_out": timed_out, "elapsed_s": elapsed,
              "completion_limit_s": timeout_s, "cleanup_limit_s": CLEANUP_SECONDS,
              "remaining_processes": remaining, "failure": failure,
              "stdout_sha256": _sha(folder/"stdout.log"), "stderr_sha256": _sha(folder/"stderr.log"),
              "worker_evidence_sha256": {f"worker_{w['rank']}.json": _sha(folder/f"worker_{w['rank']}.json") for w in workers}}
    (folder/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def validate_receipt(path, expected_source_sha256):
    path = Path(path)
    report = json.loads(path.read_text())
    if (report.get("protocol") != PROTOCOL or report.get("passed") is not True or
            report.get("source_sha256") != expected_source_sha256 or report.get("world_size") != WORLD_SIZE or
            not isinstance(report.get("host"), str) or not report["host"] or
            report.get("all_workers_initialized") is not True or report.get("sentinel") != SENTINEL or
            report.get("timed_out") is not False or report.get("failure") is not None or
            report.get("completion_limit_s") != COMPLETION_SECONDS or report.get("cleanup_limit_s") != CLEANUP_SECONDS or
            type(report.get("elapsed_s")) not in (float, int) or not 0 < report["elapsed_s"] <= COMPLETION_SECONDS or
            type(report.get("launcher_returncode")) is not int or report["launcher_returncode"] == 0 or
            report.get("remaining_processes") != []):
        raise ValueError("rank-exit receipt did not satisfy the frozen completion/cleanup criteria")
    workers = worker_evidence(path.parent)
    if report.get("workers") != workers or report.get("worker_evidence_sha256") != {
            f"worker_{w['rank']}.json": _sha(path.parent/f"worker_{w['rank']}.json") for w in workers}:
        raise ValueError("rank-exit worker evidence changed")
    logs = ""
    for name in ("stdout", "stderr"):
        p = path.parent/(name+".log")
        if report.get(name+"_sha256") != _sha(p):
            raise ValueError("rank-exit log evidence changed")
        logs += p.read_text()
    if SENTINEL not in logs:
        raise ValueError("rank-exit intentional exception marker missing")
    tracked = report.get("tracked_processes")
    if (not isinstance(tracked, list) or not tracked or
            not {report.get("launcher_pid"), *(w["pid"] for w in workers)} <= {r.get("pid") for r in tracked}):
        raise ValueError("rank-exit receipt lacks observed process identities")
    # The CUDA harness immediately rechecks on the same node. A later CPU
    # readout may be on another node: its PID namespace cannot attest the probe
    # node's processes. It authenticates the recorded completion evidence only.
    if report["host"] == socket.gethostname():
        table = process_table()
        if any(_same_process(identity, table) for identity in tracked):
            raise ValueError("rank-exit launcher/worker process remains, including zombies")
    return report


def worker_main(folder):
    import torch
    import torch.distributed as dist
    rank, local_rank, world_size = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world_size != WORLD_SIZE or torch.cuda.device_count() != WORLD_SIZE:
        raise ValueError("rank-exit probe requires exactly four CUDA ranks/devices")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    # Materialize NCCL, then attest initialization from every rank before failure.
    dist.barrier()
    identity = process_table()[os.getpid()]
    evidence = {**identity, "rank": rank, "world_size": world_size,
                "cuda_initialized": torch.cuda.is_initialized(),
                "nccl_initialized": dist.is_initialized() and dist.get_backend() == "nccl"}
    folder = Path(folder)
    temporary = folder/f"worker_{rank}.tmp"
    temporary.write_text(json.dumps(evidence, sort_keys=True)+"\n")
    temporary.replace(folder/f"worker_{rank}.json")
    dist.barrier()  # all four evidence files exist before rank 0 can raise
    if rank == 0:
        print(SENTINEL, flush=True)
        raise RuntimeError(SENTINEL)
    # The other three ranks must be terminated by the elastic launcher (or
    # observe the failed communicator), rather than complete normally.
    dist.all_reduce(torch.ones(1, device=f"cuda:{local_rank}"))
    raise RuntimeError("rank-exit peer collective unexpectedly completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=COMPLETION_SECONDS, choices=(COMPLETION_SECONDS,))
    args = parser.parse_args()
    if args.worker:
        if args.evidence_dir is None or args.out_dir is not None:
            parser.error("worker requires only --evidence-dir")
        worker_main(args.evidence_dir)
        return
    if args.out_dir is None or args.evidence_dir is not None:
        parser.error("parent requires --out-dir")
    from contrastive_prepare import source_digest
    frozen_source = source_digest()
    report = run_probe(args.out_dir, expected_source_sha256=frozen_source, timeout_s=args.timeout_s)
    if source_digest() != frozen_source:
        raise ValueError("probe source changed during execution")
    if report["passed"]:
        validate_receipt(args.out_dir/"report.json", expected_source_sha256=frozen_source)
    print(json.dumps({"passed": report["passed"], "report": str(args.out_dir/"report.json")}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
