"""GPT regression for fd874d7: detached elastic-like workers must not survive.

These CPU-only tests intentionally fail against fd874d7. All processes belong
to the test, and every test cleans up its own detached process groups in finally.
No cluster, data, CUDA, training or production-code changes are involved.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/logsquare_initialization_diagnostic.py"
spec = importlib.util.spec_from_file_location("logsquare_timeout_review", SOURCE)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def elastic_like_launcher(directory):
    """A launcher whose worker, like an elastic worker, has a new session."""
    worker_pid = directory / "worker.pid"
    launcher_pid = directory / "launcher.pid"
    heartbeat = directory / "worker.heartbeat"
    worker = (
        "import os,time\n"
        f"open({str(worker_pid)!r}, 'w').write(str(os.getpid()))\n"
        "while True:\n"
        f"    open({str(heartbeat)!r}, 'w').write(str(time.monotonic_ns()))\n"
        "    time.sleep(0.02)\n"
    )
    launcher = (
        "import os,subprocess,sys,time\n"
        f"open({str(launcher_pid)!r}, 'w').write(str(os.getpid()))\n"
        f"worker = subprocess.Popen([sys.executable, '-c', {worker!r}], start_new_session=True)\n"
        "worker.wait()\n"
    )
    return launcher, (launcher_pid, worker_pid), heartbeat


def worker_is_still_executing(heartbeat):
    """Check forward progress, not kill(pid, 0), which also sees zombies."""
    assert heartbeat.is_file(), "worker never started; reproduction is inconclusive"
    before = heartbeat.read_text()
    time.sleep(0.2)
    return heartbeat.read_text() != before


def cleanup_own_process_groups(pid_files):
    # Each of these PIDs was written by a process created by this test in its
    # own session. No global process scan, group zero, or unrelated PID is used.
    for path in reversed(pid_files):
        if path.exists():
            pid = int(path.read_text())
            assert pid > 1
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_inner_timeout_terminates_detached_elastic_like_worker(tmp_path):
    launcher, pid_files, heartbeat = elastic_like_launcher(tmp_path)
    try:
        status, timed_out = D.run_bounded(
            [sys.executable, "-c", launcher], 1,
            tmp_path / "stdout", tmp_path / "stderr",
        )
        assert timed_out and status != 0
        assert not worker_is_still_executing(heartbeat), (
            "run_bounded returned after its timeout, but the detached "
            "elastic-like worker is still executing"
        )
    finally:
        cleanup_own_process_groups(pid_files)


def test_outer_execute_timeout_terminates_nested_launcher_and_worker(tmp_path, monkeypatch):
    launcher, pid_files, heartbeat = elastic_like_launcher(tmp_path)
    # This child stands in for rank_entry_child, using the real production
    # run_bounded with a longer inner deadline. The real execute's outer
    # subprocess.run timeout fires first, as when preparation consumed the
    # unbounded 30-second gap between the real 300/270-second deadlines.
    helper = tmp_path / "rank_entry_helper.py"
    helper.write_text(
        "import importlib.util,sys\n"
        f"s = importlib.util.spec_from_file_location('nested_diagnostic', {str(SOURCE)!r})\n"
        "d = importlib.util.module_from_spec(s); s.loader.exec_module(d)\n"
        f"d.run_bounded([sys.executable, '-c', {launcher!r}], 30, "
        f"{str(tmp_path / 'nested.stdout')!r}, {str(tmp_path / 'nested.stderr')!r})\n"
    )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"test": "outer_timeout"}))
    monkeypatch.setattr(D, "VARIANTS", (D.RANK_VARIANT,))
    monkeypatch.setattr(D, "__file__", str(helper))
    monkeypatch.setattr(D, "code_identity", lambda expected: expected)
    monkeypatch.setattr(
        D, "validate_plan",
        lambda _: ({"child_timeout_seconds": 1, "rank_entry_timeout_seconds": 1}, None, None, {}, None),
    )
    try:
        report = D.execute(plan, "a" * 40, tmp_path / "output")
        assert report["complete"] is False
        assert report["variants"][D.RANK_VARIANT]["error"] == "child timed out"
        assert not worker_is_still_executing(heartbeat), (
            "execute returned after its outer timeout, but the nested "
            "launcher's detached worker is still executing"
        )
    finally:
        cleanup_own_process_groups(pid_files)
