"""CPU-only tests of abrupt-exit evidence and PID cleanup, never CUDA claims."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("rank_exit_test", Path(__file__).resolve().parents[2]/"scripts/contrastive_rank_exit.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def fake_receipt(path):
    path.mkdir()
    workers = [{"rank": r, "world_size": 4, "pid": 10000+r, "ppid": 9000, "pgid": 10000+r,
                "start_token": "Mon Nov 16 12:00:00 2026", "stat": "S",
                "cuda_initialized": True, "nccl_initialized": True} for r in range(4)]
    for w in workers:
        (path/f"worker_{w['rank']}.json").write_text(json.dumps(w))
    (path/"stdout.log").write_text(E.SENTINEL+"\n")
    (path/"stderr.log").write_text("rank 0 exited with an intentional error\n")
    report = {"protocol": E.PROTOCOL, "passed": True, "source_sha256": "a"*64,
              "host": E.socket.gethostname(),
              "world_size": 4, "all_workers_initialized": True, "workers": workers,
              "tracked_processes": [{"pid": 9000, "start_token": "Mon Nov 16 12:00:00 2026"}, *workers],
              "launcher_pid": 9000, "launcher_returncode": 1, "sentinel": E.SENTINEL,
              "timed_out": False, "elapsed_s": 12., "completion_limit_s": 60, "cleanup_limit_s": 20,
              "remaining_processes": [], "failure": None,
              "stdout_sha256": E._sha(path/"stdout.log"), "stderr_sha256": E._sha(path/"stderr.log"),
              "worker_evidence_sha256": {f"worker_{w['rank']}.json": E._sha(path/f"worker_{w['rank']}.json") for w in workers}}
    (path/"report.json").write_text(json.dumps(report))
    return report


def test_process_table_parses_identity_and_keeps_zombies(monkeypatch):
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a,0,
        " 123 1 123 Z Mon Nov 16 12:00:00 2026\n 456 123 456 S Mon Nov 16 12:00:01 2026\n", ""))
    table = E.process_table()
    assert table[123]["stat"] == "Z"
    tracked = {}
    E._observe_children(table, 123, tracked)
    assert set(tracked) == {123,456}
    assert E._same_process(tracked[123], table)
    reused = copy.deepcopy(table); reused[123]["start_token"] = "Mon Nov 16 12:00:05 2026"
    assert not E._same_process(tracked[123], reused)
    E._observe_children(reused,123,tracked)
    assert tracked[123]["start_token"] != reused[123]["start_token"]


def test_later_readout_on_other_host_does_not_inspect_unrelated_pid_namespace(tmp_path,monkeypatch):
    fake_receipt(tmp_path/"probe")
    monkeypatch.setattr(E.socket,"gethostname",lambda:"different-readout-node")
    monkeypatch.setattr(E,"process_table",lambda:pytest.fail("PIDs on another host have no evidential meaning"))
    assert E.validate_receipt(tmp_path/"probe/report.json","a"*64)["passed"]


@pytest.mark.parametrize("error",[FileNotFoundError("ps"),subprocess.TimeoutExpired("ps",5)])
def test_ps_missing_or_timed_out_fails_closed(monkeypatch,error):
    def fail(*args,**kwargs): raise error
    monkeypatch.setattr(E.subprocess,"run",fail)
    with pytest.raises(RuntimeError,match="cleanup"):
        E.process_table()


def test_complete_receipt_accepts_only_original_processes_gone(tmp_path,monkeypatch):
    report = fake_receipt(tmp_path/"probe")
    monkeypatch.setattr(E,"process_table",lambda:{10000:{"pid":10000,"start_token":"different birth"}})
    assert E.validate_receipt(tmp_path/"probe/report.json","a"*64) == report
    monkeypatch.setattr(E,"process_table",lambda:{10000:report["workers"][0]})
    with pytest.raises(ValueError,match="including zombies"):
        E.validate_receipt(tmp_path/"probe/report.json","a"*64)


@pytest.mark.parametrize("change",["zero_exit","too_slow","missing_worker","no_cuda","no_marker","changed_log","wrong_source","timeout"])
def test_nonzero_alone_cannot_pass(tmp_path,monkeypatch,change):
    folder=tmp_path/"probe"; report=fake_receipt(folder)
    monkeypatch.setattr(E,"process_table",lambda:{})
    source="a"*64
    if change=="zero_exit": report["launcher_returncode"]=0
    elif change=="too_slow": report["elapsed_s"]=60.1
    elif change=="missing_worker": (folder/"worker_3.json").unlink()
    elif change=="no_cuda":
        w=report["workers"][0];w["cuda_initialized"]=False;(folder/"worker_0.json").write_text(json.dumps(w))
    elif change=="no_marker":
        (folder/"stdout.log").write_text("some other startup error");report["stdout_sha256"]=E._sha(folder/"stdout.log")
    elif change=="changed_log": (folder/"stderr.log").write_text("modified")
    elif change=="wrong_source": source="b"*64
    else: report["timed_out"]=True
    (folder/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        E.validate_receipt(folder/"report.json",source)


def test_probe_runs_no_child_if_process_inspection_unavailable(tmp_path,monkeypatch):
    def fail(): raise RuntimeError("ps unavailable")
    monkeypatch.setattr(E,"process_table",fail)
    monkeypatch.setattr(E.subprocess,"Popen",lambda *a,**k: pytest.fail("must not start torchrun"))
    with pytest.raises(RuntimeError,match="ps"):
        E.run_probe(tmp_path/"probe",expected_source_sha256="a"*64)


def test_cpu_child_startup_failure_rejected_with_mocked_process_inspection(tmp_path,monkeypatch):
    # Exercise parent supervision with an ordinary failing CPU child. This is
    # deliberately rejected: only real four-rank CUDA evidence can pass. Process
    # inspection is mocked here because ps is forbidden by the Mac sandbox;
    # parser and unavailable-ps behavior have separate tests above.
    monkeypatch.setattr(E,"process_table",lambda:{})
    monkeypatch.setattr(E,"_launch_command",lambda folder:[E.sys.executable,"-c","raise RuntimeError('startup failure')"])
    report=E.run_probe(tmp_path/"probe",expected_source_sha256="a"*64)
    assert not report["passed"] and not report["all_workers_initialized"]
    assert not report["remaining_processes"]
    assert "four workers" in report["failure"]
