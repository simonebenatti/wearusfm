"""Preflight job cells cannot change their reviewed stage, inputs or allocation."""
import copy
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_job as J
import contrastive_run as runner


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "wearusfm_runs").mkdir(parents=True)
    roots = [tmp_path / "data1", tmp_path / "data2"]
    for path in roots: path.mkdir()
    monkeypatch.setenv("WORK", str(work))
    monkeypatch.setattr(J, "DATA_ROOTS", set(roots))
    monkeypatch.setattr(J, "source_digest", lambda: "a" * 64)
    monkeypatch.setattr(J, "validate_input_bindings", lambda _: None)
    monkeypatch.setattr(runner, "validate_recipe", lambda *args: None)
    def pin(name, payload=None):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({} if payload is None else payload))
        return {"path": str(path), "sha256": runner.sha(path)}
    def build(mode):
        job = {"mode": mode, "status": "APPROVED_FOR_PREFLIGHT", "source_sha256": "a" * 64,
               "review": pin(mode + "/review.json"), "out": str(work / "wearusfm_runs" / mode),
               "config": pin(mode + "/config.json"), "roots": list(map(str, roots)),
               "inputs": {key: pin(mode + "/" + key + ".json") for key in ("manifest", "scales", "window_seconds", "pose_stats")},
               "audit": pin(mode + "/audit.json"), "cohort": pin(mode + "/cohort.json"),
               "calibration_report": pin(mode + "/calibration.json"),
               "calibration_summary": pin(mode + "/cal/summary.json"),
               "cuda_report": pin(mode + "/cuda.json"), "resume_config": pin(mode + "/full.json")}
        job["calibration_checkpoint"] = pin(mode + "/checkpoint.pt")
        release = {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration" if mode == "calibration" else "benchmark",
                   "phase": mode, "source_sha256": job["source_sha256"], "config_sha256": job["config"]["sha256"],
                   "time_limit_s": 1800 if mode == "split" else 1770,
                   "inputs": {k: v["sha256"] for k, v in job["inputs"].items()},
                   "audit_sha256": job["audit"]["sha256"], "cohort_sha256": job["cohort"]["sha256"],
                   "calibration_sha256": job["calibration_report"]["sha256"]}
        job["release"] = pin(mode + "/release.json", release)
        job["full"] = {arm: pin(mode + "/full/" + arm + "/summary.json") for arm in ("C0", "CN", "CR")}
        job["resumed"] = {arm: pin(mode + "/resumed/" + arm + "/summary.json") for arm in ("CN", "CR")}
        return job
    return build, pin


@pytest.mark.parametrize("mode,kind,script", [("audit", "cpu", "contrastive_audit.py"),
    ("readout", "cpu", "contrastive_benchmark_readout.py"), ("cuda", "gpu", "contrastive_cuda_check.py"),
    ("calibration", "gpu", "contrastive_run.py"), ("full", "gpu", "contrastive_run.py"),
    ("split", "gpu", "contrastive_split_benchmark.py")])
def test_stage_commands_use_frozen_absolute_scripts_and_fixed_resources(job_env, mode, kind, script):
    job = job_env[0](mode)
    command = J.build_command(job, kind, 1800 if kind == "gpu" else 3600)
    assert str(ROOT / "scripts" / script) in command
    assert "--nproc_per_node=4" in command if mode in {"cuda", "calibration", "full"} else "--nproc_per_node=4" not in command
    assert "--stage" not in command or command[command.index("--stage") + 1] != "training"
    if mode == "split": assert command[command.index("--total-time-limit-s") + 1] == "1800"


@pytest.mark.parametrize("field,value", [("status", "REVIEW_REQUIRED"), ("source_sha256", "b" * 64), ("mode", "training")])
def test_jobs_reject_unreviewed_source_status_or_stage(job_env, field, value):
    job = job_env[0]("full")
    job[field] = value
    with pytest.raises(ValueError, match="wrong mode"):
        J.build_command(job, "gpu", 1800)


def test_jobs_reject_wrong_allocation_kind_time_and_input_pin(job_env):
    job = job_env[0]("full")
    for kind, seconds in (("cpu", 3600), ("gpu", 3600), ("gpu", 1799)):
        with pytest.raises(ValueError): J.build_command(job, kind, seconds)
    Path(job["inputs"]["manifest"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="pin mismatch"):
        J.build_command(job, "gpu", 1800)


def test_jobs_reject_output_escape_existing_output_and_wrong_roots(job_env, tmp_path):
    job = job_env[0]("audit")
    bad = copy.deepcopy(job)
    bad["out"] = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="fresh output"):
        J.build_command(bad, "cpu", 3600)
    Path(job["out"]).mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        J.build_command(job, "cpu", 3600)
    job = job_env[0]("cuda")
    job["roots"] = [str(tmp_path)]
    with pytest.raises(ValueError, match="data roots"):
        J.build_command(job, "gpu", 1800)


def test_job_mode_cannot_run_a_different_release_phase(job_env):
    job = job_env[0]("full")
    release = J._read(job["release"]["path"])
    release["phase"] = "split"
    job["release"] = job_env[1]("changed_release.json", release)
    with pytest.raises(ValueError, match="job stage"):
        J.build_command(job, "gpu", 1800)


def test_calibration_evidence_requires_the_actual_summary_file(job_env):
    job = job_env[0]("full")
    job["calibration_summary"] = job_env[1]("unrelated.json")
    with pytest.raises(ValueError, match="summary.json"):
        J.build_command(job, "gpu", 1800)


def test_readout_requires_all_full_arms_and_both_resumed_arms(job_env):
    job = job_env[0]("readout")
    job["resumed"].pop("CR")
    with pytest.raises(ValueError, match="arm set"):
        J.build_command(job, "cpu", 3600)


def test_allocation_guard_checks_gpu_cpu_memory_and_nodes(monkeypatch):
    for key, value in {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "32", "SLURM_MEM_PER_NODE": str(240 * 1024)}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(J.torch.cuda, "device_count", lambda: 4)
    J.validate_allocation("gpu")
    for key, value in (("SLURM_JOB_NUM_NODES", "2"), ("SLURM_NTASKS", "2"), ("SLURM_CPUS_PER_TASK", "8"), ("SLURM_MEM_PER_NODE", "1")):
        with monkeypatch.context() as patch:
            patch.setenv(key, value)
            with pytest.raises(ValueError, match="allocation"): J.validate_allocation("gpu")
    monkeypatch.setattr(J.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="four GPUs"): J.validate_allocation("gpu")


def test_cpu_allocation_accepts_30_gib_and_rejects_32_gib(monkeypatch):
    for key, value in {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8",
                       "SLURM_MEM_PER_NODE": str(30 * 1024)}.items():
        monkeypatch.setenv(key, value)
    J.validate_allocation("cpu")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(32 * 1024))
    with pytest.raises(ValueError, match="allocation"):
        J.validate_allocation("cpu")


def test_cuda_command_requires_the_same_invocations_rank_exit_receipt(job_env):
    job = job_env[0]("cuda")
    command = J.build_command(job, "gpu", 1800)
    assert command[command.index("--rank-exit-report") + 1] == str(J.rank_exit_report(job["out"]))


def test_failed_rank_probe_prevents_receipt_acceptance(job_env, monkeypatch):
    import types
    import contrastive_split_benchmark as split
    observed = []
    monkeypatch.setitem(sys.modules, "contrastive_rank_exit", types.SimpleNamespace(validate_receipt=lambda *a: observed.append(a)))
    def failed(*args):
        raise RuntimeError("rank probe failed")
    monkeypatch.setattr(split, "checked_command", failed)
    with pytest.raises(RuntimeError, match="rank probe failed"):
        J.prepare_cuda_probe(job_env[0]("cuda"))
    assert observed == []
