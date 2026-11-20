"""Skeptic 1 for R6-1: cell arm vs executed config/release arm.

Real: preflight_plan.json cells, bind_review/validate_review_binding (verify_git=True,
synthetic review committed only in the temp clone), validate_audit_accounting with a
synthetic COMPLETED 0:0 sacct row, make_job, build_command.
Stubbed (orthogonal, physical Leonardo files unavailable): J.validate_input_bindings,
J.DATA_ROOTS, and the science-plan input_sha256 seen by make_job (local input hashes).
"""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import contrastive_job as J
import contrastive_release as REL
import contrastive_prepare as P

import subprocess
REVIEW_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=P.ROOT, text=True).strip()
FRESH = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118"
C0 = "CONTRAST-BENCH-C0-targets-v3-20261118"
CNA = "CONTRAST-RELOAD-CN-A-targets-v3-20261118"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(path, payload=None, text=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else json.dumps({} if payload is None else payload))
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


@pytest.fixture
def env(tmp_path, monkeypatch):
    binding = REL.bind_review(REVIEW_COMMIT, "docs/reviews/sk1_probe_review.md",
                              P.ROOT / "docs/reviews/sk1_probe_review.md")
    source = P.source_digest()
    audit = pin(tmp_path / "audit/report.json", {"source_sha256": source})
    sacct_path = tmp_path / "audit/sacct.tsv"
    sacct_path.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n12345|COMPLETED|0:0|7000|8|120\n")
    accounting = pin(tmp_path / "audit/accounting.json", {
        "schema": "contrastive-audit-accounting-v1", "audit_sha256": audit["sha256"], "source_sha256": source,
        "logical_id": "CONTRAST-AUDIT-targets-v3-20261118", "job_id": "12345",
        "sacct": {"path": str(sacct_path), "sha256": sha(sacct_path)}})
    REL.validate_audit_accounting(Path(audit["path"]), accounting)  # real gate accepts the synthetic receipt
    roots = [(tmp_path / "d1"), (tmp_path / "d2")]
    for r in roots:
        r.mkdir()
    monkeypatch.setattr(J, "DATA_ROOTS", {r.resolve() for r in roots})
    monkeypatch.setattr(J, "validate_input_bindings", lambda _: None)
    monkeypatch.setenv("WORK", "/leonardo_work/IscrB_WearUsFM")
    inputs = {k: pin(tmp_path / ("inputs/" + k + ".json"), {"k": k}) for k in ("manifest", "scales", "window_seconds", "pose_stats")}
    real_read = REL._read
    def read(path):
        value = real_read(path)
        if Path(path).resolve() == P.PLAN.resolve():
            value = copy.deepcopy(value)
            value["input_sha256"] = {k: v["sha256"] for k, v in inputs.items()}
        return value
    monkeypatch.setattr(REL, "_read", read)
    common = {"audit": audit, "audit_sacct": accounting, "inputs": inputs, "roots": [str(r) for r in roots],
              "cohort": pin(tmp_path / "cohort/cohort.json", {"c": 1}),
              "calibration_report": pin(tmp_path / "cal/calibration.json", {"cal": 1}),
              "calibration_summary": pin(tmp_path / "cal/run/summary.json", {"s": 1}),
              "cuda_report": pin(tmp_path / "cuda/report.json", {"cuda": 1})}

    def evidence(executed_arm, phase, tag):
        ev = copy.deepcopy(common)
        ev["config"] = pin(tmp_path / ("cfg/" + tag + "_" + executed_arm + ".json"),
                           {"max_steps": 576, "contrastive": {"arm": executed_arm}})
        if phase.startswith("resume"):
            ev["fork_checkpoint"] = pin(tmp_path / ("full_" + executed_arm + "/checkpoint_step0000544.pt"), text="ckpt")
            ev["continuous_summary"] = pin(tmp_path / ("full_" + executed_arm + "/summary.json"), {"steps": 576})
        release = {"status": "APPROVED_FOR_PREPARATION", "stage": "benchmark", "phase": phase, "arm": executed_arm,
                   "source_sha256": source, "config_sha256": ev["config"]["sha256"], "time_limit_s": 1770,
                   "world_size": 4, "workers_per_rank": 6, "review_pin": binding["review_commit"],
                   "inputs": {k: v["sha256"] for k, v in inputs.items()}, "audit_sha256": audit["sha256"],
                   "package_review": binding, "audit_sacct": accounting, "cuda_report": ev["cuda_report"],
                   "cohort_sha256": ev["cohort"]["sha256"], "calibration_sha256": ev["calibration_report"]["sha256"]}
        if phase.startswith("resume"):
            release.update(fork_checkpoint=ev["fork_checkpoint"], continuous_summary=ev["continuous_summary"])
        ev["release"] = pin(tmp_path / ("rel/" + tag + "_" + executed_arm + "_" + phase + ".json"), release)
        return ev
    return binding, evidence


def executed_arm(command):
    return json.loads(Path(command[command.index("--config") + 1]).read_text())["contrastive"]["arm"]


def release_arm(command):
    return json.loads(Path(command[command.index("--release") + 1]).read_text())["arm"]


def test_c0_cell_runs_cn_config_and_release(env):
    binding, evidence = env
    job = REL.make_job(binding, C0, evidence("CN", "full", "a"), FRESH + "/bench_C0")
    assert job["id"] == C0 and job["arm"] == "C0" and job["mode"] == "full"
    command = J.build_command(job, "gpu", 1800)
    print("\n[SK1] cell", job["id"], "job.arm", job["arm"], "-> executed config arm", executed_arm(command),
          "release arm", release_arm(command), "script", Path(command[command.index("--config") - 1]).name)
    assert executed_arm(command) == "CN" and release_arm(command) == "CN"


def test_cn_reload_a_cell_runs_cr_reload_a(env):
    binding, evidence = env
    job = REL.make_job(binding, CNA, evidence("CR", "resume_a", "b"), FRESH + "/reload_CN_A")
    assert job["arm"] == "CN" and job["repeat"] == "a"
    command = J.build_command(job, "gpu", 1800)
    print("\n[SK1] cell", job["id"], "job.arm", job["arm"], "-> executed config arm", executed_arm(command),
          "release arm", release_arm(command))
    assert executed_arm(command) == "CR" and release_arm(command) == "CR"


def test_controls_repeat_is_bound_arm_label_is_bound_but_only_as_label(env):
    binding, evidence = env
    # repeat: a release of the other repeat is rejected by expected_phase
    job = REL.make_job(binding, CNA, evidence("CN", "resume_b", "c"), FRESH + "/reload_CN_A_b")
    with pytest.raises(ValueError, match="job stage/config/time/input pins differ"):
        J.build_command(job, "gpu", 1800)
    # arm label: the evidence cannot set it, and a changed label is rejected by the cell check
    ev = evidence("CN", "full", "d")
    ev["arm"] = "CN"
    with pytest.raises(ValueError, match="evidence cannot override"):
        REL.make_job(binding, C0, ev, FRESH + "/bench_C0_d")
    job = REL.make_job(binding, C0, evidence("C0", "full", "e"), FRESH + "/bench_C0_e")
    assert executed_arm(J.build_command(job, "gpu", 1800)) == "C0"  # positive control: matching arm
    job["arm"] = "CN"
    with pytest.raises(ValueError, match="reviewed fixed preflight cell"):
        J.build_command(job, "gpu", 1800)
    print("\n[SK1] controls: repeat bound via phase; arm label bound; executed arm not bound")
