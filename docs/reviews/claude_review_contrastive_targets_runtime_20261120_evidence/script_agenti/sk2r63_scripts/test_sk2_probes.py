"""Skeptic-2 probes (R6-3): each test passes on 0efa4d45 and targets one surviving mutation."""
import copy, json, sys
from pathlib import Path
from types import SimpleNamespace
import pytest

import contrastive_benchmark_readout as B
import contrastive_target_resume as T
import contrastive_release as CR
import contrastive_run as runner
import contrastive_job as J
from test_contrastive_targets_readout import readout_fixture
import test_contrastive_run as TR
from test_contrastive_job import job_env  # noqa: F401  (pytest fixture)


def raiser(msg):
    def f(*a, **k):
        raise ValueError(msg)
    return f


# ---------- readout (B01 B02 B02x B04 B05 B11 B14 B16)
@pytest.mark.parametrize("status,expected,key", [("FAILED_REQUIRES_REVIEW", "FAILED_REQUIRES_REVIEW", "failure_reasons"),
                                                 ("INCONCLUSIVE", "INCONCLUSIVE_REQUIRES_REVIEW", "inconclusive_reasons")])
def test_sk2_resume_outcome_propagates(tmp_path, monkeypatch, status, expected, key):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(T, "compare_three_paths", lambda *a, **k: {"status": status, "passed": False})
    report = B.read_preflight(**kwargs)
    other = "inconclusive_reasons" if key == "failure_reasons" else "failure_reasons"
    assert report["status"] == expected
    assert sum("common-checkpoint" in r for r in report[key]) == 2
    assert not any("common-checkpoint" in r for r in report[other])


def test_sk2_reload_origin_suffix_is_enforced(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    inner = B._run_evidence
    def shifted(folder, arm, cohort, cfg, prov, first_update=1, **kw):
        s, c, records, pins = inner(folder, arm, cohort, cfg, prov, first_update=first_update, **kw)
        if Path(folder).name == "CN_B":
            records = [dict(r, origin=[r["step"] + 1000]) for r in records]
        return s, c, records, pins
    monkeypatch.setattr(B, "_run_evidence", shifted)
    with pytest.raises(ValueError, match="continuous suffix"):
        B.read_preflight(**kwargs)


def test_sk2_each_reload_read_with_its_own_phase(tmp_path, monkeypatch):
    kwargs, _, _, calls, _ = readout_fixture(tmp_path, monkeypatch)
    B.read_preflight(**kwargs)
    phases = {name: kw.get("expected_phase", "full") for name, _, kw in calls}
    assert phases == {"C0": "full", "CN": "full", "CR": "full", "CN_A": "resume_a", "CN_B": "resume_b",
                      "CR_A": "resume_a", "CR_B": "resume_b"}


def test_sk2_uniform_wrong_profile_is_inconclusive(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    for folder in [*kwargs["full_dirs"].values(), *kwargs["resumed_dirs"].values()]:
        for rank in range(4):
            p = Path(folder) / f"numeric_profile.rank{rank}.json"
            row = json.loads(p.read_text()); row["flags"]["matmul_tf32"] = True; p.write_text(json.dumps(row))
    report = B.read_preflight(**kwargs)
    assert report["status"] == "INCONCLUSIVE_REQUIRES_REVIEW" and report["failure_reasons"] == []
    assert len(report["numeric_context_across_runs"]["missing"]) == 28


def test_sk2_drift_outside_tolerance_stays_descriptive(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    inner = B._run_evidence
    def drifted(folder, arm, cohort, cfg, prov, first_update=1, **kw):
        s, c, records, pins = inner(folder, arm, cohort, cfg, prov, first_update=first_update, **kw)
        if Path(folder).name == "CN":
            records = [dict(r, total=r["total"] * 1.5) for r in records]
        return s, c, records, pins
    monkeypatch.setattr(B, "_run_evidence", drifted)
    report = B.read_preflight(**kwargs)
    assert report["trajectory_drift_descriptive"]["CN"]["fields"]["total"]["outside_count"] == 256
    assert report["status"] == "PASSED_FOR_REVIEW"


def test_sk2_cuda_rank_numeric_profile_is_required(tmp_path):
    from contrastive_cuda_check import PROTOCOL, CRITERIA, TECHNICAL_OVERRIDES
    from contrastive_numeric_profile import numeric_profile, NumericProfileUnavailable
    import hashlib
    cal = {'contrastive': {'calibration_only': True, 'nce_weight': 1., 'rnc_weight': 1.}}
    canonical = hashlib.sha256(json.dumps(cal, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cohort = {'source_sha256': 's'*64, 'initial_state_sha256': 'i'*64}
    inputs = {k: k+'-pin' for k in ('manifest', 'scales', 'window_seconds', 'pose_stats')}
    keys = ('total', 'pose', 'contrast_nce', 'contrast_rnc', 'contrast_global_grad_base', 'contrast_global_grad_nce',
            'contrast_global_grad_rnc', 'gradient', 'student', 'teacher')
    prec = {n: {'passed': True, 'updates': [{'update': s, 'passed': True, 'errors': dict.fromkeys(keys, 0.),
             'limits': {k: (.01 if n == 'bf16' or k in ('student', 'teacher') else 1e-5) for k in keys}} for s in (1, 2)]}
            for n in ('fp32', 'bf16')}
    with numeric_profile(require_cuda=False) as flags:
        pass
    report = {'passed': True, 'source_sha256': 's'*64, 'initial_sha256': 'i'*64, 'protocol': PROTOCOL, 'criteria': CRITERIA,
              'technical_overrides': TECHNICAL_OVERRIDES, 'world_size': 4, 'inputs_unchanged': True,
              'inputs': {**inputs, 'config': 'x'}, 'pose_stats_sha256': inputs['pose_stats'], 'config_canonical_sha256': canonical,
              'ranks': [{'rank': r, 'passed': True, 'stop_checks': {'coverage': True, 'numeric': True, 'passed': True},
                         'arms': {'CN': copy.deepcopy(prec), 'CR': copy.deepcopy(prec)}, 'numeric_profile': dict(flags)} for r in range(4)]}
    path = tmp_path / 'cuda.json'
    path.write_text(json.dumps(report))
    assert B._cuda_evidence(path, cohort, inputs, cal)['passed']
    report['ranks'][2]['numeric_profile']['matmul_tf32'] = True
    path.write_text(json.dumps(report))
    with pytest.raises(NumericProfileUnavailable):
        B._cuda_evidence(path, cohort, inputs, cal)


# ---------- release (R01 R02 R03 R04 R05)
def test_sk2_release_calls_review_binding_and_accounting(tmp_path, monkeypatch):
    release, cfg, inputs, audit = TR.release_fixture(tmp_path, monkeypatch)
    runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    with monkeypatch.context() as m:
        m.setattr(CR, "validate_review_binding", raiser("sk2-binding"))
        with pytest.raises(ValueError, match="sk2-binding"):
            runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    with monkeypatch.context() as m:
        m.setattr(CR, "validate_audit_accounting", raiser("sk2-accounting"))
        with pytest.raises(ValueError, match="sk2-accounting"):
            runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    bad = copy.deepcopy(release); bad["review_pin"] = "another-reviewed-commit"
    with pytest.raises(ValueError, match="bound package review"):
        runner.validate_release(bad, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)


def bench_release(tmp_path, monkeypatch, arm, phase):
    release, cfg, inputs, audit = TR.release_fixture(tmp_path, monkeypatch)
    raw = json.loads(cfg.read_text()); raw["max_steps"] = 576; raw["contrastive"]["arm"] = arm
    cfg.write_text(json.dumps(raw))
    cal = tmp_path / "cal_report.json"; cal.write_text(json.dumps({"input_sha256": release["inputs"]}))
    cohort = tmp_path / "cohort.json"; cohort.write_text("{}")
    release.update(stage="benchmark", phase=phase, arm=arm, config_sha256=runner.sha(cfg), calibration_sha256=runner.sha(cal),
                   cohort_sha256=runner.sha(cohort), cuda_report={"fixture": True})
    monkeypatch.setattr(runner, "validate_cohort", lambda *a, **k: {"initial_state_sha256": "i"*64})
    return dict(release=release, stage="benchmark", config_path=cfg, input_paths=inputs, calibration_path=cal,
                time_limit_s=900., audit_path=audit, cohort_path=cohort, calibration_run_dir=tmp_path / "calibration")


def test_sk2_benchmark_release_requires_cuda_prerequisite(tmp_path, monkeypatch):
    args = bench_release(tmp_path, monkeypatch, "CN", "full")
    monkeypatch.setattr(CR, "validate_cuda_prerequisite", lambda *a, **k: None)
    assert runner.validate_release(**args) == {"initial_state_sha256": "i"*64}
    monkeypatch.setattr(CR, "validate_cuda_prerequisite", raiser("sk2-cuda"))
    with pytest.raises(ValueError, match="sk2-cuda"):
        runner.validate_release(**args)


def test_sk2_c0_reload_is_rejected(tmp_path, monkeypatch):
    args = bench_release(tmp_path, monkeypatch, "C0", "resume_a")
    monkeypatch.setattr(CR, "validate_cuda_prerequisite", lambda *a, **k: None)
    with pytest.raises(ValueError, match="only CN/CR"):
        runner.validate_release(**args)


# ---------- validate_target_fork (R07 R08) and main (R06 R09)
def fork_env(tmp_path, monkeypatch):
    run = tmp_path / "CN_full"; run.mkdir()
    ckpt = run / "checkpoint_step0000544.pt"; ckpt.write_bytes(b"fork")
    cohort = {"initial_state_sha256": "i"*64, "calibration_sha256": "c"*64}
    release = {"arm": "CN", "cohort_sha256": "h"*64, "inputs": {"manifest": "m"*64}}
    prov = {"source_sha256": runner.source_digest(), "calibration_sha256": "c"*64, "cohort_sha256": "h"*64, "input_sha256": release["inputs"]}
    summary = run / "summary.json"
    summary.write_text(json.dumps({"steps": 576, "stopped": "max_steps", "world_size": 4,
        "contrastive": {"config": {"arm": "CN"}, "initial_sha256": "i"*64, "provenance": prov}}))
    release["fork_checkpoint"] = {"path": str(ckpt), "sha256": runner.sha(ckpt)}
    release["continuous_summary"] = {"path": str(summary), "sha256": runner.sha(summary)}
    monkeypatch.setattr(T, "validate_fork", lambda *a, **k: None)
    return release, cohort, run, prov


def test_sk2_target_fork_name_and_provenance(tmp_path, monkeypatch):
    release, cohort, run, prov = fork_env(tmp_path, monkeypatch)
    assert runner.validate_target_fork(release, None, cohort).name == "checkpoint_step0000544.pt"
    other = run / "checkpoint_step0000543.pt"; other.write_bytes(b"fork")
    bad = copy.deepcopy(release); bad["fork_checkpoint"] = {"path": str(other), "sha256": runner.sha(other)}
    with pytest.raises(ValueError, match="complete matching continuous576"):
        runner.validate_target_fork(bad, None, cohort)
    bad = copy.deepcopy(release); bad["cohort_sha256"] = "z"*64
    with pytest.raises(ValueError, match="provenance differs"):
        runner.validate_target_fork(bad, None, cohort)


def main_env(tmp_path, monkeypatch, fork_hook):
    cfgp = tmp_path / "cfg.json"; cfgp.write_text("{}")
    rel = tmp_path / "release.json"; rel.write_text(json.dumps({"stage": "benchmark", "phase": "resume_a", "inputs": {"manifest": "m"*64}}))
    root = tmp_path / "root"; root.mkdir()
    cfg = SimpleNamespace(loader=SimpleNamespace(pose_root=str(tmp_path)), nested=None)
    monkeypatch.setattr(runner, "validate_recipe", lambda *a, **k: cfg)
    monkeypatch.setattr(runner, "validate_release", lambda *a, **k: {"initial_state_sha256": "i"*64})
    monkeypatch.setattr(runner.R, "check_input_files", lambda *a, **k: None)
    monkeypatch.setattr(runner, "validate_target_fork", fork_hook)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["contrastive_run", "--stage", "benchmark", "--config", str(cfgp), "--validate-only",
        "--release", str(rel), "--manifest", "m", "--root", str(root), "--scales", "s", "--window-seconds", "w",
        "--out-dir", str(out), "--time-limit-s", "1770"])
    return out


def test_sk2_main_reload_validates_fork(tmp_path, monkeypatch):
    main_env(tmp_path, monkeypatch, raiser("sk2-fork"))
    with pytest.raises(ValueError, match="sk2-fork"):
        runner.main()


def test_sk2_main_reload_requires_fresh_output(tmp_path, monkeypatch, capsys):
    out = main_env(tmp_path, monkeypatch, lambda *a: None)
    runner.main()
    assert json.loads(capsys.readouterr().out)["release_valid"] is True
    out.mkdir()
    with pytest.raises(ValueError, match="fresh output directory"):
        runner.main()


# ---------- dispatcher (J01-J08)
def test_sk2_job_calls_review_binding_and_accounting(job_env, monkeypatch):
    job = job_env[0]("full")
    J.build_command(job, "gpu", 1800)
    with monkeypatch.context() as m:
        m.setattr(CR, "validate_review_binding", raiser("sk2-binding"))
        with pytest.raises(ValueError, match="sk2-binding"):
            J.build_command(job, "gpu", 1800)
    with monkeypatch.context() as m:
        m.setattr(CR, "validate_audit_accounting", raiser("sk2-accounting"))
        with pytest.raises(ValueError, match="sk2-accounting"):
            J.build_command(job, "gpu", 1800)


def test_sk2_job_review_must_be_the_bound_review(job_env):
    job = job_env[0]("full")
    job["review"] = job_env[1]("other/review.json", {"other": True})
    with pytest.raises(ValueError, match="bound package review"):
        J.build_command(job, "gpu", 1800)


@pytest.mark.parametrize("field,value", [("arm", "CN"), ("repeat", "b")])
def test_sk2_job_must_match_its_cell(job_env, field, value):
    job = job_env[0]("full")
    job[field] = value
    with pytest.raises(ValueError, match="reviewed fixed preflight cell"):
        J.build_command(job, "gpu", 1800)


def test_sk2_job_output_bound_to_cohort_root(job_env):
    job = job_env[0]("full")
    pre = Path(CR.PREFLIGHT); plan = json.loads(pre.read_text())
    plan["fresh_root"] = str(Path(plan["fresh_root"]) / "cohort"); pre.write_text(json.dumps(plan))
    job["out"] = str(Path(plan["fresh_root"]).parent / "elsewhere")
    with pytest.raises(ValueError, match="fresh output"):
        J.build_command(job, "gpu", 1800)
    job["out"] = str(Path(plan["fresh_root"]) / "full")
    J.build_command(job, "gpu", 1800)


def test_sk2_job_accounting_must_be_the_released_one(job_env):
    job = job_env[0]("full")
    job["audit_sacct"] = job_env[1]("other/audit_accounting.json", {"other": True})
    with pytest.raises(ValueError, match="review/audit accounting differs"):
        J.build_command(job, "gpu", 1800)


def test_sk2_resume_job_fork_must_be_the_released_one(job_env):
    job = job_env[0]("resume")
    job["fork_checkpoint"] = job_env[1]("resume/common2/checkpoint_step0000544.pt", {"other": True})
    with pytest.raises(ValueError, match="exact common checkpoint"):
        J.build_command(job, "gpu", 1800)


def test_sk2_resume_b_cannot_run_resume_a_release(job_env):
    job = job_env[0]("resume")
    pre = Path(CR.PREFLIGHT); plan = json.loads(pre.read_text())
    plan["cells"][0]["repeat"] = "b"; pre.write_text(json.dumps(plan))
    job["repeat"] = "b"
    with pytest.raises(ValueError, match="job stage/config/time/input pins"):
        J.build_command(job, "gpu", 1800)
