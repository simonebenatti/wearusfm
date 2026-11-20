"""Skeptic 2 for R5-5: isolating probes for guards reported as untested.

Copied into a throw-away clone only; never into the read-only copy.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

import contrastive_release as C
import contrastive_benchmark_readout as B
from contrastive_numeric_profile import numeric_profile, NumericProfileUnavailable
from test_contrastive_release import package, accounting, dump  # noqa: F401  (fixture reuse)
from test_contrastive_targets_readout import readout_fixture, numeric_folder, write

REAL_RUN_EVIDENCE = B._run_evidence  # captured before any fixture stubs it


# ---------------------------------------------------------------- accounting
def test_sk2_C4d_audit_internal_source_is_checked(package):
    audit, receipt, _ = accounting(package)
    dump(audit, {"source_sha256": "0" * 64, "passed": True})
    data = json.loads(receipt.read_text())
    data["audit_sha256"] = C.file_sha(audit)          # accounting re-bound to this audit
    dump(receipt, data)                                 # accounting source is still the current one
    with pytest.raises(ValueError, match="not bound to this completed new-cohort audit"):
        C.validate_audit_accounting(audit, receipt)


def test_sk2_C4g_missing_timelimit_column(package):
    audit, receipt, tsv = accounting(package)
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS\n123|COMPLETED|0:0|7200|8\n")
    data = json.loads(receipt.read_text())
    data["sacct"] = C.pin(tsv)
    dump(receipt, data)
    with pytest.raises(ValueError, match="unambiguous allocation fields"):
        C.validate_audit_accounting(audit, receipt)


@pytest.mark.parametrize("job_id", ["123.batch", 123, ""])
def test_sk2_C4h_job_id_must_be_allocation_digits(package, job_id):
    audit, receipt, _ = accounting(package)
    data = json.loads(receipt.read_text())
    data["job_id"] = job_id
    dump(receipt, data)
    with pytest.raises(ValueError, match="job ID missing or changed"):
        C.validate_audit_accounting(audit, receipt)


def test_sk2_C4h_real_sacct_step_rows_have_blank_timelimit(package):
    """Same step selection with the Leonardo sacct format (blank TimelimitRaw on steps)."""
    audit, receipt, tsv = accounting(package)
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n"
                   "123|COMPLETED|0:0|7200|8|120\n123.batch|COMPLETED|0:0|7199|8|\n")
    data = json.loads(receipt.read_text())
    data["sacct"] = C.pin(tsv)
    data["job_id"] = "123.batch"
    dump(receipt, data)
    with pytest.raises(ValueError):
        C.validate_audit_accounting(audit, receipt)


# ----------------------------------------------------------- numeric context
def _set_flag_everywhere(kwargs, field, value):
    for folder in [*kwargs["full_dirs"].values(), *kwargs["resumed_dirs"].values()]:
        for rank in range(4):
            path = folder / f"numeric_profile.rank{rank}.json"
            row = json.loads(path.read_text())
            row["flags"][field] = value
            write(path, row)


def test_sk2_C5c_coherent_noncompliant_28_is_inconclusive(tmp_path, monkeypatch):
    kwargs, _, _, calls, _ = readout_fixture(tmp_path, monkeypatch)
    _set_flag_everywhere(kwargs, "matmul_tf32", True)
    report = B.read_preflight(**kwargs)
    context = report["numeric_context_across_runs"]
    print("SK2_C5C_STATUS", report["status"], context["status"], len(context["missing"]), len(context["differences"]))
    assert report["status"] == "INCONCLUSIVE_REQUIRES_REVIEW"
    assert context["status"] == "INCONCLUSIVE" and context["differences"] == [] and len(context["missing"]) == 28
    assert report["failure_reasons"] == [] and calls == [] and report["training_approved"] is False


def test_sk2_C5c_backstop_run_evidence_raises_not_inconclusive(tmp_path):
    cfg, provenance, cohort = numeric_folder(tmp_path)
    for rank in range(4):
        path = tmp_path / f"numeric_profile.rank{rank}.json"
        row = json.loads(path.read_text())
        row["flags"]["matmul_tf32"] = True
        write(path, row)
    with pytest.raises(NumericProfileUnavailable, match="matmul_tf32"):
        B._run_evidence(tmp_path, "CN", cohort, cfg, provenance)


def test_sk2_C5f_run_evidence_rank_flags_must_agree(tmp_path):
    cfg, provenance, cohort = numeric_folder(tmp_path)
    path = tmp_path / "numeric_profile.rank1.json"
    row = json.loads(path.read_text())
    row["flags"]["bf16_reduced_precision_reduction"] = not row["flags"]["bf16_reduced_precision_reduction"]
    write(path, row)
    with pytest.raises(ValueError, match="differ across ranks"):
        B._run_evidence(tmp_path, "CN", cohort, cfg, provenance)


def _null_keys(flags):
    return sorted(key for key, value in flags.items() if value is None)


def test_sk2_222_reachable_when_a_null_flag_is_absent_on_one_rank(tmp_path, monkeypatch):
    """Cross-run reader uses .get(); per-run reader uses dict equality."""
    fixture_root = tmp_path / "readout"
    fixture_root.mkdir()
    kwargs, _, _, calls, _ = readout_fixture(fixture_root, monkeypatch)
    path = kwargs["full_dirs"]["C0"] / "numeric_profile.rank1.json"
    row = json.loads(path.read_text())
    nulls = _null_keys(row["flags"])
    print("SK2_NULL_KEYS", nulls)
    assert nulls, "needs at least one null-valued flag in the recorded profile"
    del row["flags"][nulls[0]]
    write(path, row)
    context = B.numeric_context_across_runs(kwargs["full_dirs"], kwargs["resumed_dirs"])
    print("SK2_222_CONTEXT", context["status"], context["differences"], context["missing"])
    assert context["status"] == "PASSED"           # so read_preflight goes on to _run_evidence
    run_root = tmp_path / "run"
    run_root.mkdir()
    cfg, provenance, cohort = numeric_folder(run_root)
    path = run_root / "numeric_profile.rank1.json"
    row = json.loads(path.read_text())
    del row["flags"][nulls[0]]
    write(path, row)
    with pytest.raises(ValueError, match="production numeric flags differ across ranks"):
        REAL_RUN_EVIDENCE(run_root, "CN", cohort, cfg, provenance)


# ------------------------------------------------------------- CUDA receipt
def _cuda_report(tmp_path):
    from contrastive_cuda_check import PROTOCOL, CRITERIA, TECHNICAL_OVERRIDES
    calibration_cfg = {"contrastive": {"calibration_only": True, "nce_weight": 1., "rnc_weight": 1.}}
    canonical = hashlib.sha256(json.dumps(calibration_cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cohort = {"source_sha256": "s" * 64, "initial_state_sha256": "i" * 64}
    inputs = {k: k + "-pin" for k in ("manifest", "scales", "window_seconds", "pose_stats")}
    keys = ("total", "pose", "contrast_nce", "contrast_rnc", "contrast_global_grad_base", "contrast_global_grad_nce",
            "contrast_global_grad_rnc", "gradient", "student", "teacher")
    precisions = {}
    for name in ("fp32", "bf16"):
        limits = {k: (.01 if name == "bf16" or k in ("student", "teacher") else 1e-5) for k in keys}
        precisions[name] = {"passed": True, "updates": [{"update": step, "passed": True, "errors": dict.fromkeys(keys, 0.),
                                                         "limits": limits} for step in (1, 2)]}
    report = {"passed": True, "source_sha256": cohort["source_sha256"], "initial_sha256": cohort["initial_state_sha256"],
              "protocol": PROTOCOL, "criteria": CRITERIA, "technical_overrides": TECHNICAL_OVERRIDES, "world_size": 4,
              "inputs_unchanged": True, "inputs": {**inputs, "config": "byte-pin"},
              "pose_stats_sha256": inputs["pose_stats"], "config_canonical_sha256": canonical,
              "ranks": [{"rank": rank, "passed": True, "stop_checks": {"coverage": True, "numeric": True, "passed": True},
                         "arms": {"CN": copy.deepcopy(precisions), "CR": copy.deepcopy(precisions)}} for rank in range(4)]}
    with numeric_profile(require_cuda=False) as flags:
        for row in report["ranks"]:
            row["numeric_profile"] = dict(flags)
    return report, cohort, inputs, calibration_cfg


def test_sk2_C5m_cuda_rank_profile_is_checked(tmp_path):
    report, cohort, inputs, calibration_cfg = _cuda_report(tmp_path)
    path = tmp_path / "cuda.json"
    path.write_text(json.dumps(report))
    assert B._cuda_evidence(path, cohort, inputs, calibration_cfg)["passed"]
    report["ranks"][2]["numeric_profile"]["matmul_tf32"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(NumericProfileUnavailable, match="matmul_tf32"):
        B._cuda_evidence(path, cohort, inputs, calibration_cfg)
