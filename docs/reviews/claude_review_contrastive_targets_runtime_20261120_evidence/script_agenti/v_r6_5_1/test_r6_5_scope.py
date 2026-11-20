"""R6-5 skeptic probe: is the package-review binding stage-agnostic at the training gate?

Builds a REAL binding (real bind_review / validate_review_binding on a temporary git
package, same fixture logic as tests/cpu/test_contrastive_release.py) and feeds it to
the REAL contrastive_run.validate_release(stage='training'). Unrelated heavy gates
(audit, accounting, cohort, readout recomputation) are stubbed; the review binding and
validate_measured_budget's own checks are real.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import contrastive_release as C
import contrastive_run as runner
import contrastive_benchmark_readout as readout


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "scripts/contrastive_one.py"
    source.parent.mkdir()
    source.write_text("# reviewed execution\n")
    digest = hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest()
    science = dump(root / "science.json", {"input_sha256": {"manifest": "1" * 64}})
    preflight = dump(root / "preflight.json", {"source_sha256": digest, "training_authorized": False,
                     "science_plan": {"path": "science.json", "sha256": C.file_sha(science)},
                     "cells": [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200}]})
    monkeypatch.setattr(C, "ROOT", root)
    monkeypatch.setattr(C, "PLAN", science)
    monkeypatch.setattr(C, "PREFLIGHT", preflight)
    monkeypatch.setattr(C, "source_paths", lambda: [source])
    monkeypatch.setattr(C, "source_digest", lambda: hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest())

    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
                                       cwd=root, stderr=subprocess.PIPE).decode().strip()
    git("init", "-q"); git("add", "."); git("commit", "-qm", "Reviewed package")
    code = git("rev-parse", "HEAD")
    payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": code, "source_sha256": digest,
               "science_plan_sha256": C.file_sha(science), "preflight_plan_sha256": C.file_sha(preflight)}
    review = root / "review.md"
    review.write_text("# Preflight-only review (prose says: approves ONLY the preflight)\n\n```contrastive-release-review\n"
                      + json.dumps(payload) + "\n```\n")
    git("add", "review.md"); git("commit", "-qm", "Actual package review")
    binding = C.bind_review(git("rev-parse", "HEAD"), "review.md", review)
    return {"root": root, "payload": payload, "review": review, "binding": binding, "git": git, "code": code}


def test_A_payload_cannot_even_carry_a_scope(package):
    scoped = dict(package["payload"], scope="preflight")
    text = "```contrastive-release-review\n" + json.dumps(scoped) + "\n```\n"
    with pytest.raises(ValueError, match="not an approved target-cohort package review"):
        C.review_payload(text)
    print("A: payload with 'scope' key rejected by review_payload (fixed key set)")


def test_B_binding_signature_has_no_stage():
    import inspect
    sig = inspect.signature(C.validate_review_binding)
    print("B: validate_review_binding signature:", sig)
    assert list(sig.parameters) == ["binding", "verify_git"]
    print("B: binding keys produced by bind_review are fixed; no stage/scope key")


def _training_release(tmp_path, monkeypatch, binding, *, sci_pin, budget_pin):
    """Everything except the review binding and the measured-budget logic is stubbed."""
    monkeypatch.setattr(runner, "validate_input_bindings", lambda value: None)
    monkeypatch.setattr(runner, "validate_audit", lambda *a, **k: None)
    monkeypatch.setattr(C, "validate_audit_accounting", lambda *a, **k: None)
    cohort = {"source_sha256": runner.source_digest(), "calibration_sha256": "c" * 64,
              "initial_state_sha256": "d" * 64}
    monkeypatch.setattr(runner, "validate_cohort", lambda *a, **k: cohort)
    inputs = {}
    for name in ("manifest", "scales", "window_seconds", "pose_stats"):
        inputs[name] = dump(tmp_path / "in" / (name + ".json"), {})
    pins = {name: runner.sha(p) for name, p in inputs.items()}
    config = dump(tmp_path / "CN.5000.config.json", {"contrastive": {"arm": "CN"}, "max_steps": 5000})
    audit = dump(tmp_path / "audit.json", {"fixture": "audit"})
    calibration = dump(tmp_path / "calibration.json", {"input_sha256": pins})
    cohort_file = dump(tmp_path / "cohort.json", {"fixture": "cohort"})
    release = {"status": "APPROVED_FOR_TRAINING", "stage": "training", "phase": "initial", "arm": "CN",
               "source_sha256": runner.source_digest(), "config_sha256": runner.sha(config),
               "world_size": 4, "workers_per_rank": 6, "inputs": pins,
               "review_pin": binding["review_commit"], "package_review": binding,
               "time_limit_s": 3600.0, "audit_sha256": runner.sha(audit), "audit_sacct": {"fixture": True},
               "calibration_sha256": runner.sha(calibration), "cohort_sha256": runner.sha(cohort_file),
               "approved_training_gpu_hours": 100.0,
               "scientific_decision_pin": sci_pin, "budget_decision_pin": budget_pin}
    summaries = {}
    for arm in ("C0", "CN", "CR"):
        summaries[arm] = dump(tmp_path / "bench" / arm / "summary.json", {
            "steps": 576, "stopped": "max_steps", "world_size": 4,
            "contrastive": {"config": {"arm": arm}, "initial_sha256": cohort["initial_state_sha256"],
                            "provenance": {"source_sha256": cohort["source_sha256"],
                                           "calibration_sha256": cohort["calibration_sha256"],
                                           "cohort_sha256": release["cohort_sha256"]}}})
    receipt = {"status": "PASSED_FOR_REVIEW", "failure_reasons": [], "training_approved": False,
               "source_sha256": cohort["source_sha256"], "calibration_sha256": cohort["calibration_sha256"],
               "initial_state_sha256": cohort["initial_state_sha256"], "cohort_sha256": release["cohort_sha256"],
               "benchmark_summaries": {arm: runner.sha(p) for arm, p in summaries.items()},
               "training_allocations_gpu_hours": {"C0": 20.0, "CN": 20.0, "CR": 20.0},
               "training_allocations_gpu_hours_by_phase": {a: {"5000": 5.0, "19618": 15.0} for a in ("C0", "CN", "CR")}}
    monkeypatch.setattr(readout, "authenticate_preflight", lambda path: receipt)
    budget = dump(tmp_path / "budget.json", receipt)
    release["measured_budget_sha256"] = runner.sha(budget)
    return release, config, inputs, audit, calibration, cohort_file, budget, summaries


def test_C_same_preflight_binding_passes_training_gate_with_dummy_decisions(tmp_path, package, monkeypatch):
    calls = []
    real = C.validate_review_binding
    def spy(binding, **kw):
        calls.append((sorted(binding), kw))
        return real(binding, **kw)
    monkeypatch.setattr(C, "validate_review_binding", spy)
    release, config, inputs, audit, calibration, cohort_file, budget, summaries = _training_release(
        tmp_path, monkeypatch, package["binding"], sci_pin="x", budget_pin="y")
    runner.validate_release(release, "training", config, inputs, calibration, 3600.0, audit_path=audit,
                            cohort_path=cohort_file, calibration_run_dir=tmp_path, budget_path=budget,
                            benchmark_paths=summaries)
    print("C: validate_release(stage='training') PASSED with the preflight binding; decision pins 'x'/'y'")
    print("C: validate_review_binding calls:", calls)
    assert len(calls) == 1 and calls[0][1] == {}


def test_D_empty_decision_pin_is_the_only_training_specific_human_gate(tmp_path, package, monkeypatch):
    release, config, inputs, audit, calibration, cohort_file, budget, summaries = _training_release(
        tmp_path, monkeypatch, package["binding"], sci_pin="", budget_pin="y")
    with pytest.raises(ValueError, match="pinned scientific and budget decisions"):
        runner.validate_release(release, "training", config, inputs, calibration, 3600.0, audit_path=audit,
                                cohort_path=cohort_file, calibration_run_dir=tmp_path, budget_path=budget,
                                benchmark_paths=summaries)
    print("D: empty scientific_decision_pin rejected (only truthiness checked)")


def test_E_without_measured_budget_training_is_refused(tmp_path, package, monkeypatch):
    release, config, inputs, audit, calibration, cohort_file, budget, summaries = _training_release(
        tmp_path, monkeypatch, package["binding"], sci_pin="x", budget_pin="y")
    with pytest.raises(ValueError, match="measured three-arm benchmark"):
        runner.validate_release(release, "training", config, inputs, calibration, 3600.0, audit_path=audit,
                                cohort_path=cohort_file, calibration_run_dir=tmp_path, budget_path=None,
                                benchmark_paths=summaries)
    print("E: no measured budget -> refused")


def test_F_dispatcher_cannot_express_training(package):
    import contrastive_job as J
    assert "training" not in J.MODES
    for evidence in ({"mode": "training"}, {"status": "APPROVED_FOR_TRAINING"}):
        with pytest.raises(ValueError, match="override"):
            C.make_job(package["binding"], "new-audit", evidence, "/remote/out")
    print("F: MODES =", sorted(J.MODES), "; make_job forbids mode/status override; cells only from preflight")
