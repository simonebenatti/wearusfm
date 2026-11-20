"""Skeptic 1 (R3-M4): behavioural probes. Each probe asserts the ORIGINAL guard outcome.

On the unmutated clone every probe must pass. A mutant that a probe kills is observable
(non-equivalent); a mutant that no probe kills is a candidate equivalent mutant.
Run with PYTHONPATH=<clone>/src:<clone>/tests/cpu:<clone>/scripts and SK1_CLONE=<clone>.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

import contrastive_release as C
import contrastive_job as J
import contrastive_run as runner
from test_contrastive_job import job_env  # noqa: F401  (fixture reuse)
from test_contrastive_run import release_fixture

CLONE = Path(os.environ["SK1_CLONE"]).resolve()
assert Path(C.__file__).resolve().is_relative_to(CLONE)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


class Pkg:
    """Same construction as the `package` fixture of test_contrastive_release.py, parametrised."""

    def __init__(self, tmp_path, monkeypatch, *, cells=None, preflight_source=None, extra=None):
        self.root = root = tmp_path / "repo"
        root.mkdir()
        self.source = root / "scripts/contrastive_one.py"
        self.source.parent.mkdir()
        self.source.write_text("# reviewed execution\n")
        self.science = dump(root / "science.json", {"input_sha256": {"manifest": "1" * 64}})
        self.cells = cells or [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200}]
        self.preflight = dump(root / "preflight.json", {"source_sha256": preflight_source or self.digest(),
                              "science_plan": {"path": "science.json", "sha256": C.file_sha(self.science)},
                              "cells": self.cells})
        for name, value in (extra or {}).items():
            dump(root / name, value)
        monkeypatch.setattr(C, "ROOT", root)
        monkeypatch.setattr(C, "PLAN", self.science)
        monkeypatch.setattr(C, "PREFLIGHT", self.preflight)
        monkeypatch.setattr(C, "source_paths", lambda: [self.source])
        monkeypatch.setattr(C, "source_digest", self.digest)
        self.git("init", "-q")
        self.commit("Reviewed package")
        self.code = self.git("rev-parse", "HEAD")

    def digest(self):
        return hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(self.source.read_bytes()).digest()).hexdigest()

    def git(self, *args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args],
                                       cwd=self.root, stderr=subprocess.PIPE).decode().strip()

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def payload(self, **change):
        value = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": self.code,
                 "source_sha256": self.digest(), "science_plan_sha256": C.file_sha(self.science),
                 "preflight_plan_sha256": C.file_sha(self.preflight)}
        value.update(change)
        return value

    def review(self, text):
        self.review_path = self.root / "review.md"
        self.review_path.write_text(text)
        return self.commit("Actual package review")

    def block(self, payload):
        return "```contrastive-release-review\n" + json.dumps(payload) + "\n```\n"

    def bind(self, review_commit):
        return C.bind_review(review_commit, "review.md", self.review_path)


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    return lambda **kw: Pkg(tmp_path, monkeypatch, **kw)


def valid(pkg):
    p = pkg()
    head = p.review("# Review\n\n" + p.block(p.payload()))
    return p, p.bind(head)


# ---------------------------------------------------------------- parser
def test_P_MU1_two_blocks_refused(pkg):
    p = pkg()
    head = p.review("# Review\n\n" + p.block(p.payload()) + "\n" + p.block(p.payload()))
    with pytest.raises(ValueError, match="exactly one"):
        p.bind(head)


def test_P_MU2_rejected_status_refused(pkg):
    p = pkg()
    head = p.review(p.block(p.payload(status="REJECTED")))
    with pytest.raises(ValueError, match="approved target-cohort"):
        p.bind(head)


def test_P_MU3_extra_key_refused(pkg):
    p = pkg()
    head = p.review(p.block(p.payload(notes="anything")))
    with pytest.raises(ValueError, match="approved target-cohort"):
        p.bind(head)


# ---------------------------------------------------------------- binding
def test_P_MU5_plans_at_reviewed_commit_must_equal_current(pkg):
    p = pkg()
    reviewed = p.code
    dump(p.science, {"input_sha256": {"manifest": "2" * 64}})          # plan changed after the reviewed commit
    dump(p.preflight, {"source_sha256": p.digest(), "science_plan": {"path": "science.json", "sha256": C.file_sha(p.science)},
                       "cells": p.cells})
    p.commit("plans changed after review")
    head = p.review(p.block(p.payload(reviewed_commit=reviewed)))     # current hashes, older commit
    with pytest.raises(ValueError, match="reviewed plan bytes differ"):
        p.bind(head)


def test_P_MU6_binding_payload_must_match_copy(pkg):
    p, binding = valid(pkg)
    bad = copy.deepcopy(binding)
    bad["review_payload"]["reviewed_commit"] = "f" * 40
    with pytest.raises(ValueError, match="differs from authenticated original"):
        C.validate_review_binding(bad)


def test_P_MU8_binding_to_other_plan_path_refused(pkg):
    p = pkg(extra={"other/science.json": {"input_sha256": {"manifest": "1" * 64}}})
    assert C.file_sha(p.root / "other/science.json") == C.file_sha(p.science)
    head = p.review(p.block(p.payload()))
    binding = p.bind(head)
    bad = copy.deepcopy(binding)
    bad["science_plan"] = C.pin(p.root / "other/science.json", relative=True)
    with pytest.raises(ValueError, match="another cohort"):
        C.validate_review_binding(bad)


def test_P_MU9_preflight_internal_source_must_match(pkg):
    p = pkg(preflight_source="0" * 64)
    head = p.review(p.block(p.payload()))
    with pytest.raises(ValueError, match="preflight source/science-plan binding differs"):
        p.bind(head)


def test_P_MU32_review_copy_outside_checkout_refused(pkg, tmp_path):
    p, binding = valid(pkg)
    outside = tmp_path / "outside" / "review.md"
    outside.parent.mkdir()
    outside.write_bytes(p.review_path.read_bytes())
    bad = copy.deepcopy(binding)
    bad["review"] = C.pin(outside)
    with pytest.raises(ValueError, match="inside the frozen checkout"):
        C.validate_review_binding(bad)


def test_P_MU35_MU39_source_differs_at_reviewed_commit_refused(pkg):
    # Reviewed commit A has source v1 but a preflight that already embeds the digest of v2.
    p = pkg()
    p.source.write_text("# executed v2\n")
    v2 = p.digest()
    p.source.write_text("# reviewed execution\n")
    dump(p.preflight, {"source_sha256": v2, "science_plan": {"path": "science.json", "sha256": C.file_sha(p.science)},
                       "cells": p.cells})
    reviewed = p.commit("A: v1 sources, preflight embedding v2")
    p.source.write_text("# executed v2\n")
    p.commit("B: v2 sources")
    head = p.review(p.block(p.payload(reviewed_commit=reviewed)))
    with pytest.raises(ValueError, match="execution sources differ|reviewed source inventory differs"):
        p.bind(head)


@pytest.mark.parametrize("path", ["../review.md", "ABS"])
def test_P_MU36_uncontained_review_path_refused(pkg, path):
    p = pkg()
    head = p.review(p.block(p.payload()))
    target = str(p.review_path) if path == "ABS" else path
    with pytest.raises(ValueError):
        C.bind_review(head, target, p.review_path)


# ---------------------------------------------------------------- accounting
def accounting(p, *, rows=None, header="JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw", job_id="123",
               audit_source=None):
    audit = dump(p.root / "audit.json", {"source_sha256": audit_source or C.source_digest(), "passed": True})
    tsv = p.root / "sacct.tsv"
    rows = rows or ["123|COMPLETED|0:0|7200|8|120", "123.batch|COMPLETED|0:0|7199|8|120"]
    tsv.write_text(header + "\n" + "\n".join(rows) + "\n")
    receipt = dump(p.root / "accounting.json", {"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(),
                   "audit_sha256": C.file_sha(audit), "logical_id": "new-audit", "job_id": job_id, "sacct": C.pin(tsv)})
    return audit, receipt


def test_P_baseline_accounting_accepted(pkg):
    p = pkg()
    audit, receipt = accounting(p)
    assert C.validate_audit_accounting(audit, receipt)["state"] == "COMPLETED"


def test_P_MU19_audit_report_from_other_source_refused(pkg):
    p = pkg()
    audit, receipt = accounting(p, audit_source="0" * 64)
    with pytest.raises(ValueError, match="not bound"):
        C.validate_audit_accounting(audit, receipt)


def test_P_MU33_missing_column_refused_any(pkg):
    p = pkg()
    audit, receipt = accounting(p, header="JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS",
                                rows=["123|COMPLETED|0:0|7200|8"])
    with pytest.raises(Exception):
        C.validate_audit_accounting(audit, receipt)


def test_P_MU33_missing_column_refused_valueerror(pkg):
    p = pkg()
    audit, receipt = accounting(p, header="JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS",
                                rows=["123|COMPLETED|0:0|7200|8"])
    with pytest.raises(ValueError, match="unambiguous allocation fields"):
        C.validate_audit_accounting(audit, receipt)


def test_P_MU37_step_id_cannot_replace_timed_out_allocation(pkg):
    p = pkg()
    audit, receipt = accounting(p, job_id="123.batch",
                                rows=["123|TIMEOUT|0:0|7380|8|120", "123.batch|COMPLETED|0:0|7199|8|120"])
    with pytest.raises(ValueError, match="job ID"):
        C.validate_audit_accounting(audit, receipt)


def test_P_MU21_make_job_calibration_requires_completed_audit(pkg):
    cells = [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200},
             {"id": "new-cal", "mode": "calibration", "arm": None, "seconds": 1800}]
    p = pkg(cells=cells)
    head = p.review(p.block(p.payload()))
    binding = p.bind(head)
    audit, receipt = accounting(p, rows=["123|TIMEOUT|0:0|7380|8|120"])
    with pytest.raises(ValueError, match="COMPLETED 0:0"):
        C.make_job(binding, "new-cal", {"audit": C.pin(audit), "audit_sacct": C.pin(receipt)}, "/remote/new-output")


# ---------------------------------------------------------------- dispatcher
def spy(name):
    def raising(*args, **kwargs):
        raise ValueError("SPY-" + name)
    return raising


def test_P_MU22_dispatcher_calls_binding_guard(job_env, monkeypatch):
    monkeypatch.setattr(C, "validate_review_binding", spy("binding"))
    with pytest.raises(ValueError, match="SPY-binding"):
        J.build_command(job_env[0]("audit"), "cpu", 7200)


def test_P_MU23_dispatcher_job_review_must_be_bound(job_env):
    job = job_env[0]("audit")
    job["review"] = job_env[1]("audit/other_review.json", {"other": True})
    with pytest.raises(ValueError, match="job review differs"):
        J.build_command(job, "cpu", 7200)


def test_P_MU24_dispatcher_job_must_equal_reviewed_cell(job_env):
    job = job_env[0]("audit")
    job["arm"] = "CN"
    with pytest.raises(ValueError, match="reviewed fixed preflight cell"):
        J.build_command(job, "cpu", 7200)


@pytest.mark.parametrize("mode", ["calibration", "cuda"])
def test_P_MU25_dispatcher_calls_accounting_gate(job_env, monkeypatch, mode):
    monkeypatch.setattr(C, "validate_audit_accounting", spy("accounting"))
    with pytest.raises(ValueError, match="SPY-accounting"):
        J.build_command(job_env[0](mode), "gpu", 1800)


def test_P_MU26_dispatcher_output_inside_cohort_root(job_env):
    job = job_env[0]("audit")
    preflight = json.loads(C.PREFLIGHT.read_text())
    work_runs = Path(preflight["fresh_root"])
    preflight["fresh_root"] = str(work_runs / "cohort")
    C.PREFLIGHT.write_text(json.dumps(preflight))
    job["out"] = str(work_runs / "elsewhere")
    with pytest.raises(ValueError, match="fresh output"):
        J.build_command(job, "cpu", 7200)


def test_P_MU27_dispatcher_release_must_carry_same_binding(job_env):
    job = job_env[0]("full")
    release = json.loads(Path(job["release"]["path"]).read_text())
    release["package_review"] = {"fixture": True, "another": "review"}
    job["release"] = job_env[1]("full/release_other.json", release)
    with pytest.raises(ValueError, match="review/audit accounting differs"):
        J.build_command(job, "gpu", 1800)


# ---------------------------------------------------------------- runtime
def test_P_MU28_runtime_calls_binding_guard(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(C, "validate_review_binding", spy("binding"))
    with pytest.raises(ValueError, match="SPY-binding"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)


def test_P_MU29_runtime_review_pin_bound(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    release["review_pin"] = "another-commit"
    with pytest.raises(ValueError, match="review commit differs"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)


def test_P_MU30_runtime_calls_accounting_gate(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(C, "validate_audit_accounting", spy("accounting"))
    with pytest.raises(ValueError, match="SPY-accounting"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)


def test_P_MU34_runtime_benchmark_calls_cuda_prerequisite(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    raw = json.loads(Path(cfg).read_text())
    calibration = dump(tmp_path / "calibration_report.json", {"input_sha256": release["inputs"]})
    cohort = dump(tmp_path / "cohort.json", {"cohort": True})
    monkeypatch.setattr(runner, "validate_cohort", lambda *a: {"source_sha256": "x"})
    monkeypatch.setattr(C, "validate_cuda_prerequisite", spy("cuda"))
    release.update(stage="benchmark", phase="full", arm=raw["contrastive"]["arm"],
                   calibration_sha256=runner.sha(calibration), cohort_sha256=runner.sha(cohort))
    with pytest.raises(ValueError, match="SPY-cuda"):
        runner.validate_release(release, "benchmark", cfg, inputs, calibration, time_limit_s=900., audit_path=audit,
                                cohort_path=cohort, calibration_run_dir=tmp_path / "calibration")
