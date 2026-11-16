"""Local regressions of the conditional recovery wrapper; no remote operations."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_p1_recovery as R
DIAGNOSTIC = ROOT / "results/infra/scaling_300m_20261104/p1_recovery_20261116/evidence/run_final/report.json"


def save(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": R.digest(path), "bytes": path.stat().st_size}


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUTPUT", str(tmp_path / "p1-r2"))
    sources = {k: R.digest(ROOT / k) for k in R.SOURCE_PATHS}
    diagnostic = json.loads(DIAGNOSTIC.read_bytes())
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_bytes(DIAGNOSTIC.read_bytes())
    diagnostic_pin = {"path": str(diagnostic_path), "sha256": R.digest(diagnostic_path),
                      "bytes": diagnostic_path.stat().st_size}
    tests = {"protocol": R.CPU_TEST_PROTOCOL, "complete": True, "passed": True, "failed": 0,
             "counts": {"passed": 40}, "source_sha256": sources}
    from scaling_p1_recovery_plans import make_recovery_manifest
    extraction = {"protocol": R.V.PROTOCOL, "step": R.STEP, "checkpoints": {R.ARM: {"path": "/checkpoint.pt"}}}
    manifest = make_recovery_manifest(save(tmp_path / "extraction.json", extraction),
        diagnostic_pin, diagnostic, save(tmp_path / "tests.json", tests), tests, sources)
    pin = save(tmp_path / "recovery.json", manifest)
    return {"manifest": manifest, "pin": pin, "diagnostic": diagnostic, "tests": tests, "sources": sources}


def test_manifest_requires_actual_pinned_diagnostic_and_full_test_sources(release):
    data, states = R.load_manifest(release["pin"]["path"], release["pin"]["sha256"])
    assert data == release["manifest"]
    assert set(states) == {"recovery_manifest", "recovery_diagnostic", "recovery_cpu_tests", "recovery_extraction_plan",
                           *("recovery_source_" + name for name in R.SOURCE_PATHS)}


def test_changed_actual_diagnostic_bytes_fail_before_inference(release):
    path = Path(release["manifest"]["diagnostic_report"]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="recovery evidence pin mismatch"):
        R.load_manifest(release["pin"]["path"], release["pin"]["sha256"])


@pytest.mark.parametrize("record,key,value", [
    ("diagnostic", "complete", False), ("diagnostic", "complete", 1),
    ("diagnostic", "outcome", None), ("diagnostic", "outcome", "NULL"),
    ("diagnostic", "outcome", "ERROR"), ("diagnostic", "outcome", "INCOMPLETE"),
    ("diagnostic", "outcome", "INTEGRITY_FAIL"), ("diagnostic", "readers_status", "READERS_AFFECTED"),
    ("diagnostic", "readers_status", "INCOMPLETE"), ("diagnostic", "protocol", "other"),
    ("diagnostic", "errors", [{"message": "I/O failure"}]), ("diagnostic", "errors", None),
    ("diagnostic", "candidate_accepts_all_benign_cases", False),
    ("diagnostic", "candidate_accepts_all_benign_cases", 1),
    ("diagnostic", "cycles_completed", []), ("diagnostic", "fixture_results", []),
    ("diagnostic", "reader_rounds", []), ("manifest", "release_criterion", None),
    ("manifest", "release_criterion", "previous-criterion"),
    ("manifest", "diagnostic_outcome", "METADATA_ONLY_REPRODUCED"),
    ("manifest", "diagnostic_report", {"sha256": "0" * 64}),
    ("tests", "complete", False), ("tests", "passed", False), ("tests", "passed", 40),
    ("tests", "failed", 1), ("tests", "failed", False), ("tests", "counts", {"passed": 0}),
    ("tests", "source_sha256", {}), ("manifest", "source_sha256", {}),
    ("manifest", "output_barrier", "sync"), ("manifest", "failed_attempts", []),
    ("manifest", "out", "/tmp/adopt-previous"), ("manifest", "logical_job_id", "old-job"),
    ("manifest", "task", "p2"), ("manifest", "step", 49046),
])
def test_no_release_on_failed_missing_or_changed_evidence(release, record, key, value):
    bad = copy.deepcopy(release)
    bad[record][key] = value
    with pytest.raises(ValueError):
        R.validate_manifest(bad["manifest"], bad["diagnostic"], bad["tests"], bad["sources"])


def test_changed_output_policy_since_diagnostic_rejected_even_if_recovery_tests_match(release):
    release["manifest"]["diagnostic_source_sha256"]["scripts/scaling_output_identity.py"] = "f" * 64
    release["diagnostic"]["source_sha256"]["scripts/scaling_output_identity.py"] = "f" * 64
    with pytest.raises(ValueError, match="changed after diagnostic"):
        R.validate_manifest(release["manifest"], release["diagnostic"], release["tests"], release["sources"])


def test_actual_complete_null_report_is_explicitly_admitted_by_v2(release):
    assert R.digest(DIAGNOSTIC) == R.RELEASE_DIAGNOSTIC_SHA256
    assert release["diagnostic"]["outcome"] == "NOT_REPRODUCED"
    assert release["manifest"]["release_criterion"] == R.RELEASE_CRITERION
    assert R.validate_manifest(release["manifest"], release["diagnostic"], release["tests"], release["sources"])


def test_pure_gate_outcome_enum_keeps_metadata_only_valid(release):
    # Outcome-branch unit test only: actual file loading still requires the fixed
    # bytes of the approved NOT_REPRODUCED report, tested separately above.
    release["diagnostic"]["outcome"] = "METADATA_ONLY_REPRODUCED"
    release["manifest"]["diagnostic_outcome"] = "METADATA_ONLY_REPRODUCED"
    assert R.validate_manifest(release["manifest"], release["diagnostic"], release["tests"], release["sources"])


@pytest.mark.parametrize("field", ["release_criterion", "diagnostic_outcome", "diagnostic_report"])
def test_new_criterion_fields_cannot_be_omitted(release, field):
    release["manifest"].pop(field)
    with pytest.raises(ValueError):
        R.validate_manifest(release["manifest"], release["diagnostic"], release["tests"], release["sources"])


def test_null_report_is_not_a_valid_null_result(release):
    with pytest.raises(ValueError, match="complete diagnostic report"):
        R.validate_manifest(release["manifest"], None, release["tests"], release["sources"])


@pytest.fixture
def execution(release, tmp_path, monkeypatch):
    v = R.V
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"accepted checkpoint")
    split = tmp_path / "splits.json"
    split.write_text("{}")
    stats, contract = tmp_path / "pose_stats.json", tmp_path / "contract.json"
    stats.write_text("{}"); contract.write_text("{}")
    references = {}
    for ds in v.DATASETS["p1"]:
        file = tmp_path / (ds + "-reference.npz")
        file.write_bytes(b"reference")
        references[ds] = {"path": str(file), "sha256": R.digest(file)}
    plan = {"checkpoints": {R.ARM: {"path": str(checkpoint)}}, "classification_references": references}
    calls = []
    code = "a" * 40
    def context(path, arm, step, expected):
        calls.append(("context", path, arm, step, expected))
        return plan, {"plan": v.P.file_identity(path), "pose_stats": v.P.file_identity(stats),
                      "training_source_contract": v.P.file_identity(contract), "splits": v.P.file_identity(split)}, \
            {"checkpoint_sha256": R.digest(checkpoint), "model_config": {"dim": 896}}, code
    monkeypatch.setattr(v, "load_context", context)
    monkeypatch.setattr(v, "extraction_commands", lambda stage, plan, cp, out, task, device: {"p1": ["frozen-extractor", str(out)]})
    monkeypatch.setattr(v.E, "classification_sources", lambda plan, task: {str(checkpoint): "same"})
    monkeypatch.setattr(v.E, "source_states", lambda paths: {str(checkpoint): "same"})
    def run(command, **kwargs):
        calls.append(("producer", command, kwargs["cwd"], kwargs["env"]["WEARUSFM_CODE_SHA"]))
        for ds in v.DATASETS["p1"]:
            (Path(command[1]) / (ds + ".npz")).write_bytes(ds.encode())
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(R.subprocess, "run", run)
    def validate(path, reference, ds, cp, cp_sha, actual_code, splits, arm):
        calls.append(("validate", ds, cp, cp_sha, actual_code, arm))
        return {"dataset": ds, "task": "p1", "input_sha256": "3" * 64,
                "splits_sha256": "4" * 64, "extraction_recipe": {"unchanged": True}}
    monkeypatch.setattr(v, "verify_classification_cache", validate)
    import scaling_train as ST
    monkeypatch.setattr(ST, "check_source_contract", lambda path: calls.append(("source_contract", path)))
    monkeypatch.setattr(v.P, "code_identity", lambda expected: code)
    return release, calls, checkpoint


def test_one_new_producer_three_cache_finalizations_and_original_final_guards(execution):
    release, calls, checkpoint = execution
    result = R.execute(release["pin"]["path"], release["pin"]["sha256"], "a" * 40, device="cpu")
    assert list(result) == list(R.V.DATASETS["p1"])
    assert sum(x[0] == "producer" for x in calls) == 1
    assert [x[1] for x in calls if x[0] == "validate"] == list(R.V.DATASETS["p1"])
    assert sum(x[0] == "source_contract" for x in calls) == 1
    receipt = json.loads((Path(R.OUTPUT) / "receipt.json").read_text())
    assert receipt["complete"] is True and receipt["checkpoint_unchanged"] is True
    assert receipt["recovery_manifest_sha256"] == release["pin"]["sha256"]
    assert receipt["failed_attempts_preserved"] == R.FAILED_ATTEMPTS
    assert receipt["output_barrier"] == "none"
    assert receipt["release_criterion"] == R.RELEASE_CRITERION
    assert receipt["diagnostic_outcome"] == "NOT_REPRODUCED"
    assert receipt["diagnostic_report"] == release["manifest"]["diagnostic_report"]
    for out in result.values():
        provenance = json.loads(Path(out["provenance_path"]).read_text())
        for key in ("release_criterion", "diagnostic_outcome", "diagnostic_report"):
            assert provenance[key] == receipt[key]
        assert out["output_identity_audit"]["complete"] is True
        assert out["output_identity_audit"]["extra_reads"] == 0
        assert len(out["output_identity_audit"]["observations"]) == 2
    assert checkpoint.read_bytes() == b"accepted checkpoint"
    with pytest.raises(ValueError, match="must be new"):
        R.execute(release["pin"]["path"], release["pin"]["sha256"], "a" * 40, device="cpu")
    assert sum(x[0] == "producer" for x in calls) == 1


@pytest.mark.parametrize("failure", ["producer", "cache", "input", "STOP"])
def test_failure_never_produces_complete_receipt_or_retries(execution, monkeypatch, failure):
    release, calls, checkpoint = execution
    if failure == "producer":
        def broken(*args, **kwargs):
            calls.append(("producer",))
            raise subprocess.CalledProcessError(1, args[0])
        monkeypatch.setattr(R.subprocess, "run", broken)
    elif failure == "cache":
        monkeypatch.setattr(R.O, "finalized_output_identity", lambda *a, **kw: (_ for _ in ()).throw(ValueError("identity failure")))
    elif failure == "input":
        monkeypatch.setattr(R.V.P, "check_unchanged", lambda states: (_ for _ in ()).throw(ValueError("input changed")))
    else:
        (checkpoint.parent / "STOP").write_text("stop")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        R.execute(release["pin"]["path"], release["pin"]["sha256"], "a" * 40, device="cpu")
    assert not (Path(R.OUTPUT) / "receipt.json").exists()
    assert sum(x[0] == "producer" for x in calls) == 1
