"""Bounded protocol tests: small files, injected time, no remote or GPU access."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
import scaling_p1_identity_diagnostic as D
import scaling_output_identity as I


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 30
        self.waits.append(seconds)
        self.now += seconds


class Frozen:
    hash_fn = staticmethod(D.file_sha)

    def __init__(self):
        self.calls = []

    def probe(self, path, digest, size, initial=None):
        self.calls.append((str(path), initial is not None))
        assert self.hash_fn(path) == digest and path.stat().st_size == size
        state = path.stat()
        pose = [state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns]
        return {"errors": [], "classification_identity": {"sha256": digest, "bytes": size,
            "path": str(path), "inode": state.st_ino, "mtime_ns": state.st_mtime_ns, "ctime_ns": state.st_ctime_ns},
            "pose_verify": {str(path): pose}, "pose_final_state": pose}


@pytest.fixture
def setup(tmp_path):
    inputs = []
    for i, name in enumerate(D.IDS):
        p = tmp_path/(name+".npz")
        p.write_bytes(bytes([i])*127)
        inputs.append({"id": name, "path": str(p), "bytes": p.stat().st_size,
                       "sha256": D.file_sha(p) if name in D.ACCEPTED else None,
                       "pin_kind": "accepted" if name in D.ACCEPTED else "diagnostic"})
    return {"protocol": D.PROTOCOL, "inputs": inputs, "frozen_code_dir": str(tmp_path/"frozen"),
            "frozen_code_sha": "a"*40, "frozen_sources": {"scripts/tokenizer_preflight.py": "b"*64},
            "candidate_tests": {"path": "tests.json", "sha256": "c"*64, "passed": True}}


def small_settings(**extra):
    return D.Settings(**{"target_sizes": (131072, 65536, 147456), "age_seconds": 720,
                         "pause_seconds": .01, "deadline_seconds": 1500,
                         "fixture_cap_bytes": 4_000_000, **extra})


def execute(tmp_path, manifest, *, observer=I, settings=None, clock=None, frozen=None, **extra):
    clock = clock or Clock()
    frozen = frozen or Frozen()
    barriers = []
    report = D.run(manifest, tmp_path/"diagnostic", "d"*40,
        settings=settings or small_settings(), clock=clock, sleep=clock.sleep,
        frozen=frozen, observer=observer, runtime=lambda *a: {"test_environment": True},
        metadata=lambda *a: {"test_metadata": True}, use_alarm=False,
        sync=lambda: barriers.append("sync"), fsync=lambda fd: barriers.append("fsync"),
        test_evidence={"complete": True, "passed": True}, **extra)
    return report, clock, frozen, barriers


def observer(legacy=None, descriptor=None, assess=None):
    return SimpleNamespace(observe_legacy=legacy or I.observe_legacy,
        observe_descriptor=descriptor or I.observe_descriptor, assess_pair=assess or I.assess_pair,
        IdentityError=I.IdentityError, stat_record=I.stat_record)


def test_fixed_matrix_and_real_input_coverage_are_complete_without_reproduction(tmp_path, setup):
    report, clock, frozen, barriers = execute(tmp_path, setup)
    assert report["complete"] is True
    assert report["outcome"] == "NOT_REPRODUCED"
    assert report["readers_status"] == "READERS_CLEAR_IN_TEST"
    assert report["automatic_p1_release"] is False
    assert [(x["age"], x["mode"]) for x in report["cycles_completed"]] == list(D.MATRIX)
    assert len(report["fixture_results"]) == 18
    assert [r["id"].split("_")[-1] for r in report["fixture_results"]] == list(D.NAMES)*6
    assert len(frozen.calls) == 12 and sum(x[1] for x in frozen.calls) == 6
    assert barriers.count("sync") == 2 and barriers.count("fsync") == 6
    assert all(x["age_at_primary_seconds"] >= 720 for x in report["fixture_results"]
               if x["age"] == "aged" and not x["id"].endswith("db6"))
    assert report["fixture_results"][-1]["global_sync_calls_before_primary"] == 2
    assert max(clock.waits) <= 30 and sum(clock.waits) >= 720
    events = [json.loads(x) for x in (tmp_path/"diagnostic/events.jsonl").read_text().splitlines()]
    assert [e["sequence"] for e in events] == list(range(1,len(events)+1))
    assert all(isinstance(e["kind"], str) for e in events)
    assert events[-1]["kind"] == "report"
    assert report["events_sha256"] == D.file_sha(tmp_path/"diagnostic/events.jsonl")
    assert json.loads((tmp_path/"diagnostic/report.json").read_text()) == report


def test_metadata_only_reproduction_needs_exact_oracles_and_clear_readers(tmp_path, setup):
    def legacy(path, *args, **kwargs):
        value = I.observe_legacy(path, *args, **kwargs)
        if Path(path).name == "fresh_A_db2.npz":
            value["stat_before"]["mtime_ns"] -= 1
        return value
    report, *_ = execute(tmp_path, setup, observer=observer(legacy=legacy))
    assert report["complete"] is True
    assert report["outcome"] == "METADATA_ONLY_REPRODUCED"
    assert report["metadata_only_cases"] == ["fresh_A_db2"]
    assert report["candidate_accepts_all_benign_cases"] is True
    assert report["readers_status"] == "READERS_CLEAR_IN_TEST"
    assert report["automatic_p1_release"] is False


def test_real_reader_metadata_drift_is_not_accepted_as_clear(tmp_path, setup):
    def legacy(path, *args, **kwargs):
        value = I.observe_legacy(path, *args, **kwargs)
        if Path(path).name == "pose_val.npz":
            value["stat_before"]["ctime_ns"] -= 1
        return value
    report, *_ = execute(tmp_path, setup, observer=observer(legacy=legacy))
    assert report["complete"] is True
    assert report["readers_status"] == "READERS_AFFECTED"
    assert report["outcome"] == "NOT_REPRODUCED"


def test_between_round_reader_drift_is_detected(tmp_path, setup):
    counts = {}
    def legacy(path, *args, **kwargs):
        name = Path(path).name
        if name == "pose_val.npz":
            counts[name] = counts.get(name, 0)+1
            if counts[name] == 2:
                s = Path(path).stat()
                os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns+1_000_000))
        return I.observe_legacy(path, *args, **kwargs)
    report, *_ = execute(tmp_path, setup, observer=observer(legacy=legacy))
    assert report["complete"] is True
    assert report["readers_status"] == "READERS_AFFECTED"


def test_digest_drift_is_integrity_failure_and_preserves_events(tmp_path, setup):
    def descriptor(path, **kwargs):
        value = I.observe_descriptor(path, **kwargs)
        if Path(path).name == "fresh_A_db2.npz":
            value["sha256"] = "0"*64
        return value
    report, *_ = execute(tmp_path, setup, observer=observer(descriptor=descriptor))
    assert report["outcome"] == "INTEGRITY_FAIL" and report["complete"] is False
    assert len(report["cycles_completed"]) == 0
    assert (tmp_path/"diagnostic/report.json").exists()
    events = (tmp_path/"diagnostic/events.jsonl").read_text()
    assert '"kind": "primary_descriptor"' in events and '"kind": "integrity_failure"' in events


def test_real_pin_mismatch_stops_before_fixture_writes(tmp_path, setup):
    setup["inputs"][3]["sha256"] = "0"*64
    report, *_ = execute(tmp_path, setup)
    assert report["outcome"] == "INTEGRITY_FAIL"
    assert report["fixture_bytes_written"] == 0


def test_deadline_while_waiting_for_aged_files_is_incomplete_not_integrity_fail(tmp_path, setup):
    report, clock, *_ = execute(tmp_path, setup, settings=small_settings(deadline_seconds=500))
    assert report["outcome"] == "INCOMPLETE" and report["complete"] is False
    assert report["errors"][-1]["type"] == "DeadlineExceeded"
    assert len(report["cycles_completed"]) == 3
    assert clock.now <= 1500.00001


def test_deadline_inside_historical_hash_is_incomplete(tmp_path, setup):
    class TimedFrozen(Frozen):
        @staticmethod
        def hash_fn(path):
            if Path(path).name == "fresh_A_db2.npz":
                raise D.DeadlineExceeded("mid hash")
            return D.file_sha(path)
    report, *_ = execute(tmp_path, setup, frozen=TimedFrozen())
    assert report["outcome"] == "INCOMPLETE" and report["complete"] is False
    assert report["errors"][-1]["type"] == "DeadlineExceeded"


def test_validator_is_between_the_two_primary_hashes(tmp_path):
    path = tmp_path/"ordered.bin"
    path.write_bytes(b"content")
    clock = Clock()
    deadline = D.Deadline(100, clock, clock.sleep)
    actions = []
    def legacy(*args, **kwargs):
        actions.append("first")
        return I.observe_legacy(*args, **kwargs)
    def descriptor(*args, **kwargs):
        actions.append("second")
        return I.observe_descriptor(*args, **kwargs)
    def validate():
        actions.append("validate")
        return {"passed": True}
    _, _, result = D.observe_pair(path, D.file_sha(path), 7, D.file_sha, deadline,
        lambda *a: None, 2, observer(legacy, descriptor), validate)
    assert actions == ["first", "validate", "second"] and result["passed"]
    assert clock.waits == [2]


def test_mutation_during_validation_is_detected_by_second_hash(tmp_path):
    path = tmp_path/"mutant.bin"
    path.write_bytes(b"before!")
    digest = D.file_sha(path)
    s = path.stat()
    clock = Clock()
    def mutate():
        path.write_bytes(b"after!!")
        os.utime(path, ns=(s.st_atime_ns,s.st_mtime_ns))
        return {"passed": True}
    with pytest.raises(D.IntegrityFailure, match="digest"):
        D.observe_pair(path, digest, 7, D.file_sha, D.Deadline(100,clock,clock.sleep),
                       lambda *a: None, 2, validator=mutate)


def test_candidate_rejection_is_incomplete_even_with_integral_fixtures(tmp_path, setup):
    def reject(*args, **kwargs):
        raise I.IdentityError("SECOND_INSTABILITY", "test rejection")
    report, *_ = execute(tmp_path, setup, observer=observer(assess=reject))
    assert report["outcome"] == "INCOMPLETE" and report["complete"] is False
    assert report["candidate_accepts_all_benign_cases"] is False


@pytest.mark.parametrize("mutation", [
    lambda m: m["inputs"].reverse(),
    lambda m: m["inputs"].pop(),
    lambda m: m["inputs"][0].update(pin_kind="accepted"),
    lambda m: m["inputs"][3].update(sha256=None),
    lambda m: m["inputs"][1].update(path=m["inputs"][0]["path"]),
    lambda m: m["inputs"][0].update(path="relative.npz"),
    lambda m: m["candidate_tests"].update(passed=False),
])
def test_manifest_refuses_unplanned_or_unpinned_inputs(setup, mutation):
    mutation(setup)
    with pytest.raises(ValueError):
        D.validate_manifest(setup)


def test_runtime_test_evidence_is_pinned_to_helper(tmp_path):
    p = tmp_path/"tests.json"
    value = {"complete": True, "passed": True,
             "source_sha256": {"scripts/scaling_output_identity.py": D.file_sha(Path(I.__file__))}}
    p.write_text(json.dumps(value))
    manifest = {"candidate_tests": {"path": str(p), "sha256": D.file_sha(p)}}
    assert D.candidate_test_evidence(manifest)["passed"] is True
    value["source_sha256"]["scripts/scaling_output_identity.py"] = "0"*64
    p.write_text(json.dumps(value))
    manifest["candidate_tests"]["sha256"] = D.file_sha(p)
    with pytest.raises(ValueError, match="another helper"):
        D.candidate_test_evidence(manifest)


def test_production_constants_and_cli_do_not_expose_test_shortcuts():
    assert D.Settings() == D.Settings((1934763898,493636450,2191580650),720,2,1500,32_000_000_000)
    with pytest.raises(SystemExit):
        D.main(["--small-fixtures"])


def test_statx_is_observed_only_after_primary_sequence(tmp_path, setup):
    report, *_ = execute(tmp_path, setup)
    events = [json.loads(line) for line in (tmp_path/"diagnostic/events.jsonl").read_text().splitlines()]
    first_fixture = next(x for x in report["fixture_results"] if x["id"] == "fresh_A_db2")
    assert first_fixture["post_observation_metadata"] == {"test_metadata": True}
    relevant = [e["kind"] for e in events if e["kind"] in
                ("primary_legacy", "validation_begin", "validation_end", "primary_descriptor", "fixture_complete")]
    n = relevant.index("validation_begin")
    assert relevant[n-1:n+4] == ["primary_legacy", "validation_begin", "validation_end", "primary_descriptor", "fixture_complete"]


@pytest.mark.parametrize("code", ["PATH_IDENTITY", "CONTENT_CHANGED", "OBJECT_CHANGED"])
def test_hard_candidate_refusal_stops_as_integrity_failure(tmp_path, setup, code):
    def reject(*args, **kwargs):
        raise I.IdentityError(code, "test hard refusal")
    report, *_ = execute(tmp_path, setup, observer=observer(assess=reject))
    assert report["complete"] is False
    assert report["outcome"] == "INTEGRITY_FAIL"
    assert report["cycles_completed"] == []


def real_readers():
    # Exercise the actual unchanged guard functions and dispatcher import order.
    import importlib
    value = D.FrozenReaders.__new__(D.FrozenReaders)
    value.compare = importlib.import_module("scaling_compare19618")
    value.P = importlib.import_module("tokenizer_preflight")
    value.Q = importlib.import_module("tokenizer_pose_readout")
    value.hash_fn = value.P.R.sha256_file
    return value


def test_real_frozen_classification_and_pose_guards_detect_between_round_drift(tmp_path):
    value = real_readers()
    path = tmp_path/"actual-guard.npz"
    D.np.savez(path, payload=D.np.arange(31, dtype=D.np.uint8))
    digest, size = D.file_sha(path), path.stat().st_size
    initial = value.probe(path, digest, size)
    assert initial["errors"] == []
    assert initial["classification_identity"]["sha256"] == digest
    assert initial["pose_verify"][str(path)] == value.Q.T.file_state(path)
    state = path.stat()
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns+1_000_000))
    after = value.probe(path, digest, size, initial)
    assert {x["stage"] for x in after["errors"]} == {
        "classification_check_unchanged", "pose_final_state"}


@pytest.mark.parametrize("target", ["file_identity", "check_unchanged"])
def test_real_frozen_adapter_never_swallows_deadline(tmp_path, monkeypatch, target):
    value = real_readers()
    path = tmp_path/"timeout.npz"
    path.write_bytes(b"fixture")
    def fail(*args, **kwargs):
        raise D.DeadlineExceeded("in hash or guard")
    monkeypatch.setattr(value.P, target, fail)
    with pytest.raises(D.DeadlineExceeded):
        value.probe(path, D.file_sha(path), path.stat().st_size)


def test_real_frozen_adapter_classifies_io_subclasses(tmp_path, monkeypatch):
    value = real_readers()
    path = tmp_path/"io.npz"
    path.write_bytes(b"fixture")
    def fail(*args, **kwargs):
        raise FileNotFoundError("disappeared during hash")
    monkeypatch.setattr(value.P, "file_identity", fail)
    result = value.probe(path, D.file_sha(path), path.stat().st_size)
    assert result["errors"][0]["type"] == "FileNotFoundError"
    assert result["errors"][0]["kind"] == "io"


@pytest.mark.parametrize("field,expected", [("mtime_ns", "READERS_AFFECTED"), ("inode", "INTEGRITY_FAIL"),
                                          ("sha256", "INTEGRITY_FAIL")])
def test_frozen_guard_bridge_detects_changes_after_primary(tmp_path, setup, field, expected):
    class Drifting(Frozen):
        def probe(self, path, digest, size, initial=None):
            value = super().probe(path, digest, size, initial)
            if initial is not None and path.name == "control30m_db3.npz":
                identity = value["classification_identity"]
                identity[field] = "0"*64 if field == "sha256" else identity[field]+1
            return value
    report, *_ = execute(tmp_path, setup, frozen=Drifting())
    if expected == "READERS_AFFECTED":
        assert report["complete"] is True
        assert report["readers_status"] == expected
        assert report["reader_rounds"][1]["inputs"][-1]["guard_boundary_drift"] == ["mtime_ns"]
    else:
        assert report["complete"] is False
        assert report["outcome"] == expected


def test_frozen_classification_digest_is_checked_against_pin(tmp_path, monkeypatch):
    value = real_readers()
    path = tmp_path/"wrong-classification-digest.npz"
    path.write_bytes(b"fixture")
    real = value.P.file_identity
    monkeypatch.setattr(value.P, "file_identity", lambda p: {**real(p), "sha256": "0"*64})
    with pytest.raises(D.IntegrityFailure, match="expected SHA or size"):
        value.probe(path, D.file_sha(path), path.stat().st_size)
