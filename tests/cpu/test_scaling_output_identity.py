"""Real small-file validation boundaries and synthetic filesystem observations.

No Lustre behavior is emulated as a proven cause. The mutants test the exact
acceptance contract and the maximum number of complete rereads.
"""
import ast
import copy
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scaling_output_identity_test", ROOT / "scripts/scaling_output_identity.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "cache.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("synthetic.npy", bytes(range(256)) * 4)
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        return {"members": archive.namelist()}


def states(observation):
    names = ("stat_before", "stat_after") if observation["kind"] == "legacy" else (
        "path_before", "stat_before", "stat_after", "path_after")
    return [observation[name] for name in names]


def times(observation, offset):
    result = copy.deepcopy(observation)
    for value in states(result):
        for key in S.TIME_FIELDS:
            value[key] += offset
    return result


def observations(cache):
    return S.observe_legacy(cache, sha), S.observe_descriptor(cache)


def fake_observer(monkeypatch, values):
    pending, seen = iter(copy.deepcopy(values)), []
    def observe(*args, **kwargs):
        item = next(pending)
        seen.append(item)
        return item
    monkeypatch.setattr(S, "observe_descriptor", observe)
    monkeypatch.setattr(S.time, "sleep", lambda seconds: None)
    return seen


def test_module_is_standard_library_only_and_frozen_input_helper_is_untouched():
    tree = ast.parse((ROOT / "scripts/scaling_output_identity.py").read_text())
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert names <= {"__future__", "datetime", "hashlib", "os", "pathlib", "stat", "time"}
    contract = json.loads((ROOT / "results/infra/scaling_300m_20261104/training_source_contract.json").read_text())
    assert sha(ROOT / "scripts/tokenizer_preflight.py") == contract["scientific_source_sha256"]["scripts/tokenizer_preflight.py"]


def test_legacy_observer_does_exactly_stat_hash_stat_without_path_probes(cache, monkeypatch):
    calls, original_stat = [], Path.stat
    def record_stat(path, *args, **kwargs):
        calls.append("stat")
        return original_stat(path, *args, **kwargs)
    def record_hash(path):
        calls.append("hash")
        return sha(path)
    monkeypatch.setattr(Path, "stat", record_stat)
    monkeypatch.setattr(Path, "lstat", lambda *args, **kwargs: pytest.fail("no legacy lstat"))
    monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: pytest.fail("no legacy resolve"))
    result = S.observe_legacy(cache, record_hash)
    assert calls == ["stat", "hash", "stat"]
    assert result["error"] is None and result["byte_count"] is None
    assert result["sha256"] == sha(cache)


def test_descriptor_observation_counts_real_bytes_and_binds_path(cache):
    result = S.observe_descriptor(cache, chunk_bytes=31)
    assert result["error"] is None and result["byte_count"] == cache.stat().st_size
    assert result["sha256"] == sha(cache) and result["canonical_path"] == str(cache.resolve())
    assert all(state["ino"] == cache.stat().st_ino for state in states(result))


def test_descriptor_does_not_open_a_fifo_or_directory(cache, monkeypatch):
    cache.unlink()
    os.mkfifo(cache)
    monkeypatch.setattr(S.os, "open", lambda *args: pytest.fail("nonregular object must not be opened"))
    result = S.observe_descriptor(cache)
    assert result["error"]["code"] == "OBJECT_TYPE" and result["sha256"] is None


def test_stable_pair_accepts_exact_pin_and_preserves_input_observations(cache):
    first, second = observations(cache)
    saved = copy.deepcopy([first, second])
    result = S.assess_pair(first, second, sha(cache), cache.stat().st_size)
    assert result["accepted"] and result["reason"] == "stable"
    assert result["original_guard_passed"] and not result["timestamp_transition"]
    assert [first, second] == saved


def test_only_atime_change_is_ignored_for_integrity(cache):
    first, second = observations(cache)
    first["stat_after"]["atime_ns"] += 100
    second["path_before"]["atime_ns"] += 1
    second["stat_after"]["atime_ns"] += 999
    result = S.assess_pair(first, second)
    assert result["reason"] == "stable" and result["original_guard_passed"]


@pytest.mark.parametrize("field", S.TIME_FIELDS)
def test_timestamp_only_change_in_first_read_is_explicitly_audited(cache, field):
    first, second = observations(cache)
    first["stat_after"][field] += 1
    result = S.assess_pair(first, second, sha(cache))
    assert result["accepted"] and result["timestamp_transition"]
    assert result["reason"] == "timestamp_settled" and not result["original_guard_passed"]


@pytest.mark.parametrize("field", S.HARD_FIELDS)
@pytest.mark.parametrize("which", (0, 1))
def test_protected_field_changes_during_either_read_are_rejected(cache, field, which):
    pair = list(observations(cache))
    pair[which]["stat_after"][field] += 1
    with pytest.raises(S.IdentityError):
        S.assess_pair(*pair)


@pytest.mark.parametrize("field", ("dev", "ino", "mode", "uid", "gid", "nlink"))
def test_protected_field_change_between_stable_reads_is_rejected_even_same_hash(cache, field):
    first, second = observations(cache)
    for value in states(second):
        value[field] += 1
    with pytest.raises(S.IdentityError, match="object changed"):
        S.assess_pair(first, second)


def test_wrong_object_type_is_rejected(cache):
    first, second = observations(cache)
    for value in states(second):
        value["mode"] = stat.S_IFDIR | 0o700
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "OBJECT_TYPE"


@pytest.mark.parametrize("field", S.TIME_FIELDS)
def test_second_read_timestamp_instability_is_not_retried(cache, field):
    first, second = observations(cache)
    second["stat_after"][field] += 1
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "SECOND_INSTABILITY"


def test_content_mismatch_and_wrong_pin_are_rejected(cache):
    first, second = observations(cache)
    second["sha256"] = "f" * 64
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "CONTENT_CHANGED"
    first, second = observations(cache)
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second, "f" * 64)
    assert exc.value.code == "CONTENT_CHANGED"
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second, expected_bytes=cache.stat().st_size + 1)
    assert exc.value.code == "SIZE_CHANGED"


def test_short_read_and_noncanonical_or_changed_paths_are_rejected(cache):
    first, second = observations(cache)
    second["byte_count"] -= 1
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "SHORT_READ"
    first, second = observations(cache)
    second["canonical_path"] = second["path"] + "-alias"
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "PATH_IDENTITY"
    second["path"] = second["canonical_path"]
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "PATH_IDENTITY"


def test_malformed_stat_or_digest_evidence_never_passes(cache):
    first, second = observations(cache)
    second["stat_after"]["ctime_ns"] = True
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "STAT_SCHEMA"
    first, second = observations(cache)
    second["sha256"] = None
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(first, second)
    assert exc.value.code == "DIGEST_SCHEMA"


def test_io_error_is_recorded_and_rejected_without_successful_digest(cache, monkeypatch):
    def fail(*args):
        raise OSError(errno.EIO, "injected read error")
    monkeypatch.setattr(S.os, "read", fail)
    result = S.observe_descriptor(cache)
    assert result["error"]["code"] == "IO_ERROR" and result["error"]["errno"] == errno.EIO
    assert result["sha256"] is None
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(S.observe_legacy(cache, sha), result)
    assert exc.value.code == "IO_ERROR"


def test_deadline_mid_hash_records_incomplete_and_closes_descriptor(cache, monkeypatch):
    calls, closed = [], []
    original_close = S.os.close
    def check():
        calls.append(1)
        if len(calls) == 4:
            raise TimeoutError("diagnostic deadline")
    def close(fd):
        closed.append(fd)
        return original_close(fd)
    monkeypatch.setattr(S.os, "close", close)
    result = S.observe_descriptor(cache, chunk_bytes=1, check_deadline=check)
    assert result["error"]["code"] == "DEADLINE" and result["sha256"] is None
    assert len(closed) == 1


def test_observer_records_start_and_finish_but_does_not_hide_programming_errors(cache):
    events = []
    result = S.observe_legacy(cache, sha, event=events.append)
    assert [e["event"] for e in events] == ["identity_observation_start", "identity_observation_end"]
    assert events[-1]["observation"] == result
    with pytest.raises(RuntimeError, match="programming"):
        S.observe_legacy(cache, lambda path: (_ for _ in ()).throw(RuntimeError("programming")))


def test_real_finalizer_preserves_two_boundaries_and_validator_details(cache):
    result = S.finalized_output_identity(cache, validate, expected_sha256=sha(cache), expected_bytes=cache.stat().st_size)
    assert result["details"] == {"members": ["synthetic.npy"]}
    assert result["identity"]["sha256"] == sha(cache)
    assert result["audit"]["extra_reads"] == 0 and len(result["audit"]["observations"]) == 2
    assert result["audit"]["validator_completed"] and result["audit"]["complete"]
    assert "mtime_ns" not in result["identity"] and "ctime_ns" not in result["identity"]


@pytest.mark.parametrize("boundary", ("before", "after", "after_internal"))
def test_single_timestamp_settling_uses_exactly_one_extra_hash(cache, monkeypatch, boundary):
    base = S.observe_descriptor(cache)
    moved = times(base, 100)
    if boundary == "before":
        first = copy.deepcopy(base)
        first["stat_after"]["ctime_ns"] += 1
        sequence = [first, moved, copy.deepcopy(moved)]
    else:
        after = copy.deepcopy(moved)
        if boundary == "after_internal":
            after["stat_before"]["ctime_ns"] -= 1
        sequence = [base, after, copy.deepcopy(moved)]
    seen = fake_observer(monkeypatch, sequence)
    result = S.finalized_output_identity(cache, validate)
    assert len(seen) == 3 and result["audit"]["extra_reads"] == 1
    assert result["audit"]["timestamp_transition"] and result["audit"]["complete"]


def test_two_separate_timestamp_events_never_get_two_extra_reads(cache, monkeypatch):
    base = S.observe_descriptor(cache)
    first = copy.deepcopy(base)
    first["stat_after"]["ctime_ns"] += 1
    sequence = [first, times(base, 10), times(base, 20), times(base, 20)]
    seen = fake_observer(monkeypatch, sequence)
    with pytest.raises(S.IdentityError) as exc:
        S.finalized_output_identity(cache, validate)
    assert exc.value.code == "RETRY_EXHAUSTED" and len(seen) == 3
    assert exc.value.audit["extra_reads"] == 1 and not exc.value.audit["complete"]


def test_unstable_extra_read_stops_before_validation(cache, monkeypatch):
    base = S.observe_descriptor(cache)
    first, second = copy.deepcopy(base), copy.deepcopy(base)
    first["stat_after"]["ctime_ns"] += 1
    second["stat_after"]["ctime_ns"] += 2
    seen = fake_observer(monkeypatch, [first, second])
    with pytest.raises(S.IdentityError) as exc:
        S.finalized_output_identity(cache, lambda p: pytest.fail("validator must not start"))
    assert exc.value.code == "SECOND_INSTABILITY" and len(seen) == 2
    assert not exc.value.audit["validator_started"]


@pytest.mark.parametrize("change", ("same_size_bytes", "append", "truncate", "replacement", "permission", "symlink"))
def test_actual_mutation_during_validation_is_rejected_without_retry(cache, change):
    old = cache.stat()
    original = cache.read_bytes()
    def mutate(path):
        if change == "same_size_bytes":
            path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        elif change == "append":
            with path.open("ab") as f:
                f.write(b"extra")
        elif change == "truncate":
            with path.open("r+b") as f:
                f.truncate(20)
        elif change == "replacement":
            other = path.with_suffix(".replacement")
            other.write_bytes(original)
            other.replace(path)
        elif change == "permission":
            path.chmod(stat.S_IMODE(old.st_mode) ^ stat.S_IWUSR)
        elif change == "symlink":
            other = path.with_suffix(".target")
            other.write_bytes(original)
            path.unlink()
            path.symlink_to(other)
        return {}
    with pytest.raises(S.IdentityError) as exc:
        S.finalized_output_identity(cache, mutate)
    assert exc.value.audit["extra_reads"] == 0
    assert not exc.value.audit["complete"]


def test_replacement_during_an_open_descriptor_hash_is_detected(cache, monkeypatch):
    original_read = S.os.read
    fired = []
    def read(fd, count):
        data = original_read(fd, count)
        if not fired:
            fired.append(True)
            replacement = cache.with_suffix(".new")
            replacement.write_bytes(cache.read_bytes())
            replacement.replace(cache)
        return data
    monkeypatch.setattr(S.os, "read", read)
    observation = S.observe_descriptor(cache)
    with pytest.raises(S.IdentityError) as exc:
        S.assess_pair(observation, observation)
    assert exc.value.code in ("OBJECT_CHANGED", "OBJECT_TYPE")


def test_short_read_in_real_reader_is_detected_before_validator(cache, monkeypatch):
    original_read = S.os.read
    def read(fd, count):
        return original_read(fd, count)[:4]
    monkeypatch.setattr(S.os, "read", read)
    with pytest.raises(S.IdentityError) as exc:
        S.finalized_output_identity(cache, lambda p: pytest.fail("short-read cannot validate"))
    assert exc.value.code == "SHORT_READ" and len(exc.value.audit["observations"]) == 1


def test_validator_failure_is_not_retried_or_marked_complete(cache):
    def fail(path):
        raise ValueError("NPZ CRC failed")
    with pytest.raises(S.IdentityError) as exc:
        S.finalized_output_identity(cache, fail)
    assert exc.value.code == "VALIDATION_ERROR" and not exc.value.audit["validator_completed"]
    assert len(exc.value.audit["observations"]) == 1


def test_finalizer_refuses_a_different_pause_policy(cache):
    with pytest.raises(S.IdentityError) as exc:
        S.finalized_output_identity(cache, validate, pause_seconds=0)
    assert exc.value.code == "PAUSE_POLICY"
