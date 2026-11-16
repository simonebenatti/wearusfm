"""Bounded identity observations for *new, closed outputs* of the P1 producer.

This standard-library-only module does not replace any frozen input guard.
The diagnostic observer preserves the historical stat/hash/stat sequence.
Production finalization has two validation boundaries and at most one extra
complete hash in total. Timestamp observations remain in an audit; they are
not silently turned into an assertion that every attribute stayed unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import stat
import time

POLICY = "scaling-output-identity-20261116-v1"
STAT_FIELDS = ("dev", "ino", "size", "mode", "uid", "gid", "nlink", "atime_ns", "mtime_ns", "ctime_ns")
HARD_FIELDS = ("dev", "ino", "size", "mode", "uid", "gid", "nlink")
TIME_FIELDS = ("mtime_ns", "ctime_ns")
LEGACY_FIELDS = ("ino", "size", "mtime_ns", "ctime_ns")
PAUSE_SECONDS = 2.0
CHUNK_BYTES = 1024 * 1024


class IdentityError(ValueError):
    """A fail-closed identity rejection, with the evidence collected so far."""

    def __init__(self, code, message, audit=None):
        super().__init__(message)
        self.code = code
        self.audit = audit or {}


def _require(ok, code, message, audit=None):
    if not ok:
        raise IdentityError(code, message, audit)


def stat_record(value):
    """Integer stat values; atime is observed but never an integrity criterion."""
    return {name: int(getattr(value, "st_" + name)) for name in STAT_FIELDS}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _check(check_deadline):
    if check_deadline is not None:
        check_deadline()


def _event(event, value):
    if event is not None:
        event(value)


def _start(path, kind, event):
    result = {"kind": kind, "path": os.path.abspath(os.fspath(path)),
              "started_utc": _utc(), "started_monotonic": time.monotonic(),
              "sha256": None, "byte_count": None, "stat_before": None, "stat_after": None,
              "path_before": None, "path_after": None, "canonical_path": None, "error": None}
    _event(event, {"event": "identity_observation_start", "kind": kind,
                   "path": result["path"], "started_utc": result["started_utc"]})
    return result


def _finish(result, event):
    result["finished_monotonic"] = time.monotonic()
    result["finished_utc"] = _utc()
    result["duration_seconds"] = result["finished_monotonic"] - result["started_monotonic"]
    before, after = result["stat_before"], result["stat_after"]
    result["changed_fields"] = ([k for k in STAT_FIELDS if before[k] != after[k]]
                                if before is not None and after is not None else None)
    _event(event, {"event": "identity_observation_end", "observation": result})
    return result


def _error(exc):
    return {"code": "DEADLINE" if isinstance(exc, TimeoutError) else "IO_ERROR",
            "type": type(exc).__name__, "errno": getattr(exc, "errno", None), "message": str(exc)}


def observe_legacy(path, hash_fn, *, check_deadline=None, event=None):
    """Observe exactly Path.stat -> historical hash_fn(path) -> Path.stat.

    No resolve, lstat, descriptor probe or extra open precedes this sequence.
    Drift is returned, not raised. OSError/TimeoutError is recorded; unexpected
    programming errors propagate. The supplied hash function owns any checks
    inside its chunks. No byte-count claim is invented from stat size.
    """
    path = Path(path)
    result = _start(path, "legacy", event)
    try:
        _check(check_deadline)
        result["stat_before"] = stat_record(path.stat())
        result["sha256"] = hash_fn(path)
        result["stat_after"] = stat_record(path.stat())
        _check(check_deadline)
    except OSError as exc:
        result["error"] = _error(exc)
    return _finish(result, event)


def observe_descriptor(path, *, check_deadline=None, event=None, chunk_bytes=CHUNK_BYTES):
    """Hash one descriptor and bind it to the path at both boundaries.

    This is the second diagnostic observation or a production observation;
    it must not be substituted for the first historical observation. lstat
    and O_NOFOLLOW reject final-component symlinks; canonical-path equality
    additionally rejects symlinked parents when the assessment is made.
    """
    _require(type(chunk_bytes) is int and chunk_bytes > 0, "CHUNK_SIZE", "invalid read chunk size")
    path = Path(path)
    result = _start(path, "descriptor", event)
    fd = None
    try:
        _check(check_deadline)
        result["path_before"] = stat_record(path.lstat())
        if not stat.S_ISREG(result["path_before"]["mode"]):
            result["error"] = {"code": "OBJECT_TYPE", "type": "NonRegularFile", "errno": None,
                               "message": "path does not name a regular file"}
            return _finish(result, event)
        flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) |
                 getattr(os, "O_NONBLOCK", 0))
        fd = os.open(path, flags)
        result["stat_before"] = stat_record(os.fstat(fd))
        if not stat.S_ISREG(result["stat_before"]["mode"]):
            result["error"] = {"code": "OBJECT_TYPE", "type": "NonRegularFile", "errno": None,
                               "message": "opened descriptor is not a regular file"}
        else:
            digest, count = hashlib.sha256(), 0
            while True:
                _check(check_deadline)
                data = os.read(fd, chunk_bytes)
                if not data:
                    break
                digest.update(data)
                count += len(data)
            result["sha256"], result["byte_count"] = digest.hexdigest(), count
            result["stat_after"] = stat_record(os.fstat(fd))
            result["path_after"] = stat_record(path.lstat())
            result["canonical_path"] = str(path.resolve(strict=True))
            _check(check_deadline)
    except OSError as exc:
        result["error"] = _error(exc)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError as exc:
                result["error"] = result["error"] or _error(exc)
    return _finish(result, event)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _snapshots(observation):
    names = ("stat_before", "stat_after")
    if observation.get("kind") == "descriptor":
        names = ("path_before", "stat_before", "stat_after", "path_after")
    return [observation.get(name) for name in names]


def _same(states, fields):
    return all(all(state[k] == states[0][k] for k in fields) for state in states[1:])


def _validate_observation(observation, expected_sha256=None, expected_bytes=None):
    audit = {"observations": [observation]}
    _require(isinstance(observation, dict), "OBSERVATION_SCHEMA", "observation is not an object")
    error = observation.get("error")
    if error is not None:
        raise IdentityError(error.get("code", "IO_ERROR"), error.get("message", "observation failed"), audit)
    _require(observation.get("kind") in ("legacy", "descriptor"), "OBSERVATION_SCHEMA", "unknown observation kind", audit)
    path = observation.get("path")
    _require(isinstance(path, str) and os.path.isabs(path) and os.path.normpath(path) == path,
             "PATH_IDENTITY", "path must be absolute and normalized", audit)
    states = _snapshots(observation)
    for value in states:
        _require(isinstance(value, dict) and all(type(value.get(k)) is int for k in STAT_FIELDS),
                 "STAT_SCHEMA", "incomplete or noninteger stat evidence", audit)
        _require(stat.S_ISREG(value["mode"]) and value["size"] >= 0 and value["nlink"] >= 1,
                 "OBJECT_TYPE", "object is not a regular linked file", audit)
    _require(_same(states, HARD_FIELDS), "OBJECT_CHANGED", "object or protected attributes changed", audit)
    _require(_digest(observation.get("sha256")), "DIGEST_SCHEMA", "missing or invalid complete SHA256", audit)
    if observation["kind"] == "descriptor":
        _require(observation.get("canonical_path") == path, "PATH_IDENTITY", "path is not canonical", audit)
        _require(type(observation.get("byte_count")) is int and observation["byte_count"] == states[0]["size"],
                 "SHORT_READ", "read byte count does not equal the stable file size", audit)
    else:
        _require(observation.get("byte_count") is None, "OBSERVATION_SCHEMA", "legacy byte count must be unknown", audit)
    if expected_sha256 is not None:
        _require(_digest(expected_sha256), "PIN_SCHEMA", "invalid expected SHA256", audit)
        _require(observation["sha256"] == expected_sha256, "CONTENT_CHANGED", "digest differs from fixed expected digest", audit)
    if expected_bytes is not None:
        _require(type(expected_bytes) is int and expected_bytes >= 0, "PIN_SCHEMA", "invalid expected byte count", audit)
        _require(states[0]["size"] == expected_bytes, "SIZE_CHANGED", "size differs from fixed expected byte count", audit)
    return states


def _pair(first, second, expected_sha256=None, expected_bytes=None):
    states = (_validate_observation(first, expected_sha256, expected_bytes) +
              _validate_observation(second, expected_sha256, expected_bytes))
    audit = {"observations": [first, second]}
    _require(first["path"] == second["path"], "PATH_IDENTITY", "path changed between observations", audit)
    _require(_same(states, HARD_FIELDS), "OBJECT_CHANGED", "object changed between observations", audit)
    _require(first["sha256"] == second["sha256"], "CONTENT_CHANGED", "complete hashes differ", audit)
    return states


def _identity(observation):
    state = observation["stat_after"]
    return {"path": observation["path"], "sha256": observation["sha256"], "bytes": state["size"],
            "dev": state["dev"], "inode": state["ino"], "mode": state["mode"],
            "uid": state["uid"], "gid": state["gid"], "nlink": state["nlink"]}


def assess_pair(first, second, expected_sha256=None, expected_bytes=None):
    """Pure assessment of two recorded reads; never retries or changes a pin.

    The second read must be descriptor-based and stable in its timestamps as
    well as its protected object fields. Only mtime/ctime may settle from the
    first read. Atime is audit-only. Rejections carry .code and .audit.
    """
    states = _pair(first, second, expected_sha256, expected_bytes)
    audit = {"observations": [first, second]}
    _require(second.get("kind") == "descriptor", "SECOND_READ_KIND", "second read requires a descriptor", audit)
    _require(_same(_snapshots(second), TIME_FIELDS), "SECOND_INSTABILITY", "second read has unstable timestamps", audit)
    transition = not _same(states, TIME_FIELDS)
    return {"accepted": True, "identity": _identity(second), "timestamp_transition": transition,
            "original_guard_passed": _same([first["stat_before"], first["stat_after"]], LEGACY_FIELDS),
            "reason": "timestamp_settled" if transition else "stable", "policy": POLICY}


def finalized_output_identity(path, validator, *, expected_sha256=None, expected_bytes=None,
                              pause_seconds=PAUSE_SECONDS, check_deadline=None, event=None):
    """Validate a newly produced output, using at most three complete hashes.

    A caller must establish a finished producer and exclusive output ownership.
    This performs no sync, file mutation, receipt write or change to input guards.
    The supplied validator is called once and its returned details are preserved.
    """
    _require(type(pause_seconds) in (int, float) and pause_seconds == PAUSE_SECONDS,
             "PAUSE_POLICY", "the frozen reread pause is exactly two seconds")
    audit = {"policy": POLICY, "observations": [], "extra_reads": 0, "timestamp_transition": False,
             "validator_started": False, "validator_completed": False}

    def observe():
        value = observe_descriptor(path, check_deadline=check_deadline, event=event)
        audit["observations"].append(value)
        _validate_observation(value, expected_sha256, expected_bytes)
        return value

    def settle(first):
        _require(audit["extra_reads"] == 0, "RETRY_EXHAUSTED", "only one extra complete read is permitted", audit)
        audit["extra_reads"] = 1
        _check(check_deadline)
        time.sleep(pause_seconds)
        _check(check_deadline)
        second = observe()
        assess_pair(first, second, expected_sha256, expected_bytes)
        audit["timestamp_transition"] = True
        return second

    try:
        initial = observe()
        before = settle(initial) if not _same(_snapshots(initial), TIME_FIELDS) else initial
        _check(check_deadline)
        audit["validator_started"] = True
        details = validator(Path(path))
        audit["validator_completed"] = True
        _check(check_deadline)
        after = observe()
        states = _pair(before, after, expected_sha256, expected_bytes)
        if not _same(states, TIME_FIELDS):
            after = settle(after)
            _pair(before, after, expected_sha256, expected_bytes)
        # All prior observations, including a rejected first timestamp sample,
        # remain bound to the final byte/object identity; only time may differ.
        for previous in audit["observations"]:
            _pair(previous, after, expected_sha256, expected_bytes)
        audit["complete"] = True
        return {"identity": _identity(after), "details": details, "audit": audit}
    except IdentityError as exc:
        audit["complete"] = False
        exc.audit = audit
        raise
    except OSError as exc:
        audit["complete"] = False
        error = _error(exc)
        raise IdentityError(error["code"], error["message"], audit) from exc
    except Exception as exc:
        audit["complete"] = False
        raise IdentityError("VALIDATION_ERROR", f"{type(exc).__name__}: {exc}", audit) from exc
