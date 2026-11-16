"""Bounded, non-adopting P1 file-identity experiment. No model inference or job submission.

Production dimensions, order, age and deadline are fixed by the approved protocol.
Small sizes and fake clocks are injectable through the Python API only, for CPU tests.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import signal
import stat
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np  # Preserve the frozen readers' NumPy-first bootstrap.

import scaling_output_identity as I

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "SCALING-P1-IDENTITY-DIAG-v1"
IDS = ("p1_db6_failed", "p1r1_db3_failed", "p1r1_db2_completed",
       "pose_val", "p2_epn", "control30m_db3")
ACCEPTED = {"pose_val", "p2_epn", "control30m_db3"}
NAMES = ("db2", "db3", "db6")
MATRIX = (("fresh", "A"), ("fresh", "F"), ("fresh", "S"),
          ("aged", "S"), ("aged", "F"), ("aged", "A"))
OBJECT_FIELDS = ("dev", "ino", "size", "mode", "uid", "gid", "nlink")
STATE_FIELDS = OBJECT_FIELDS + ("mtime_ns", "ctime_ns")
CHUNK = 1 << 20


class DeadlineExceeded(TimeoutError):
    pass


class IntegrityFailure(RuntimeError):
    pass


class CandidatePolicyFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    target_sizes: tuple = (1934763898, 493636450, 2191580650)
    age_seconds: float = 720.0
    pause_seconds: float = 2.0
    deadline_seconds: float = 1500.0
    fixture_cap_bytes: int = 32_000_000_000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


class Deadline:
    def __init__(self, seconds, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.start = clock()
        self.end = self.start + seconds

    def check(self):
        if self.clock() >= self.end:
            raise DeadlineExceeded("application deadline reached")

    def wait(self, seconds, event):
        target = self.clock() + seconds
        while self.clock() < target:
            self.check()
            wait = min(30.0, target - self.clock(), self.end - self.clock())
            event("wait", {"seconds": wait, "remaining_seconds": self.end-self.clock()})
            self.sleep(wait)
        self.check()


@contextmanager
def alarm(deadline, enabled):
    if not enabled:
        yield
        return
    require(hasattr(signal, "setitimer"), "production requires a POSIX application timer")
    old = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "another application alarm is already active")
    def expired(*_):
        raise DeadlineExceeded("application timer expired")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline.end-deadline.clock()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class Journal:
    def __init__(self, out, deadline):
        self.path = out / "events.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.deadline = deadline
        self.index = 0

    def __call__(self, kind, details=None):
        self.index += 1
        value = {"sequence": self.index, "utc": utc(), "monotonic": self.deadline.clock(),
                 "elapsed_seconds": self.deadline.clock()-self.deadline.start,
                 "kind": kind, "details": details or {}}
        self.stream.write(json.dumps(value, sort_keys=True, default=str)+"\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def bounded_command(argv, deadline):
    deadline.check()
    try:
        r = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=min(10.0, max(0.01, deadline.end-deadline.clock())), check=False)
        return {"argv": argv, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except TimeoutError:
        raise
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"argv": argv, "unavailable": type(e).__name__+": "+str(e)}


def post_metadata(path, deadline):
    """Post-observation only: never call before the primary identity sequence."""
    result = {"layout": bounded_command(["lfs", "getstripe", str(path)], deadline),
              "fid": bounded_command(["lfs", "path2fid", str(path)], deadline)}
    if platform.system() != "Linux":
        result["statx_force_sync"] = {"unavailable": "Linux statx is unavailable"}
        return result
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        call = libc.statx
        call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
        call.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(256)
        deadline.check()
        rc = call(-100, os.fsencode(path), 0x2000, 0x7ff, ctypes.byref(buffer))
        if rc:
            err = ctypes.get_errno()
            result["statx_force_sync"] = {"unavailable": os.strerror(err), "errno": err}
        else:
            data = buffer.raw
            def timestamp(offset):
                sec, ns = struct.unpack_from("=qI", data, offset)
                return sec*1_000_000_000+ns
            result["statx_force_sync"] = {"mask": struct.unpack_from("=I", data, 0)[0],
                "ino": struct.unpack_from("=Q", data, 32)[0], "size": struct.unpack_from("=Q", data, 40)[0],
                "atime_ns": timestamp(64), "ctime_ns": timestamp(96), "mtime_ns": timestamp(112)}
    except TimeoutError:
        raise
    except (AttributeError, OSError) as e:
        result["statx_force_sync"] = {"unavailable": type(e).__name__+": "+str(e)}
    return result


def environment(out, deadline):
    return {"python": platform.python_version(), "numpy": np.__version__, "hostname": platform.node(),
            "kernel": platform.release(), "platform": platform.platform(),
            "allocation": {k: os.environ.get(k, "unavailable") for k in
                ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS")},
            "thread_environment": {k: os.environ.get(k, "unavailable") for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "directory_stat": I.stat_record(out.stat()),
            "filesystem": bounded_command(["stat", "-f", str(out)], deadline),
            "mount": bounded_command(["findmnt", "-T", str(out), "-J"], deadline),
            "lustre_version": bounded_command(["lfs", "--version"], deadline),
            "directory_layout": bounded_command(["lfs", "getstripe", str(out)], deadline),
            "own_child_processes": "Only the read-only metadata commands recorded in this report; no background writers"}


def validate_manifest(manifest):
    require(manifest.get("protocol") == PROTOCOL, "wrong diagnostic protocol")
    require(isinstance(manifest.get("inputs"), list) and
            [v.get("id") for v in manifest["inputs"]] == list(IDS), "six preselected inputs required in order")
    paths = []
    for item in manifest["inputs"]:
        path = item.get("path")
        require(isinstance(path, str) and Path(path).is_absolute() and os.path.normpath(path) == path,
                "input paths must be absolute normalized paths")
        paths.append(path)
        require(type(item.get("bytes")) is int and item["bytes"] > 0, "positive input size required")
        require(item.get("pin_kind") == ("accepted" if item["id"] in ACCEPTED else "diagnostic"),
                "failed jobs may only carry diagnostic pins")
        digest = item.get("sha256")
        require(digest is None or (isinstance(digest, str) and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest)), "invalid input SHA")
        require(item["pin_kind"] != "accepted" or digest is not None, "accepted inputs need historical SHA")
    require(len(set(paths)) == 6, "duplicate input paths")
    require(Path(manifest.get("frozen_code_dir", "")).is_absolute(), "frozen checkout path required")
    require(isinstance(manifest.get("frozen_sources"), dict) and manifest["frozen_sources"], "frozen source pins required")
    require(isinstance(manifest.get("frozen_code_sha"), str) and len(manifest["frozen_code_sha"]) == 40,
            "frozen checkout SHA required")
    tests = manifest.get("candidate_tests", {})
    require(tests.get("passed") is True and isinstance(tests.get("path"), str) and
            isinstance(tests.get("sha256"), str) and len(tests["sha256"]) == 64,
            "pinned candidate CPU test attestation required")


class FrozenReaders:
    def __init__(self, manifest):
        root = Path(manifest["frozen_code_dir"])
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        require(actual == manifest["frozen_code_sha"], "frozen checkout commit differs")
        require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=root), "frozen checkout is dirty")
        for name, digest in manifest["frozen_sources"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "invalid frozen source path")
            require(file_sha(root/name) == digest, "frozen source differs: "+name)
        sys.path[:0] = [str(root/"scripts"), str(root/"src")]
        # This is the frozen CPU dispatcher's bootstrap, before any reader calls.
        self.compare = importlib.import_module("scaling_compare19618")
        self.P = importlib.import_module("tokenizer_preflight")
        self.Q = importlib.import_module("tokenizer_pose_readout")
        require(Path(self.P.__file__).resolve() == (root/"scripts/tokenizer_preflight.py").resolve() and
                Path(self.Q.__file__).resolve() == (root/"scripts/tokenizer_pose_readout.py").resolve(),
                "frozen modules were already imported from another checkout")
        self.hash_fn = self.P.R.sha256_file

    def probe(self, path, digest, size, initial=None):
        result = {"classification": {}, "pose": {}, "errors": []}
        for key, action in (
            ("classification_identity", lambda: self.P.file_identity(path)),
            ("pose_verify", lambda: self.Q.verify_file(path, digest, size))):
            try:
                result[key] = action()
            except TimeoutError:
                raise
            except (ValueError, OSError) as e:
                result["errors"].append({"stage": key, "type": type(e).__name__, "message": str(e),
                                       "kind": "io" if isinstance(e, OSError) else "guard"})
        state = result.get("classification_identity")
        if state is not None:
            if state.get("sha256") != digest or state.get("bytes") != size:
                raise IntegrityFailure("frozen classification identity differs from expected SHA or size")
            try:
                self.P.check_unchanged({"cache": initial["classification_identity"] if initial and
                    "classification_identity" in initial else state})
            except TimeoutError:
                raise
            except (ValueError, OSError) as e:
                result["errors"].append({"stage": "classification_check_unchanged", "type": type(e).__name__, "message": str(e),
                                       "kind": "io" if isinstance(e, OSError) else "guard"})
        result["pose_final_state"] = self.Q.T.file_state(path)
        for prior in (result.get("pose_verify"), initial.get("pose_verify") if initial else None):
            if prior and prior.get(str(path)) != result["pose_final_state"]:
                result["errors"].append({"stage": "pose_final_state", "type": "ValueError", "message": "state drift"})
        return result


def candidate_test_evidence(manifest):
    pin = manifest["candidate_tests"]
    path = Path(pin["path"])
    if not path.is_absolute():
        path = ROOT/path
    require(file_sha(path) == pin["sha256"], "candidate test report SHA differs")
    value = json.loads(path.read_text())
    require(value.get("complete") is True and value.get("passed") is True, "candidate CPU tests did not pass")
    require(value.get("source_sha256", {}).get("scripts/scaling_output_identity.py") == file_sha(Path(I.__file__)),
            "candidate test report belongs to another helper")
    return {"path": str(path), "sha256": pin["sha256"], "complete": True, "passed": True}


def payload(length, salt, deadline):
    array = np.empty(length, dtype=np.uint8)
    for start in range(0, length, CHUNK):
        deadline.check()
        count = min(CHUNK, length-start)
        array[start:start+count] = np.arange(count, dtype=np.uint8)+np.uint8((start+salt) % 256)
    return array


def buffer_digest(buffer, deadline):
    h = hashlib.sha256()
    view = buffer.getbuffer()
    try:
        for start in range(0, len(view), CHUNK):
            deadline.check()
            h.update(view[start:start+CHUNK])
        return h.hexdigest(), len(view)
    finally:
        view.release()


def write_fixture(path, fixture_id, target, salt, deadline, event, written, cap):
    deadline.check()
    array = payload(max(1, target-1024), salt, deadline)
    values = {"payload": array, "fixture_id": np.asarray(fixture_id)}
    event("oracle_begin", {"fixture": fixture_id, "target_bytes": target})
    with io.BytesIO() as oracle:
        np.savez(oracle, **values)
        expected_sha, expected_bytes = buffer_digest(oracle, deadline)
    require(abs(expected_bytes-target) <= .01*target, "fixture does not meet predeclared size tolerance")
    require(written+expected_bytes <= cap, "fixture write cap exceeded")
    deadline.check()
    event("write_begin", {"fixture": fixture_id, "path": str(path), "expected_sha256": expected_sha,
                          "expected_bytes": expected_bytes, "salt": salt})
    started = deadline.clock()
    np.savez(path, **values)
    closed = deadline.clock()
    # No stat, resolve, reopen or extra hash before the primary observation.
    event("write_closed", {"fixture": fixture_id, "duration_seconds": closed-started})
    return {"id": fixture_id, "path": str(path), "sha256": expected_sha, "bytes": expected_bytes,
            "salt": salt, "payload_length": len(array), "closed_monotonic": closed,
            "write_seconds": closed-started}


def verify_synthetic_npz(record, deadline):
    started = deadline.clock()
    with zipfile.ZipFile(record["path"]) as z:
        names = z.namelist()
        if sorted(names) != ["fixture_id.npy", "payload.npy"] or any(i.compress_type != zipfile.ZIP_STORED for i in z.infolist()):
            raise IntegrityFailure("synthetic NPZ schema or compression changed")
        with z.open("payload.npy") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise IntegrityFailure("unexpected NPY version")
            if shape != (record["payload_length"],) or fortran or dtype != np.dtype("u1"):
                raise IntegrityFailure("synthetic payload header differs")
            offset = 0
            while offset < record["payload_length"]:
                deadline.check()
                data = f.read(min(CHUNK, record["payload_length"]-offset))
                if not data:
                    raise IntegrityFailure("short synthetic payload")
                expected = np.arange(len(data), dtype=np.uint8)+np.uint8((offset+record["salt"]) % 256)
                if not np.array_equal(np.frombuffer(data, dtype=np.uint8), expected):
                    raise IntegrityFailure("synthetic payload bytes differ")
                offset += len(data)
            if f.read(1):
                raise IntegrityFailure("synthetic payload has trailing bytes")
        with z.open("fixture_id.npy") as f:
            if np.lib.format.read_array(f, allow_pickle=False).item() != record["id"]:
                raise IntegrityFailure("fixture identity differs")
    return {"passed": True, "duration_seconds": deadline.clock()-started}


def apply_barrier(mode, records, deadline, event, sync=os.sync, fsync=os.fsync):
    deadline.check()
    start = deadline.clock()
    event("barrier_begin", {"mode": mode, "paths": [r["path"] for r in records]})
    if mode == "F":
        for record in records:
            deadline.check()
            fd = os.open(record["path"], os.O_RDONLY)
            try:
                fsync(fd)
            finally:
                os.close(fd)
    elif mode == "S":
        sync()
    elif mode != "A":
        raise ValueError("unregistered barrier")
    deadline.check()
    value = {"mode": mode, "duration_seconds": deadline.clock()-start}
    event("barrier_end", value)
    return value


def check_observations(first, second, expected_sha, expected_bytes):
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise IntegrityFailure("missing complete reference digest")
    for item in (first, second):
        if item.get("error"):
            error = item["error"]
            if error.get("type") in ("DeadlineExceeded", "TimeoutError"):
                raise DeadlineExceeded(str(error))
            raise IntegrityFailure("observation failed: "+str(error))
        if item.get("sha256") != expected_sha:
            raise IntegrityFailure("digest differs from oracle or pin")
        if any(item[field]["size"] != expected_bytes for field in ("stat_before", "stat_after")):
            raise IntegrityFailure("size differs from oracle or pin")
    if second.get("byte_count") != expected_bytes:
        raise IntegrityFailure("descriptor byte count differs")
    if first.get("path") != second.get("path") or second.get("canonical_path") != second.get("path"):
        raise IntegrityFailure("path identity differs")
    states = [first["stat_before"], first["stat_after"], second["stat_before"], second["stat_after"],
              second["path_before"], second["path_after"]]
    if any(any(s[k] != states[0][k] for k in OBJECT_FIELDS) for s in states[1:]):
        raise IntegrityFailure("object identity changed")
    if not stat.S_ISREG(states[0]["mode"]):
        raise IntegrityFailure("not a regular file")


def observe_pair(path, expected_sha, expected_bytes, hash_fn, deadline, event, pause, observer=I, validator=None):
    event("primary_begin", {"path": str(path)})
    observer_event = lambda value: event("identity_observer", value)
    first = observer.observe_legacy(path, hash_fn, check_deadline=deadline.check, event=observer_event)
    event("primary_legacy", first)
    deadline.wait(pause, event)
    validation = None
    if validator is not None:
        deadline.check()
        event("validation_begin", {"path": str(path)})
        validation = validator()
        event("validation_end", validation)
    second = observer.observe_descriptor(path, check_deadline=deadline.check, event=observer_event)
    event("primary_descriptor", second)
    digest = expected_sha if expected_sha is not None else first.get("sha256")
    check_observations(first, second, digest, expected_bytes)
    return first, second, validation


def frozen_observation_bridge(actual, reference, path, digest, size):
    """Compare already returned guard states; no additional filesystem reads."""
    states = []
    classification = actual.get("classification_identity")
    if classification is not None:
        if (classification.get("sha256") != digest or classification.get("bytes") != size or
                classification.get("path") != str(path)):
            raise IntegrityFailure("classification guard identity differs from primary observation")
        states.append({key: classification[name] for key, name in
                       (("ino", "inode"), ("size", "bytes"), ("mtime_ns", "mtime_ns"), ("ctime_ns", "ctime_ns"))})
    pose = actual.get("pose_verify")
    values = [pose[str(path)]] if pose is not None else []
    values.append(actual["pose_final_state"])
    for value in values:
        if not isinstance(value, list) or len(value) != 5:
            raise IntegrityFailure("invalid frozen pose state")
        states.append(dict(zip(("dev", "ino", "size", "mtime_ns", "ctime_ns"), value)))
    changed = set()
    for value in states:
        for key, number in value.items():
            if number != reference[key]:
                if key in OBJECT_FIELDS:
                    raise IntegrityFailure("frozen guard object changed after primary observation: "+key)
                changed.add(key)
    return sorted(changed)


def reader_round(inputs, frozen, deadline, event, pause, initial=None, observer=I):
    results = []
    for item in inputs:
        deadline.check()
        event("reader_begin", {"id": item["id"], "round": "after" if initial else "before"})
        first, second, _ = observe_pair(Path(item["path"]), item["sha256"], item["bytes"],
            frozen.hash_fn, deadline, event, pause, observer)
        previous = next((r for r in initial or [] if r["id"] == item["id"]), None)
        digest = item["sha256"] or first["sha256"]
        if previous and digest != previous["first"]["sha256"]:
            raise IntegrityFailure("real input bytes changed between rounds")
        if previous and any(second["stat_after"][k] != previous["second"]["stat_after"][k] for k in OBJECT_FIELDS):
            raise IntegrityFailure("real input object changed between rounds")
        deadline.check()
        actual = frozen.probe(Path(item["path"]), digest, item["bytes"], previous.get("frozen") if previous else None)
        for error in actual.get("errors", []):
            if error.get("kind") == "io" or error["type"] == "OSError" or "SHA mismatch" in error["message"] or "size mismatch" in error["message"]:
                raise IntegrityFailure("frozen reader integrity rejection: "+str(error))
        guard_drift = frozen_observation_bridge(actual, second["stat_after"],
                                               Path(item["path"]), digest, item["bytes"])
        all_states = [first["stat_before"], first["stat_after"], second["stat_before"], second["stat_after"]]
        affected = bool(actual.get("errors")) or bool(guard_drift) or any(any(s[k] != all_states[0][k] for k in STATE_FIELDS) for s in all_states[1:])
        if previous:
            affected |= any(second["stat_after"][k] != previous["second"]["stat_after"][k] for k in STATE_FIELDS)
        record = {"id": item["id"], "pin_kind": item["pin_kind"], "first": first, "second": second,
                  "frozen": actual, "guard_boundary_drift": guard_drift, "affected": affected, "adoption": False}
        results.append(record)
        event("reader_complete", record)
    return results


def run(manifest, out, expected_code_sha, *, settings=Settings(), clock=time.monotonic,
        sleep=time.sleep, frozen=None, observer=I, runtime=environment,
        metadata=post_metadata, use_alarm=True, sync=os.sync, fsync=os.fsync,
        test_evidence=None):
    """The injectable arguments are for bounded local tests; the CLI exposes none."""
    validate_manifest(manifest)
    out = Path(out)
    require(out.is_absolute() and not out.exists() and not out.is_symlink(), "new absolute diagnostic output required")
    require(os.path.normpath(str(out)) == str(out) and out.resolve() == out, "canonical diagnostic output required")
    require(not any(out == Path(x["path"]) or out in Path(x["path"]).parents for x in manifest["inputs"]),
            "output would contain a real input")
    deadline = Deadline(settings.deadline_seconds, clock, sleep)
    out.mkdir(parents=True, exist_ok=False)
    event = Journal(out, deadline)
    report = {"protocol": PROTOCOL, "code_sha": expected_code_sha, "complete": False, "outcome": "INCOMPLETE",
              "readers_status": "INCOMPLETE", "automatic_p1_release": False, "fixture_results": [],
              "reader_rounds": [], "cycles_completed": [], "errors": [], "started_utc": utc(),
              "source_sha256": {"scripts/scaling_p1_identity_diagnostic.py": file_sha(Path(__file__)),
                                "scripts/scaling_output_identity.py": file_sha(Path(I.__file__))},
              "frozen_source_sha256": manifest["frozen_sources"], "manifest": manifest,
              "limits": {"deadline_seconds": settings.deadline_seconds, "fixture_cap_bytes": settings.fixture_cap_bytes},
              "kernel_uninterruptible_io_limit": "SIGALRM cannot guarantee interruption of a kernel-uninterruptible operation; SLURM timeout means INCOMPLETE"}
    written = 0
    sync_count = 0
    try:
        with alarm(deadline, use_alarm):
            report["candidate_tests"] = test_evidence if test_evidence is not None else candidate_test_evidence(manifest)
            require(report["candidate_tests"].get("passed") is True, "candidate tests not passed")
            frozen = FrozenReaders(manifest) if frozen is None else frozen
            report["environment"] = runtime(out, deadline)
            event("environment", report["environment"])
            # Tiny byte-level oracle qualification before any large fixture.
            tiny = {"payload": np.arange(17, dtype=np.uint8), "fixture_id": np.asarray("oracle-qualification")}
            with io.BytesIO() as memory:
                np.savez(memory, **tiny)
                expected, _ = buffer_digest(memory, deadline)
            # Cross a ZIP timestamp tick; agreement in the same tick is insufficient.
            deadline.wait(2.0, event)
            tiny_path = out/"oracle_qualification.npz"
            np.savez(tiny_path, **tiny)
            if frozen.hash_fn(tiny_path) != expected:
                raise IntegrityFailure("BytesIO/path NPZ oracle qualification failed")
            report["oracle_qualification"] = {"passed": True, "sha256": expected}
            before = reader_round(manifest["inputs"], frozen, deadline, event, settings.pause_seconds, observer=observer)
            report["reader_rounds"].append({"phase": "before", "inputs": before})
            fixtures = out/"fixtures"
            fixtures.mkdir()
            aged = {}
            for age, mode in MATRIX[3:]:
                records = []
                for index in (0, 1):
                    name = f"{age}_{mode}_{NAMES[index]}"
                    rec = write_fixture(fixtures/(name+".npz"), name, settings.target_sizes[index],
                        (ord(mode)+index) % 256, deadline, event, written, settings.fixture_cap_bytes)
                    written += rec["bytes"]
                    records.append(rec)
                aged[mode] = records
            for age, mode in MATRIX:
                deadline.check()
                records = aged.get(mode, []).copy() if age == "aged" else []
                if age == "aged":
                    required = max(r["closed_monotonic"]+settings.age_seconds for r in records)
                    deadline.wait(max(0, required-clock()), event)
                for index in ((2,) if age == "aged" else (0, 1, 2)):
                    name = f"{age}_{mode}_{NAMES[index]}"
                    rec = write_fixture(fixtures/(name+".npz"), name, settings.target_sizes[index],
                        (ord(mode)+index) % 256, deadline, event, written, settings.fixture_cap_bytes)
                    written += rec["bytes"]
                    records.append(rec)
                barrier = apply_barrier(mode, records, deadline, event, sync, fsync)
                if mode == "S":
                    sync_count += 1
                for rec in records:
                    primary_started = clock()
                    first, second, verified = observe_pair(Path(rec["path"]), rec["sha256"], rec["bytes"],
                        frozen.hash_fn, deadline, event, settings.pause_seconds, observer,
                        validator=lambda: verify_synthetic_npz(rec, deadline))
                    changed = [k for k in STATE_FIELDS if first["stat_before"][k] != first["stat_after"][k]]
                    try:
                        candidate = observer.assess_pair(first, second, expected_sha256=rec["sha256"], expected_bytes=rec["bytes"])
                    except observer.IdentityError as e:
                        # A second timestamp drift is a correct refusal, not a successful correction.
                        if e.code != "SECOND_INSTABILITY":
                            raise IntegrityFailure("candidate integrity refusal: "+e.code) from e
                        candidate = {"accepted": False, "code": e.code, "audit": e.audit}
                    validation_drift = [k for k in STATE_FIELDS if first["stat_after"][k] != second["stat_before"][k]]
                    record = {**rec, "age": age, "mode": mode, "barrier": barrier,
                              "global_sync_calls_before_primary": sync_count,
                              "age_at_primary_seconds": primary_started-rec["closed_monotonic"],
                              "first": first, "second": second, "legacy_changed_fields": changed,
                              "candidate": candidate, "validation": verified,
                              "validation_state_after": second["stat_after"], "validation_drift": validation_drift,
                              "post_observation_metadata": metadata(Path(rec["path"]), deadline)}
                    report["fixture_results"].append(record)
                    event("fixture_complete", record)
                report["cycles_completed"].append({"age": age, "mode": mode})
            after = reader_round(manifest["inputs"], frozen, deadline, event, settings.pause_seconds, before, observer)
            report["reader_rounds"].append({"phase": "after", "inputs": after})
            report["readers_status"] = "READERS_AFFECTED" if any(r["affected"] for r in before+after) else "READERS_CLEAR_IN_TEST"
            report["complete"] = True
            reproduced = [r for r in report["fixture_results"] if r["legacy_changed_fields"] and
                          set(r["legacy_changed_fields"]) <= {"mtime_ns", "ctime_ns"}]
            benign_accepted = all(r["candidate"].get("accepted") is True
                                  for r in report["fixture_results"])
            report["metadata_only_cases"] = [r["id"] for r in reproduced]
            report["candidate_accepts_all_benign_cases"] = benign_accepted
            report["outcome"] = "METADATA_ONLY_REPRODUCED" if reproduced and benign_accepted else "NOT_REPRODUCED"
            if not benign_accepted:
                report["outcome"] = "INCOMPLETE"
                report["complete"] = False
                report["errors"].append({"type": "CandidatePolicyFailure", "message": "metadata drift reproduced but candidate or validation boundaries not clear"})
    except TimeoutError as e:
        report["outcome"] = "INCOMPLETE"
        report["errors"].append({"type": type(e).__name__, "message": str(e)})
        event("incomplete", report["errors"][-1])
    except (IntegrityFailure, OSError, zipfile.BadZipFile) as e:
        report["outcome"] = "INTEGRITY_FAIL"
        report["readers_status"] = "INTEGRITY_FAIL" if not report["reader_rounds"] else report["readers_status"]
        report["errors"].append({"type": type(e).__name__, "message": str(e)})
        event("integrity_failure", report["errors"][-1])
    except Exception as e:
        report["outcome"] = "INCOMPLETE"
        report["errors"].append({"type": type(e).__name__, "message": str(e)})
        event("incomplete", report["errors"][-1])
    finally:
        report["finished_utc"] = utc()
        report["elapsed_seconds"] = clock()-deadline.start
        report["fixture_bytes_written"] = written
        event("report", {k: report[k] for k in ("complete", "outcome", "readers_status")})
        event.close()
        report["events_sha256"] = file_sha(out/"events.jsonl")
        with (out/"report.json").open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    args = parser.parse_args(argv)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(actual == args.expected_code_sha and not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT),
            "diagnostic checkout is not the frozen clean commit")
    require(file_sha(args.manifest) == args.manifest_sha256, "diagnostic manifest changed")
    require(str(args.out).startswith("/leonardo_scratch/") and "/wearusfm_runs/diag/" in str(args.out),
            "production output must be a new diagnostic directory on SCRATCH")
    report = run(json.loads(args.manifest.read_text()), args.out, actual)
    print(json.dumps({k: report[k] for k in ("complete", "outcome", "readers_status")}))
    return 0 if report["complete"] and report["outcome"] in ("METADATA_ONLY_REPRODUCED", "NOT_REPRODUCED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
