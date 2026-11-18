#!/usr/bin/env python3
"""Download a frozen OSF SEEDS manifest into an external-only data root.

No discovery, preprocessing, extraction or pretraining registration occurs here.
Existing complete files are verified and reused; incomplete ``.part`` files restart
from byte zero. A corrupt completed file is an error, never silently overwritten.
Manifest schema: schema_version=1, dataset="SEEDS", source_node="wa3qk",
pretraining_allowed=false, files=[{path, size, sha256, md5, url, ...}].
At least SHA256 is required; MD5, when provided, must agree as well.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK = 1024 * 1024
POLICY_NAME = ".seeds_external_only.json"
POLICY = {"dataset": "SEEDS", "pretraining_allowed": False, "purpose": "external_downstream_benchmark"}


class IntegrityError(ValueError):
    """A payload differs from the frozen metadata; do not retry automatically."""


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _no_symlinks(path: Path) -> None:
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError(f"symlink forbidden: {parent}")


def _relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("invalid relative path")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts) or PurePosixPath(name).is_absolute():
        raise ValueError(f"unsafe relative path: {name!r}")
    if any(p == POLICY_NAME or p.endswith(".part") for p in parts):
        raise ValueError(f"reserved path: {name!r}")
    return PurePosixPath(name)


def validate_manifest(manifest: dict) -> list[dict]:
    if (manifest.get("schema_version") != 1 or manifest.get("dataset") != "SEEDS"
            or manifest.get("source_node") != "wa3qk" or manifest.get("pretraining_allowed") is not False):
        raise ValueError("manifest must identify SEEDS/wa3qk and forbid pretraining")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest contains no files")
    paths, folded = set(), set()
    for entry in files:
        path = str(_relative(entry["path"]))
        if path.casefold() in folded:
            raise ValueError(f"duplicate path: {path}")
        paths.add(path)
        folded.add(path.casefold())
        if type(entry.get("size")) is not int or entry["size"] < 0:
            raise ValueError(f"invalid size: {path}")
        for key, length, required in (("sha256", 64, True), ("md5", 32, False)):
            value = entry.get(key)
            if (required or value is not None) and not (isinstance(value, str) and re.fullmatch(f"[a-f0-9]{{{length}}}", value)):
                raise ValueError(f"invalid {key}: {path}")
        url = urllib.parse.urlsplit(entry["url"])
        if (url.scheme != "https" or url.netloc != "osf.io" or url.query or url.fragment
                or not re.fullmatch(r"/download/[a-zA-Z0-9]+/?", url.path)):
            raise ValueError(f"expected an official OSF download URL: {path}")
    for name in paths:
        if any(str(parent).casefold() in folded for parent in PurePosixPath(name).parents):
            raise ValueError(f"file/directory collision: {name}")
    return files


def _hashes(path: Path) -> dict:
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def _assert_hashes(actual: dict, entry: dict) -> None:
    for key in ("size", "sha256", "md5"):
        if entry.get(key) is not None and actual[key] != entry[key]:
            raise IntegrityError(f"{entry['path']}: {key} differs from frozen manifest")


def _target(root: Path, entry: dict) -> Path:
    target = root.joinpath(*_relative(entry["path"]).parts)
    _no_symlinks(target)
    _no_symlinks(target.with_name(target.name + ".part"))
    return target


def _initialize_root(destination: Path) -> Path:
    root = destination.absolute()
    _no_symlinks(root)
    if tuple(root.parts[-2:]) != ("external_benchmarks", "seeds"):
        raise ValueError("destination must end in external_benchmarks/seeds")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / POLICY_NAME
    _no_symlinks(marker)
    if marker.exists():
        if json.loads(marker.read_text()) != POLICY:
            raise ValueError("existing SEEDS exclusion policy differs")
    else:
        with marker.open("x") as handle:
            json.dump(POLICY, handle, indent=2)
            handle.write("\n")
    return root


def _retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or 500 <= error.code <= 599
    return isinstance(error, (urllib.error.URLError, TimeoutError, socket.timeout,
                              ConnectionError, http.client.IncompleteRead, http.client.RemoteDisconnected))


def _download(entry: dict, root: Path, timeout: float, retries: int) -> dict:
    started, attempts, network_bytes = time.monotonic(), [], 0
    result = {"path": entry["path"], "status": "failed", "attempts": attempts}
    try:
        target = _target(root, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        _no_symlinks(target)
        part = target.with_name(target.name + ".part")
        if target.exists():
            actual = _hashes(target)
            _assert_hashes(actual, entry)
            result.update(status="verified_existing", **actual)
            return result
        for attempt in range(retries + 1):
            bytes_read = 0
            try:
                _no_symlinks(part)
                request = urllib.request.Request(entry["url"], headers={"User-Agent": "WearUsFM-SEEDS/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as handle:
                    if response.getcode() != 200:
                        raise IntegrityError(f"unexpected HTTP status: {response.getcode()}")
                    if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                        raise IntegrityError("download redirected outside HTTPS")
                    for chunk in iter(lambda: response.read(CHUNK), b""):
                        bytes_read += len(chunk)
                        if bytes_read > entry["size"]:
                            raise IntegrityError("download exceeds expected size")
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                actual = _hashes(part)
                _assert_hashes(actual, entry)
                # link is an atomic no-clobber publication on the same filesystem.
                os.link(part, target)
                part.unlink()
                attempts.append({"attempt": attempt + 1, "bytes_read": bytes_read, "status": "verified"})
                network_bytes += bytes_read
                result.update(status="downloaded_verified", **actual)
                return result
            except Exception as error:
                network_bytes += bytes_read
                attempts.append({"attempt": attempt + 1, "bytes_read": bytes_read,
                                 "error": f"{type(error).__name__}: {error}"})
                if not _retryable(error) or attempt == retries:
                    raise
                time.sleep(min(2 ** attempt, 4))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["network_bytes"] = network_bytes
        result["elapsed_seconds"] = time.monotonic() - started
    return result


def run(manifest_path: Path, destination: Path, receipt_path: Path, *, max_bytes: int,
        subjects: set[int] | None = None, max_file_bytes: int | None = None,
        workers: int = 1, timeout: float = 120, retries: int = 2,
        free_margin_bytes: int = 1024 ** 3) -> dict:
    """Fetch a declared subset; stop scheduling files after the first failure."""
    if not 1 <= workers <= 4 or not 0 <= retries <= 2 or timeout <= 0 or max_bytes < 0 or free_margin_bytes < 0:
        raise ValueError("invalid download limits")
    if subjects is not None and (not subjects or any(type(s) is not int or not 1 <= s <= 25 for s in subjects)):
        raise ValueError("subjects must be integers 1..25")
    if max_file_bytes is not None and max_file_bytes < 0:
        raise ValueError("max_file_bytes must be nonnegative")
    raw = manifest_path.read_bytes()
    files = validate_manifest(json.loads(raw))
    selected = []
    for entry in files:
        match = re.match(r"^subj(\d{2})/", entry["path"])
        if subjects is not None and (match is None or int(match.group(1)) not in subjects):
            continue
        if max_file_bytes is None or entry["size"] <= max_file_bytes:
            selected.append(entry)
    if not selected:
        raise ValueError("no files match the explicit selection")
    selected.sort(key=lambda x: x["path"])
    total = sum(entry["size"] for entry in selected)
    if total > max_bytes:
        raise ValueError(f"selection is {total} bytes, above --max-bytes={max_bytes}")
    _no_symlinks(receipt_path.absolute())
    if receipt_path.exists():
        raise FileExistsError(f"receipt already exists: {receipt_path}")
    root = _initialize_root(destination)
    if receipt_path.absolute().is_relative_to(root):
        raise ValueError("receipt must be outside the SEEDS payload directory")
    # Validate all selected paths before opening any network connection.
    remaining = 0
    for entry in selected:
        target = _target(root, entry)
        if target.exists():
            if not target.is_file():
                raise ValueError(f"target is not a regular file: {target}")
            _assert_hashes(_hashes(target), entry)
        else:
            remaining += entry["size"]
    free = shutil.disk_usage(root).free
    if free < remaining + free_margin_bytes:
        raise ValueError(f"insufficient space: {free} free; need {remaining} + margin {free_margin_bytes}")
    report = {"schema_version": 1, "dataset": "SEEDS", "pretraining_allowed": False,
              "manifest_sha256": hashlib.sha256(raw).hexdigest(), "destination": str(root),
              "started_utc": _utc(), "status": "running", "manifest_files": len(files),
              "selected_files": len(selected), "selected_bytes": total,
              "whole_manifest_selected": len(selected) == len(files),
              "subjects_filter": sorted(subjects) if subjects is not None else None,
              "max_file_bytes": max_file_bytes, "max_bytes": max_bytes, "workers": workers,
              "timeout_seconds": timeout, "transport_retries": retries,
              "free_bytes_before": free, "free_margin_bytes": free_margin_bytes, "files": []}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temp = receipt_path.with_name(receipt_path.name + ".part")
        _no_symlinks(temp)
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(receipt_path)

    started = time.monotonic()
    save()
    iterator, failed = iter(selected), False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()

        def schedule() -> bool:
            entry = next(iterator, None)
            if entry is None:
                return False
            pending.add(pool.submit(_download, entry, root, timeout, retries))
            return True

        for _ in range(workers):
            schedule()
        while pending:
            completed, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                report["files"].append(result)
                failed |= result["status"] == "failed"
                print(f"{result['status']} {result['path']}", flush=True)
            save()
            if not failed:
                for _ in completed:
                    schedule()
    report["files"].sort(key=lambda x: x["path"])
    report.update(status="failed" if failed else "complete", completed_utc=_utc(),
                  elapsed_seconds=time.monotonic() - started,
                  verified_files=sum(x["status"] != "failed" for x in report["files"]),
                  network_bytes=sum(x["network_bytes"] for x in report["files"]))
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--subjects", help="comma-separated IDs; ancillary files excluded by this filter")
    parser.add_argument("--max-file-bytes", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--free-margin-bytes", type=int, default=1024 ** 3)
    args = parser.parse_args()
    try:
        subjects = {int(s) for s in args.subjects.split(",")} if args.subjects else None
        report = run(args.manifest, args.destination, args.receipt, max_bytes=args.max_bytes,
                     subjects=subjects, max_file_bytes=args.max_file_bytes, workers=args.workers,
                     timeout=args.timeout, retries=args.retries, free_margin_bytes=args.free_margin_bytes)
    except Exception as error:
        parser.exit(1, f"SEEDS download stopped: {type(error).__name__}: {error}\n")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
