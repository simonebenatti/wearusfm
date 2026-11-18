"""Authenticate one frozen historical evidence set after a reviewed source repair.

This is not a source-digest alias. The original and current source inventories,
all changed paths, the unchanged scientific plan and the exact evidence are
checked independently. No caller-supplied manifest or runtime override exists.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "contrastive-source-compatibility-20261118-v1"
HISTORICAL_SOURCE_SHA256 = "ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b"
MANIFEST_RELATIVE_PATH = "results/infra/contrastive_comparison_20261116/compatibility_20261118/manifest.json"
PLAN_RELATIVE_PATH = "results/infra/contrastive_comparison_20261116/plan.json"
HELPER_PATH = "scripts/contrastive_source_compat.py"
ALLOWED_CHANGED_SOURCES = frozenset({
    "scripts/contrastive_cuda_check.py", "scripts/contrastive_run.py",
    "scripts/contrastive_benchmark_readout.py", "scripts/contrastive_prepare.py", HELPER_PATH,
})
RUN_FILES = ("config.json", "summary.json", "metrics.jsonl")


def _require(condition, message):
    if not condition:
        raise ValueError("historical evidence compatibility: " + message)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _bytes(path):
    try:
        return Path(path).read_bytes()
    except (OSError, TypeError) as exc:
        raise ValueError("historical evidence compatibility: unreadable file " + str(path)) from exc


def _canonical_json(value):
    def check(item):
        if isinstance(item, dict):
            _require(all(isinstance(k, str) for k in item), "JSON keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, float):
            _require(math.isfinite(item), "nonfinite JSON number")
        else:
            _require(item is None or type(item) in (bool, int, str), "unsupported JSON value")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("historical evidence compatibility: nonfinite JSON constant " + value)
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=bad_constant)
    _canonical_json(result)  # Also reject numeric overflow such as 1e999.
    return result


def _relative_path(name):
    _require(isinstance(name, str) and bool(name), "missing relative source/document path")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and path.as_posix() == name and ".." not in path.parts and "\\" not in name,
             "source/document path must be canonical and repository-relative")
    return path


def _repository_file(name):
    _relative_path(name)
    path = ROOT / name
    _require(path.resolve().is_relative_to(ROOT.resolve()), "source/document escapes the frozen checkout")
    return path


def _source_order(names):
    groups = [[], [], []]
    trailing = ("scripts/train_jepa.py", "scripts/slurm/lib_code_dir.sh")
    for name in names:
        path = _relative_path(name)
        parts = path.parts
        if name.startswith("src/wearusfm/") and path.suffix == ".py":
            groups[0].append(name)
        elif len(parts) == 2 and parts[0] == "scripts" and parts[1].startswith("contrastive_") and path.suffix == ".py":
            groups[1].append(name)
        elif len(parts) == 3 and parts[:2] == ("scripts", "slurm") and parts[2].startswith("contrastive") and path.suffix == ".sbatch":
            groups[2].append(name)
        else:
            _require(name in trailing, "unexpected execution source path " + name)
    _require(all(name in names for name in trailing), "missing fixed execution helper source")
    # pathlib orders path components; sorting full path strings can differ.
    return [name for group in groups for name in sorted(group, key=lambda n: PurePosixPath(n).parts)] + list(trailing)


def _source_inventory():
    paths = sorted((ROOT / "src/wearusfm").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("contrastive_*.py"))
    paths += sorted((ROOT / "scripts/slurm").glob("contrastive*.sbatch"))
    paths += [ROOT / "scripts/train_jepa.py", ROOT / "scripts/slurm/lib_code_dir.sh"]
    return [{"path": str(path.relative_to(ROOT)), "sha256": _sha256(_bytes(_repository_file(str(path.relative_to(ROOT)))))}
            for path in paths]


def _source_pins(value, label):
    _require(isinstance(value, list) and bool(value), label + " must be an ordered source inventory")
    for row in value:
        _require(isinstance(row, dict) and set(row) == {"path", "sha256"} and _hex_digest(row.get("sha256")),
                 "invalid " + label + " source pin")
        _relative_path(row["path"])
    names = [row["path"] for row in value]
    _require(len(names) == len(set(names)), "duplicate " + label + " source path")
    _require(names == _source_order(names), label + " source order differs from source_digest")
    return {row["path"]: row["sha256"] for row in value}


def _inventory_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode() + b"\0" + bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def _pin(value, label, key="sha256"):
    _require(isinstance(value, dict) and set(value) == {key} and _hex_digest(value.get(key)), "invalid " + label + " artifact pin")
    return value[key]


def validate_historical_evidence(kind, payload_or_folder):
    """Return provenance only for this exact reviewed source transition/evidence."""
    _require(kind in ("audit", "calibration_run", "calibration_receipt"), "unsupported evidence kind")
    manifest_path = _repository_file(MANIFEST_RELATIVE_PATH)
    raw_manifest = _bytes(manifest_path)
    manifest = _read_json(raw_manifest)
    required = {"protocol", "historical_source_sha256", "current_source_sha256", "historical_sources", "current_sources",
                "artifacts", "frozen_documents"}
    _require(isinstance(manifest, dict) and set(manifest) == required, "incomplete or unexpected manifest fields")
    _require(manifest["protocol"] == PROTOCOL, "wrong manifest protocol")
    _require(manifest["historical_source_sha256"] == HISTORICAL_SOURCE_SHA256, "unapproved historical source identity")
    _require(_hex_digest(manifest["current_source_sha256"]), "invalid current source identity")
    old = _source_pins(manifest["historical_sources"], "historical")
    current = _source_pins(manifest["current_sources"], "current")
    _require(_inventory_digest(manifest["historical_sources"]) == HISTORICAL_SOURCE_SHA256,
             "historical source inventory does not authenticate the frozen producer")
    actual = _source_inventory()
    _require(actual == manifest["current_sources"], "current execution source inventory/pins differ")
    actual_digest = _inventory_digest(actual)
    _require(actual_digest == manifest["current_source_sha256"], "current source digest differs from manifest")
    _require(set(current) - set(old) == {HELPER_PATH} and not set(old) - set(current),
             "only the compatibility helper may be added; no source may be removed")
    changed = {name for name in set(old) | set(current) if old.get(name) != current.get(name)}
    _require(changed == ALLOWED_CHANGED_SOURCES, "changed execution sources differ from the five reviewed paths")

    documents = manifest["frozen_documents"]
    _require(isinstance(documents, dict) and PLAN_RELATIVE_PATH in documents, "missing frozen scientific plan")
    for name, pin in documents.items():
        _require(_sha256(_bytes(_repository_file(name))) == _pin(pin, name), "frozen document changed: " + name)

    artifacts = manifest["artifacts"]
    _require(isinstance(artifacts, dict) and set(artifacts) == {"audit", "calibration_run", "calibration_receipt"},
             "incomplete exact historical artifact set")
    audit_pin = _pin(artifacts["audit"], "audit")
    receipt_pin = _pin(artifacts["calibration_receipt"], "calibration_receipt", "canonical_sha256")
    run_pins = artifacts["calibration_run"]
    _require(isinstance(run_pins, dict) and set(run_pins) == set(RUN_FILES), "calibration run must pin exactly its three evidence files")
    run_pins = {name: _pin(run_pins[name], name) for name in RUN_FILES}
    if kind == "calibration_receipt":
        _require(isinstance(payload_or_folder, dict), "calibration receipt must be a JSON dictionary")
        artifact_sha = _sha256(_canonical_json(payload_or_folder))
        _require(artifact_sha == receipt_pin, "historical calibration receipt changed")
    elif kind == "audit":
        _require(isinstance(payload_or_folder, (str, Path)), "audit evidence must be a file path")
        artifact_sha = _sha256(_bytes(payload_or_folder))
        _require(artifact_sha == audit_pin, "historical audit bytes changed")
    else:
        _require(isinstance(payload_or_folder, (str, Path)), "calibration evidence must be a folder path")
        artifact_sha = {name: _sha256(_bytes(Path(payload_or_folder) / name)) for name in RUN_FILES}
        _require(artifact_sha == run_pins, "historical calibration run bytes changed")

    # Lazy import avoids a cycle with the guarded preparation/receipt reader.
    import contrastive_prepare as preparation
    _require(Path(preparation.ROOT).resolve() == ROOT.resolve(), "reader imported from a different checkout")
    _require(preparation.source_digest() == actual_digest, "actual source_digest differs from authenticated source inventory")
    _require(_bytes(manifest_path) == raw_manifest, "compatibility manifest changed during validation")
    return {"protocol": PROTOCOL, "kind": kind, "manifest_sha256": _sha256(raw_manifest),
            "historical_source_sha256": HISTORICAL_SOURCE_SHA256, "current_source_sha256": actual_digest,
            "artifact_sha256": artifact_sha}
