"""Exact historical evidence cannot authorize arbitrary source or data drift."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compatibility_under_test", REPO / "scripts/contrastive_source_compat.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(rows):
    return sha(b"".join(row["path"].encode() + b"\0" + bytes.fromhex(row["sha256"]) for row in rows))


@pytest.fixture
def scene(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    paths = ["src/wearusfm/fm.py", "src/wearusfm/model/__init__.py", "src/wearusfm/model.py",
             "scripts/contrastive_audit.py", "scripts/contrastive_benchmark_readout.py",
             "scripts/contrastive_cuda_check.py", "scripts/contrastive_prepare.py", "scripts/contrastive_run.py",
             "scripts/contrastive_source_compat.py", "scripts/slurm/contrastive_preflight_cpu.sbatch",
             "scripts/train_jepa.py", "scripts/slurm/lib_code_dir.sh"]
    old, current = [], []
    for name in paths:
        historical = ("historical: " + name + "\n").encode()
        if name != C.HELPER_PATH:
            old.append({"path": name, "sha256": sha(historical)})
        content = ("repair: " + name + "\n").encode() if name in C.ALLOWED_CHANGED_SOURCES else historical
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        current.append({"path": name, "sha256": sha(content)})
    plan = root / C.PLAN_RELATIVE_PATH
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_bytes(b'{"scientific_recipe":"unchanged"}\n')
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    audit = evidence / "audit.json"
    audit.write_bytes(b'{"audit":"exact historical bytes"}\n')
    run = evidence / "calibration"
    run.mkdir()
    for name in C.RUN_FILES:
        (run / name).write_bytes((name + ": exact historical bytes\n").encode())
    receipt = {"status": "PASSED_FOR_REVIEW", "weights": {"nce": .123, "rnc": .234},
               "reader_provenance": {"scope": "original", "name": "verifica è"}}
    manifest = {"protocol": C.PROTOCOL, "historical_source_sha256": digest(old), "current_source_sha256": digest(current),
                "historical_sources": old, "current_sources": current,
                "frozen_documents": {C.PLAN_RELATIVE_PATH: {"sha256": sha(plan.read_bytes())}},
                "artifacts": {"audit": {"sha256": sha(audit.read_bytes())},
                              "calibration_run": {name: {"sha256": sha((run / name).read_bytes())} for name in C.RUN_FILES},
                              "calibration_receipt": {"canonical_sha256": sha(canonical(receipt))}}}
    path = root / C.MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        path.write_bytes(canonical(manifest))
    save()
    # Test-only roots and synthetic digest; production has no override parameter.
    monkeypatch.setattr(C, "ROOT", root)
    monkeypatch.setattr(C, "HISTORICAL_SOURCE_SHA256", digest(old))
    monkeypatch.setitem(sys.modules, "contrastive_prepare", SimpleNamespace(ROOT=root, source_digest=lambda: digest(manifest["current_sources"])))
    def repin_current(name, content):
        (root / name).write_bytes(content)
        next(row for row in manifest["current_sources"] if row["path"] == name)["sha256"] = sha(content)
        manifest["current_source_sha256"] = digest(manifest["current_sources"])
        save()
    return SimpleNamespace(root=root, manifest=manifest, manifest_path=path, audit=audit, run=run, receipt=receipt,
                           save=save, repin_current=repin_current)


def test_production_producer_digest_is_the_single_hardcoded_reviewed_identity():
    assert C.HISTORICAL_SOURCE_SHA256 == "ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b"


@pytest.mark.parametrize("kind", ["audit", "calibration_run", "calibration_receipt"])
def test_exact_evidence_returns_distinct_historical_and_current_identities(scene, kind):
    payload = {"audit": scene.audit, "calibration_run": scene.run, "calibration_receipt": scene.receipt}[kind]
    result = C.validate_historical_evidence(kind, payload)
    assert result["kind"] == kind and result["protocol"] == C.PROTOCOL
    assert result["historical_source_sha256"] == scene.manifest["historical_source_sha256"]
    assert result["current_source_sha256"] == scene.manifest["current_source_sha256"]
    assert result["historical_source_sha256"] != result["current_source_sha256"]
    assert result["manifest_sha256"] == sha(scene.manifest_path.read_bytes())


def test_copied_audit_bytes_can_relocate_without_changing_identity(scene, tmp_path):
    copy_path = tmp_path / "relocated.json"
    copy_path.write_bytes(scene.audit.read_bytes())
    assert C.validate_historical_evidence("audit", copy_path) == C.validate_historical_evidence("audit", scene.audit)


@pytest.mark.parametrize("kind", ["audit", "config.json", "summary.json", "metrics.jsonl", "receipt", "reader_provenance"])
def test_any_historical_evidence_mutation_is_rejected(scene, kind):
    if kind == "audit":
        scene.audit.write_bytes(scene.audit.read_bytes() + b" ")
        evidence_kind, payload = "audit", scene.audit
    elif kind in C.RUN_FILES:
        path = scene.run / kind
        path.write_bytes(path.read_bytes() + b" ")
        evidence_kind, payload = "calibration_run", scene.run
    else:
        changed = copy.deepcopy(scene.receipt)
        if kind == "receipt": changed["weights"]["nce"] += .001
        else: changed["reader_provenance"]["scope"] = "new reader"
        evidence_kind, payload = "calibration_receipt", changed
    with pytest.raises(ValueError, match="historical .* changed"):
        C.validate_historical_evidence(evidence_kind, payload)


@pytest.mark.parametrize("field,value", [("protocol", "another-protocol"), ("historical_source_sha256", "0" * 64),
                                         ("current_source_sha256", "0" * 64)])
def test_manifest_identity_cannot_be_retargeted(scene, field, value):
    scene.manifest[field] = value
    scene.save()
    with pytest.raises(ValueError): C.validate_historical_evidence("audit", scene.audit)


def test_old_source_bytes_are_authenticated_by_the_original_digest(scene):
    scene.manifest["historical_sources"][0]["sha256"] = "f" * 64
    scene.save()
    with pytest.raises(ValueError, match="historical source inventory"):
        C.validate_historical_evidence("audit", scene.audit)


@pytest.mark.parametrize("field", ["historical_sources", "current_sources"])
def test_source_order_is_part_of_the_authenticated_identity(scene, field):
    rows = scene.manifest[field]
    rows[0], rows[1] = rows[1], rows[0]
    scene.save()
    with pytest.raises(ValueError, match="source order"):
        C.validate_historical_evidence("audit", scene.audit)


def test_missing_current_inventory_entry_is_rejected(scene):
    scene.manifest["current_sources"].pop(0)
    scene.save()
    with pytest.raises(ValueError, match="current execution source inventory"):
        C.validate_historical_evidence("audit", scene.audit)


def test_unlisted_new_execution_source_is_rejected(scene):
    (scene.root / "src/wearusfm/extra.py").write_text("unexpected scientific source")
    with pytest.raises(ValueError, match="current execution source inventory"):
        C.validate_historical_evidence("audit", scene.audit)


def test_repinning_a_changed_scientific_source_does_not_expand_permission(scene):
    scene.repin_current("src/wearusfm/fm.py", b"scientific behavior changed\n")
    with pytest.raises(ValueError, match="five reviewed paths"):
        C.validate_historical_evidence("audit", scene.audit)


def test_exactly_all_five_reviewed_paths_must_differ(scene):
    name = "scripts/contrastive_run.py"
    scene.repin_current(name, ("historical: " + name + "\n").encode())
    with pytest.raises(ValueError, match="five reviewed paths"):
        C.validate_historical_evidence("audit", scene.audit)


def test_repinning_a_source_deletion_is_rejected(scene):
    name = "scripts/contrastive_audit.py"
    (scene.root / name).unlink()
    scene.manifest["current_sources"] = [row for row in scene.manifest["current_sources"] if row["path"] != name]
    scene.manifest["current_source_sha256"] = digest(scene.manifest["current_sources"])
    scene.save()
    with pytest.raises(ValueError, match="no source may be removed"):
        C.validate_historical_evidence("audit", scene.audit)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_scientific_plan_is_required_and_physically_pinned(scene, mutation):
    if mutation == "missing":
        scene.manifest["frozen_documents"] = {}
        scene.save()
    else:
        (scene.root / C.PLAN_RELATIVE_PATH).write_text('{"scientific_recipe":"changed"}')
    with pytest.raises(ValueError, match="frozen (scientific plan|document)"):
        C.validate_historical_evidence("audit", scene.audit)


def test_lazy_source_digest_cannot_alias_the_historical_identity(scene, monkeypatch):
    monkeypatch.setattr(sys.modules["contrastive_prepare"], "source_digest", lambda: scene.manifest["historical_source_sha256"])
    with pytest.raises(ValueError, match="actual source_digest"):
        C.validate_historical_evidence("audit", scene.audit)


def test_lazy_reader_must_come_from_the_same_checkout(scene, monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["contrastive_prepare"], "ROOT", tmp_path / "other_checkout")
    with pytest.raises(ValueError, match="different checkout"):
        C.validate_historical_evidence("audit", scene.audit)


@pytest.mark.parametrize("bad", [b'{"protocol":1,"protocol":2}', b'{"number":NaN}', b'{"number":1e999}'])
def test_manifest_json_rejects_duplicate_or_nonfinite_values(scene, bad):
    scene.manifest_path.write_bytes(bad)
    with pytest.raises(ValueError): C.validate_historical_evidence("audit", scene.audit)


def test_manifest_cannot_read_frozen_documents_outside_checkout(scene):
    scene.manifest["frozen_documents"]["../outside.json"] = {"sha256": "f" * 64}
    scene.save()
    with pytest.raises(ValueError, match="repository-relative"):
        C.validate_historical_evidence("audit", scene.audit)


def test_missing_pinned_calibration_file_fails_closed(scene):
    (scene.run / "metrics.jsonl").unlink()
    with pytest.raises(ValueError, match="unreadable file"):
        C.validate_historical_evidence("calibration_run", scene.run)


def test_unsupported_kind_and_payload_types_are_rejected(scene):
    with pytest.raises(ValueError, match="unsupported evidence kind"):
        C.validate_historical_evidence("arbitrary", scene.audit)
    with pytest.raises(ValueError, match="file path"):
        C.validate_historical_evidence("audit", {})
    with pytest.raises(ValueError, match="JSON dictionary"):
        C.validate_historical_evidence("calibration_receipt", [])
