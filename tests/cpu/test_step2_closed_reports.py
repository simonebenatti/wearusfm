"""The secondary comparison must preserve paired counts and closed provenance."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import compare_step2_closed_reports as C


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    roots = {c: tmp_path / c for c in C.CAMPAIGNS}
    for campaign, root in roots.items():
        protocol, source, variant, features = C.CAMPAIGNS[campaign]
        score, hit = (.5, [1, 1]) if campaign == "step2a" else (.75, [2, 1])
        hashes = {}
        for task, datasets in C.DATASETS.items():
            for seed in (0, 1):
                arm = f"B_seed{seed}"
                counts = {"classes": [0, 1], "subjects": {s: {"n": [2, 2], "correct": hit} for s in ("u0", "u1", "u2")}}
                row = {"test_bacc": score, "paired_counts": counts}
                data = {ds: copy.deepcopy(row) if task == "p1" else {"test_bacc": score,
                        "splits": [dict(copy.deepcopy(row), split_seed=k) for k in (0, 1, 2)]} for ds in datasets}
                cache = {ds: {"input_sha256": "same-input", "rows": {"test": {"n": 12, "labels_subjects_sha256": "same-rows"}},
                              "sha256": campaign + "-different-features"} for ds in datasets}
                identity = {"protocol": protocol, "source_commit": source, "task": task, "arm": arm,
                            "variant": variant, "which": "teacher", "probe_seed": 0, "split_sha256": "same-split"}
                identity["source_checkpoint_sha256"] = C.CHECKPOINTS[arm]
                report = {task: score, "datasets": data, "feature_set": features,
                          "provenance": {"identity": identity, "cache": cache}}
                if campaign == "step2a":
                    report["provenance"]["downstream"] = {"identity": {
                        "arm": arm, "which": "teacher", "checkpoint_sha256": C.CHECKPOINTS[arm]}}
                hashes[f"{task}/{arm}/{variant}"] = _write(root / task / arm / f"{variant}.json", report)
        _write(root / "comparison.json", {"protocol": protocol, "source_commit": source,
                                          "thresholds": C.THRESHOLDS, "checkpoint_sha256": C.CHECKPOINTS,
                                          "report_sha256": hashes})
    return roots


def test_secondary_uses_signed_rule_and_accepts_distinct_feature_hashes(tmp_path):
    result = C.replay(_fixture(tmp_path), n_boot=20)
    assert result["decision"]["positive"]
    assert len(result["report_sha256"]) == 8
    for row in result["paired"]["tasks"].values():
        assert row["delta"] == pytest.approx(.25)
        assert row["ci95"] == pytest.approx([.25, .25])
        assert "practical_guard_pass" not in row


@pytest.mark.parametrize("field", ["hash", "protocol", "source", "threshold", "arm", "features", "input", "rows", "split", "cache", "teacher"])
def test_rejects_closed_report_or_pairing_drift(tmp_path, field):
    roots = _fixture(tmp_path)
    root = roots["step2c"]
    report_path, manifest_path = root / "p1/B_seed0/RPW.json", root / "comparison.json"
    report, manifest = json.loads(report_path.read_text()), json.loads(manifest_path.read_text())
    identity, cache = report["provenance"]["identity"], report["provenance"]["cache"]
    if field == "protocol":
        manifest["protocol"] = "another"
    elif field == "source":
        manifest["source_commit"] = "another"
    elif field == "threshold":
        manifest["thresholds"] = {**C.THRESHOLDS, "useful_delta": 0}
    elif field == "arm":
        identity["arm"] = "A_seed0"
    elif field == "features":
        report["feature_set"] = "mask"
    elif field == "input":
        cache["ninapro_db2"]["input_sha256"] = "different"
    elif field == "rows":
        cache["ninapro_db2"]["rows"]["test"]["labels_subjects_sha256"] = "different"
    elif field == "split":
        identity["split_sha256"] = "different"
    elif field == "cache":
        report["provenance"]["cache"] = {}
    elif field == "teacher":
        identity["source_checkpoint_sha256"] = C.CHECKPOINTS["B_seed1"]
    else:
        report["p1"] = 0
    digest = _write(report_path, report)
    if field != "hash":
        manifest["report_sha256"]["p1/B_seed0/RPW"] = digest
    _write(manifest_path, manifest)
    with pytest.raises(ValueError):
        C.replay(roots, n_boot=5)


@pytest.mark.parametrize("delta,lower,seeds,expected", [
    (.01, .001, [.001, .019], True), (.0099, .001, [.001, .019], False),
    (.02, 0, [.01, .03], False), (.02, .001, [0, .04], False),
])
def test_useful_requires_all_three_conditions(delta, lower, seeds, expected):
    row = {"delta": delta, "ci95": [lower, .1], "delta_by_model_seed": seeds}
    result = C.decision({"tasks": {"p1": row, "p2": {**row, "delta": 0}}})
    assert result["positive"] is expected


@pytest.mark.parametrize("delta,expected", [(-.01, True), (-.01001, False)])
def test_other_task_must_pass_signed_non_harm_guard(delta, expected):
    good = {"delta": .02, "ci95": [.01, .03], "delta_by_model_seed": [.01, .03]}
    result = C.decision({"tasks": {"p1": good, "p2": {**good, "delta": delta}}})
    assert result["positive"] is expected
