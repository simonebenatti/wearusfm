"""Independent synthetic MATLAB fixtures for the external SEEDS integrity audit."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

SPEC = importlib.util.spec_from_file_location("seeds_audit", Path(__file__).parents[2] / "scripts/seeds_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def trial(subject=1, session=1, movement=1, repetition=1, n_emg=158, n_glove=20):
    # Known signal range intentionally extends both below zero and above one.
    angles = np.tile(np.linspace(-.25, 1.25, n_glove, dtype=np.float32), (18, 1))
    return {
        "subject": f"subj{subject:02d}", "session": session, "date": "20170622",
        "movement": f"movement-{movement}", "repetition": repetition, "speed": "slow",
        "fs_emg": 2048., "fs_glove": 256.,
        "emg": np.zeros((134, n_emg), dtype=np.float32), "glove": angles,
        "channels_emg": np.asarray([f"EMG{i:03}" for i in range(134)]),
        "channels_glove": np.asarray([f"sensor{i:02}" for i in range(18)]),
    }


def trial_name(subject=1, session=1, movement=1, repetition=1):
    return f"subj{subject:02d}/detop_exp01_subj{subject:02d}_Sess{session}_{movement:02d}_{repetition:02d}.mat"


def calibration():
    return {"calibration_min": np.arange(18, dtype=np.int32) - 10,
            "calibration_max": np.arange(18, dtype=np.int32) + 10}


def payload(data):
    out = io.BytesIO()
    savemat(out, data)
    return out.getvalue()


def entry(name, content):
    return {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "md5": hashlib.md5(content).hexdigest(), "url": "https://osf.io/download/abc12/"}


def setup(tmp_path, files):
    root = tmp_path / "external_benchmarks/seeds"
    root.mkdir(parents=True)
    (root / ".seeds_external_only.json").write_text(json.dumps({
        "dataset": "SEEDS", "pretraining_allowed": False, "purpose": "external_downstream_benchmark"}))
    entries = []
    for name, data in files:
        content = payload(data)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(entry(name, content))
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "dataset": "SEEDS", "source_node": "wa3qk",
                                    "pretraining_allowed": False, "files": entries}))
    return manifest, root


def test_known_target_range_preserved_and_noninteger_ratio_accepted(tmp_path):
    manifest, root = setup(tmp_path, [(trial_name(), trial()),
                                     ("subj01/subj01_calibration_values.mat", calibration())])
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.mat")}
    report = audit.run(manifest, root, sample=10)
    assert report["status"] == "partial" and not report["full_dataset_pass"]
    assert report["summary"]["verified_trial_files"] == 1
    assert report["summary"]["verified_calibration_files"] == 1
    row = next(x for x in report["files"] if x["kind"] == "trial")
    assert row["glove_range"] == [-.25, 1.25]
    assert row["glove_values_below_zero"] > 0 and row["glove_values_above_one"] > 0
    assert row["duration_difference_glove_samples"] == -.25
    assert not row["exact_sample_ratio_8"]
    assert not report["alignment"]["author_helpers_equivalent"]
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.mat")} == before
    json.dumps(report, allow_nan=False)


def test_partial_manifest_cannot_pass_full_audit(tmp_path):
    manifest, root = setup(tmp_path, [(trial_name(), trial())])
    result = audit.run(manifest, root)
    assert result["status"] == "failed" and not result["full_dataset_pass"]
    assert result["gates"]["all_selected_files_pass_structure_and_source_hashes"]
    assert not result["gates"]["complete_published_manifest"]
    assert len(result["manifest_inventory"]["missing_trial_keys"]) == 5849


@pytest.mark.parametrize("field,value,message", [
    ("subject", "subj02", "subject differs"),
    ("session", 2, "session differs"),
    ("fs_glove", 60., "sampling frequencies"),
    ("fs_emg", 2000., "sampling frequencies"),
    ("speed", "normal", "speed"),
    ("date", "20170230", "day is out of range"),
    ("glove", np.zeros((20, 20), dtype=np.float32), "shape"),
    ("emg", np.zeros((133, 158), dtype=np.float32), "shape"),
    ("glove", np.full((18, 20), np.nan, dtype=np.float32), "non-finite"),
    ("emg", np.full((134, 158), np.inf, dtype=np.float32), "non-finite"),
    ("emg", np.zeros((134, 158), dtype=np.complex64), "real floating"),
    ("channels_glove", np.array(["same"] * 18), "repeated"),
])
def test_noisy_or_inconsistent_mat_fails_even_when_download_hash_is_valid(tmp_path, field, value, message):
    data = trial()
    data[field] = value
    manifest, root = setup(tmp_path, [(trial_name(), data)])
    report = audit.run(manifest, root, sample=1)
    assert report["status"] == "failed"
    assert message in report["files"][0]["error"]


@pytest.mark.parametrize("n_emg,valid", [(153, True), (159, True), (160, True), (167, True), (152, False), (168, False)])
def test_duration_gate_is_strictly_less_than_one_glove_sample(tmp_path, n_emg, valid):
    manifest, root = setup(tmp_path, [(trial_name(), trial(n_emg=n_emg))])
    report = audit.run(manifest, root, sample=1)
    assert (report["status"] == "partial") is valid
    if not valid:
        assert "duration discrepancy" in report["files"][0]["error"]


@pytest.mark.parametrize("change", ["equal", "reverse", "nan", "wrong_shape"])
def test_bad_calibration_never_passes(tmp_path, change):
    data = calibration()
    if change == "equal":
        data["calibration_max"][3] = data["calibration_min"][3]
    elif change == "reverse":
        data["calibration_max"][3] = data["calibration_min"][3] - 1
    elif change == "nan":
        data["calibration_min"] = np.full(18, np.nan)
    else:
        data["calibration_min"] = np.zeros(17)
    manifest, root = setup(tmp_path, [("subj01/subj01_calibration_values.mat", data)])
    assert audit.run(manifest, root, sample=1)["status"] == "failed"


def test_channel_reordering_and_movement_relabeling_detected_across_trials(tmp_path):
    second, third = trial(repetition=2), trial(repetition=3)
    second["channels_emg"] = second["channels_emg"][::-1].copy()
    third["movement"] = "different-movement"
    manifest, root = setup(tmp_path, [(trial_name(), trial()), (trial_name(repetition=2), second),
                                     (trial_name(repetition=3), third)])
    report = audit.run(manifest, root, sample=3)
    assert report["summary"]["failed_files"] == 2
    assert "channels_emg" in report["files"][1]["error"]
    assert "movement ID" in report["files"][2]["error"]


def test_source_corruption_fails_before_parsing(tmp_path):
    name = trial_name()
    manifest, root = setup(tmp_path, [(name, trial())])
    path = root / name
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    report = audit.run(manifest, root, sample=1)
    assert report["status"] == "failed"
    assert "differs from frozen manifest" in report["files"][0]["error"]


def test_identical_trial_payloads_reported_only_when_requested(tmp_path):
    data = trial()
    manifest, root = setup(tmp_path, [(trial_name(), data), (trial_name(repetition=2), data)])
    # savemat's timestamps are identical within the same second; guarantee bytes.
    first, second = root / trial_name(), root / trial_name(repetition=2)
    second.write_bytes(first.read_bytes())
    j = json.loads(manifest.read_text())
    j["files"][1] = entry(trial_name(repetition=2), first.read_bytes())
    manifest.write_text(json.dumps(j))
    assert audit.run(manifest, root, sample=2)["duplicate_trial_hashes"] == []
    result = audit.run(manifest, root, sample=2, check_duplicates=True)
    assert result["status"] == "failed" and len(result["duplicate_trial_hashes"]) == 1


def test_directory_identity_mismatch_rejected(tmp_path):
    manifest, root = setup(tmp_path, [(trial_name().replace("subj01/", "subj02/"), trial())])
    report = audit.run(manifest, root, sample=1)
    assert report["status"] == "failed"
    assert "directory" in report["manifest_inventory"]["errors"][0]["error"]


def test_complete_synthetic_payload_can_pass_and_sample_never_can(tmp_path):
    def files():
        for s in range(1, 26):
            yield f"subj{s:02d}/subj{s:02d}_calibration_values.mat", calibration()
            for session in range(1, 4):
                for movement in range(1, 14):
                    for repetition in range(1, 7):
                        yield trial_name(s, session, movement, repetition), trial(s, session, movement, repetition, 14, 2)
    manifest, root = setup(tmp_path, files())
    full = audit.run(manifest, root, check_duplicates=True)
    assert full["status"] == "passed" and full["full_dataset_pass"]
    assert full["summary"]["verified_trial_files"] == 5850
    assert full["summary"]["verified_calibration_files"] == 25
    assert set(full["manifest_inventory"]["trial_counts_by_subject"].values()) == {234}
    subset = audit.run(manifest, root, sample=2)
    assert subset["status"] == "partial" and not subset["full_dataset_pass"]
    # A complete manifest with one missing payload must fail a full audit.
    (root / trial_name(subject=25, session=3, movement=13, repetition=6)).unlink()
    missing = audit.run(manifest, root)
    assert missing["gates"]["complete_published_manifest"]
    assert missing["status"] == "failed" and missing["summary"]["failed_files"] == 1


def test_external_root_policy_required(tmp_path):
    manifest, root = setup(tmp_path, [(trial_name(), trial())])
    (root / ".seeds_external_only.json").write_text('{"pretraining_allowed":true}')
    with pytest.raises(ValueError, match="external-only policy"):
        audit.run(manifest, root, sample=1)
