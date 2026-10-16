"""Regressioni della review C1-C4: provenienza, pairing e decisioni fail-closed."""

import json

import numpy as np
import pytest

from test_calibration_selection import CP, CS, DATASETS, _cand, _features, _splits


def _rewrite_npz(path, update):
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    update(arrays)
    np.savez(path, **arrays)


@pytest.mark.parametrize("value", [0.0, 0.1, 1000.0])
def test_constant_features_have_zero_effective_rank(value):
    d = CP.feature_diagnostics(np.full((30, 16), value, dtype=np.float32))
    assert d["effective_rank"] == 0.0
    assert d["constant_columns"] == 16


@pytest.mark.parametrize("field", ["splits_sha256", "extraction_recipe"])
def test_probe_refuses_stale_extraction_metadata(tmp_path, field):
    sp, splits = _splits(tmp_path)
    features = _features(tmp_path, splits)
    def change(z):
        z[field] = np.asarray("wrong" if field == "splits_sha256" else json.dumps({"encoder": "student"}))
    _rewrite_npz(features / f"{DATASETS[0]}.npz", change)
    with pytest.raises(ValueError, match="cache"):
        CP.main(["--stage", "probe", "--features", str(features), "--splits", str(sp), "--out", str(tmp_path / "out.json")])


def test_oof_tracks_original_rows(tmp_path):
    sp, splits = _splits(tmp_path)
    features = _features(tmp_path, splits)
    out = tmp_path / "out.json"
    CP.main(["--stage", "probe", "--features", str(features), "--splits", str(sp), "--out", str(out)])
    report = json.loads(out.read_text())
    CS.load_oof(report)
    with np.load(report["oof"]) as oof:
        for ds in DATASETS:
            with np.load(features / f"{ds}.npz") as cache:
                rows = oof[f"{ds}_row"]
                assert len(np.unique(rows)) == len(rows)
                np.testing.assert_array_equal(cache["y"][rows], oof[f"{ds}_y"])
                np.testing.assert_array_equal(cache["s"][rows], oof[f"{ds}_s"])


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_checkpoint_bytes_must_match_the_probe(tmp_path, mutation):
    _, rd, rp, steps = _cand(tmp_path, 3e-4, 0.8)
    checkpoint = rd / CS.CHECKPOINT
    if mutation == "missing":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"different checkpoint")
    adm = CS.admissibility(rd, json.loads(rp.read_text()), steps)
    assert not adm["admissible"]
    assert any("checkpoint" in p for p in adm["problems"])


@pytest.mark.parametrize("field", ["fold_seed", "splits_sha256", "probe_seed", "input_sha256"])
def test_pairing_checked_even_when_no_bootstrap_is_needed(tmp_path, field):
    candidates = [_cand(tmp_path, 1e-4, 0.8), _cand(tmp_path, 3e-4, 0.8), _cand(tmp_path, 1e-3, 0.8, stop=True)]
    # C1 would retain reference + sole admissible alternative, without comparing scores.
    rp = candidates[0][2]
    rep = json.loads(rp.read_text())
    if field == "input_sha256":
        rep["datasets"][DATASETS[0]][field] = "b" * 64
    else:
        rep[field] = "changed"
    rp.write_text(json.dumps(rep))
    with pytest.raises(ValueError, match="provenienza non accoppiata"):
        CS.select("C1", candidates)


@pytest.mark.parametrize("mutation", ["duplicates", "hash", "labels"])
def test_oof_integrity_checked_before_selection(tmp_path, mutation):
    candidates = [_cand(tmp_path, 1e-4, 0.8), _cand(tmp_path, 3e-4, 0.8), _cand(tmp_path, 1e-3, 0.8, stop=True)]
    rp = candidates[0][2]
    rep = json.loads(rp.read_text())
    def change(z):
        if mutation == "labels":
            z[f"{DATASETS[0]}_y"][0] = 2
        elif mutation == "duplicates":
            z[f"{DATASETS[0]}_row"][0] = z[f"{DATASETS[0]}_row"][1]
        else:
            z[f"{DATASETS[0]}_row"] += 1
    _rewrite_npz(rep["oof"], change)
    rep["oof_sha256"] = CS._sha256(rep["oof"])
    rp.write_text(json.dumps(rep))
    with pytest.raises(ValueError, match="OOF"):
        CS.select("C1", candidates)


def test_invalid_bootstrap_is_not_a_tie():
    a = {"d": (np.array([1, 2, 1, 2]), np.array([1, 2, 1, 2]), np.array(["s1", "s1", "s2", "s2"]))}
    with pytest.raises(ValueError, match="repliche valide"):
        CS.paired_difference(a, a, {"d": 1.0}, n_boot=1)


def test_merge_cannot_advance_past_a_stop_or_incomplete_confirmation():
    with pytest.raises(ValueError, match="precedente ha deciso stop"):
        CS.merge({"C1": {"stop": "failed"}}, {"C2": {"confirmed": []}})
    with pytest.raises(ValueError, match="conferma C2 incompleta"):
        CS.merge({"C1": {"finalists": [1e-4, 3e-4]}, "C2": {"confirmed": [3e-4]}}, {"C3": {"proposed_300m_lr": 3e-4}})
