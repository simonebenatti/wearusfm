import json

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.data.processed import load_session, validate_session
from wearusfm.ingest.ninapro_db8 import (
    EFFECTIVE_BAND_HZ,
    build_montage_metadata,
    ingest_subject,
    load_subject,
    power_fraction_above,
    scan_db8,
)


def _mat(subject_field, n, zero_cols=(), label_n=None, exercise=1, seed=0):
    rng = np.random.default_rng(seed)
    emg = rng.normal(scale=2e-5, size=(n, 16)).astype(np.float32)
    emg[:, list(zero_cols)] = 0.0
    lab = np.zeros((n if label_n is None else label_n, 1), np.int8)
    lab[len(lab) // 3 :, 0] = 4
    lab[2 * len(lab) // 3 :, 0] = 7
    return {"emg": emg, "acc": np.zeros((n, 48), np.float32), "glove": np.zeros((n, 18), np.float32), "stimulus": lab, "restimulus": lab,
            "repetition": lab, "rerepetition": lab, "subject": np.array([[subject_field]], float), "exercise": np.array([[exercise]], float)}


def _write(root, subject, subject_field=None, zero_cols=(), lengths=(900, 800, 300), skip=None, **kw):
    d = root / "DB8"
    d.mkdir(parents=True, exist_ok=True)
    for k, n in zip((1, 2, 3), lengths):
        if k != skip:
            sio.savemat(d / f"S{subject}_E1_A{k}.mat", _mat(subject if subject_field is None else subject_field, n, zero_cols, seed=k, **kw))
    return scan_db8(root)[subject]


def test_scan_groups_acquisitions_by_subject(tmp_path):
    _write(tmp_path, 1)
    _write(tmp_path, 11, subject_field=101)
    (tmp_path / "DB8" / "S1_E1_A1.txt").write_text("non e' un .mat")
    got = scan_db8(tmp_path)
    assert set(got) == {1, 11} and sorted(got[1]) == [1, 2, 3] and got[11][3].name == "S11_E1_A3.mat"


def test_load_subject_refuses_missing_acquisition_and_wrong_exercise(tmp_path):
    with pytest.raises(ValueError, match="acquisizioni"):
        load_subject(_write(tmp_path / "a", 1, skip=2), 1)
    with pytest.raises(ValueError, match="exercise"):
        load_subject(_write(tmp_path / "b", 1, exercise=2), 1)


def test_montage_rings_for_able_bodied_sparse_nominal_for_amputees():
    able = build_montage_metadata(3)
    assert [(g.group_id, g.topology.value, g.symmetry) for g in able.groups] == [("ring8_row1", "ring", "D_8"), ("ring8_row2", "ring", "D_8")]
    assert [c.sensor_coords.channel_index for g in able.groups for c in g.channels] == list(range(16))
    assert able.groups[1].channels[2].sensor_coords.ring_angle_deg == 90.0
    ch = able.groups[0].channels[0]
    assert ch.native_fs_hz == 2000.0 and tuple(ch.effective_band_hz) == (0.0, 555.5) and ch.chirality.value == "right"
    assert not ch.anatomical_identity.nominal
    amp = build_montage_metadata(12)
    assert all(g.topology.value == "sparse" and g.symmetry == "none" for g in amp.groups)
    assert all(c.sensor_coords.ring_angle_deg is None and c.anatomical_identity.nominal for g in amp.groups for c in g.channels)


def test_ingest_amputee_concatenates_acquisitions_drops_empty_columns_and_reads_back(tmp_path):
    paths = _write(tmp_path / "raw", 11, subject_field=101, zero_cols=(13, 14, 15))
    r = ingest_subject(paths, tmp_path / "out", 11)
    assert r["discarded_channels"] == [13, 14, 15] and r["samples_per_acquisition"] == [900, 800, 300] and r["n_samples"] == 2000
    assert r["subject_field_in_file"] == [101] and r["movements_per_acquisition"] == [2, 2, 2]
    out = tmp_path / "out" / "s11" / "session1"
    meta = json.loads((out / "metadata.json").read_text())
    assert [(t["acquisition"], t["offset"], t["n_samples"]) for t in meta["trials"]] == [(1, 0, 900), (2, 900, 800), (3, 1700, 300)]
    assert meta["native_fs_hz"] == 2000.0 and meta["acquisition_fs_hz"] == 1111.0 and meta["nominal_anatomy"] is True
    s = load_session(out, "ninapro_db8", "s11", "session1")
    assert [seg.shape for seg in s.segments] == [(900, 16), (800, 16), (300, 16)] and s.qc_valid.tolist() == [True] * 13 + [False] * 3
    assert not validate_session(out)
    raw = sio.loadmat(paths[2])["emg"].astype(np.float64)
    assert np.allclose(s.segments[1][:, :13], raw[:, :13], atol=np.abs(raw).max() / 30000)  # l'acquisizione 2 sta nel suo segmento
    lab = np.load(out / "labels.npz")
    assert lab["restimulus"].shape == (2000,) and set(np.unique(lab["restimulus"])) == {0, 4, 7}


def test_label_shorter_than_emg_is_padded_and_reported(tmp_path):
    paths = _write(tmp_path / "raw", 2, label_n=None)
    m = {k: v for k, v in sio.loadmat(paths[1]).items() if not k.startswith("__")}
    m["restimulus"] = m["restimulus"][:-3]
    sio.savemat(paths[1], m)
    r = ingest_subject(paths, tmp_path / "out", 2)
    assert r["label_length_adjustments"] == {"1": {"restimulus": -3}}


def test_power_fraction_above_band_low_for_lowpassed_and_none_for_zero_channel():
    from scipy import signal

    rng = np.random.default_rng(0)
    low = signal.sosfiltfilt(signal.butter(10, 500, fs=2000, output="sos"), rng.normal(size=(20000, 1)), axis=0)
    x = np.concatenate([low, rng.normal(size=(20000, 1)), np.zeros((20000, 1))], axis=1)
    f = power_fraction_above(x, 2000.0, EFFECTIVE_BAND_HZ[1])
    assert f[0] < 1e-3 and 0.35 < f[1] < 0.55 and f[2] is None
