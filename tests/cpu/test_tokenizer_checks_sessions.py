import json

import numpy as np
import pytest

from wearusfm.tokenizer_checks.sessions import (
    SessionData,
    allocate_quota,
    choose_sessions,
    discover_sessions,
    draw_from_session,
    load_session,
    qc_valid_from_metadata,
    to_canonical,
)


def _write_session(tmp_path, data, fs, qc, scale=None):
    d = tmp_path / "sess"
    d.mkdir(exist_ok=True)
    np.save(d / "data_int16.npy", data)
    channels = [{"qc_valid": bool(q)} for q in qc]
    meta = {"montage": {"groups": [{"channels": channels[:2]}, {"channels": channels[2:]}]}, "native_fs_hz": fs}
    if scale is not None:
        meta["int16_scale"] = scale
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


def test_qc_valid_from_metadata_order_and_length_check():
    meta = {"montage": {"groups": [{"channels": [{"qc_valid": True}, {"qc_valid": False}]}, {"channels": [{"qc_valid": True}]}]}}
    assert qc_valid_from_metadata(meta, 3).tolist() == [True, False, True]
    with pytest.raises(ValueError):
        qc_valid_from_metadata(meta, 4)


def test_load_session_3d_2d_and_scale(tmp_path):
    data3 = (np.arange(3 * 100 * 4) % 50).astype(np.int16).reshape(3, 100, 4)
    d = _write_session(tmp_path, data3, 1000, [1, 1, 0, 1], scale=2.0)
    s = load_session(d, "ds", "s1", "a")
    assert len(s.segments) == 3 and s.segments[0].shape == (100, 4)
    assert np.allclose(s.segments[0], data3[0] / 2.0)
    assert s.fs == 1000.0 and s.qc_valid.tolist() == [True, True, False, True]
    d2 = _write_session(tmp_path, data3[0], 2048, [1, 1, 1, 1])
    s2 = load_session(d2, "ds", "s1", "b")
    assert len(s2.segments) == 1 and s2.segments[0].shape == (100, 4) and s2.fs == 2048.0


def _canonical_session(rng, seconds=8, fs=2000, n_trials=2, qc=(1, 1, 1, 0), subject="s1", session="a"):
    segs = [rng.normal(scale=50.0, size=(int(seconds * fs), 4)) for _ in range(n_trials)]
    return SessionData("ds", subject, session, float(fs), segs, np.asarray(qc, dtype=bool))


def test_to_canonical_scales_session_and_shapes():
    rng = np.random.default_rng(0)
    cs = to_canonical(_canonical_session(rng))
    assert cs.stream.shape == (4, 2 * (8000 // 200), 200)  # 8 s a 1 kHz = 40 patch per prova
    valid = cs.stream[cs.qc_valid]
    assert valid.std() == pytest.approx(1.0, rel=0.02)  # scala di sessione sui canali QC-validi


def test_to_canonical_rejects_low_fs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="non e' eleggibile"):
        to_canonical(_canonical_session(rng, fs=200))


def test_draw_from_session_qc_channels_and_provenance():
    rng = np.random.default_rng(1)
    cs = to_canonical(_canonical_session(rng, subject="s0", session="x"))
    x, prov = draw_from_session(cs, 12, rng)
    assert x.shape == (12, 3200) and x.dtype == np.float32
    assert all(p[2] != 3 for p in prov)  # il canale 3 e' scartato dal QC
    subj, sess, ch, a0 = prov[0]
    assert np.array_equal(x[0], cs.stream[ch, a0 : a0 + 16].reshape(-1))


def test_draw_from_session_short_session_returns_empty():
    rng = np.random.default_rng(2)
    short = to_canonical(_canonical_session(rng, seconds=1.0, n_trials=1))  # 5 patch < 16
    x, prov = draw_from_session(short, 4, rng)
    assert x.shape[0] == 0 and prov == []


def test_discover_choose_and_allocate(tmp_path):
    for subj, sess in [("s1", "a"), ("s1", "b"), ("s1", "c"), ("s2", "a"), ("s3", "x/y")]:
        d = tmp_path / subj / sess
        d.mkdir(parents=True)
        (d / "metadata.json").write_text("{}")
    (tmp_path / "s4").mkdir()
    (tmp_path / "s4" / "metadata.json").write_text("{}")  # stile CapgMyo: sessione = cartella soggetto
    found = discover_sessions(tmp_path, "ds")
    assert {(s, ss) for s, ss, _ in found} == {("s1", "a"), ("s1", "b"), ("s1", "c"), ("s2", "a"), ("s3", "x/y"), ("s4", "s")}
    rng = np.random.default_rng(0)
    chosen = choose_sessions(found, 2, rng)
    assert sum(1 for s, _, _ in chosen if s == "s1") == 2 and len(chosen) == 5
    quota = allocate_quota(chosen, 40, rng)
    per_subject = {}
    for (subj, _), n in quota.items():
        per_subject[subj] = per_subject.get(subj, 0) + n
    assert sum(quota.values()) == 40 and set(per_subject.values()) == {10}  # 4 soggetti, parti uguali
