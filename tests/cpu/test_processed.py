import json

import numpy as np
import pytest

from wearusfm.data.processed import (
    discover_sessions,
    load_session,
    read_scale,
    session_summary,
    split_segments,
    validate_session,
)


def _write(d, data, *, qc=(1, 1, 1, 1), fs=1000.0, scale=None, trials=None, shape="auto", extra=None, labels=None):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data_int16.npy", data)
    ch = [{"qc_valid": bool(q)} for q in qc]
    meta = {
        "montage": {"groups": [{"channels": ch[:2]}, {"channels": ch[2:]}]},
        "native_fs_hz": fs,
        "shape": list(data.shape) if shape == "auto" else shape,
        "n_channels_discarded_by_qc": int(sum(1 for q in qc if not q)),
    }
    if scale is not None:
        meta["int16_scale"] = scale
    if trials is not None:
        meta["trials"] = trials
    meta.update(extra or {})
    (d / "metadata.json").write_text(json.dumps(meta))
    if labels is not None:
        np.savez(d / "labels.npz", **labels)
    return d


def _data(shape=(3, 100, 4)):
    return (np.arange(int(np.prod(shape))) % 40).astype(np.int16).reshape(shape)


def test_valid_layouts_pass(tmp_path):
    assert validate_session(_write(tmp_path / "a", _data())) == []  # (n_prove, T, C), int16 nativo senza scala
    assert validate_session(_write(tmp_path / "b", _data((100, 4)), scale=3.5)) == []  # continuo, scala scalare
    assert validate_session(_write(tmp_path / "c", _data((100, 4)), scale=[1.0, 2.0, 3.0, 4.0])) == []  # scala per canale
    tr = [{"offset": 0, "n_samples": 60}, {"offset": 60, "n_samples": 40}]
    lab = {"stimulus": np.zeros(100, dtype=np.int16)}
    assert validate_session(_write(tmp_path / "d", _data((100, 4)), scale=2.0, trials=tr, labels=lab)) == []


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"data": _data().astype(np.int32)}, "dtype"),
        ({"data": _data((5,))}, "dimensioni"),
        ({"shape": [3, 100, 5]}, "shape"),
        ({"qc": (1, 1, 1)}, "canali nel montaggio"),
        ({"scale": [1.0, 2.0]}, "int16_scale ha 2 valori"),
        ({"scale": [1.0, 2.0, -1.0, 1.0]}, "int16_scale"),
        ({"scale": 0.0}, "int16_scale"),
        ({"fs": -5.0}, "native_fs_hz"),
        ({"extra": {"n_channels_discarded_by_qc": 3}}, "n_channels_discarded_by_qc"),
        ({"trials": [{}, {}]}, "prove nel sidecar"),
    ],
)
def test_violations_are_reported(tmp_path, kwargs, needle):
    data = kwargs.pop("data", _data())
    problems = validate_session(_write(tmp_path / "x", data, **kwargs))
    assert any(needle in p for p in problems), problems


def test_trials_offsets_and_labels_are_checked_on_concatenated_layout(tmp_path):
    gap = [{"offset": 0, "n_samples": 50}, {"offset": 60, "n_samples": 40}]
    assert any("non contigue" in p for p in validate_session(_write(tmp_path / "g", _data((100, 4)), trials=gap)))
    short = [{"offset": 0, "n_samples": 50}, {"offset": 50, "n_samples": 40}]
    assert any("coprono 90" in p for p in validate_session(_write(tmp_path / "s", _data((100, 4)), trials=short)))
    lab = {"stimulus": np.zeros(99, dtype=np.int16)}
    assert any("labels.npz[stimulus]" in p for p in validate_session(_write(tmp_path / "l", _data((100, 4)), labels=lab)))


def test_missing_files_and_broken_files(tmp_path):
    d = tmp_path / "e"
    d.mkdir()
    assert sorted(validate_session(d)) == ["manca data_int16.npy", "manca metadata.json"]
    _write(tmp_path / "f", _data())
    (tmp_path / "f" / "metadata.json").write_text("{non json")
    assert "non e' JSON valido" in validate_session(tmp_path / "f")[0]
    _write(tmp_path / "h", _data())
    (tmp_path / "h" / "data_int16.npy").write_bytes(b"non un npy")
    assert "illeggibile" in validate_session(tmp_path / "h")[0]
    (tmp_path / "f" / "metadata.json").write_text("{}")
    assert any("manca la chiave `montage`" in p for p in validate_session(tmp_path / "f"))


def test_read_scale_and_split_segments(tmp_path):
    assert read_scale({}) is None and read_scale({"int16_scale": 2}) == 2
    assert read_scale({"int16_scale": [1, 2]}).tolist() == [1.0, 2.0]
    arr = _data((100, 4))
    segs = split_segments(arr, {"trials": [{"offset": 0, "n_samples": 30}, {"offset": 30, "n_samples": 70}]})
    assert [s.shape for s in segs] == [(30, 4), (70, 4)] and segs[1].dtype == np.float64
    assert len(split_segments(arr, {})) == 1 and len(split_segments(_data(), {})) == 3
    with pytest.raises(ValueError):
        split_segments(np.zeros(5), {})


def test_load_session_per_channel_scale_and_trial_split(tmp_path):
    data = _data((100, 4))
    scale = [1.0, 2.0, 4.0, 8.0]
    tr = [{"offset": 0, "n_samples": 30}, {"offset": 30, "n_samples": 70}]
    s = load_session(_write(tmp_path / "p", data, scale=scale, trials=tr), "ds", "s1", "a")
    assert [x.shape for x in s.segments] == [(30, 4), (70, 4)]
    assert np.allclose(s.segments[0], data[:30] / np.asarray(scale))


def test_discover_sessions_layouts(tmp_path):
    _write(tmp_path / "s01" / "session1", _data())
    _write(tmp_path / "s01" / "session2", _data())
    _write(tmp_path / "s02", _data())  # sessione nella cartella del soggetto (CapgMyo)
    _write(tmp_path / "s03" / "D1" / "T2", _data())
    found = {(subj, sess) for subj, sess, _ in discover_sessions(tmp_path, "ds")}
    assert found == {("s01", "session1"), ("s01", "session2"), ("s02", "s"), ("s03", "D1/T2")}


def test_session_summary(tmp_path):
    s = session_summary(_write(tmp_path / "a", _data((3, 1000, 4)), qc=(1, 0, 1, 1), fs=1000.0))
    assert s["n_samples"] == 3000 and s["n_channels"] == 4 and s["n_channels_valid"] == 3
    assert s["hours"] == pytest.approx(3.0 / 3600) and s["layout"] == "trials3d" and s["scale"] == "none"
    tr = [{"offset": 0, "n_samples": 100}]
    c = session_summary(_write(tmp_path / "b", _data((100, 4)), scale=[1, 1, 1, 1], trials=tr))
    assert c["layout"] == "concatenated_trials" and c["scale"] == "per_channel"
    assert session_summary(_write(tmp_path / "c", _data((100, 4)), scale=1.0))["layout"] == "continuous"
