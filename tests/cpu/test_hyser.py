import json

import numpy as np
import pytest

from wearusfm.ingest.hyser import (
    ARRAY_NAMES,
    channel_layout,
    ingest_group,
    read_physical,
    scan_hyser,
)
from wearusfm.ingest.grabmyo import parse_hea

NAMES = [f"{a}-{i}-{j}" for a in ARRAY_NAMES for i in range(8, 0, -1) for j in range(8, 0, -1)]


def _write_record(path, stem, n, *, gain_scale, seed):
    """Registrazione WFDB sintetica a 256 canali, con guadagno e baseline DIVERSI per file e per canale."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(scale=1.0, size=(n, 256)) * (0.01 + 3.0 * rng.random(256))  # ampiezze fisiche diverse (~300x)
    # come nei file reali: il guadagno di ogni canale porta il suo picco a ~ fondo scala (qui 20000, piu' la baseline)
    gains = 20000.0 / np.abs(truth).max(axis=0) * gain_scale
    base = rng.integers(-9000, 9000, size=256)
    codes = np.round(truth * gains + base).astype("<i2")
    lines = [f"{stem} 256 2048 {n}"]
    for k, nm in enumerate(NAMES):
        lines.append(f"{stem}.dat 16 {gains[k]:.4f}({base[k]})/V 0 0 0 0 0 {nm}")
    (path / f"{stem}.hea").write_text("\n".join(lines) + "\n")
    codes.tofile(path / f"{stem}.dat")
    return truth


def _make_session(root, sub="1dof_dataset", subject=1, session=1, stems=("1dof_raw_finger1_sample1", "1dof_raw_finger1_sample2"), n=120):
    d = root / sub / f"subject{subject:02d}_session{session}"
    d.mkdir(parents=True)
    truths = [_write_record(d, st, n, gain_scale=1.0 + 0.1 * k, seed=k) for k, st in enumerate(stems)]
    # dati che NON devono entrare: forza e preprocess
    (d / "1dof_force_finger1_sample1.hea").write_text("1dof_force_finger1_sample1 5 100 10\n")
    (d / "1dof_preprocess_finger1_sample1.hea").write_text("junk\n")
    return d, truths


def test_scan_takes_only_raw_and_groups_pr_by_kind(tmp_path):
    _make_session(tmp_path)
    pr = tmp_path / "pr_dataset" / "subject01_session1"
    pr.mkdir(parents=True)
    for st in ("dynamic_raw_sample1", "dynamic_raw_sample2", "dynamic_raw_sample10", "maintenance_raw_sample1"):
        _write_record(pr, st, 40, gain_scale=1.0, seed=3)
    (pr / "label_dynamic.txt").write_text("1,2,3")
    (pr / "label_maintenance.txt").write_text("1,1")
    groups = scan_hyser(tmp_path)
    keys = {(g.key, len(g.hea_paths), len(g.label_files)) for g in groups}
    assert keys == {("1dof", 2, 0), ("pr_dynamic", 3, 1), ("pr_maintenance", 1, 1)}
    dyn = next(g for g in groups if g.key == "pr_dynamic")
    assert [p.stem for p in dyn.hea_paths] == ["dynamic_raw_sample1", "dynamic_raw_sample2", "dynamic_raw_sample10"]  # ordine naturale


def test_channel_layout_checks_contiguity_and_completeness(tmp_path):
    d, _ = _make_session(tmp_path)
    h = parse_hea(d / "1dof_raw_finger1_sample1.hea")
    lay = channel_layout(h)
    assert lay[0] == ("ED", 8, 8) and lay[63] == ("ED", 1, 1) and lay[64][0] == "EP" and lay[-1] == ("FP", 1, 1)
    bad = h.__class__(h.record_name, h.n_channels, h.fs_hz, h.n_samples, h.channels[64:] + h.channels[:64])
    assert channel_layout(bad)[0][0] == "EP"  # ordine diverso ma ancora contiguo: accettato (l'ordine lo decide il file)
    broken = h.__class__(h.record_name, h.n_channels, h.fs_hz, h.n_samples, h.channels[:10] + h.channels[64:74] + h.channels[10:64] + h.channels[74:])
    with pytest.raises(ValueError, match="non contigue"):
        channel_layout(broken)


def test_read_physical_uses_each_files_own_gain_and_baseline(tmp_path):
    d, truths = _make_session(tmp_path, n=200)
    for k, st in enumerate(("1dof_raw_finger1_sample1", "1dof_raw_finger1_sample2")):
        x, _ = read_physical(d / f"{st}.hea")
        gains = np.array([c.gain for c in parse_hea(d / f"{st}.hea").channels])
        # errore di quantizzazione <= mezzo passo del PROPRIO file
        assert np.all(np.abs(x - truths[k]) <= 0.51 / gains[None, :] + 1e-5)


def test_ingest_group_per_channel_scale_preserves_quiet_channels_and_records_trials(tmp_path):
    _make_session(tmp_path / "raw", n=200)
    (g,) = scan_hyser(tmp_path / "raw")
    res = ingest_group(g, tmp_path / "out")
    d = tmp_path / "out" / "s01" / "session1_1dof"
    data = np.load(d / "data_int16.npy")
    meta = json.loads((d / "metadata.json").read_text())
    assert data.shape == (400, 256) and data.dtype == np.int16 and res["n_trials"] == 2
    assert [t["offset"] for t in meta["trials"]] == [0, 200] and meta["int16_scale_per_channel"] is True
    scale = np.asarray(meta["int16_scale"])
    assert scale.shape == (256,) and scale.max() / scale.min() > 20  # scale molto diverse: canali con ampiezze diverse
    # ogni canale usa (quasi) tutta la dinamica int16, anche quelli quieti
    assert (np.abs(data).max(axis=0) >= 31000).all()
    # ricostruzione fedele alla verita' fisica
    x1, _ = read_physical(tmp_path / "raw" / "1dof_dataset" / "subject01_session1" / "1dof_raw_finger1_sample1.hea")
    rec = data[:200].astype(np.float64) / scale
    assert np.abs(rec - x1).max() <= (0.51 / scale).max() + 1e-6
    # montaggio: 4 griglie 8x8 con riga/colonna, qc_valid in ordine di colonna
    groups = meta["montage"]["groups"]
    assert [x["group_id"] for x in groups] == list(ARRAY_NAMES) and all(len(x["channels"]) == 64 for x in groups)
    assert groups[0]["topology"] == "grid_2d" and groups[0]["channels"][0]["sensor_coords"]["grid_row"] == 7
    assert groups[0]["channels"][0]["anatomical_identity"]["region"] == "forearm_distal"
    assert groups[3]["channels"][0]["anatomical_identity"]["region"] == "forearm_proximal"  # FP
    assert sum(c["qc_valid"] for x in groups for c in x["channels"]) == 256


def test_pr_label_files_are_copied_verbatim(tmp_path):
    pr = tmp_path / "raw" / "pr_dataset" / "subject02_session2"
    pr.mkdir(parents=True)
    _write_record(pr, "dynamic_raw_sample1", 50, gain_scale=1.0, seed=1)
    (pr / "label_dynamic.txt").write_text("1,1,1,2,2,2")
    (g,) = scan_hyser(tmp_path / "raw")
    ingest_group(g, tmp_path / "out")
    out = tmp_path / "out" / "s02" / "session2_pr_dynamic"
    assert (out / "label_dynamic.txt").read_text() == "1,1,1,2,2,2"
    assert json.loads((out / "metadata.json").read_text())["label_files"] == ["label_dynamic.txt"]
