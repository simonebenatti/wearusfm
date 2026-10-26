"""Target di posa per finestra e per patch (R3, `docs/pilot_posa_v1.md` §2): griglia del front-end e centri dei blocchi, blocchi non validi
mascherati, standardizzazione per lato o insieme, sessione senza posa = errore, `PretrainBatch.pose` dal loader vero su un albero sintetico,
micro-batch, e default del loader bit per bit invariato senza `pose_root`."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from test_pretraining_loader import _cfg, _tree  # noqa: F401 -- albero sintetico con una sessione di emg2pose (u1/sessA, 24.000 campioni)
from wearusfm.data import pose_targets as PT
from wearusfm.data import pretraining_loader as L

FS = 2000.0


class _Win:
    def __init__(self, start, n_patches, trial=None):
        self.start, self.n_patches, self.trial = start, n_patches, trial


def _stats(root: Path, mean=None, std=None, per_side=False, by_side=None):
    mean = [0.0] * 20 if mean is None else mean
    std = [1.0] * 20 if std is None else std
    st = {"all": {"n": 100, "mean": list(mean), "std": list(std)}, "side_criterion": {"per_side": per_side},
          "by_side": by_side or {}, "joints": [f"j{i}" for i in range(20)], "split": "train"}
    root.mkdir(parents=True, exist_ok=True)
    (root / PT.STATS_FILE).write_text(json.dumps(st))
    return st


def _pose_tree(root: Path, n_samples: int, slope=0.001, invalid_blocks=(), per_joint_offset=True):
    """Posa lineare nel tempo: angolo_j(campione i) = slope * i + j (cosi' la media per patch e' il centro della patch, in campioni)."""
    nb = n_samples // PT.BLOCK
    j = np.arange(nb)[:, None]
    centers = j * PT.BLOCK + (PT.BLOCK - 1) / 2.0  # media dei 10 campioni del blocco
    blocks = (slope * centers + (np.arange(20)[None, :] if per_joint_offset else 0)).astype(np.float16)
    valid = np.ones(nb, dtype=bool)
    valid[list(invalid_blocks)] = False
    d = root / "u1" / "sessA"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / PT.POSE_FILE, blocks)
    np.save(d / PT.VALID_FILE, valid)
    return blocks, valid


def test_window_pose_averages_the_five_block_centres_of_each_patch_on_the_frontend_grid(tmp_path):
    blocks, valid = _pose_tree(tmp_path, 24000, invalid_blocks=[12, 13])
    mean, std = np.zeros(20, dtype=np.float32), np.ones(20, dtype=np.float32)
    # finestra che comincia al campione 37 (non multiplo di 10): patch da 50 campioni, 5 centri per patch
    tgt, mask = PT.window_pose(blocks, valid, _Win(37, 40), FS, 25.0, mean, std)
    assert tgt.shape == (40, 20) and mask.shape == (40, 20) and mask.dtype == bool
    bounds = PT.patch_bounds(40, 25.0, FS) + 37
    assert bounds[1] - bounds[0] == 50 and bounds[-1] == 37 + 2000
    for p in (0, 7, 39):
        lo, hi = bounds[p], bounds[p + 1]
        js = [j for j in range(blocks.shape[0]) if lo <= j * 10 + 5 < hi and valid[j]]
        assert len(js) == 5
        expected = np.asarray(blocks[js], dtype=np.float64).mean(axis=0)
        assert np.allclose(tgt[p], expected, atol=2e-2) and mask[p].all()  # float16 sui blocchi
        # il centro della patch (in campioni) e' la media dei 5 centri a meno dello sfasamento, al piu' mezzo blocco (2,5 ms = 5 campioni)
        assert abs(tgt[p, 0] / 0.001 - (lo + hi - 1) / 2.0) <= 5.0 + 0.5
    # i blocchi 12 e 13 (campioni 120-139) cadono nella patch 1 (87-137) e 2 (137-187): le due patch hanno meno di 5 blocchi ma restano valide
    p1 = [j for j in range(blocks.shape[0]) if bounds[1] <= j * 10 + 5 < bounds[2] and valid[j]]
    assert len(p1) == 3 and mask[1].all() and np.allclose(tgt[1], np.asarray(blocks[p1], dtype=np.float64).mean(axis=0), atol=2e-2)


def test_window_pose_masks_patches_without_valid_blocks_standardizes_and_rejects_trials(tmp_path):
    blocks, valid = _pose_tree(tmp_path, 4000, invalid_blocks=range(10, 20))  # campioni 100-199 non validi
    mean, std = np.full(20, 2.0, dtype=np.float32), np.full(20, 4.0, dtype=np.float32)
    tgt, mask = PT.window_pose(blocks, valid, _Win(100, 8), FS, 25.0, mean, std)  # patch 0 e 1 = campioni 100-199: nessun blocco valido
    assert not mask[0].any() and not mask[1].any() and mask[2:].all() and np.all(tgt[:2] == 0)
    raw = np.asarray(blocks[20:25], dtype=np.float64).mean(axis=0)  # patch 2 = campioni 200-249: blocchi 20-24
    assert np.allclose(tgt[2], (raw - 2.0) / 4.0, atol=1e-2)
    # finestra oltre la fine della posa: le patch senza blocchi sono mascherate, nessun errore
    tgt2, mask2 = PT.window_pose(blocks, valid, _Win(3900, 8), FS, 25.0, mean, std)
    assert mask2[:2].all() and not mask2[2:].any()
    with pytest.raises(ValueError, match="prove"):
        PT.window_pose(blocks, valid, _Win(0, 4, trial=0), FS, 25.0, mean, std)


def test_store_reads_stats_picks_the_side_and_fails_loudly_without_pose_files(tmp_path):
    _pose_tree(tmp_path / "pose", 24000)
    _stats(tmp_path / "pose", mean=[1.0] * 20, std=[2.0] * 20)
    store = PT.PoseStore(tmp_path / "pose")
    assert store.per_side is False and len(store.stats_sha256) == 64 and store.applies({"dataset": "emg2pose"}) and \
        not store.applies({"dataset": "ninapro_db2"})

    class V:  # vista minima di una sessione
        def __init__(self, path, side, n):
            self.path, self.meta, self.fs = Path(path), {"csv_row": {"side": side}}, FS
            self.arr = np.zeros((n, 16), dtype=np.int16)
    v = V(tmp_path / "processed" / "emg2pose" / "u1" / "sessA", "left", 24000)
    out = store.window({"dataset": "emg2pose"}, v, _Win(0, 4), 25.0)
    assert out is not None and out[0].shape == (4, 20) and out[1].all() and np.allclose(out[2], 2.0)  # (target, maschera, deviazioni usate)
    assert store.window({"dataset": "ninapro_db2"}, v, _Win(0, 4), 25.0) is None
    with pytest.raises(FileNotFoundError, match="posa mancante"):
        store.window({"dataset": "emg2pose"}, V(tmp_path / "processed" / "emg2pose" / "u9" / "nope", "left", 24000), _Win(0, 4), 25.0)
    with pytest.raises(ValueError, match="blocchi di posa"):
        store.window({"dataset": "emg2pose"}, V(tmp_path / "processed" / "emg2pose" / "u1" / "sessA", "left", 30000), _Win(0, 4), 25.0)
    # per lato: le statistiche del lato della sessione; un lato senza statistiche e' un errore
    _stats(tmp_path / "pose", per_side=True, by_side={"left": {"mean": [0.0] * 20, "std": [1.0] * 20}, "right": {"mean": [5.0] * 20, "std": [1.0] * 20}})
    ps = PT.PoseStore(tmp_path / "pose")
    l_t, _, l_sd = ps.window({"dataset": "emg2pose"}, V(v.path, "left", 24000), _Win(0, 2), 25.0)
    r_t, _, r_sd = ps.window({"dataset": "emg2pose"}, V(v.path, "right", 24000), _Win(0, 2), 25.0)
    assert np.allclose(l_t - r_t, 5.0, atol=1e-3) and np.allclose(l_sd, 1.0) and np.allclose(r_sd, 1.0)
    # mano non sollecitata esclusa (--pose-unprompted off): target a zero e maschera falsa, deviazioni comunque riportate
    class VU(V):
        def __init__(self, path, side, n, moving):
            super().__init__(path, side, n)
            self.meta["csv_row"]["moving_hand"] = moving
    off = PT.PoseStore(tmp_path / "pose", include_unprompted=False)
    t_u, m_u, _ = off.window({"dataset": "emg2pose"}, VU(v.path, "left", 24000, "right"), _Win(0, 3), 25.0)
    assert not m_u.any() and np.all(t_u == 0) and PT.PoseStore.unprompted(VU(v.path, "left", 24000, "right"))
    t_p, m_p, _ = off.window({"dataset": "emg2pose"}, VU(v.path, "left", 24000, "both"), _Win(0, 3), 25.0)
    assert m_p.all() and not PT.PoseStore.unprompted(VU(v.path, "left", 24000, "left"))
    with pytest.raises(ValueError, match="2000"):
        PT.window_pose(np.zeros((10, 20), np.float16), np.ones(10, bool), _Win(0, 1), 1000.0, 25.0, np.zeros(20, np.float32), np.ones(20, np.float32))
    with pytest.raises(ValueError, match="lato"):
        ps.window({"dataset": "emg2pose"}, V(v.path, "", 24000), _Win(0, 2), 25.0)
    (tmp_path / "pose" / PT.STATS_FILE).write_text(json.dumps({"all": {"mean": None}}))
    with pytest.raises(ValueError, match="vuote"):
        PT.PoseStore(tmp_path / "pose")


def test_loader_attaches_pose_only_to_emg2pose_windows_and_the_default_stays_bit_identical(tmp_path):
    root, mpath = _tree(tmp_path)
    _pose_tree(tmp_path / "pose", 24000, invalid_blocks=range(900, 910))
    _stats(tmp_path / "pose")
    idx = L.ManifestIndex.load(mpath, [root])
    plain = L.PretrainLoader(idx, _cfg())
    with_pose = L.PretrainLoader(idx, _cfg(pose_root=str(tmp_path / "pose")))
    assert plain.pose_store is None and with_pose.pose_store is not None
    b0 = plain.batch(6, np.random.default_rng(3))
    b1 = with_pose.batch(6, np.random.default_rng(3))
    # il default non cambia: stesse finestre, segnali, maschere e target delle ancore, e `pose` None
    assert b0.pose is None and b1.pose is not None and len(b1.pose) == 6
    assert [(w.start, w.stop) for w in b0.windows] == [(w.start, w.stop) for w in b1.windows]
    assert all(np.array_equal(x, y) for x, y in zip(b0.signals, b1.signals)) and np.array_equal(b0.visible, b1.visible)
    assert all(np.array_equal(a.log_rms, b.log_rms) for a, b in zip(b0.anchor_targets, b1.anchor_targets))
    n_pose = 0
    for row, win, pose, n_p in zip(b1.rows, b1.windows, b1.pose, b1.n_patches):
        if row["dataset"] == "emg2pose":
            n_pose += 1
            tgt, mask, sd = pose
            assert tgt.shape == (n_p, 20) and mask.shape == (n_p, 20) and tgt.dtype == np.float32 and sd.shape == (20,)
            # posa lineare nel tempo: il target della patch p cresce col centro della patch (a meno della finestra)
            valid_p = np.flatnonzero(mask[:, 0])
            if len(valid_p) > 2:
                assert np.all(np.diff(tgt[valid_p, 0]) > 0)
        else:
            assert pose is None
    assert n_pose >= 1
    # micro-batch: la posa segue i campioni
    parts = L.split_batch(b1, 2)
    assert [len(p.pose) for p in parts] == [3, 3] and parts[0].pose == b1.pose[:3] and parts[1].pose == b1.pose[3:]
    assert all(p.pose is None for p in L.split_batch(b0, 2))


def test_loader_stops_on_an_emg2pose_session_without_pose(tmp_path):
    root, mpath = _tree(tmp_path)
    _stats(tmp_path / "pose")  # statistiche presenti, nessun file di posa per u1/sessA
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), _cfg(pose_root=str(tmp_path / "pose")))
    with pytest.raises(FileNotFoundError, match="posa mancante"):
        for seed in range(20):
            loader.batch(4, np.random.default_rng(seed))


def test_replace_config_keeps_pose_root_optional():
    cfg = _cfg()
    assert cfg.pose_root is None and replace(cfg, pose_root="/x").pose_root == "/x"
