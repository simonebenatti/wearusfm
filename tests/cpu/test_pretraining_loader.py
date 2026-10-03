"""Dataloader del pretraining su un albero sintetico con i montaggi veri degli adattatori e un manifest prodotto dal costruttore vero:
sessione continua con un salto dell'asse dei tempi e un tratto costante (emg2pose), sessione a prove (NinaPro DB2), array 3D a prove da 1 s (CapgMyo)."""

import gzip
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from wearusfm.data import pretraining_loader as L
from wearusfm.ingest import capgmyo, emg2pose, ninapro_std
from wearusfm.ingest.common import montage_to_dict
from wearusfm.training.masking import MaskSpec

_spec = importlib.util.spec_from_file_location("build_manifest", Path(__file__).resolve().parents[2] / "scripts" / "build_manifest.py")
BM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BM)

SIGMA = 3.0  # ampiezza fisica del rumore sintetico
GAP_AT = 9000
RUN = (2, 15000, 17000)  # canale, inizio, fine del tratto costante


def _write(d: Path, data: np.ndarray, meta: dict):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data_int16.npy", data)
    (d / "metadata.json").write_text(json.dumps(meta))


def _tree(tmp_path: Path):
    rng = np.random.default_rng(0)
    root = tmp_path / "processed"
    m = emg2pose.build_montage_metadata("u1", "sessA", "left")
    x = rng.normal(scale=SIGMA, size=(24000, 16))  # 12 s a 2 kHz
    x[RUN[1]:RUN[2], RUN[0]] = 0.0
    _write(root / "emg2pose" / "u1" / "sessA", np.round(x * 1000).astype(np.int16),
           {"montage": montage_to_dict(m, [True] * 16), "native_fs_hz": 2000.0, "shape": [24000, 16], "int16_scale": 1000.0,
            "time_axis": {"n_gaps": 1, "gaps": [{"index": GAP_AT, "dt_s": 0.01}], "duration_s": 12.01, "dt_max_s": 0.01},
            "constant_runs": [{"channel": RUN[0], "start": RUN[1], "n_samples": RUN[2] - RUN[1]}]})
    m = ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right")
    x = rng.normal(scale=SIGMA, size=(18000, 12))
    trials = [{"offset": 0, "n_samples": 6000}, {"offset": 6000, "n_samples": 6000}, {"offset": 12000, "n_samples": 6000}]
    _write(root / "ninapro_db2" / "s01" / "session1", np.round(x * 1000).astype(np.int16),
           {"montage": montage_to_dict(m, [True] * 12), "native_fs_hz": 2000.0, "shape": [18000, 12], "int16_scale": 1000.0, "trials": trials})
    m = capgmyo.build_montage_metadata(1)
    x = rng.normal(scale=SIGMA, size=(10, 1000, 128))
    _write(root / "capgmyo" / "s01", np.round(x * 1000).astype(np.int16),
           {"montage": montage_to_dict(m, [True] * 128), "native_fs_hz": 1000.0, "shape": [10, 1000, 128], "int16_scale": 1000.0})
    splits = {"datasets": {"emg2pose": {"pretraining": ["u1"], "test": []}, "ninapro_db2": {"pretraining": ["s01"], "test": []},
                           "capgmyo": {"pretraining": ["s01"], "test": []}}}
    rows, summary = BM.build([root], splits, {"A": 0.3, "B": 0.5, "C": 0.2}, 0.5, 1e9, 25.0)
    params = {"quota": {"A": 0.3, "B": 0.5, "C": 0.2}}
    doc = {"params": params, "columns": list(BM.asdict(rows[0]).keys()), "rows": [list(BM.asdict(r).values()) for r in rows]}
    path = tmp_path / "manifest.json.gz"
    path.write_bytes(gzip.compress(json.dumps(doc).encode()))
    return root, path


def _cfg(**kw):
    base = dict(min_window_s=1.0, max_window_s=4.0, split_at_gaps=True, mask=MaskSpec.d10_proposal(0.5), k_neighbors=4, filter_band_hz=None)
    base.update(kw)
    return L.LoaderConfig(**base)


def test_windows_are_aligned_inside_spans_and_avoid_gaps_and_constant_runs(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    loader = L.PretrainLoader(idx, _cfg())
    rng = np.random.default_rng(1)
    row = next(r for r in idx.rows if r["dataset"] == "emg2pose")
    view = loader.view(row)
    assert [(s, e) for _, s, e, _ in view.spans] == [(0, GAP_AT), (GAP_AT, 24000)]  # spezzata al salto
    for _ in range(300):
        w = L.sample_window(view, loader.cfg, rng)
        assert w.start % 400 == 0  # 200 ms a 2 kHz dall'inizio della prova (ancora 0, anche dopo il salto)
        assert not (w.start < GAP_AT < w.stop)  # non attraversa il salto
        assert not (w.start < RUN[2] + 400 and w.stop > RUN[1] - 400)  # non tocca il tratto costante, con 200 ms di margine per lato
        assert 40 <= w.n_patches <= 160 and w.stop - w.start == w.n_patches * 50
    db2 = loader.view(next(r for r in idx.rows if r["dataset"] == "ninapro_db2"))
    for _ in range(100):
        w = L.sample_window(db2, loader.cfg, rng)
        trial = next(t for t in range(3) if t * 6000 <= w.start < (t + 1) * 6000)
        assert w.stop <= (trial + 1) * 6000 and (w.start - trial * 6000) % 400 == 0  # dentro la prova, griglia dall'inizio della prova
    cap = loader.view(next(r for r in idx.rows if r["dataset"] == "capgmyo"))
    w = L.sample_window(cap, loader.cfg, rng)
    assert w.trial is not None and w.n_patches == 40 and (w.start, w.stop) == (0, 1000)  # prove da 1 s: contesto variabile, minimo 1 s
    assert L.sample_window(cap, _cfg(min_window_s=2.0), rng) is None  # con 2 s di minimo CapgMyo non da' finestre (D10)
    joined = L.PretrainLoader(idx, _cfg(split_at_gaps=False)).view(row)
    assert [(s, e) for _, s, e, _ in joined.spans] == [(0, 24000)]


def test_scale_sampling_and_batch(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    loader = L.PretrainLoader(idx, _cfg())
    rng = np.random.default_rng(2)
    row = next(r for r in idx.rows if r["dataset"] == "ninapro_db2")
    s = loader.scale(row, loader.view(row), rng)
    assert s == pytest.approx(0.6745 * SIGMA, rel=0.05)  # MAD di una gaussiana = 0,6745 sigma
    counts = {}
    for _ in range(3000):
        r = idx.rows[int(rng.choice(len(idx.rows), p=idx.weights))]
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    for r, w in zip(idx.rows, idx.weights):
        assert counts[r["dataset"]] / 3000 == pytest.approx(w, abs=0.03)  # campionamento coi pesi del manifest
    b = loader.batch(6, rng)
    c_tot = sum(b.counts)
    p_max = max(b.n_patches)
    assert b.visible.shape == b.kind.shape == (c_tot, p_max) and b.sets.index.shape[0] == c_tot and len(b.signals) == 6
    off = np.cumsum([0, *b.counts])
    for i, (x, n) in enumerate(zip(b.signals, b.n_patches)):
        assert x.shape[1] == math.ceil(n * 0.025 * b.fs[i] - 1e-9)
        assert b.visible[off[i]:off[i + 1], n:].all()  # padding visibile, mai nascosto
        assert b.anchor_targets[i].log_rms.shape == (b.counts[i], n)
        mad = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
        assert np.median(mad) == pytest.approx(1.0, rel=0.25)  # normalizzata con la scala di sessione (stima su tratti sparsi)
    assert b.rvq_on.all()  # i tre dataset hanno l'ancora accesa


def test_filter_removes_mains_and_stays_inside_the_span(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    cfg = _cfg(filter_band_hz=(20.0, 450.0))
    view = L.PretrainLoader(idx, cfg).view(next(r for r in idx.rows if r["dataset"] == "ninapro_db2"))
    t = np.arange(6000) / 2000.0
    hum = 50.0 * np.sin(2 * np.pi * 50.0 * t)
    raw = view.read(None, 0, 6000) + hum[None, :]

    class Fake:  # stessa sessione, con 50 Hz di rete sommati alla prima prova
        fs, scale = 2000.0, None

        def read(self, trial, a, b):
            return raw[:, a:b]

    y = L.read_window(Fake(), None, 1000, 5000, (0, 6000), cfg)
    p50 = lambda z: np.abs(np.fft.rfft(z, axis=-1))[:, int(50 * z.shape[1] / 2000)].mean()  # noqa: E731
    assert y.shape == (12, 4000) and p50(y) < 0.01 * p50(raw[:, 1000:5000])  # rete abbattuta di oltre 40 dB


def test_session_without_a_scale_is_skipped_and_counted(tmp_path):
    root, mpath = _tree(tmp_path)
    d = root / "ninapro_db2" / "s01" / "session1"
    np.save(d / "data_int16.npy", np.zeros((18000, 12), dtype=np.int16))  # tutto zero: MAD nullo, la scala non esiste
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), _cfg())
    b = loader.batch(8, np.random.default_rng(0))
    assert all(r["dataset"] != "ninapro_db2" for r in b.rows) and "ninapro_db2/s01/session1" in loader.skipped
    assert "scala di sessione nulla" in loader.skipped["ninapro_db2/s01/session1"]


def test_block_reading_reads_once_per_block_and_keeps_session_weights(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    loader = L.PretrainLoader(idx, _cfg())
    rng = np.random.default_rng(4)
    counts = {}
    n = 4000
    for _ in range(n):
        row, blk, win = loader.sample_from_blocks(rng)
        assert blk.data_lo <= win.start and win.stop <= blk.data_lo + blk.data.shape[1] and win.span[0] <= win.start and win.stop <= win.span[1]
        if row["dataset"] == "emg2pose":
            assert not (win.start < GAP_AT < win.stop) and not (win.start < RUN[2] + 400 and win.stop > RUN[1] - 400)
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    assert loader.blocks_read <= n / 8 + loader.cfg.block_pool + 5  # una lettura ogni ~8 finestre, non una per finestra
    for r, w in zip(idx.rows, idx.weights):
        assert counts[r["dataset"]] / n == pytest.approx(w, abs=0.06)  # la probabilita' delle sessioni resta quella del manifest


def test_window_from_a_block_equals_the_direct_read(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    cfg = _cfg(filter_band_hz=(20.0, 450.0))
    loader = L.PretrainLoader(idx, cfg)
    rng = np.random.default_rng(6)
    for _ in range(20):
        row, blk, win = loader.sample_from_blocks(rng)
        direct = L.read_window(blk.view, win.trial, win.start, win.stop, win.span, cfg)
        from_block = L.read_window_block(blk, win, cfg)
        m = int(round(cfg.filter_margin_s * blk.view.fs))
        # il blocco contiene sempre il margine intero della lettura diretta: stesso filtro, stesso risultato, senza eccezioni
        assert blk.data_lo <= max(win.span[0], win.start - m) and blk.data_lo + blk.data.shape[1] >= min(win.span[1], win.stop + m)
        assert np.allclose(from_block, direct, atol=1e-5)


def test_scale_ignores_tiny_spans_between_close_gaps(tmp_path):
    root, mpath = _tree(tmp_path)
    meta_path = root / "emg2pose" / "u1" / "sessA" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["time_axis"]["gaps"] = [{"index": 9000, "dt_s": 0.01}, {"index": 9005, "dt_s": 0.01}, {"index": 20000, "dt_s": 0.01},
                                 {"index": 20003, "dt_s": 0.01}]  # tratti di 5 e 3 campioni: il filtro li rifiuterebbe
    meta["time_axis"]["n_gaps"] = 4
    meta_path.write_text(json.dumps(meta))
    view = L.SessionView.open(meta_path.parent, True)
    assert any(e - s < 10 for _, s, e, _ in view.spans)
    cfg = _cfg(filter_band_hz=(20.0, 450.0))
    for seed in range(30):
        assert L.estimate_session_scale(view, cfg, np.random.default_rng(seed)) > 0  # prima: ValueError «padlen» quando pescava un tratto minuscolo


def test_blocks_give_the_same_window_distribution_as_direct_reads(tmp_path):
    """Review del 03/10: la prima lettura a blocchi copriva poco l'inizio dei tratti e troppo la fine, e dava finestre corte in eccesso. Con le
    tessere la distribuzione degli inizi e delle lunghezze dentro la sessione e' quella della lettura diretta."""
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    w = np.asarray([1.0 if r["dataset"] == "emg2pose" else 0.0 for r in idx.rows])
    idx = L.ManifestIndex([r for r, x in zip(idx.rows, w) if x], np.ones(1), idx.roots)  # solo la sessione di emg2pose (due tratti, un buco)
    cfg = _cfg(block_s=3.0)  # tessere da 3 s: tratti da 4,5 e 7,5 s, quindi ultime tessere parziali
    direct, blocks = L.PretrainLoader(idx, cfg), L.PretrainLoader(idx, cfg)
    view = direct.view(idx.rows[0])
    rng_d, rng_b = np.random.default_rng(10), np.random.default_rng(11)
    n = 6000
    d = [L.sample_window(view, cfg, rng_d) for _ in range(n)]
    b = [blocks.sample_from_blocks(rng_b)[2] for _ in range(n)]
    edges = np.arange(0, 24001, 1200)
    hd, _ = np.histogram([x.start for x in d], edges)
    hb, _ = np.histogram([x.start for x in b], edges)
    nz = hd > 0
    assert set(np.nonzero(hb)[0]) <= set(np.nonzero(hd)[0])  # nessun inizio fuori da quelli ammessi
    assert np.all(np.abs(hb[nz] - hd[nz]) <= 4 * np.sqrt(hd[nz]) + 10)  # stessa copertura del tempo, a meno del rumore di campionamento
    short_d = np.mean([x.n_patches < 160 for x in d])
    short_b = np.mean([x.n_patches < 160 for x in b])
    assert abs(short_b - short_d) < 0.03  # stessa quota di finestre corte (prima: molte di piu' coi blocchi)


def test_vectorized_acceptance_equals_window_at_for_every_start(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    for kw in ({}, {"run_margin_s": 0.2, "max_window_s": 2.5}, {"min_window_s": 0.5, "patch_ms": 20.0}):
        cfg = _cfg(**kw)
        loader = L.PretrainLoader(idx, cfg)
        for row in idx.rows:
            view = loader.view(row)
            got = {(c[0], c[1], int(k)) for c in L.admissible_starts(view, cfg) for k in c[5]}
            fs = view.fs
            align = cfg.align_ms / 1000.0 * fs
            want = set()
            for trial, s, e, anchor in view.spans:
                k0 = math.ceil((s - anchor) / align - 1e-9)
                k1 = math.floor((e - cfg.min_window_s * fs - anchor) / align + 1e-9)
                for k in range(k0, k1 + 1):
                    if L.window_at(view, cfg, (trial, s, e, anchor), k) is not None:
                        want.add((trial, s, k))
            assert got == want and got
