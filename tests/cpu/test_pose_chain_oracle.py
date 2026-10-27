"""Controllo 0 della diagnosi di GPT sul collaudo dbg di R3 (27/10/2026, `docs/reviews/r3_dbg_diagnosis_20261027.md`): la catena dei bersagli di
posa su geometria REALE, dal file grezzo ai tensori del trainer, contro un oracolo indipendente calcolato dagli indici grezzi; abbinamento
sbagliato rifiutato; apprendibilita' della testa su feature note (`scripts/pose_head_synthetic.py`).

Catena provata: HDF5 (angoli a 2 kHz) -> `ingest_pose_recording` (zeri + margine M = 280, blocchi di 10) -> albero della posa -> `PoseStore.window`
sulle finestre vere del loader (`fixed_windows`: griglia a 200 ms, patch da 25 ms, 4 s) -> `_pose_tensors` (riempimento alla larghezza del batch).
Oracolo: campione non valido se tutto a zero o entro M campioni da una riga tutta a zero (forza bruta); blocco valido se tutti i 10 campioni lo
sono; bersaglio della patch = media dei blocchi validi col centro (10 j + 5) in [inizio, fine) della patch, standardizzato."""

import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ingest_emg2pose_pose import _hdf5_bytes  # noqa: E402
from test_pose_targets import _stats  # noqa: E402
from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.harness import pose_eval as PE  # noqa: E402
from wearusfm.ingest import emg2pose  # noqa: E402
from wearusfm.ingest import emg2pose_pose as PP  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.training import run as R  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import pose_head_synthetic as PHS  # noqa: E402

FS, N, M = 2000.0, 40000, 280  # 20 s a 2 kHz; margine firmato il 27/10/2026
MEAN, STD = [1.0] * 20, [0.35] * 20


def _raw_angles(n, seed, zero_runs=()):
    """Posa lenta nota (sinusoidi per giunto) piu' rumore, con tratti tutti a zero (fallimenti della cinematica inversa)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    a = (0.6 * np.sin(2 * np.pi * (0.3 + 0.07 * np.arange(20))[None, :] * t[:, None] + rng.normal(size=20)[None, :]) + 1.0
         + 0.02 * rng.normal(size=(n, 20))).astype(np.float32)
    for s, e in zero_runs:
        a[s:e] = 0.0
    return a


def _emg_tree(root: Path, user: str, stem: str, side: str, n: int, seed: int, moving="both") -> Path:
    """Albero EMG processato di emg2pose come lo scrive l'ingest dell'EMG (sidecar con `csv_row`)."""
    rng = np.random.default_rng(seed)
    m = emg2pose.build_montage_metadata(user, stem, side)
    x = rng.normal(scale=3.0, size=(n, 16))
    d = root / f"u{user}" / stem
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data_int16.npy", np.round(x * 1000).astype(np.int16))
    (d / "metadata.json").write_text(json.dumps({"montage": montage_to_dict(m, [True] * 16), "native_fs_hz": FS, "shape": [n, 16],
                                                 "int16_scale": 1000.0, "csv_row": {"side": side, "moving_hand": moving, "user": user}}))
    return d


def _ingest(pose_root: Path, user: str, stem: str, angles: np.ndarray, side: str):
    ref = emg2pose.RecordingRef(member=f"emg2pose_data/{stem}.hdf5", stem=stem)
    return PP.ingest_pose_recording(io.BytesIO(_hdf5_bytes(angles)), ref, pose_root,
                                    {"user": user, "side": side, "moving_hand": "both", "split": "train"}, zero_margin_samples=M, measure=False)


def _oracle(angles: np.ndarray, win, mean, std, patch_samples=50):
    n = angles.shape[0]
    zero = np.all(np.abs(angles) <= PP.ZERO_ATOL, axis=1)
    bad = zero | ~np.all(np.isfinite(angles), axis=1)  # righe non finite: non valide, NON dilatate (contratto dell'ingest)
    for z in np.flatnonzero(zero):  # forza bruta: entro M campioni (inclusi) da una riga a zero
        bad[max(0, z - M):min(n, z + M + 1)] = True
    P = int(win.n_patches)
    bounds = int(win.start) + np.ceil(np.arange(P + 1) * patch_samples - 1e-9).astype(np.int64)
    tgt, msk = np.zeros((P, 20), np.float32), np.zeros((P, 20), bool)
    mean, std = np.asarray(mean), np.asarray(std)
    for p in range(P):
        vals = []
        for j in range(bounds[p] // 10 - 1, min(n // 10, bounds[p + 1] // 10 + 2)):
            if j < 0:
                continue
            c = 10 * j + 5
            if bounds[p] <= c < bounds[p + 1] and not bad[10 * j:10 * j + 10].any():
                vals.append(angles[10 * j:10 * j + 10].astype(np.float64).mean(axis=0))
        if vals:
            tgt[p] = (np.mean(vals, axis=0) - mean) / std
            msk[p] = True
    return tgt, msk


@pytest.fixture
def chain(tmp_path):
    root, pose_root = tmp_path / "processed" / "emg2pose", tmp_path / "pose"
    a1 = _raw_angles(N, 1, zero_runs=[(5000, 5400), (25000, 25020)])
    a1[12000:12003, 4] = np.nan  # righe non finite: il blocco non vale, ma niente margine attorno
    a1[30000, 7] = np.inf
    specs = {("1", "2021-01-01-1-h1-cv-emg-pose-A@2-recording-0_left", "left"): a1,
             ("2", "2021-01-02-1-h2-cv-emg-pose-A@2-recording-0_right", "right"): _raw_angles(N, 2, zero_runs=[(39000, 39700)])}
    views = {}
    for (user, stem, side), ang in specs.items():
        d = _emg_tree(root, user, stem, side, N, seed=int(user))
        out = _ingest(pose_root, user, stem, ang, side)
        assert out["n_blocks"] == N // 10 and out["n_zero_rows"] == sum(e - s for s, e in ([(5000, 5400), (25000, 25020)] if user == "1" else [(39000, 39700)]))
        assert out["n_nonfinite_rows"] == (4 if user == "1" else 0)
        views[user] = (d, ang, stem, side)
    # statistiche PER LATO (criterio `per_side`): l'oracolo deve usare quelle del lato della finestra
    _stats(pose_root, mean=MEAN, std=STD, per_side=True, by_side={"left": {"mean": MEAN, "std": STD}, "right": {"mean": [1.2] * 20, "std": [0.5] * 20}})
    return root, pose_root, views


def test_targets_from_raw_to_trainer_tensors_match_an_independent_oracle(chain):
    root, pose_root, views = chain
    store = PT.PoseStore(pose_root)
    assert store.per_side
    cfg = L.signed_config()
    assert cfg.patch_ms == 25.0  # 50 campioni a 2 kHz: 5 blocchi di posa per patch
    n_checked, n_masked_patches, n_full = 0, 0, 0
    items, n_patches = [], []
    for user, (d, ang, stem, side) in views.items():
        view = L.SessionView.open(d, True)
        mean_side, std_side = store.norm(view)
        assert np.allclose(std_side, STD if side == "left" else 0.5)
        wins = PE.fixed_windows(view, cfg, 10, 4.0 if user == "1" else 2.0, PE.SEED_WINDOWS)  # utente 2: finestre corte (80 patch)
        assert len(wins) >= 3
        for w in wins:
            got = store.window({"dataset": "emg2pose"}, view, w, cfg.patch_ms)
            assert got is not None
            t, m, sd = got
            o_t, o_m = _oracle(ang, w, mean_side, std_side)
            assert m.shape == (w.n_patches, 20) and np.array_equal(m, o_m), (user, w.start)
            assert np.allclose(t[m], o_t[o_m], atol=6e-3), (user, w.start, float(np.abs(t[m] - o_t[o_m]).max()))  # blocchi in float16
            assert np.isfinite(t).all() and np.allclose(sd, std_side)
            n_checked += 1
            n_masked_patches += int((~m[:, 0]).sum())
            n_full += int(m.all())
            items.append((t, m, sd))
            n_patches.append(int(w.n_patches))
    assert n_checked >= 6 and n_masked_patches > 0 and n_full > 0  # finestre con margine mascherato e finestre piene
    # righe non finite (utente 1, campioni 12.000-12.002 e 30.000): invalidano il loro blocco, non la patch (restano 4 blocchi su 5) e non
    # dilatano: finestra esplicita che le contiene, uguale all'oracolo e tutta valida
    d1, ang1, _, _ = views["1"]
    view1 = L.SessionView.open(d1, True)
    mean1, std1 = store.norm(view1)
    for start in (11600, 29800):
        w = L.Window(None, start, start + 8000, (0, N), 160)
        t, m, _ = store.window({"dataset": "emg2pose"}, view1, w, cfg.patch_ms)
        o_t, o_m = _oracle(ang1, w, mean1, std1)
        assert m.all() and np.array_equal(m, o_m) and np.allclose(t, o_t, atol=6e-3) and np.isfinite(t).all()
    # _pose_tensors: riempimento alla larghezza del batch con maschera falsa, una finestra senza posa nel mezzo
    batch = SimpleNamespace(pose=[items[0], None, items[-1]], n_patches=[n_patches[0], n_patches[0], n_patches[-1]])
    tgt, msk, std = R._pose_tensors(batch, "cpu")
    p_max = max(batch.n_patches)
    assert n_patches[-1] < p_max  # la finestra corta: il riempimento oltre le sue patch e' davvero controllato
    assert tgt.shape == (3, p_max, 20) and msk.shape == (3, p_max, 20) and std.shape == (3, 20)
    assert torch.equal(msk[0, :n_patches[0]], torch.from_numpy(items[0][1])) and not msk[1].any()
    assert torch.equal(msk[2, :n_patches[-1]], torch.from_numpy(items[-1][1])) and not msk[2, n_patches[-1]:].any() and not tgt[2, n_patches[-1]:].any()
    assert torch.allclose(tgt[2, :n_patches[-1]], torch.from_numpy(items[-1][0])) and torch.allclose(std[2], torch.from_numpy(items[-1][2]))
    # la perdita iniziale (testa a zero) sul supporto valido e' la media dei quadrati dei bersagli: ~ varianza standardizzata, non esattamente 1
    sq = float((tgt[msk] ** 2).mean())
    assert 0.3 < sq < 3.0


def test_wrong_pairing_and_wrong_side_are_refused_and_unprompted_is_masked(chain):
    root, pose_root, views = chain
    d1, _, stem1, _ = views["1"]
    d2, _, stem2, _ = views["2"]
    cfg = L.signed_config()
    # posa dell'utente 2 (stessa lunghezza) copiata nella cartella dell'utente 1: forma e lunghezza passano, l'identita' no
    swapped = pose_root.parent / "pose_swapped"
    shutil.copytree(pose_root, swapped)
    for f in (PT.POSE_FILE, PT.VALID_FILE, PT.META_FILE):
        shutil.copy(pose_root / "u2" / stem2 / f, swapped / "u1" / stem1 / f)
    store = PT.PoseStore(swapped)
    view1 = L.SessionView.open(d1, True)
    w = PE.fixed_windows(view1, cfg, 1, 4.0, PE.SEED_WINDOWS)[0]
    with pytest.raises(ValueError, match="pose_meta.json di u2"):
        store.window({"dataset": "emg2pose"}, view1, w, cfg.patch_ms)
    # stesso utente e stem, lato della posa diverso da quello dell'EMG
    sided = pose_root.parent / "pose_sided"
    shutil.copytree(pose_root, sided)
    meta_path = sided / "u1" / stem1 / PT.META_FILE
    meta = json.loads(meta_path.read_text())
    meta["side"] = "right"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="lato dell'EMG"):
        PT.PoseStore(sided).window({"dataset": "emg2pose"}, view1, w, cfg.patch_ms)
    # senza il sidecar della posa la guardia non scatta (alberi scritti prima del 27/10): si usano forma e lunghezza come prima
    bare = pose_root.parent / "pose_bare"
    shutil.copytree(pose_root, bare)
    (bare / "u1" / stem1 / PT.META_FILE).unlink()
    assert PT.PoseStore(bare).window({"dataset": "emg2pose"}, view1, w, cfg.patch_ms) is not None
    # mano non sollecitata: con include_unprompted=False la finestra ha bersaglio nullo e maschera vuota; con True (firmato) ha i bersagli
    d_unp = _emg_tree(root, "1", stem1 + "_unp", "left", N, seed=7, moving="right")
    shutil.copytree(pose_root / "u1" / stem1, pose_root / "u1" / (stem1 + "_unp"))
    meta_u = json.loads((pose_root / "u1" / (stem1 + "_unp") / PT.META_FILE).read_text())
    meta_u["stem"] = stem1 + "_unp"
    (pose_root / "u1" / (stem1 + "_unp") / PT.META_FILE).write_text(json.dumps(meta_u))
    view_u = L.SessionView.open(d_unp, True)
    assert PT.PoseStore.unprompted(view_u)
    t0, m0, _ = PT.PoseStore(pose_root, include_unprompted=False).window({"dataset": "emg2pose"}, view_u, w, cfg.patch_ms)
    assert not m0.any() and not t0.any()
    t1, m1, _ = PT.PoseStore(pose_root).window({"dataset": "emg2pose"}, view_u, w, cfg.patch_ms)
    assert m1.any()


def test_zero_head_learns_a_known_linear_target_only_with_enough_signal_and_lr_dose():
    """Calibrazione del controllo (non una previsione del dataset vero): a feature fisse e isotrope, con lo schedule del collaudo (dose di lr
    pari a ~671 passi di picco) la testa a zero scende del 10% solo con segnale forte; con lo schedule di R3 fino a 5.000 passi anche con
    segnale moderato; senza segnale mai. I numeri (passi al 10%) stanno in `scripts/pose_head_synthetic.py --out`."""
    from wearusfm.training import run as R
    cfg = PHS.schedule_cfg("dbg")
    assert all(PHS.lr_at(s, "dbg") == R._lr_at(s - 1, cfg) for s in range(1, 1301))  # lo stesso schedule del trainer, passo per passo
    assert abs(sum(PHS.lr_at(s, "dbg") for s in range(1, 1301)) / 1e-4 - 671) < 1e-6 and PHS.lr_at(1041, "dbg") == 1e-4 and PHS.lr_at(1042, "dbg") < 1e-4
    assert PHS.lr_at(1962, "r3") == 1e-4 and PHS.lr_at(5000, "r3") == 1e-4 and PHS.lr_at(1, "r3") < 1e-6
    strong = PHS.simulate(1.0, 7, "dbg", 0, 1300, eval_every=100)
    assert abs(strong["lr_dose_peak_steps"] - 671) < 1e-6 and strong["mse_end"] <= 0.9 * strong["mse_0"] and strong["first_step_with_10pct"] <= 1300
    null = PHS.simulate(0.0, 7, "dbg", 0, 1300, eval_every=100)
    assert null["ratio_end"] > 0.95 and null["first_step_with_10pct"] is None
    # segnale moderato (q = 0,2): con 7 finestre da 10 istanti indipendenti (70 per passo) il collaudo dbg basta; con 7 istanti per passo (P = 1)
    # no, e con lo schedule di R3 fino a 5.000 passi si'. Scenari illustrativi: nel dataset vero il numero efficace di istanti e q non sono misurati
    mod_p10 = PHS.simulate(0.2, 7, "dbg", 0, 1300, eval_every=100)
    mod_p1_dbg = PHS.simulate(0.2, 7, "dbg", 0, 1300, eval_every=100, P=1)
    mod_p1_r3 = PHS.simulate(0.2, 7, "r3", 0, 5000, eval_every=500, P=1)
    assert mod_p10["instants_per_step"] == 70 and mod_p10["ratio_end"] <= 0.9
    assert mod_p1_dbg["instants_per_step"] == 7 and mod_p1_dbg["ratio_end"] > 0.9
    assert mod_p1_r3["ratio_end"] <= 0.9 and mod_p1_r3["first_step_with_10pct"] is not None
    assert strong["optimum_ratio_at_fixed_features"] == 0.0 and strong["mse_end"] < 0.5  # teacher sulle feature normalizzate: l'ottimo e' 1 - q
