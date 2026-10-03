"""Codici dell'ancora RVQ per il pretraining (`data.rvq_codes`): stessa vista canonica di V2, blocchi da 16 patch per canale, allineamento con le
finestre del dataloader. Il tokenizer vero gira su Leonardo; qui un finto `encode` che dipende solo dalla patch."""

import json

import numpy as np
import pytest

from wearusfm.data import pretraining_loader as L
from wearusfm.data import rvq_codes as RC
from wearusfm.data.processed import load_session
from wearusfm.tokenizer_checks.sessions import to_canonical

from test_pretraining_loader import _cfg, _tree


def fake_encode(x):
    """(B, n_time*200) -> (B, n_time): un codice per patch che dipende solo da quella patch (energia), tra 0 e 8191."""
    p = x.reshape(len(x), -1, 200)
    return (np.sqrt((p ** 2).mean(axis=-1)) * 997).astype(np.int64) % 8192


def _session(root, row):
    path = L.ManifestIndex([row], np.ones(1), [root]).path_of(row)
    return load_session(path, row["dataset"], row["subject"], row["session"])


def test_canonical_trials_are_v2_canonical_view_trial_by_trial(tmp_path):
    root, mpath = _tree(tmp_path)
    for row in L.ManifestIndex.load(mpath, [root]).rows:
        s = _session(root, row)
        trials, scale = RC.canonical_trials(s.segments, s.fs, s.qc_valid)
        cs = to_canonical(s)
        assert scale == pytest.approx(cs.scale, rel=1e-12)
        assert np.array_equal(np.concatenate([t for t in trials if t is not None], axis=1), cs.stream)  # stesso stream di V2, bit per bit


def test_session_codes_per_patch_blocks_and_qc(tmp_path):
    root, mpath = _tree(tmp_path)
    row = next(r for r in L.ManifestIndex.load(mpath, [root]).rows if r["dataset"] == "ninapro_db2")  # 3 prove da 3 s = 15 patch ciascuna
    s = _session(root, row)
    qc = s.qc_valid.copy()
    qc[3] = False
    trials, _ = RC.canonical_trials(s.segments, s.fs, qc)
    calls = []
    out = RC.session_codes(trials, qc, lambda x: calls.append(x.shape) or fake_encode(x), factor=2.0, batch=7)
    assert out["trial_offset"].tolist() == [0, 15, 30] and out["trial_patches"].tolist() == [15, 15, 15]
    assert (out["codes"][3] == -1).all() and (out["codes"][np.flatnonzero(qc)] >= 0).all()  # canale scartato: nessun codice
    for ti, t in enumerate(trials):
        want = fake_encode(t[0].reshape(1, -1) * 2.0)[0]  # il finto codice dipende solo dalla patch: stesso valore qualunque sia il blocco
        assert np.array_equal(out["codes"][0, 15 * ti: 15 * ti + 15], want)
    assert {c[1] for c in calls} == {15 * 200}  # prove da 15 patch: un blocco solo per prova, mai a cavallo di due prove
    assert RC.blocks_of(40) == [(0, 16), (16, 32), (32, 40)]


def test_canonical_patch_p_is_the_200ms_block_p_of_the_trial(tmp_path):
    """Allineamento indipendente dal codice di `window_codes`: un impulso nel mezzo del blocco nativo p (dall'inizio della prova) cade nella patch
    canonica p della stessa prova."""
    rng = np.random.default_rng(0)
    for fs in (2000.0, 1000.0, 1926.0, 1111.0):
        n = int(6 * fs)
        seg = rng.normal(scale=0.01, size=(n, 2))
        for p in (0, 7, 23):
            x = seg.copy()
            x[int((p + 0.5) * 0.2 * fs), :] += 50.0
            trials, _ = RC.canonical_trials([x], fs, np.ones(2, dtype=bool))
            energy = (trials[0][0] ** 2).sum(axis=-1)
            assert int(energy.argmax()) == p, (fs, p)


def test_window_codes_follow_the_trial_grid_and_mark_missing(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    loader = L.PretrainLoader(idx, _cfg())
    rng = np.random.default_rng(3)
    for row in idx.rows:
        view = loader.view(row)
        s = _session(root, row)
        trials, _ = RC.canonical_trials(s.segments, s.fs, s.qc_valid)
        entry = RC.session_codes(trials, s.qc_valid, fake_encode)
        entry["codes"] = np.tile(np.arange(entry["codes"].shape[1], dtype=np.int16), (len(s.qc_valid), 1))  # codice = indice di patch
        for _ in range(50):
            win = L.sample_window(view, loader.cfg, rng)
            ti, anchor = RC.trial_of(view, win)
            got = RC.window_codes(entry, view, win, len(s.qc_valid))
            k0 = round((win.start - anchor) / (0.2 * view.fs))
            j = np.arange(k0, k0 + win.n_patches // 8)
            inside = j < entry["trial_patches"][ti]
            assert got.shape == (len(s.qc_valid), win.n_patches // 8)
            assert (got[:, inside] == entry["trial_offset"][ti] + j[inside]).all() and (got[:, ~inside] == -1).all()
        assert (RC.window_codes(None, view, win, 3) == -1).all()  # sessione senza codici: nessun target
    packed = RC.pack_codes([np.zeros((2, 3), dtype=np.int64), np.ones((1, 5), dtype=np.int64)])
    assert packed.shape == (3, 5) and (packed[:2, 3:] == -1).all()


def test_loader_batch_carries_codes_only_with_a_code_root(tmp_path):
    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    codes_root = tmp_path / "codes"
    for row in idx.rows:
        s = _session(root, row)
        trials, _ = RC.canonical_trials(s.segments, s.fs, s.qc_valid)
        p = RC.code_path(codes_root, row["dataset"], row["subject"], row["session"])
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **RC.session_codes(trials, s.qc_valid, fake_encode))
    assert L.PretrainLoader(idx, _cfg()).batch(4, np.random.default_rng(0)).rvq_codes is None
    b = L.PretrainLoader(idx, _cfg(rvq_codes_root=str(codes_root))).batch(6, np.random.default_rng(0))
    assert len(b.rvq_codes) == 6
    for c, n, row in zip(b.rvq_codes, b.n_patches, b.rows):
        assert c.shape[1] == n // 8
        assert (c >= 0).any() == (row["rvq"] == "on")  # nessun codice dove l'ancora e' spenta
    assert json.dumps(L.LoaderConfig.__dataclass_fields__["rvq_codes_root"].default) == "null"
