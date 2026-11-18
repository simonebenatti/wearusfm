"""TEMPONet su emg2pose (`docs/temponet_emg2pose_plan.md`): protocollo ufficiale contro reimplementazioni letterali del codice di emg2pose,
istanti, scala dell'ingresso, blocchi di training, EMA e alpha contro quelle di DB8, sintesi per utente, cancello, passo 0 su un tar sintetico."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wearusfm.harness import db8_regression as db8
from wearusfm.harness import emg2pose_regression as R
from wearusfm.ingest import emg2pose_pose as PP

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import emg2pose_eval_targets  # noqa: E402
from test_ingest_emg2pose_pose import _csv, _hdf5_bytes, _tar  # noqa: E402


# --- riferimenti letterali del codice di emg2pose (letto il 10/11/2026) ----------------------------------------------------------------

def _ref_get_contiguous_ones(binary_vector):
    if (binary_vector == 0).all():
        return []
    ones = np.where(binary_vector)[0]
    boundaries = np.where(np.diff(ones) != 1)[0]
    return [(ones[i], ones[j]) for i, j in zip(np.insert(boundaries + 1, 0, 0), np.append(boundaries, len(ones) - 1))]


def _ref_windows(no_ik_failure, window_length, stride):
    """WindowedEmgDataset(window_length, stride, skip_ik_failures=True): blocchi, finestre per blocco, offset (precompute_windows)."""
    blocks = [(t0, t1) for t0, t1 in _ref_get_contiguous_ones(no_ik_failure) if (t1 - t0) >= window_length]
    lens = [(b[1] - b[0] - window_length) // stride + 1 for b in blocks]
    cumsum = np.cumsum([0] + lens)
    out = []
    for idx in range(int(cumsum[-1])):
        block_idx = np.searchsorted(cumsum, idx, "right") - 1
        start_idx, _ = blocks[block_idx]
        out.append(start_idx + (idx - cumsum[block_idx]) * stride)
    return out


def test_contiguous_ones_matches_emg2pose_and_keeps_the_end_inclusive():
    assert R.contiguous_ones(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    assert R.contiguous_ones(np.zeros(5)) == [] and R.contiguous_ones(np.ones(4)) == [(0, 3)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = rng.random(rng.integers(1, 80)) < rng.random()
        assert R.contiguous_ones(b) == [(int(a), int(c)) for a, c in _ref_get_contiguous_ones(b)]


def test_no_ik_failure_is_the_all_zero_row_with_isclose():
    a = np.ones((5, 20), np.float32)
    a[1] = 0.0
    a[2] = 0.0
    a[2, 7] = 1e-9  # dentro la tolleranza di np.isclose: ancora un fallimento
    a[3] = 0.0
    a[3, 0] = 1e-7  # fuori tolleranza: valido
    assert R.no_ik_failure(a).tolist() == [True, False, False, True, True]


def test_official_windows_match_the_windowed_dataset_of_emg2pose():
    rng = np.random.default_rng(1)
    for _ in range(300):
        mask = rng.random(rng.integers(1, 400)) < 0.97
        mask[rng.integers(0, mask.size, size=rng.integers(0, 4))] = False
        got = R.official_window_starts(mask, predicted=40, context=7, stride=40)
        assert got.tolist() == [int(o) for o in _ref_windows(mask, 47, 40)]
    run = np.ones(10_255, bool)  # t1 - t0 = 10.254 < 10.255: nessuna finestra
    assert R.official_window_starts(run).size == 0
    assert R.official_window_starts(np.ones(10_256, bool)).tolist() == [0]
    assert R.official_window_starts(np.ones(30_256, bool)).tolist() == [0, 10_000, 20_000]


def test_eval_instants_have_the_context_inside_the_window():
    inst = R.eval_instants(np.array([0, 10_000]))
    assert inst.shape == (2, 1000) and inst[0, 0] == 255 and inst[0, -1] == 255 + 9_990 and inst[1, 0] == 10_255
    assert (inst[:, -1] < np.array([0, 10_000]) + 10_255).all()  # dentro la finestra effettiva


def test_scale_exponent_rule():
    assert R.scale_exponent(0.0123) == 3 and R.scale_exponent(12.5) == 0 and R.scale_exponent(31.0) == -1
    assert R.scale_exponent(3.0) == 1  # 3 e 30 entrambi ammessi: vince 30, piu' vicino a 10 in scala logaritmica
    with pytest.raises(ValueError):
        R.scale_exponent(0.0)


def test_train_blocks_need_history_validity_and_the_stride():
    valid = np.ones(100, bool)
    valid[40] = False
    j = R.train_blocks(valid, n_samples=1000, stride_blocks=8)
    assert j.tolist() == [32, 48, 56, 64, 72, 80, 88, 96]  # 0-24 senza storia (10 j + 9 < 255), 40 non valido
    assert (R.block_end(j) >= 255).all() and R.train_blocks(valid, n_samples=900, stride_blocks=8).max() == 88


def test_ema_and_alpha_match_the_db8_versions():
    import torch

    rng = np.random.default_rng(2)
    pred = rng.normal(size=(6, 30, 4))
    target = pred + rng.normal(scale=0.5, size=pred.shape)
    alpha = np.array([0.0, 0.3, 0.9, 0.99])
    got = R.ema_windows(torch.tensor(pred), torch.tensor(alpha)).numpy()
    for w in range(6):
        np.testing.assert_allclose(got[w], db8.ema(pred[w], alpha), rtol=1e-12, atol=1e-12)
    grid = np.round(np.arange(0.0, 1.0, 0.05), 2)
    a_got, mae_got = R.tune_alpha_windows(torch.tensor(pred), torch.tensor(target), grid)
    a_ref, mae_ref = db8.tune_alpha([(pred[w], target[w]) for w in range(6)], grid)
    np.testing.assert_allclose(mae_got, mae_ref, rtol=1e-10)
    assert a_got.tolist() == a_ref.tolist()


def test_user_summary_gate_and_reading():
    s = R.summarize_users({"a": (np.deg2rad(10.0) * 4, 4), "b": (np.deg2rad(14.0) * 1, 1)})
    assert s["mean_over_users_deg"] == pytest.approx(12.0) and s["pooled_deg"] == pytest.approx(10.8)
    assert s["sd_over_users_deg"] == pytest.approx(np.std([10.0, 14.0], ddof=1))
    g = R.gate({"a": (np.deg2rad(8.9) * 2, 2)}, {"a": (np.deg2rad(10.0) * 2, 2)})
    assert g["passed"] and g["ratio"] == pytest.approx(0.89)
    assert not R.gate({"a": (np.deg2rad(9.0), 1)}, {"a": (np.deg2rad(10.0), 1)})["passed"]  # 0,9 non e' < 0,9
    assert R.reading(12.0, "user") == "sotto vemg2pose" and R.reading(12.2, "user") == "fra vemg2pose e NeuroPose"
    assert R.reading(18.0, "stage") == "fra NeuroPose e SensingDynamics" and R.reading(19.0, "user_stage") == "sopra SensingDynamics"


# --- passo 0 su dati sintetici ----------------------------------------------------------------------------------------------------------

def _angles(n, zero_runs=(), seed=0):
    t = np.arange(n) / 2000.0
    rng = np.random.default_rng(seed)
    a = (0.5 * np.sin(2 * np.pi * (0.3 + 0.05 * np.arange(20))[None, :] * t[:, None] + rng.normal(size=20)) + 0.7).astype(np.float32)
    for s, e in zero_runs:
        a[s:e] = 0.0
    return a


def _synthetic(tmp_path, specs):
    """specs: [(stem, user, split, generalization, n, zero_runs)] -> tar, CSV, processato e pose a 200 Hz."""
    recs, rows = {}, []
    proc, pose = tmp_path / "proc", tmp_path / "pose"
    rng = np.random.default_rng(5)
    for stem, user, split, gen, n, zr in specs:
        ang = _angles(n, zr, seed=len(recs))
        recs[stem] = _hdf5_bytes(ang)
        rows.append({"user": user, "filename": stem, "split": split, "generalization": gen, "side": "left", "moving_hand": "left",
                     "stage": "s1", "session": stem})
        d = proc / f"u{user}" / stem
        d.mkdir(parents=True)
        emg = rng.normal(scale=0.02, size=(n, 16))
        scale = 32767 / np.abs(emg).max()
        np.save(d / "data_int16.npy", np.round(emg * scale).astype(np.int16))
        (d / "metadata.json").write_text(json.dumps({"shape": [n, 16], "int16_scale": scale}))
        p = pose / f"u{user}" / stem
        p.mkdir(parents=True)
        blocks, valid = PP.block_reduce(ang, ~PP.invalid_rows(ang, 0))
        np.save(p / "pose_200hz_f16.npy", blocks)
        np.save(p / "pose_valid.npy", valid)
    tar = _tar(tmp_path / "e.tar", recs)
    csv = _csv(tmp_path / "meta.csv", rows)
    return tar, csv, proc, pose


def test_eval_targets_end_to_end(tmp_path):
    specs = [("v1", "uval", "val", "user", 25_000, [(12_000, 12_050)]),  # tratti 0-11.999 (1 finestra) e 12.050-24.999 (1 finestra)
             ("t1", "utest", "test", "user", 21_000, []),  # 0-20.999: finestre a 0 e 10.000
             ("t2", "utrain", "test", "stage", 9_000, []),  # troppo corta: nessuna finestra
             ("tr1", "utrain", "train", "none", 6_000, []), ("tr2", "utrain", "train", "none", 6_000, [])]
    tar, csv, proc, pose = _synthetic(tmp_path, specs)
    out = tmp_path / "out"
    assert emg2pose_eval_targets.main(["--tar", str(tar), "--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose),
                                       "--out", str(out), "--n-scale", "2"]) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["complete"] and not m["problems"] and set(m["recordings"]) == {"v1", "t1", "t2"}
    assert m["totals"]["val/user"]["windows"] == 2 and m["totals"]["test/user"]["windows"] == 2
    assert m["totals"]["test/stage"]["recordings_without_windows"] == 1
    z = np.load(out / "test" / "uutest" / "t1.npz")
    assert z["starts"].tolist() == [0, 10_000] and z["targets"].shape == (2, 1000, 20)
    ang = _angles(21_000, (), seed=1)
    np.testing.assert_array_equal(z["targets"], ang[R.eval_instants(z["starts"])])
    v = np.load(out / "val" / "uuval" / "v1.npz")
    assert v["starts"].tolist() == [0, 12_050]
    assert m["block_check"]["max_rad"] < 0.01  # medie di 10 campioni di un segnale lento, allineate per indice
    assert 3 <= m["scale"]["median_channel_sd_scaled"] <= 30 and m["scale"]["exponent_k"] == 3  # deviazione 0,02 -> 20
    assert emg2pose_eval_targets.main(["--tar", str(tar), "--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose),
                                       "--out", str(out)]) == 2  # cartella non vuota


def test_eval_targets_flags_misalignment(tmp_path):
    tar, csv, proc, pose = _synthetic(tmp_path, [("t1", "u1", "test", "user", 12_000, []), ("tr", "u2", "train", "none", 3_000, [])])
    meta = proc / "uu1" / "t1" / "metadata.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "shape": [11_999, 16]}))
    out = tmp_path / "out"
    assert emg2pose_eval_targets.main(["--tar", str(tar), "--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose),
                                       "--out", str(out), "--n-scale", "1"]) == 1
    assert "processato" in json.loads((out / "manifest.json").read_text())["problems"][0]


# --- training e valutazione (CPU) -------------------------------------------------------------------------------------------------------

import temponet_emg2pose as TE  # noqa: E402


def _dataset(tmp_path):
    specs = [("tr1", "ua", "train", "none", 30_000, [(9_000, 9_100)]), ("tr2", "ub", "train", "none", 24_000, []),
             ("va1", "uv", "val", "user", 21_000, []), ("te1", "ut", "test", "user", 21_000, []),
             ("te2", "ua", "test", "stage", 11_000, [])]
    tar, csv, proc, pose = _synthetic(tmp_path, specs)
    (pose / "pose_stats.json").write_text(json.dumps({"split": "train", "all": {"mean": [0.7] * 20, "std": [0.4] * 20}}))
    targets = tmp_path / "targets"
    assert emg2pose_eval_targets.main(["--tar", str(tar), "--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose),
                                       "--out", str(targets), "--n-scale", "2"]) == 0
    return csv, proc, pose, targets / "manifest.json"


def test_host_store_gather_and_feeder_chunks(tmp_path):
    import torch

    csv, proc, pose, _ = _dataset(tmp_path)
    rows = [r for r in R.read_rows(csv) if r["split"] == "train"]
    st = TE.HostStore(rows, proc, 3)
    raw0 = np.load(proc / "uua" / "tr1" / "data_int16.npy")
    raw1 = np.load(proc / "uub" / "tr2" / "data_int16.npy")
    ends = torch.tensor([255, 29_999, 30_000 + 300])
    w = st.gather(ends).numpy()
    np.testing.assert_array_equal(w[0], raw0[0:256])
    np.testing.assert_array_equal(w[1], raw0[29_744:30_000])
    np.testing.assert_array_equal(w[2], raw1[45:301])
    meta0 = json.loads((proc / "uua" / "tr1" / "metadata.json").read_text())
    assert float(st.factor[0]) == pytest.approx(1e3 / meta0["int16_scale"], rel=1e-6)
    e, fw, ys = TE.block_index(st, pose, 8)
    assert len(e) == len(fw) == len(ys) and int(e.min()) >= 255
    order = torch.randperm(len(e), generator=torch.Generator().manual_seed(0))
    fd = TE.Feeder(st, e, fw, ys, batch=4, k=3, pin=False)
    fd.start(order, steps=7)
    seen = 0
    for b, c in fd.chunks():
        m = c * 4
        sel = order[seen * 4 : seen * 4 + m]
        np.testing.assert_array_equal(fd.bx[b][:m].numpy(), st.gather(e[sel]).numpy())
        np.testing.assert_array_equal(fd.by[b][:m].numpy(), ys[sel].numpy())
        seen += c
        fd.release(b)
    assert seen == 7


@pytest.mark.parametrize("loss", ["l1", "smoothl1", "mse"])
def test_stepper_matches_a_plain_loop_on_scaled_input_and_standardized_target(loss):
    import torch

    rng = np.random.default_rng(3)
    x = torch.from_numpy(rng.integers(-3000, 3000, size=(3, 4, 256, 16)).astype(np.int16))
    f = torch.from_numpy(rng.uniform(0.01, 0.02, size=(3, 4)).astype(np.float32))
    y = torch.from_numpy(rng.normal(0.7, 0.4, size=(3, 4, 20)).astype(np.float16))
    mu, sd = np.full(20, 0.7), np.full(20, 0.4)
    cpu = torch.device("cpu")
    m1, o1, l1 = TE.build(5, cpu, 1e-3, dropout=0.0, loss=loss)
    st = TE.Stepper(m1, o1, l1, mu, sd, 4, False, cpu)
    for i in range(3):
        st.step(x[i], f[i], y[i])
    m2, o2, l2 = TE.build(5, cpu, 1e-3, dropout=0.0, loss=loss)
    total = 0.0
    for i in range(3):
        xi = x[i].permute(0, 2, 1).float() * f[i][:, None, None]
        yi = (y[i].float() - torch.tensor(mu, dtype=torch.float32)) / torch.tensor(sd, dtype=torch.float32)
        o2.zero_grad(set_to_none=True)
        loss = l2(m2(xi), yi)
        loss.backward()
        o2.step()
        total += float(loss.detach())
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    assert st.take_total() == pytest.approx(total)


def test_train_then_eval_end_to_end_on_cpu(tmp_path, monkeypatch):
    csv, proc, pose, manifest = _dataset(tmp_path)
    common = ["--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose), "--targets-manifest", str(manifest), "--device", "cpu"]
    run = tmp_path / "run"
    assert TE.main(["train", *common, "--out", str(run), "--epochs", "2", "--batch", "16", "--chunk-steps", "3", "--max-steps-per-epoch", "7"]) == 0
    r = json.loads((run / "run.json").read_text())
    assert r["status"] == "ok" and r["scale_exponent_k"] == 3 and r["parameters"] == 71_412 and r["steps_per_epoch"] == 7
    h = json.loads((run / "history.json").read_text())
    assert [row["epoch"] for row in h] == [1, 2] and all(np.isfinite(row["val_mae_raw_deg"]) for row in h)
    assert (run / "model_final.pt").is_file() and len(list((run / "checkpoints").glob("*.pt"))) == 2
    ev = tmp_path / "eval_val"
    assert TE.main(["eval", *common, "--out", str(ev), "--model-dir", str(run), "--val-only"]) == 0
    e = json.loads((ev / "eval.json").read_text())
    assert e["test"].startswith("non calcolato") and len(e["alpha"]) == 20 and e["val"]["windows"] == 2 and "ratio" in e["gate"]
    assert "Test non calcolato" in (ev / "eval.md").read_text()
    monkeypatch.setattr(TE.R, "gate", lambda m, c: {"model_deg": 1.0, "constant_deg": 2.0, "ratio": 0.5, "max_ratio": 0.9, "passed": True})
    ev2 = tmp_path / "eval_test"
    assert TE.main(["eval", *common, "--out", str(ev2), "--model-dir", str(run)]) == 0
    t = json.loads((ev2 / "eval.json").read_text())["test"]
    assert t["windows"] == 3 and set(t["conditions"]) == set(R.CONDITIONS)
    assert t["conditions"]["user"]["ema"]["n_users"] == 1 and t["conditions"]["stage"]["ema"]["n_users"] == 1
    assert t["conditions"]["user_stage"]["ema"]["mean_over_users_deg"] is None and t["conditions"]["user"]["reading"]
    assert "| user | 1 |" in (ev2 / "eval.md").read_text()


# --- studio delle loss (18/11/2026) -----------------------------------------------------------------------------------------------------

def test_partner_blocks_need_contiguous_validity_and_history():
    valid = np.ones(100, bool)
    valid[40] = False
    valid[46] = False
    j = R.train_blocks(valid, n_samples=1000, stride_blocks=8)
    assert j.tolist() == [32, 48, 56, 64, 72, 80, 88, 96]
    jp, ok = TE.partner_blocks(valid, j, 4)
    assert ok.tolist() == [True, False, True, True, True, True, True, True]  # 48: il blocco 46, fra 44 e 48, non e' valido
    assert jp.tolist() == [28, 48, 52, 60, 68, 76, 84, 92]
    jp8, ok8 = TE.partner_blocks(valid, j, 8)  # 32 -> 24: 10 x 24 + 9 = 249 < 255, senza storia; 48 -> 40 non valido
    assert ok8.tolist() == [False, False, True, True, True, True, True, True] and jp8.tolist() == [32, 48, 48, 56, 64, 72, 80, 88]
    with pytest.raises(ValueError):
        TE.partner_blocks(valid, j, 0)


def test_block_index_with_partners_keeps_the_anchors(tmp_path):
    import torch

    csv, proc, pose, _ = _dataset(tmp_path)
    rows = [r for r in R.read_rows(csv) if r["split"] == "train"]
    st = TE.HostStore(rows, proc, 3)
    e, fw, ys = TE.block_index(st, pose, 8)
    e2, fw2, ys2, (pe, py, pm) = TE.block_index(st, pose, 8, 4)
    assert torch.equal(e, e2) and torch.equal(fw, fw2) and torch.equal(ys, ys2)
    ok = pm.numpy() > 0
    assert 0 < ok.sum() < len(ok)  # il tratto nullo di tr1 (campioni 9.000-9.099) toglie qualche compagna
    np.testing.assert_array_equal(pe.numpy()[ok], e.numpy()[ok] - 4 * R.BLOCK)
    np.testing.assert_array_equal(pe.numpy()[~ok], e.numpy()[~ok])
    np.testing.assert_array_equal(py.numpy()[~ok], ys.numpy()[~ok])
    for i, d in enumerate(st.dirs):
        blocks = np.load(pose / d.parent.name / d.name / "pose_200hz_f16.npy")
        sel = (e.numpy() >= st.offset[i]) & (e.numpy() < st.offset[i] + st.n[i])
        local = pe.numpy()[sel] - st.offset[i]
        assert (local >= R.WINDOW - 1).all() and ((local + 1) % R.BLOCK == 0).all()  # stessa registrazione, storia intera, fine di un blocco
        np.testing.assert_array_equal(py.numpy()[sel], blocks[(local + 1) // R.BLOCK - 1])


def test_velocity_std_matches_numpy(monkeypatch):
    import torch

    rng = np.random.default_rng(4)
    ys = torch.from_numpy(rng.normal(0.7, 0.4, size=(50, 20)).astype(np.float16))
    pys = torch.from_numpy(rng.normal(0.7, 0.4, size=(50, 20)).astype(np.float16))
    mask = torch.from_numpy((rng.random(50) < 0.7).astype(np.float32))
    monkeypatch.setattr(TE, "STATS_CHUNK", 7)  # piu' blocchi
    sd, n = TE.velocity_std(ys, (None, pys, mask))
    k = mask.numpy() > 0
    assert n == int(k.sum())
    np.testing.assert_allclose(sd, (ys.numpy()[k].astype(np.float64) - pys.numpy()[k].astype(np.float64)).std(axis=0), rtol=1e-12)
    with pytest.raises(ValueError):
        TE.velocity_std(ys, (None, ys, torch.ones(50)))  # velocita' nulla


def test_build_losses_share_the_initialization():
    import torch

    cpu = torch.device("cpu")
    ref, _, l1 = TE.build(5, cpu, 1e-3)
    assert isinstance(l1, torch.nn.L1Loss)
    for loss, cls in {"smoothl1": torch.nn.SmoothL1Loss, "mse": torch.nn.MSELoss, "mse_vel": torch.nn.MSELoss}.items():
        m, _, fn = TE.build(5, cpu, 1e-3, loss=loss)
        assert isinstance(fn, cls)
        for a, b in zip(ref.state_dict().values(), m.state_dict().values()):
            assert torch.equal(a, b)
    assert TE.build(5, cpu, 1e-3, loss="smoothl1")[2].beta == TE.SMOOTH_L1_BETA == 1.0
    with pytest.raises(ValueError):
        TE.build(5, cpu, 1e-3, loss="huber")


def _pair_batches(seed, steps, batch):
    import torch

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.integers(-3000, 3000, size=(steps, batch, 256, 16)).astype(np.int16))
    xp = torch.from_numpy(rng.integers(-3000, 3000, size=(steps, batch, 256, 16)).astype(np.int16))
    f = torch.from_numpy(rng.uniform(0.01, 0.02, size=(steps, batch)).astype(np.float32))
    y = rng.normal(0.7, 0.4, size=(steps, batch, 20))
    yp = y + rng.normal(0.0, 0.03, size=y.shape)
    return x, xp, f, torch.from_numpy(y.astype(np.float16)), torch.from_numpy(yp.astype(np.float16))


VEL_STATS = (np.full(20, 0.7), np.linspace(0.2, 0.6, 20), np.linspace(0.01, 0.05, 20))  # mu, sd della posa, sd della velocita'


def test_mse_vel_loss_and_gradient_match_an_independent_formula():
    import torch

    x, xp, f, y, yp = _pair_batches(6, 1, 4)
    m = torch.tensor([1.0, 0.0, 1.0, 1.0])
    mu, sd, sd_v = VEL_STATS
    cpu = torch.device("cpu")
    m1, o1, l1 = TE.build(5, cpu, 1e-3, dropout=0.0, loss="mse_vel")
    st = TE.Stepper(m1, o1, l1, mu, sd, 4, False, cpu, vel=(0.1, sd_v))
    for buf, val in ((st.xb, x[0]), (st.fb, f[0]), (st.yb, y[0]), (st.xpb, xp[0]), (st.ypb, yp[0]), (st.mb, m)):
        buf.copy_(val)
    zs = []

    def keep(module, inputs, output):
        output.retain_grad()
        zs.append(output)

    m1.register_forward_hook(keep)
    loss = st._loss()
    loss.backward()
    sd_t, sdv_t = torch.tensor(sd), torch.tensor(sd_v)

    def reference(z):  # formula per coppia, in float64
        pose = ((z[:4] - (y[0].double() - torch.tensor(mu)) / sd_t) ** 2).mean()
        terms = [((((z[b] - z[4 + b]) * sd_t - (y[0, b].double() - yp[0, b].double())) / sdv_t) ** 2).mean() for b in range(4) if m[b] > 0]
        vel = torch.stack(terms).mean()  # media sulle 3 coppie valide e sui 20 giunti
        return pose, vel

    # 1) la formula, sulle stesse uscite z del modello in float32 (il modello entra solo attraverso z)
    zr = zs[0].detach().double().requires_grad_()
    pose, vel = reference(zr)
    (pose + 0.1 * vel).backward()
    assert float(loss.detach()) == pytest.approx(float((pose + 0.1 * vel).detach()), rel=1e-6)
    np.testing.assert_allclose(st._parts.numpy(), [float(pose.detach()), float(vel.detach())], rtol=1e-6)
    torch.testing.assert_close(zs[0].grad.double(), zr.grad, rtol=1e-4, atol=1e-9)
    # 2) le due finestre passano insieme nel modello (BatchNorm sulle 8 finestre): gradiente dei parametri contro il modello in float64
    m2, _, _ = TE.build(5, cpu, 1e-3, dropout=0.0, loss="mse_vel")
    m2.double()
    z = m2(torch.cat([x[0].permute(0, 2, 1).double() * f[0].double()[:, None, None], xp[0].permute(0, 2, 1).double() * f[0].double()[:, None, None]]))
    pose2, vel2 = reference(z)
    (pose2 + 0.1 * vel2).backward()
    assert float(loss.detach()) == pytest.approx(float((pose2 + 0.1 * vel2).detach()), rel=1e-4)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):  # float32 contro float64 attraverso BatchNorm a 8 finestre: errore relativo in norma
        assert float((p1.grad.double() - p2.grad).norm() / p2.grad.norm()) < 1e-3


def test_stepper_mse_vel_matches_a_plain_loop():
    import torch

    x, xp, f, y, yp = _pair_batches(7, 3, 4)
    m = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])  # al secondo passo nessuna coppia valida
    mu, sd, sd_v = VEL_STATS
    cpu = torch.device("cpu")
    m1, o1, l1 = TE.build(5, cpu, 1e-3, dropout=0.0, loss="mse_vel")
    st = TE.Stepper(m1, o1, l1, mu, sd, 4, False, cpu, vel=(0.1, sd_v))
    for i in range(3):
        st.step(x[i], f[i], y[i], xp[i], yp[i], m[i])
    m2, o2, l2 = TE.build(5, cpu, 1e-3, dropout=0.0, loss="mse_vel")
    mu_t, sd_t, sdv_t = (torch.tensor(a, dtype=torch.float32) for a in (mu, sd, sd_v))
    total, parts = 0.0, [0.0, 0.0]
    for i in range(3):
        z = m2(torch.cat([x[i].permute(0, 2, 1).float() * f[i][:, None, None], xp[i].permute(0, 2, 1).float() * f[i][:, None, None]]))
        pose = l2(z[:4], (y[i].float() - mu_t) / sd_t)
        dv = ((z[:4] - z[4:]) * sd_t - (y[i].float() - yp[i].float())) / sdv_t
        vel = (m[i][:, None] * dv.square()).sum() / (m[i].sum().clamp(min=1.0) * 20)
        loss = pose + 0.1 * vel
        o2.zero_grad(set_to_none=True)
        loss.backward()
        o2.step()
        total += float(loss.detach())
        parts = [parts[0] + float(pose.detach()), parts[1] + float(vel.detach())]
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    assert st.take_total() == pytest.approx(total)
    np.testing.assert_allclose(st.take_parts(), parts, rtol=1e-6)
    assert st.take_parts() == (0.0, 0.0)  # azzerati dalla lettura


@pytest.mark.parametrize("loss", ["smoothl1", "mse", "mse_vel"])
def test_train_with_the_other_losses_then_eval_on_cpu(tmp_path, loss):
    csv, proc, pose, manifest = _dataset(tmp_path)
    common = ["--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose), "--targets-manifest", str(manifest), "--device", "cpu"]
    run = tmp_path / "run"
    assert TE.main(["train", *common, "--out", str(run), "--epochs", "1", "--batch", "16", "--chunk-steps", "3", "--max-steps-per-epoch", "5",
                    "--loss", loss]) == 0
    r = json.loads((run / "run.json").read_text())
    assert r["status"] == "ok" and r["args"]["loss"] == loss and r["parameters"] == 71_412 and r["loss_definition"]
    h = json.loads((run / "history.json").read_text())[0]
    assert np.isfinite(h[f"train_{loss}_std"]) and np.isfinite(h["val_mae_raw_deg"])
    if loss == "mse_vel":
        assert len(r["vel_sd_rad"]) == 20 and min(r["vel_sd_rad"]) > 0 and 0 < r["vel_pairs_fraction"] < 1
        assert h["train_mse_vel_std"] == pytest.approx(h["train_pose_mse_std"] + 0.1 * h["train_vel_mse_std"])
    else:
        assert "vel_sd_rad" not in r and "train_pose_mse_std" not in h
    ev = tmp_path / "eval_val"
    assert TE.main(["eval", *common, "--out", str(ev), "--model-dir", str(run), "--val-only"]) == 0
    assert json.loads((ev / "eval.json").read_text())["train_loss"] == loss


def test_mse_vel_needs_a_positive_lag_and_weight(tmp_path):
    base = ["train", "--csv", "x", "--processed-root", "x", "--pose-root", "x", "--targets-manifest", "x", "--out", str(tmp_path / "o"),
            "--loss", "mse_vel"]
    for extra in (["--vel-lag-blocks", "0"], ["--vel-weight", "0"]):
        with pytest.raises(SystemExit):
            TE.main(base + extra)


@pytest.mark.parametrize("fmt", ["pax", "gnu"])
def test_tar_index_matches_tarfile_and_reads_the_members(tmp_path, fmt):
    import io
    import os
    import tarfile

    recs = {"x" * 150 + "-recording-1_left": b"a" * 1000, "short_right": b"b" * 513, "vuota": b""}  # nome oltre i 100 caratteri ustar
    tar = tmp_path / f"t_{fmt}.tar"
    with tarfile.open(tar, "w", format=tarfile.PAX_FORMAT if fmt == "pax" else tarfile.GNU_FORMAT) as tf:
        d = tarfile.TarInfo("emg2pose_data")
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
        for stem, data in recs.items():
            info = tarfile.TarInfo(f"emg2pose_data/{stem}.hdf5")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    idx = emg2pose_eval_targets.tar_index(tar)
    with tarfile.open(tar) as tf:
        ref = {Path(m.name).stem: (m.offset_data, m.size) for m in tf if m.isfile()}
    assert idx == ref and set(idx) == set(recs)
    fd = os.open(tar, os.O_RDONLY)
    try:
        for stem, (off, size) in idx.items():
            assert emg2pose_eval_targets.read_exact(fd, off, size) == recs[stem]
    finally:
        os.close(fd)
