"""Catena della replica di TEMPONet su dati finti, su CPU: cache dai .mat, training dei tre modelli, sintesi e verdetto."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import temponet_db8  # noqa: E402
import temponet_db8_cache  # noqa: E402
import temponet_db8_summary  # noqa: E402
import temponet_db8_screen  # noqa: E402


def _raw_root(tmp_path, subjects=(1,), n=3000):
    root = tmp_path / "raw"
    (root / "DB8").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for s in subjects:
        for a in (1, 2, 3):
            t = np.arange(n) / 2000.0
            glove = (45 + 30 * np.sin(2 * np.pi * 0.5 * t)[:, None] * np.ones((1, 18))).astype(np.float32)
            emg = (rng.normal(0, 5e-5, size=(n, 16)) * (1 + np.abs(np.sin(2 * np.pi * 0.5 * t)))[:, None]).astype(np.float32)
            lab = np.zeros((n, 1), dtype=np.int8)
            sio.savemat(root / "DB8" / f"S{s}_E1_A{a}.mat", {"emg": emg, "glove": glove, "restimulus": lab, "rerepetition": lab})
    return root


@pytest.fixture()
def cache(tmp_path):
    out = tmp_path / "cache"
    assert temponet_db8_cache.main(["--raw-root", str(_raw_root(tmp_path)), "--out", str(out), "--workers", "1", "--subjects", "1"]) == 0
    return out


def test_cache_manifest(cache):
    m = json.loads((cache / "manifest.json").read_text())
    assert m["kind"] == "temponet_db8_cache_v1" and len(m["entries"]) == 3
    e = m["entries"][0]
    assert e["n_samples"] == 3000 and e["outputs"]["emg"]["shape"] == [3000, 16] and e["outputs"]["doa"]["dtype"] == "float32"
    assert len(e["source_sha256"]) == 64
    # una cartella non vuota non si riusa
    assert temponet_db8_cache.main(["--raw-root", str(cache), "--out", str(cache), "--workers", "1"]) == 2


def test_cache_trims_trailing_nan_and_records_it(tmp_path):
    root = _raw_root(tmp_path)
    src = root / "DB8" / "S1_E1_A1.mat"
    mat = sio.loadmat(src)
    mat["glove"][-1, :] = np.nan  # come S5_E1_A1 del raw reale
    sio.savemat(src, {k: v for k, v in mat.items() if not k.startswith("__")})
    out = tmp_path / "cache_nan"
    assert temponet_db8_cache.main(["--raw-root", str(root), "--out", str(out), "--workers", "1", "--subjects", "1"]) == 0
    e = json.loads((out / "manifest.json").read_text())["entries"][0]
    assert (e["acquisition"], e["n_samples_source"], e["trimmed_tail_samples"], e["n_samples"]) == (1, 3000, 1, 2999)
    assert np.isfinite(np.load(out / "S1_A1_doa.npy")).all()


def test_training_and_summary_end_to_end(cache, tmp_path):
    out = tmp_path / "run"
    args = ["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "--epochs", "2", "--batch", "16",
            "--workers", "1", "--device", "cpu"]
    assert temponet_db8.main(args) == 0
    unit = json.loads((out / "units" / "S1_seed0.json").read_text())
    assert set(unit["models"]) == {"fold1", "fold2", "full"}
    f1, full = unit["models"]["fold1"], unit["models"]["full"]
    assert f1["train_acquisitions"] == [1] and f1["val_acquisition"] == 2 and full["train_acquisitions"] == [1, 2]
    assert len(f1["history"]) == 2 and "val_mae_raw_deg" in f1["history"][0]
    assert "lr_first" not in f1["history"][0]  # learning rate costante: percorso di prima, nessuna chiave dello schedule
    assert full["n_train_windows"] == 2 * f1["n_train_windows"]
    for m in unit["models"].values():
        assert np.isfinite(m["test_ema"]["mae_deg"]) and len(m["alpha"]) == 5
    assert (out / "models" / "S1_seed0_full.pt").is_file()
    assert not (out / "units" / "S1_seed0.partial.json").exists()  # tolto a fine unita'
    (out / "units" / "S2_seed0.partial.json").write_text("{}")  # un'unita' non finita non entra nella sintesi
    assert np.load(out / "predictions" / "S1_seed0_fold2_test_raw.npy").shape == (np.load(out / "predictions" / "S1_test_target.npy").shape)
    run = json.loads((out / "run.json").read_text())
    assert run["model"]["parameters"] == 70_917 and run["status"] == {"S1_seed0": "ok"}
    assert (run["args"]["lr_schedule"], run["args"]["final_epochs"], run["args"]["final_lr"]) == ("constant", 0, None)
    assert temponet_db8_summary.main([str(out), "--seeds", "0"]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["verdict"] == "incompleta" and not s["complete"]
    expected = np.mean([unit["models"][f]["test_ema"]["mae_deg"] for f in ("fold1", "fold2")])
    assert s["primary_mae_deg"] == pytest.approx(expected)
    # la cartella delle uscite deve essere nuova
    assert temponet_db8.main(args) == 2


def test_train_step_and_model_subset(cache, tmp_path):
    out = tmp_path / "run_sub"
    assert temponet_db8.main(["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "--epochs", "1",
                              "--batch", "16", "--workers", "1", "--device", "cpu", "--train-step", "16", "--models", "fold1"]) == 0
    unit = json.loads((out / "units" / "S1_seed0.json").read_text())
    assert set(unit["models"]) == {"fold1"}
    assert unit["models"]["fold1"]["n_train_windows"] == len(range(255, 3000, 16))
    assert unit["models"]["fold1"]["test_ema"]["n_instants"] == len(range(255, 3000, 32))  # test sempre a 16 ms
    assert (out / "predictions" / "S1_test_target.npy").is_file()
    with pytest.raises(SystemExit):
        temponet_db8.main(["--cache", str(cache), "--out", str(tmp_path / "x"), "--models", "fold1", "full", "--device", "cpu"])


def test_cache_tampering_is_detected(cache, tmp_path):
    arr = np.load(cache / "S1_A1_emg.npy")
    arr[0, 0] += 1.0
    np.save(cache / "S1_A1_emg.npy", arr)
    out = tmp_path / "run"
    assert temponet_db8.main(["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "--epochs", "1",
                              "--batch", "16", "--workers", "1", "--device", "cpu"]) == 1
    assert "sha256" in (out / "units" / "S1_seed0.error").read_text()


def test_stepper_eager_matches_a_plain_training_loop(cache):
    import argparse

    import torch

    from wearusfm.harness import db8_regression as db8

    cfg = argparse.Namespace(window=256, symmetric_padding=False, lr=1e-3, batch=8)
    rec = db8.load_cached(cache, 1, 1)
    sig = temponet_db8.Signal([rec], torch.device("cpu"), 1e6, 256, 32)
    batches = sig.ends[: 5 * 8].view(5, 8)
    m1, o1, l1 = temponet_db8.build(cfg, 3, torch.device("cpu"), dropout=0.0)
    st = temponet_db8.Stepper(m1, o1, l1, sig, 8, False, torch.device("cpu"))
    for b in batches:
        st.step(b)
    m2, o2, l2 = temponet_db8.build(cfg, 3, torch.device("cpu"), dropout=0.0)
    total = 0.0
    for b in batches:
        x, y = sig.batch(b)
        o2.zero_grad(set_to_none=True)
        loss = l2(m2(x), y)
        loss.backward()
        o2.step()
        total += float(loss.detach())
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    assert st.take_total() == pytest.approx(total) and st.take_total() == 0.0


def test_target_norm_trains_on_standardized_targets_and_predicts_degrees(cache, tmp_path, capsys):
    import torch

    from wearusfm.harness import db8_regression as db8

    out = tmp_path / "run_tn"
    assert temponet_db8.main(["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "--epochs", "1",
                              "--batch", "16", "--workers", "1", "--device", "cpu", "--models", "fold1", "fold2",
                              "--target-norm", "--hide-test-in-log"]) == 0
    log = capsys.readouterr().out
    assert "validazione MAE" in log and "test MAE" not in log
    sd = torch.load(out / "models" / "S1_seed0_fold1.pt")
    rec = db8.load_cached(cache, 1, 1)
    y = rec.doa[db8.window_ends(rec.n_samples)].astype(np.float64)
    np.testing.assert_allclose(sd["mu"].numpy(), y.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(sd["sd"].numpy(), y.std(axis=0, ddof=1), rtol=1e-3)
    unit = json.loads((out / "units" / "S1_seed0.json").read_text())
    assert set(unit["models"]) == {"fold1", "fold2"} and np.isfinite(unit["models"]["fold2"]["val_ema"]["mae_deg"])
    run = json.loads((out / "run.json").read_text())
    assert run["args"]["target_norm"] and run["args"]["hide_test_in_log"]


def test_with_targets_shares_signal_and_denormalizer():
    import torch

    from wearusfm.harness import db8_regression as db8

    rec = db8.Recording(np.zeros((600, 16), np.float32), np.arange(3000, dtype=np.float32).reshape(600, 5), "x")
    sig = temponet_db8.Signal([rec], torch.device("cpu"), 1.0, 256, 32)
    other = sig.with_targets(sig.doa * 0)
    assert other.emg is sig.emg and other.ends is sig.ends and float(other.doa.abs().sum()) == 0 and float(sig.doa.sum()) > 0
    net = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.eye_(net.weight)
    den = temponet_db8.denormalizer(net, torch.tensor([10.0, -5.0]), torch.tensor([2.0, 3.0]))
    assert torch.allclose(den(torch.tensor([[1.0, 1.0]])), torch.tensor([[12.0, -2.0]]))
    assert set(den.state_dict()) == {"net.weight", "mu", "sd"}


def _val_unit(subject, val):
    return {"subject": subject, "seed": 0, "models": {f: {"val_ema": {"mae_deg": val}} for f in ("fold1", "fold2")}}


@pytest.mark.parametrize("vals,choice", [
    ({"V1": 9.5, "V3": 9.8, "V13": 9.6}, "V1"),
    ({"V1": 9.8, "V3": 9.75, "V13": 9.9}, "V0"),
    ({"V1": 9.706, "V3": 9.9, "V13": 9.9}, "V1"),
])
def test_screen_choice_rule(tmp_path, vals, choice):
    for v, val in vals.items():
        (tmp_path / v / "units").mkdir(parents=True)
        for s in range(1, 13):
            (tmp_path / v / "units" / f"S{s}_seed0.json").write_text(json.dumps(_val_unit(s, val)))
    assert temponet_db8_screen.main([str(tmp_path)]) == 0
    s = json.loads((tmp_path / "screen.json").read_text())
    assert s["choice"] == choice and s["v0_val_mae_deg"] == 9.906


def test_screen_incomplete_makes_no_choice(tmp_path):
    for v in ("V1", "V3", "V13"):
        (tmp_path / v / "units").mkdir(parents=True)
        for s in range(1, 12):  # manca S12
            (tmp_path / v / "units" / f"S{s}_seed0.json").write_text(json.dumps(_val_unit(s, 9.0)))
    assert temponet_db8_screen.main([str(tmp_path)]) == 0
    assert json.loads((tmp_path / "screen.json").read_text())["choice"] is None


def _unit(subject, seed, fold_mae, full_mae=7.0):
    def model(mae):
        return {"test_ema": {"mae_deg": mae, "mae_per_doa_deg": [mae] * 5, "acc_10deg": 0.8, "acc_15deg": 0.9},
                "test_raw": {"mae_deg": mae + 1, "mae_per_doa_deg": [mae + 1] * 5}, "alpha": [0.86] * 5}
    return {"subject": subject, "seed": seed, "models": {"fold1": model(fold_mae), "fold2": model(fold_mae), "full": model(full_mae)}}


@pytest.mark.parametrize("mae,verdict", [(6.89, "replicata"), (7.19, "replicata"), (7.25, "non replicata"),
                                         (6.50, "non coerente (migliore del pubblicato)")])
def test_summary_verdicts(mae, verdict):
    units = [_unit(s, k, mae) for s in range(1, 13) for k in (0, 1, 2)]
    s = temponet_db8_summary.summarize(units, [0, 1, 2])
    assert s["complete"] and s["verdict"] == verdict
    assert s["primary_mae_deg"] == pytest.approx(mae) and s["full_mae_deg"] == pytest.approx(7.0)
    assert s["alpha_mean"] == pytest.approx(0.86) and s["seed_sd_deg"] == pytest.approx(0.0)


def test_summary_with_full_only_for_seed_0():
    units = []
    for s in range(1, 13):
        for k in (0, 1, 2):
            u = _unit(s, k, 7.0, full_mae=6.5)
            if k:
                del u["models"]["full"]
            units.append(u)
    s = temponet_db8_summary.summarize(units, [0, 1, 2])
    assert s["complete"] and s["full_seeds"] == [0] and s["full_n_units"] == 12
    assert s["full_mae_deg"] == pytest.approx(6.5) and s["fold_seed0_mae_deg"] == pytest.approx(7.0)
    assert "solo seme 0" in temponet_db8_summary.to_markdown(s)


def test_full_seeds_option_skips_full_for_other_seeds(cache, tmp_path):
    out = tmp_path / "run_fs"
    assert temponet_db8.main(["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "1", "--epochs", "1",
                              "--batch", "16", "--workers", "1", "--device", "cpu", "--full-seeds", "0"]) == 0
    u0 = json.loads((out / "units" / "S1_seed0.json").read_text())
    u1 = json.loads((out / "units" / "S1_seed1.json").read_text())
    assert set(u0["models"]) == {"fold1", "fold2", "full"} and set(u1["models"]) == {"fold1", "fold2"}


def test_summary_separates_able_bodied_and_amputees():
    units = [_unit(s, k, 9.0 if s >= 11 else 7.0, full_mae=8.0 if s >= 11 else 6.0) for s in range(1, 13) for k in (0, 1, 2)]
    s = temponet_db8_summary.summarize(units, [0, 1, 2])
    assert s["primary_mae_deg"] == pytest.approx((10 * 7.0 + 2 * 9.0) / 12)  # il criterio resta sui 12 soggetti
    g = s["groups"]
    assert g["normodotati"]["subjects"] == list(range(1, 11)) and g["amputati"]["subjects"] == [11, 12]
    assert g["normodotati"]["fold_mae_deg"] == pytest.approx(7.0) and g["amputati"]["fold_mae_deg"] == pytest.approx(9.0)
    assert g["normodotati"]["full_mae_deg"] == pytest.approx(6.0) and g["amputati"]["full_n_units"] == 6
    md = temponet_db8_summary.to_markdown(s)
    assert "| 11 (amputato) |" in md and "| normodotati | 1-10 |" in md and "| amputati | 11-12 |" in md


def test_summary_with_full_as_primary():
    units = [_unit(s, k, 8.8, full_mae=7.0) for s in range(1, 13) for k in (0, 1, 2)]
    s = temponet_db8_summary.summarize(units, [0, 1, 2], primary_model="full")
    assert s["criterion"]["primary_model"] == "full" and s["primary_mae_deg"] == pytest.approx(7.0) and s["verdict"] == "replicata"
    assert s["fold_mae_all_deg"] == pytest.approx(8.8)
    assert "modello full" in temponet_db8_summary.to_markdown(s)
    del units[5]["models"]["full"]  # full mancante per un'unita': incompleta
    assert temponet_db8_summary.summarize(units, [0, 1, 2], primary_model="full")["verdict"] == "incompleta"
    with pytest.raises(ValueError):
        temponet_db8_summary.summarize(units, [0, 1, 2], primary_model="altro")


def test_summary_full_primary_describes_the_full_model():
    """Con la primaria full ogni riga descrittiva dichiara il modello: fold e full con valori diversi per soggetto e seme."""
    units = []
    for s in range(1, 13):
        for k in (0, 1, 2):
            u = _unit(s, k, 8.0 + 0.1 * s, full_mae=6.0 + 0.1 * s + 0.01 * k)
            u["models"]["full"]["test_ema"].update(acc_10deg=0.7, acc_15deg=0.85)
            u["models"]["full"]["alpha"] = [0.93] * 5
            units.append(u)
    s = temponet_db8_summary.summarize(units, [0, 1, 2], primary_model="full")
    fd = s["full_details"]
    assert s["primary_mae_deg"] == pytest.approx(6.65 + 0.01) and s["fold_mae_all_deg"] == pytest.approx(8.65)
    assert fd["per_subject_mae_deg"]["12"] == pytest.approx(7.21) and s["per_subject_mae_deg"]["12"] == pytest.approx(9.2)
    assert fd["acc_10deg"] == pytest.approx(0.7) and s["acc_10deg"] == pytest.approx(0.8) and fd["alpha_mean"] == pytest.approx(0.93)
    assert list(s["full_per_seed_mae_deg"].values()) == pytest.approx([6.65, 6.66, 6.67]) and fd["seed_sd_deg"] == pytest.approx(0.01)
    assert s["groups"]["amputati"]["full_acc_15deg"] == pytest.approx(0.85)
    md = temponet_db8_summary.to_markdown(s)
    assert "- `full` per seme: 6.650, 6.660, 6.670 (dev. std. 0.010)" in md
    assert "- `full`, accuratezza a 10 e 15 gradi: 0.700 e 0.850" in md and "- Fold (secondaria): 8.650 con EMA" in md
    assert "| 12 (amputato) | 7.210 | 9.200 |" in md and "| rotazione pollice | 6.660 | 8.650 |" in md
    assert "| amputati | 11-12 | 7.160 | 0.700 / 0.850 | 9.150 |" in md
    fold_md = temponet_db8_summary.to_markdown(temponet_db8_summary.summarize(units, [0, 1, 2]))
    assert "- Per seme: 8.650, 8.650, 8.650" in fold_md and "`full` per seme" not in fold_md  # modo fold invariato


def test_summary_incomplete_when_a_seed_is_missing():
    units = [_unit(s, k, 6.9) for s in range(1, 13) for k in (0, 1)]
    assert temponet_db8_summary.summarize(units, [0, 1, 2])["verdict"] == "incompleta"


# Proposta H5 (docs/temponet_db8_h5_plan.md): schedule del learning rate


def test_lr_schedule_values():
    import math

    w1 = temponet_db8.LRSchedule("cosine", 1e-4, 19, 100)
    values = [w1.value(e, i) for e in range(19) for i in range(100)]
    assert values[0] == 1e-4 and 0 < values[-1] < 1e-9 and w1.total_epochs == 19
    assert all(a > b for a, b in zip(values, values[1:]))  # decresce a ogni passo
    assert w1.value(9, 50) == pytest.approx(5e-5)  # meta' dei passi: meta' del valore iniziale
    assert values[-1] == pytest.approx(1e-4 * 0.5 * (1 + math.cos(math.pi * 1899 / 1900)))
    w2 = temponet_db8.LRSchedule("constant", 1e-4, 19, 100, final_epochs=1, final_lr=1e-5)
    assert w2.total_epochs == 20 and w2.value(0, 0) == w2.value(18, 99) == 1e-4 and w2.value(19, 0) == w2.value(19, 99) == 1e-5
    with pytest.raises(ValueError):
        temponet_db8.LRSchedule("constant", 1e-4, 19, 100, final_epochs=1)
    with pytest.raises(ValueError):
        temponet_db8.LRSchedule("step", 1e-4, 19, 100)


def test_check_lrs_compresses_the_schedule_into_the_check():
    import argparse

    base = {"lr": 1e-4, "epochs": 19}
    assert temponet_db8.check_lrs(argparse.Namespace(**base, lr_schedule="constant", final_epochs=0, final_lr=None), 60) is None
    w1 = temponet_db8.check_lrs(argparse.Namespace(**base, lr_schedule="cosine", final_epochs=0, final_lr=None), 60)
    assert len(w1) == 60 == len(set(w1)) and w1[0] == 1e-4 and w1[-1] < 1e-6
    w2 = temponet_db8.check_lrs(argparse.Namespace(**base, lr_schedule="constant", final_epochs=1, final_lr=1e-5), 60)
    assert w2 == [1e-4] * 30 + [1e-5] * 30  # il cambio cade dopo la cattura del graph (passo GRAPH_WARMUP)
    assert temponet_db8.GRAPH_WARMUP < 30


def test_tensor_lr_matches_the_float_path_and_follows_the_schedule(cache):
    import argparse

    import torch

    from wearusfm.harness import db8_regression as db8

    cpu = torch.device("cpu")
    sig = temponet_db8.Signal([db8.load_cached(cache, 1, 1)], cpu, 1e6, 256, 32)
    batches = sig.ends[: 6 * 8].view(6, 8)
    base = argparse.Namespace(window=256, symmetric_padding=False, lr=1e-3, batch=8)

    def run(cfg, lrs=None):
        model, opt, loss_fn = temponet_db8.build(cfg, 3, cpu, dropout=0.0)
        st = temponet_db8.Stepper(model, opt, loss_fn, sig, 8, False, cpu)
        for i, b in enumerate(batches):
            st.step(b, None if lrs is None else lrs[i])
        return model, opt, st

    def same(m1, m2):
        return all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))

    m_float, o_float, st_float = run(base)
    assert isinstance(o_float.param_groups[0]["lr"], float)  # senza schedule: numero float come prima
    with pytest.raises(ValueError):
        st_float.step(batches[0], 1e-3)
    # valori costanti col tensore: stessi pesi del numero float, bit per bit
    m_const, o_const, _ = run(argparse.Namespace(**vars(base), lr_schedule="constant", final_epochs=1, final_lr=1e-3), [1e-3] * 6)
    assert torch.is_tensor(o_const.param_groups[0]["lr"]) and same(m_float, m_const)
    # coseno: il tensore riscritto prima di ogni passo equivale al numero float impostato a mano prima di ogni passo
    sched = temponet_db8.LRSchedule("cosine", 1e-3, 1, 6)
    lrs = [sched.value(0, i) for i in range(6)]
    m_cos, o_cos, _ = run(argparse.Namespace(**vars(base), lr_schedule="cosine", final_epochs=0, final_lr=None), lrs)
    m_ref, o_ref, l_ref = temponet_db8.build(base, 3, cpu, dropout=0.0)
    for lr, b in zip(lrs, batches):
        o_ref.param_groups[0]["lr"] = lr
        x, y = sig.batch(b)
        o_ref.zero_grad(set_to_none=True)
        l_ref(m_ref(x), y).backward()
        o_ref.step()
    assert same(m_cos, m_ref) and not same(m_cos, m_float) and float(o_cos.param_groups[0]["lr"]) == lrs[-1]


@pytest.mark.parametrize("extra", [["--lr-schedule", "cosine"], ["--final-epochs", "1", "--final-lr", "1e-5"]])
def test_training_with_lr_schedules_end_to_end(cache, tmp_path, extra, capsys):
    import math

    out = tmp_path / "run_lr"
    assert temponet_db8.main(["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "--epochs", "2", "--batch", "16",
                              "--workers", "1", "--device", "cpu", "--models", "fold1", "fold2", "--target-norm", "--symmetric-padding",
                              "--hide-test-in-log", *extra]) == 0
    assert "test MAE" not in capsys.readouterr().out
    unit = json.loads((out / "units" / "S1_seed0.json").read_text())
    for name in ("fold1", "fold2"):
        h = unit["models"][name]["history"]
        steps = h[0]["steps"]
        if extra[0] == "--lr-schedule":  # W1: due epoche, dal valore iniziale a ~0
            assert len(h) == 2 and h[0]["lr_first"] == 1e-4 and h[0]["lr_first"] > h[0]["lr_last"] > h[1]["lr_first"] > h[1]["lr_last"]
            assert h[1]["lr_last"] == pytest.approx(1e-4 * 0.5 * (1 + math.cos(math.pi * (2 * steps - 1) / (2 * steps))))
        else:  # W2: le due epoche a 1e-4 e una in piu' a 1e-5
            assert [(r["lr_first"], r["lr_last"]) for r in h] == [(1e-4, 1e-4), (1e-4, 1e-4), (1e-5, 1e-5)]
        assert np.isfinite(unit["models"][name]["val_ema"]["mae_deg"])
    args = json.loads((out / "run.json").read_text())["args"]
    assert args["target_norm"] and args["symmetric_padding"]


def test_final_epochs_and_final_lr_go_together(cache, tmp_path):
    for k, extra in enumerate((["--final-epochs", "1"], ["--final-lr", "1e-5"], ["--final-epochs", "-1", "--final-lr", "1e-5"])):
        with pytest.raises(SystemExit):
            temponet_db8.main(["--cache", str(cache), "--out", str(tmp_path / f"x{k}"), "--device", "cpu", *extra])


@pytest.mark.parametrize("vals,choice", [
    ({"W1": 9.40, "W2": 9.50}, "W1"),
    ({"W1": 9.45, "W2": 9.44}, "V13"),
    ({"W1": 9.50, "W2": 9.428}, "W2"),  # guadagno di 0,20 esatti: in float 0,1999999999999993, ammesso
    ({"W1": 9.30, "W2": 9.30}, "W1"),  # parita' esatta: vale l'ordine W1, W2
])
def test_screen_h5_rule(tmp_path, vals, choice):
    assert 9.628 - 9.428 < 0.20  # senza la tolleranza di arrotondamento il terzo caso sarebbe respinto
    for v, val in vals.items():
        (tmp_path / v / "units").mkdir(parents=True)
        for s in range(1, 13):
            (tmp_path / v / "units" / f"S{s}_seed0.json").write_text(json.dumps(_val_unit(s, val)))
    assert temponet_db8_screen.main([str(tmp_path), "--rule", "h5"]) == 0
    s = json.loads((tmp_path / "screen.json").read_text())
    assert (s["rule"], s["base"], s["base_val_mae_deg"], s["choice"]) == ("h5", "V13", 9.628, choice)
    md = (tmp_path / "screen.md").read_text()
    assert "Base V13 (media di 9,590, job 64005316, e 9,666, job 64010753, seme 0): 9.628 gradi." in md
    assert ("la replica si chiude a 7,585 e il passo B non si fa" in md) == (choice == "V13")


def test_screen_h5_incomplete_or_missing_makes_no_choice(tmp_path):
    for v in ("W1", "W2"):
        (tmp_path / v / "units").mkdir(parents=True)
        for s in range(1, 13 if v == "W1" else 12):  # a W2 manca S12
            (tmp_path / v / "units" / f"S{s}_seed0.json").write_text(json.dumps(_val_unit(s, 9.0)))
    assert temponet_db8_screen.main([str(tmp_path), "--rule", "h5"]) == 0
    assert json.loads((tmp_path / "screen.json").read_text())["choice"] is None
    other = tmp_path / "solo_W1"
    (other / "W1" / "units").mkdir(parents=True)
    assert temponet_db8_screen.main([str(other), "--rule", "h5"]) == 0
    assert json.loads((other / "screen.json").read_text())["choice"] is None
