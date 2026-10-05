"""Ciclo di training su piu' passi (sanity JEPA): passi veri dal dataloader, diagnostiche, allarmi, checkpoint e ripresa, limite di tempo.
Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from wearusfm.training import run as R  # noqa: E402

from test_pretraining_loader import _tree  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_jepa.py"


def _lines(p):
    return [json.loads(x) for x in (p / "metrics.jsonl").read_text().splitlines()]


def test_steps_diagnostics_checkpoint_and_resume(tmp_path):
    root, mpath = _tree(tmp_path)
    out = tmp_path / "run"
    s = R.train(R.small_config(datasets=None), mpath, [root], out, log=lambda m: None)
    assert s["stopped"] == "max_steps" and s["steps"] == 4 and (out / "checkpoint.pt").exists()
    rec = _lines(out)
    assert [r["step"] for r in rec] == [1, 2, 3, 4] and all(math.isfinite(r["total"]) for r in rec)
    assert "student_collapse_ratio" in rec[1] and "teacher_erank" in rec[3] and "student_erank" not in rec[0]
    assert rec[0]["lr"] < rec[1]["lr"] <= 3e-4  # warmup
    s2 = R.train(R.small_config(datasets=None, max_steps=6), mpath, [root], out, log=lambda m: None)
    assert s2["steps"] == 6 and [r["step"] for r in _lines(out)] == [1, 2, 3, 4, 5, 6]  # ripresa dal passo 4
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    assert state["step"] == 6 and json.loads((out / "config.json").read_text())["jepa"]["target"] == "b"


def test_time_limit_stops_and_saves(tmp_path):
    root, mpath = _tree(tmp_path)
    s = R.train(R.small_config(datasets=None), mpath, [root], tmp_path / "r", time_limit_s=0.0, log=lambda m: None)
    assert s["stopped"] == "limite di tempo" and s["steps"] == 0 and (tmp_path / "r" / "checkpoint.pt").exists()


def test_alarm_rule_and_dataset_filter(tmp_path):
    mon = R.AlarmMonitor(R.AlarmRule(), dim=100)
    assert mon.update(0.01, 50) is None and mon.update(0.01, 50) is None and "collasso" in mon.update(0.01, 50)  # 3 di fila
    mon = R.AlarmMonitor(R.AlarmRule(), dim=100)
    assert mon.update(0.01, 50) is None and mon.update(0.5, 50) is None and mon.update(0.01, 50) is None  # azzerato dal recupero
    mon = R.AlarmMonitor(R.AlarmRule(), dim=100)
    assert mon.update(float("nan"), 5) is None and mon.update(0.5, 5) is None and "rango" in mon.update(0.5, 5)  # 10% di 100 = 10
    root, mpath = _tree(tmp_path)
    idx = R.load_index(mpath, [root], ("emg2pose",))
    assert [r["dataset"] for r in idx.rows] == ["emg2pose"] and idx.weights.tolist() == [1.0]
    with pytest.raises(ValueError):
        R.load_index(mpath, [root], ("emg2qwerty",))
    cfg = R.sanity_config()
    assert (cfg.jepa.target, cfg.jepa.anchor_weight, cfg.jepa.ema_momentum, cfg.model.muscle_dropout, cfg.loader.mask.ratio, cfg.datasets,
            cfg.eval_every, cfg.alarm.collapse_ratio_min, cfg.alarm.erank_fraction_min, cfg.alarm.patience) == \
        ("b", 0.2, 0.996, 0.4, 0.5, ("emg2qwerty",), 500, 0.05, 0.10, 3)  # i valori firmati il 03/10/2026


def test_cli_small_preset(tmp_path):
    root, mpath = _tree(tmp_path)
    r = subprocess.run([sys.executable, str(SCRIPT), "--preset", "small", "--manifest", str(mpath), "--root", str(root), "--out-dir",
                        str(tmp_path / "cli"), "--max-steps", "2", "--device", "cpu", "--datasets", "all"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((tmp_path / "cli" / "summary.json").read_text())["steps"] == 2


def test_alarm_writes_a_stop_that_later_jobs_respect(tmp_path):
    root, mpath = _tree(tmp_path)
    out = tmp_path / "run"
    cfg = R.small_config(datasets=None, alarm=R.AlarmRule(collapse_ratio_min=1e9, patience=1))  # allarme alla prima valutazione
    s = R.train(cfg, mpath, [root], out, log=lambda m: None)
    assert s["stopped"].startswith("allarme") and s["steps"] == 2 and (out / "STOP").exists()
    s2 = R.train(R.small_config(datasets=None, max_steps=6), mpath, [root], out, log=lambda m: None)
    assert s2["stopped"].startswith("fermato in precedenza") and s2["steps"] is None
    assert [r["step"] for r in _lines(out)] == [1, 2]  # nessun passo dopo l'allarme
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    assert "torch_rng" in state and state["torch_rng"].dtype == torch.uint8


def test_rvq_anchor_on_trains_with_precomputed_codes_and_needs_both_halves(tmp_path):
    import numpy as np

    from wearusfm.data import pretraining_loader as L
    from wearusfm.data import rvq_codes as RC
    from wearusfm.model.anchors import rvq_loss

    from test_rvq_codes import _session, fake_encode

    root, mpath = _tree(tmp_path)
    codes_root = tmp_path / "codes"
    for row in L.ManifestIndex.load(mpath, [root]).rows:
        s = _session(root, row)
        trials, _ = RC.canonical_trials(s.segments, s.fs, s.qc_valid)
        p = RC.code_path(codes_root, row["dataset"], row["subject"], row["session"])
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **RC.session_codes(trials, s.qc_valid, fake_encode))
    cfg = R.with_rvq(R.small_config(datasets=None, max_steps=3), codes_root)
    s = R.train(cfg, mpath, [root], tmp_path / "run", log=lambda m: None)
    rec = _lines(tmp_path / "run")
    assert s["steps"] == 3 and all("rvq" in r and math.isfinite(r["rvq"]) for r in rec)
    assert sum(r["rvq_targets"] for r in rec) > 0  # gli slab hanno davvero dei codici
    with pytest.raises(ValueError, match="ancora RVQ a meta'"):
        from dataclasses import replace
        R.train(replace(cfg, model=replace(cfg.model, rvq_codes=None)), mpath, [root], tmp_path / "r2", log=lambda m: None)
    logits = torch.zeros(4, 8192, requires_grad=True)
    loss = rvq_loss(logits, torch.tensor([3, -1, 5, -1]))
    assert float(loss.detach()) == pytest.approx(1.0, abs=1e-6)  # uniforme: ln(8192) / ln(8192); i -1 non contano
    assert float(rvq_loss(logits, torch.tensor([-1, -1, -1, -1])).detach()) == 0.0


def test_loss_per_window_averages_inside_each_window_first():
    """Decisione 2 del 04/10/2026: con la media per token una finestra con molti token (es. HD) pesa di piu'; per finestra pesano uguale."""
    from wearusfm.model.anchors import masked_mean

    num = torch.tensor([1.0, 1.0, 1.0, 5.0])  # campione 0: tre query con errore 1; campione 1: una query con errore 5
    den = torch.ones(4)
    s = torch.tensor([0, 0, 0, 1])
    assert float(masked_mean(num, den)) == pytest.approx(2.0)  # per token: (1+1+1+5)/4
    assert float(masked_mean(num, den, s, 2)) == pytest.approx(3.0)  # per finestra: (1 + 5)/2
    assert float(masked_mean(num, torch.tensor([1.0, 1.0, 1.0, 0.0]), s, 3)) == pytest.approx(1.0)  # campioni senza elementi validi: esclusi


def test_window1_rules_train_end_to_end(tmp_path):
    import numpy as np

    from wearusfm.data import pretraining_loader as L

    root, mpath = _tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    cfg = R.with_window1_rules(R.small_config(datasets=None, max_steps=2))
    assert cfg.jepa.loss_per_window and cfg.loader.time_weighted and cfg.loader.anchor_edge_guard_s == 0.1
    assert cfg.loader.virtual is not None and cfg.loader.virtual.p == 0.5 and cfg.loader.virtual.classes == ("C",)  # D6a, firmato il 05/10
    from dataclasses import replace

    base = L.PretrainLoader(idx, replace(cfg.loader, time_weighted=False))  # solo per aprire le sessioni
    ws = {f"{r['dataset']}/{r['subject']}/{r['session']}": L.mean_window_s(base.view(r), cfg.loader) for r in idx.rows}
    s = R.train(cfg, mpath, [root], tmp_path / "run", log=lambda m: None, window_s=ws)
    assert s["steps"] == 2 and all(math.isfinite(r["total"]) for r in _lines(tmp_path / "run"))
    assert all("ms_fast" in r and "ms_slow" in r for r in _lines(tmp_path / "run"))  # ancora multi-scala accesa nella finestra 1
    assert "rvq" not in _lines(tmp_path / "run")[0]  # e niente RVQ
    sanity = R.sanity_config()
    assert not sanity.jepa.loss_per_window and not sanity.loader.time_weighted and sanity.loader.anchor_edge_guard_s == 0.0  # sanity invariato
    assert sanity.loader.virtual is None
    assert np.isfinite(list(ws.values())).all()


def test_init_from_another_run_takes_weights_but_restarts_steps(tmp_path):
    root, mpath = _tree(tmp_path)
    R.train(R.small_config(datasets=None, max_steps=2), mpath, [root], tmp_path / "a", log=lambda m: None)
    src = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    from dataclasses import replace

    cfg = R.small_config(datasets=None, max_steps=1)
    cfg = replace(cfg, jepa=replace(cfg.jepa, rvq_targets="channel"), lr=0.0)  # lr nullo: i pesi restano quelli caricati
    s = R.train(cfg, mpath, [root], tmp_path / "b", log=lambda m: None, init_from=tmp_path / "a" / "checkpoint.pt")
    dst = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    assert s["steps"] == 1 and dst["step"] == 1
    assert all(torch.equal(src["student"][k], dst["student"][k]) for k in src["student"])


def test_anchor_candidates_extract_and_probe_end_to_end(tmp_path, monkeypatch):
    """Il percorso del confronto dei target candidati (scripts/anchor_candidates.py), fase di estrazione, con un finto tokenizer sull'albero di
    prova. Le sonde sono in test_anchor_candidates.py (senza torch: nel venv di torch sklearn va in conflitto con OpenMP)."""
    import importlib.util

    import numpy as np

    from wearusfm.data import pretraining_loader as L
    from wearusfm.data import rvq_codes as RC
    from wearusfm.tokenizer_checks import neurorvq as NR

    from test_rvq_codes import _session, fake_encode

    spec = importlib.util.spec_from_file_location("anchor_candidates_script", Path(__file__).resolve().parents[2] / "scripts" / "anchor_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root, mpath = _tree(tmp_path)
    codes_root = tmp_path / "codes"
    for row in L.ManifestIndex.load(mpath, [root]).rows:
        s = _session(root, row)
        trials, scale = RC.canonical_trials(s.segments, s.fs, s.qc_valid)
        p = RC.code_path(codes_root, row["dataset"], row["subject"], row["session"])
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, scale=np.float64(scale), **RC.session_codes(trials, s.qc_valid, fake_encode))
    R.train(R.with_rvq(R.small_config(datasets=None, max_steps=1), codes_root), mpath, [root], tmp_path / "run", log=lambda m: None)
    tok = tmp_path / "tok.pt"
    torch.save({"quantize_1.layers.0.embedding.weight": torch.randn(8192, 128)}, tok)

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def features(self, x, idx):
            b, _, n = x.shape
            t = n // 200
            return np.tile(x.reshape(b * t, 200)[:, :128][None], (4, 1, 1)).astype(np.float32)

    monkeypatch.setattr(NR, "NeuroRVQRunner", FakeRunner)
    scales = tmp_path / "scales.json"
    scales.write_text(json.dumps({"scales": {}}))
    out = tmp_path / "units.npz"
    assert mod.main(["extract", "--manifest", str(mpath), "--root", str(root), "--scales", str(scales), "--rvq-codes", str(codes_root),
                     "--checkpoint", str(tmp_path / "run" / "checkpoint.pt"), "--repo-dir", str(tmp_path), "--tokenizer-checkpoint", str(tok),
                     "--out", str(out), "--dataset", "emg2pose", "--n-users", "1", "--windows-per-user", "4", "--batch", "2",
                     "--device", "cpu", "--keep", "1.0"]) == 0
    z = np.load(out)
    n = len(z["code"])
    assert n > 0 and z["V"].shape == (n, 16) and z["c16"].shape == (n, 16) and z["ms"].shape == (n, 48) and z["nrvq"].shape == (n, 128)
    assert z["h_mask"].any() and (z["code"] >= 0).all()
