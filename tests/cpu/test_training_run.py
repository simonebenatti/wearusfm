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
