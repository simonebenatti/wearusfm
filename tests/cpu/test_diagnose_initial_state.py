"""Diagnostica dello stato iniziale per la baseline casuale (`scripts/diagnose_initial_state.py`): ricostruzioni in processi nuovi con
ambienti diversi, confronto con gli SHA iniziali di `pairing.json`, tensori diversi fra ricostruzioni, buffer diversi dal teacher finale."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from wearusfm.training import run as R

from test_spectral_step1 import _calibration, _setup

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import diagnose_initial_state as D  # noqa: E402


def _ab_runs(tmp_path):
    """Due bracci «A/B» minuscoli, stesso seme (stessa inizializzazione), con l'audit dell'accoppiamento (pairing.json)."""
    root, manifest, cfg, _, y, m = _setup(tmp_path)
    cal = tmp_path / "cal.npz"
    _calibration(cal, cfg, manifest, y, m)
    runs = tmp_path / "runs"
    for arm, w in (("A", 0.0), ("B", 0.05)):
        R.train(replace(R.with_spectral_keep(cfg, cal, w), pairing_audit=True), manifest, [root], runs / f"{arm}_seed0", log=lambda _: None)
    return runs


def test_the_trainer_replica_reproduces_the_recorded_initial_sha(tmp_path):
    runs = _ab_runs(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    report = D.diagnose(runs, [0], work, ("ab_replica", "torch_only_env1", "gpt_like_env8"))
    row = report["seeds"]["0"]
    expected = row["pairing_initial_sha256"]
    assert expected["A_teacher"] == expected["B_teacher"] == expected["A_student"]  # A e B accoppiati: stesso stato iniziale
    for v in ("ab_replica", "torch_only_env1", "gpt_like_env8"):
        r = row["variants"][v]
        assert "error" not in r, r
        assert set(r["matches"]) == set(expected), v  # sulla stessa macchina e con un thread, tutte riproducono lo SHA
        assert r["buffers_differing_from_final_teacher"] == {}
    assert row["variants"]["torch_only_env1"]["tensors_differing_from_ab_replica"] == {}
    env = row["variants"]["ab_replica"]["environment"]
    assert env["torch_threads"] == 1 and env["env"]["OMP_NUM_THREADS"] == "1"
    assert row["variants"]["gpt_like_env8"]["environment"]["env"]["OMP_NUM_THREADS"] == "8"


def test_a_buffer_changed_after_initialization_is_located(tmp_path):
    """Un buffer immutabile del teacher finale diverso dalla ricostruzione (es. il guadagno fisso del front-end) si vede per nome."""
    runs = _ab_runs(tmp_path)
    path = runs / "B_seed0" / "checkpoint.pt"
    state = torch.load(path, weights_only=False)
    state["teacher"]["tokenizer.fe_gain"] = state["teacher"]["tokenizer.fe_gain"] * (1 + 1e-12)
    torch.save(state, path)
    work = tmp_path / "work"
    work.mkdir()
    r = D.diagnose(runs, [0], work, ("ab_replica",))["seeds"]["0"]["variants"]["ab_replica"]
    assert list(r["buffers_differing_from_final_teacher"]) == ["tokenizer.fe_gain"]


def test_cli_writes_a_new_report(tmp_path):
    runs = _ab_runs(tmp_path)
    out = tmp_path / "diag" / "report.json"
    assert D.main(["--runs-root", str(runs), "--seeds", "0", "--variants", "ab_replica", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["seeds"]["0"]["variants"]["ab_replica"]["matches"]
    import pytest
    with pytest.raises(SystemExit):
        D.main(["--runs-root", str(runs), "--seeds", "0", "--out", str(out)])  # niente sovrascritture
