"""Regressioni GPT: pilot 134b77c. WEARUSFM_REVIEW_SOURCE = snapshot da verificare."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

SOURCE = Path(os.environ["WEARUSFM_REVIEW_SOURCE"])
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "tests/cpu")]

from wearusfm.model import branch_masking as BM
from wearusfm.training import run as R
from test_branch_pilot_trainer import _source_and_cfg


def test_missing_monitor_sample_breaks_consecutive_low_checks():
    alarm = BM.CollapseAlarm()
    key = "masked_wavelet_slow_variance"
    for step in range(1, 11):
        alarm.update(step, {key: 1.0})
    assert alarm.update(110, {key: 0.001}) is None
    assert alarm.update(120, {key: 0.001}) is None
    # Scala senza bersagli/contesto al controllo successivo: non e' un terzo basso,
    # e interrompe la sequenza di controlli consecutivi del protocollo.
    assert alarm.update(130, {"masked_wavelet_slow_targets": 0}) is None
    assert alarm.update(140, {key: 0.001}) is None


def test_undefined_normalized_metric_does_not_crash_progress_logging(tmp_path, monkeypatch):
    root, manifest, cfg, src = _source_and_cfg(tmp_path)
    cfg = replace(cfg, max_steps=1, model=replace(cfg.model, branch_objective="masked", branch_freeze_output=True))
    original = BM.masked_branch_objective

    def objective(*args, **kwargs):
        loss, metrics = original(*args, **kwargs)
        # Valore previsto dall'obiettivo se il riferimento costante ha varianza zero.
        metrics["masked_normalized"] = float("nan")
        return loss, metrics

    monkeypatch.setattr(BM, "masked_branch_objective", objective)
    out = tmp_path / "undefined"
    summary = R.train(cfg, manifest, [root], out, log=lambda _: None, init_from=src)
    assert summary["steps"] == 1 or summary["stopped"].startswith("allarme")
    assert (out / "summary.json").exists()


def test_pwf_collects_branch_collapse_reference(tmp_path):
    root, manifest, cfg, src = _source_and_cfg(tmp_path)
    cfg = replace(cfg, max_steps=1, model=replace(cfg.model, branch_freeze_output=True))
    out = tmp_path / "pwf"
    summary = R.train(cfg, manifest, [root], out, log=lambda _: None, init_from=src)
    assert summary["spectral_branches"]["collapse_reference"], "PWf non osserva il collasso dei rami"
    rec = json.loads((out / "metrics.jsonl").read_text().splitlines()[0])
    assert any("variance" in k for k in rec)
