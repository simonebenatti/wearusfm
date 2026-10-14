"""Ripresa con la stessa ricetta (review GPT del 14/10/2026): seme, batch, accumulo, loader e dataset diversi dal checkpoint si rifiutano
prima di creare modelli o scrivere output; si ammettono solo le variazioni operative dichiarate (estendere max_steps, registrare l'origine
dei batch). Gli ingressi dello stream (manifest, scale, durate) stanno nel checkpoint e si verificano alla ripresa; un checkpoint di prima,
senza la firma, si riprende con un avviso. Pannello fisso: l'identita' contiene l'hash del contenuto dei batch."""

import json
from dataclasses import replace

import pytest
import torch

from wearusfm.training import run as R

from test_spectral_step1 import _setup


def _cfg(tmp_path):
    root, manifest, cfg, _, _, _ = _setup(tmp_path)
    cfg = replace(cfg, model=replace(cfg.model, keep_readout=False), max_steps=1)  # senza la testa dello Step 1: la guardia generale
    return root, manifest, cfg


@pytest.mark.parametrize("change", [dict(seed=1), dict(batch_size=4), dict(lr=1e-3), dict(eval_every=3)])
def test_resume_with_another_recipe_is_refused_before_writing(tmp_path, change):
    root, manifest, cfg = _cfg(tmp_path)
    out = tmp_path / "run"
    R.train(cfg, manifest, [root], out, log=lambda _: None)
    before = (out / "metrics.jsonl").read_text()
    with pytest.raises(ValueError, match="ricetta diversa"):
        R.train(replace(cfg, max_steps=2, **change), manifest, [root], out, log=lambda _: None)
    assert (out / "metrics.jsonl").read_text() == before  # nessuna scrittura


def test_resume_with_another_loader_or_dataset_is_refused(tmp_path):
    root, manifest, cfg = _cfg(tmp_path)
    out = tmp_path / "run"
    R.train(cfg, manifest, [root], out, log=lambda _: None)
    for c in (replace(cfg, loader=replace(cfg.loader, k_neighbors=4)), replace(cfg, datasets=("ninapro_db2", "emg2pose"))):
        with pytest.raises(ValueError, match="ricetta diversa"):
            R.train(replace(c, max_steps=2), manifest, [root], out, log=lambda _: None)


def test_operational_changes_are_allowed_and_the_resume_continues(tmp_path):
    root, manifest, cfg = _cfg(tmp_path)
    out = tmp_path / "run"
    R.train(cfg, manifest, [root], out, log=lambda _: None)
    summary = R.train(replace(cfg, max_steps=2, log_batch_origin=True), manifest, [root], out, log=lambda _: None)
    assert summary["steps"] == 2
    recs = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [1, 2] and "origin" in recs[1] and "origin" not in recs[0]


def test_data_inputs_are_recorded_and_checked_at_resume(tmp_path):
    root, manifest, cfg = _cfg(tmp_path)
    out = tmp_path / "run"
    R.train(cfg, manifest, [root], out, log=lambda _: None, scales={})
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    assert state["data_signature"]["manifest_sha256"] == R.sha256_file(manifest)
    with pytest.raises(ValueError, match="dati diversi.*scales_sha256"):
        R.train(replace(cfg, max_steps=2), manifest, [root], out, log=lambda _: None, scales={"ninapro_db2/s01/session1": 2.0})
    # un checkpoint di prima, senza la firma dei dati, si riprende con un avviso
    del state["data_signature"]
    torch.save(state, out / "checkpoint.pt")
    said = []
    assert R.train(replace(cfg, max_steps=2), manifest, [root], out, log=said.append, scales={})["steps"] == 2
    assert any("senza la firma dei dati" in m for m in said)


def test_fixed_panel_identity_contains_the_content(tmp_path):
    """Stesse finestre, scale diverse: identificativi uguali, hash del contenuto diversi (la ripresa lo rifiuta)."""
    root, manifest, cfg = _cfg(tmp_path)
    cfg = replace(cfg, fixed_eval_batches=2)
    index = R.load_index(manifest, [root], cfg.datasets)
    _, a = R._fixed_eval_set(index, cfg, {}, None)
    _, b = R._fixed_eval_set(index, cfg, {"ninapro_db2/s01/session1": 2.0}, None)
    assert a["windows"] == b["windows"] and a["batch_sha256"] != b["batch_sha256"]
    path = tmp_path / "fixed_eval_set.json"
    R._write_or_check(path, a, "insieme fisso")
    with pytest.raises(ValueError, match="insieme fisso"):
        R._write_or_check(path, b, "insieme fisso")


def test_pooled_metrics_use_counts_and_skip_absent_strata():
    """Review GPT del 14/10: uno strato presente in un solo batch valeva la meta' (diviso per tutti i batch). Ora: numeratori e conteggi."""
    parts = [{"num": {"jepa": 2.0, "log_rms": 1.0}, "den": {"jepa": 4.0, "log_rms": 2.0},
              "log_num": {"jepa_topo_grid_2d": 6.0, "jepa_kind_short": 1.0}, "log_den": {"jepa_topo_grid_2d": 3.0, "jepa_kind_short": 1.0},
              "log_sum": {"jepa_targets": 4.0}},
             {"num": {"jepa": 6.0, "log_rms": 0.0}, "den": {"jepa": 4.0, "log_rms": 0.0},
              "log_num": {"jepa_kind_short": 3.0}, "log_den": {"jepa_kind_short": 1.0}, "log_sum": {"jepa_targets": 4.0}}]
    out = R._pool_terms(parts, {"jepa": 1.0, "log_rms": 0.2})
    assert out["jepa"] == pytest.approx(8.0 / 8.0) and out["log_rms"] == pytest.approx(0.5)  # log_rms solo dal primo batch
    assert out["total"] == pytest.approx(1.0 + 0.2 * 0.5)
    assert out["jepa_topo_grid_2d"] == pytest.approx(2.0)  # lo strato c'e' solo nel primo: non 1
    assert out["jepa_kind_short"] == pytest.approx(2.0) and out["jepa_targets"] == 8.0 and out["jepa_count"] == 8.0
    empty = R._pool_terms([{"num": {"jepa": 1.0, "log_rms": 0.0}, "den": {"jepa": 1.0, "log_rms": 0.0}}], {"jepa": 1.0, "log_rms": 0.2})
    assert "log_rms" not in empty and empty["total"] == pytest.approx(1.0)  # nessun target: assente, non zero
