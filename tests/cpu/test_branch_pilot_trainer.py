"""Pilot dei rami nel trainer (protocollo `rami_pilota_v1`): obiettivo mascherato a encoder congelato, strati finali fissi fuori da
ottimizzatore ed EMA, passo 0 dei rami caricato da un `checkpoint_init.pt` (come quello del 2C), allarme di collasso dei rami; il percorso del
2C (obiettivo vecchio, nessuno strato fisso) resta quello di prima (lo provano i test esistenti in `test_spectral_branches.py`)."""

import json
from dataclasses import replace

import pytest
import torch

from wearusfm.model import branch_masking as BM
from wearusfm.model import spectral_branches as SB
from wearusfm.training import run as R

from test_spectral_branches import _branch_cfg

FIXED = (".output.", ".combine.")


def _source_and_cfg(tmp_path):
    """Un teacher senza rami (l'encoder da congelare) e la configurazione dei rami a encoder congelato (come il 2C)."""
    root, manifest, cfg_b = _branch_cfg(tmp_path, freeze_encoder=True, keep=0.0)
    src_cfg = R.with_spectral_keep(replace(cfg_b, model=replace(cfg_b.model, spectral_branches=()), freeze_encoder=False,
                                           jepa=replace(cfg_b.jepa, branch_weight=0.0)), cfg_b.keep_calibration, 0.0)
    src = tmp_path / "src"
    R.train(src_cfg, manifest, [root], src, log=lambda _: None)
    return root, manifest, cfg_b, src / "checkpoint.pt"


def _fixed_keys(state, objective):
    keys = [k for k in state if k.startswith("spectral_branches.") and (any(f in k for f in FIXED) or not k.endswith(("weight", "bias")))]
    if objective == "masked":
        keys += [k for k in state if k.startswith("branch_heads.")]
    return keys


def test_masked_objective_trains_only_tokens_and_predictors(tmp_path):
    root, manifest, cfg_b, src = _source_and_cfg(tmp_path)
    cfg = replace(cfg_b, max_steps=3, model=replace(cfg_b.model, branch_objective="masked", branch_freeze_output=True))
    out = tmp_path / "masked"
    summary = R.train(cfg, manifest, [root], out, log=lambda _: None, init_from=src)
    assert summary["steps"] == 3 and summary["stopped"] == "max_steps"
    sb = summary["spectral_branches"]
    assert sb["objective"] == "masked" and sb["freeze_output"] and sb["collapse_reference"]
    init = torch.load(out / "checkpoint_init.pt", weights_only=False)
    final = torch.load(out / "checkpoint.pt", weights_only=False)
    for part in ("student", "teacher"):  # strati fissi, statistiche d'ingresso e teste: uguali al passo 0 bit per bit, anche nel teacher
        assert all(torch.equal(init[part][k], final[part][k]) for k in _fixed_keys(init[part], "masked"))
        tokens = [k for k in init[part] if k.startswith("spectral_branches.") and (".project." in k or ".temporal." in k)]
        assert tokens and any(not torch.equal(init[part][k], final[part][k]) for k in tokens)  # i token si allenano (studente ed EMA)
    assert any(k.startswith("branch_predictors.") for k in final["student"])
    recs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert all("masked" in r and r["masked_normalized"] is None or r["masked_normalized"] > 0 for r in recs)
    assert any(k.endswith("_context_per_band") for k in recs[0]) and "keep_branches" not in recs[0]
    # parametri allenati: solo token dei rami e predittori
    from wearusfm.model.fm import WearUsFM

    m = WearUsFM(cfg.model)
    SB.freeze_raw(m)
    SB.freeze_branch_layers(m)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all((n.startswith("spectral_branches.") and (".project." in n or ".temporal." in n)) or n.startswith("branch_predictors.")
                         for n in names)
    assert sb["trainable_parameters"] == sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_old_objective_with_fixed_output_and_branch_state_loaded_from_a_2c_init(tmp_path):
    """PWf: il passo 0 dei rami caricato dal `checkpoint_init.pt` di un run come il 2C; la perdita del passo 1 coincide con quella del run
    d'origine (stessi dati, stessa inizializzazione: e' il controllo del protocollo); `output` e `combine` restano quelli del passo 0."""
    root, manifest, cfg_b, src = _source_and_cfg(tmp_path)
    ref = tmp_path / "ref"
    R.train(replace(cfg_b, max_steps=2), manifest, [root], ref, log=lambda _: None, init_from=src)
    cfg_f = replace(cfg_b, max_steps=2, model=replace(cfg_b.model, branch_freeze_output=True))
    out = tmp_path / "pwf"
    R.train(cfg_f, manifest, [root], out, log=lambda _: None, init_from=src, branch_state_from=ref / "checkpoint_init.pt")
    state = json.loads((out / "spectral_branches.json").read_text())["branch_state"]
    assert state["sha256"] == R.sha256_file(ref / "checkpoint_init.pt") and state["calibration_reproduced"]["all"]
    r1 = json.loads((ref / "metrics.jsonl").read_text().splitlines()[0])
    f1 = json.loads((out / "metrics.jsonl").read_text().splitlines()[0])
    assert r1["step"] == f1["step"] == 1 and r1["keep_branches"] == f1["keep_branches"]
    init = torch.load(ref / "checkpoint_init.pt", weights_only=False)
    final = torch.load(out / "checkpoint.pt", weights_only=False)
    for part in ("student", "teacher"):
        assert all(torch.equal(init[part][k], final[part][k]) for k in _fixed_keys(init[part], "readout"))
        assert not torch.equal(init[part]["branch_heads.fused.linear.weight"], final[part]["branch_heads.fused.linear.weight"])


def test_pilot_options_are_refused_outside_the_frozen_encoder_and_with_a_wrong_state(tmp_path):
    root, manifest, cfg_b, src = _source_and_cfg(tmp_path)
    joint = replace(cfg_b, freeze_encoder=False, model=replace(cfg_b.model, branch_objective="masked"),
                    jepa=replace(cfg_b.jepa, keep_weight=0.05))
    with pytest.raises(ValueError, match="solo a encoder congelato"):
        R.train(joint, manifest, [root], tmp_path / "joint", log=lambda _: None)
    other = replace(cfg_b, max_steps=1, model=replace(cfg_b.model, spectral_branches=("psd",)))
    ref = tmp_path / "psd_only"
    R.train(other, manifest, [root], ref, log=lambda _: None, init_from=src)
    with pytest.raises(ValueError, match="rami e teste diversi"):
        R.train(replace(cfg_b, max_steps=1), manifest, [root], tmp_path / "bad", log=lambda _: None, init_from=src,
                branch_state_from=ref / "checkpoint_init.pt")


def test_branch_collapse_alarm_stops_the_run(tmp_path, monkeypatch):
    root, manifest, cfg_b, src = _source_and_cfg(tmp_path)
    cfg = replace(cfg_b, max_steps=4, model=replace(cfg_b.model, branch_objective="masked", branch_freeze_output=True))
    monkeypatch.setattr(BM.CollapseAlarm, "update", lambda self, step, metrics: "allarme di collasso dei rami: prova" if step == 2 else None)
    out = tmp_path / "alarm"
    summary = R.train(cfg, manifest, [root], out, log=lambda _: None, init_from=src)
    # l'allarme si legge dopo l'aggiornamento del passo 2 (come quello del raw): il passo conta, il run si ferma li'
    assert summary["steps"] == 2 and summary["stopped"].startswith("allarme")
    assert (out / "STOP").read_text().startswith("allarme") and "alarm" in json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])
