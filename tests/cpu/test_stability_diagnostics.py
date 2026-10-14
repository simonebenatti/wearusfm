"""Diagnostiche della stabilita' nel trainer (GPT e Claude, 14/10/2026, dopo il collaudo dello step 2C; Simone: «implementa i punti 1 e 2»):
loss su un insieme FISSO di batch del training (`fixed_eval_every`) e origine di ogni finestra in `metrics.jsonl` (`log_batch_origin`). Non
cambiano il training: pesi, teacher e metriche del passo bit per bit come senza; valori ripetibili; insieme fisso uguale alla ripresa."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from wearusfm.training import run as R

from test_spectral_branches import _branch_cfg
from test_spectral_step1 import _calibration, _setup


def _keep_cfg(tmp_path):
    root, manifest, cfg, _, y, m = _setup(tmp_path)
    cal = tmp_path / "cal.npz"
    _calibration(cal, cfg, manifest, y, m)
    return root, manifest, R.with_spectral_keep(cfg, cal, 0.05)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _same_training(a, b):
    """Stessi pesi (studente, teacher, optimizer) e stesse metriche del passo, a parte l'origine dei batch."""
    sa, sb = (torch.load(p / "checkpoint.pt", weights_only=False) for p in (a, b))
    for part in ("student", "teacher"):
        assert sa[part].keys() == sb[part].keys() and all(torch.equal(sa[part][k], sb[part][k]) for k in sa[part]), part
    assert torch.equal(sa["torch_rng"], sb["torch_rng"])
    ra, rb = _records(a / "metrics.jsonl"), _records(b / "metrics.jsonl")
    assert len(ra) == len(rb)
    timing = ("t_data_s", "t_step_s")
    for x, y in zip(ra, rb):
        y = {k: v for k, v in y.items() if k != "origin"}
        assert {k: v for k, v in x.items() if k not in timing} == {k: v for k, v in y.items() if k not in timing}


def test_diagnostics_leave_the_joint_training_unchanged(tmp_path):
    root, manifest, cfg = _keep_cfg(tmp_path)
    cfg = replace(cfg, max_steps=4)
    R.train(cfg, manifest, [root], tmp_path / "plain", log=lambda _: None)
    diag = replace(cfg, fixed_eval_every=2, fixed_eval_batches=2, log_batch_origin=True)
    R.train(diag, manifest, [root], tmp_path / "diag", log=lambda _: None)
    _same_training(tmp_path / "plain", tmp_path / "diag")
    fixed = _records(tmp_path / "diag" / "fixed_eval.jsonl")
    assert [r["step"] for r in fixed] == [0, 2, 4] and all(r["batches"] == 2 for r in fixed)
    assert all(np.isfinite(r[k]) for r in fixed for k in ("total", "jepa", "keep"))
    assert fixed[0]["total"] != fixed[-1]["total"]  # il modello e' cambiato, l'insieme no
    windows = json.loads((tmp_path / "diag" / "fixed_eval_set.json").read_text())["windows"]
    assert len(windows) == 2 * min(cfg.batch_size, 32) and all(len(w) == 4 for w in windows)
    assert not (tmp_path / "plain" / "fixed_eval.jsonl").exists() and not (tmp_path / "plain" / "origin_sessions.json").exists()


def test_batch_origin_names_sessions_and_windows(tmp_path):
    root, manifest, cfg = _keep_cfg(tmp_path)
    cfg = replace(cfg, max_steps=3, log_batch_origin=True)
    out = tmp_path / "run"
    R.train(cfg, manifest, [root], out, log=lambda _: None)
    sessions = json.loads((out / "origin_sessions.json").read_text())
    index = R.load_index(manifest, [root], cfg.datasets)
    assert sessions == [R._session_key(r) for r in index.rows]
    for rec in _records(out / "metrics.jsonl"):
        assert len(rec["origin"]) == rec["windows"]
        for pos, trial, start in rec["origin"]:
            assert 0 <= pos < len(sessions) and isinstance(start, int) and (trial is None or isinstance(trial, int))
    # coi worker del DataLoader (lo stream dipende dal loro numero, come la ripresa) l'origine attraversa il passaggio in memoria condivisa
    R.train(replace(cfg, max_steps=3), manifest, [root], tmp_path / "workers", log=lambda _: None, num_workers=2)
    for rec in _records(tmp_path / "workers" / "metrics.jsonl"):
        assert len(rec["origin"]) == rec["windows"] and all(0 <= o[0] < len(sessions) for o in rec["origin"])


def test_fixed_eval_is_repeatable_and_keeps_the_random_state(tmp_path):
    root, manifest, cfg = _keep_cfg(tmp_path)
    cfg = replace(cfg, fixed_eval_every=1, fixed_eval_batches=2)
    out = tmp_path / "run"
    R.train(replace(cfg, max_steps=1), manifest, [root], out, log=lambda _: None)
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    index = R.load_index(manifest, [root], cfg.datasets)
    batches, ident = R._fixed_eval_set(index, cfg, {}, None)
    assert ident == json.loads((out / "fixed_eval_set.json").read_text())
    student = R.WearUsFM(cfg.model)
    student.load_state_dict(state["student"])
    teacher = R.WearUsFM(cfg.model)
    teacher.load_state_dict(state["teacher"])
    student.train()
    rng = torch.get_rng_state().clone()
    amp = torch.autocast("cuda", enabled=False)
    first = R._fixed_eval_losses(student, teacher, batches, cfg, "cpu", amp)
    assert torch.equal(torch.get_rng_state(), rng) and student.training
    assert R._fixed_eval_losses(student, teacher, batches, cfg, "cpu", amp) == first
    assert first == {k: v for k, v in _records(out / "fixed_eval.jsonl")[-1].items() if k != "step"}


def test_fixed_eval_on_the_frozen_encoder_leaves_training_unchanged(tmp_path):
    root, manifest, cfg_b = _branch_cfg(tmp_path, freeze_encoder=True, keep=0.0)
    src_cfg = R.with_spectral_keep(replace(cfg_b, model=replace(cfg_b.model, spectral_branches=()), freeze_encoder=False,
                                           jepa=replace(cfg_b.jepa, branch_weight=0.0)), cfg_b.keep_calibration, 0.0)
    R.train(src_cfg, manifest, [root], tmp_path / "src", log=lambda _: None)
    init = tmp_path / "src" / "checkpoint.pt"
    cfg_b = replace(cfg_b, max_steps=3)
    R.train(cfg_b, manifest, [root], tmp_path / "plain", log=lambda _: None, init_from=init)
    R.train(replace(cfg_b, fixed_eval_every=1, fixed_eval_batches=2, log_batch_origin=True), manifest, [root], tmp_path / "diag",
            log=lambda _: None, init_from=init)
    _same_training(tmp_path / "plain", tmp_path / "diag")
    fixed = _records(tmp_path / "diag" / "fixed_eval.jsonl")
    assert [r["step"] for r in fixed] == [0, 1, 2, 3]
    assert all({"keep_branches", "keep_fused", "keep_psd", "keep_wavelet"} <= set(r) for r in fixed)
    assert fixed[0]["keep_branches"] != fixed[-1]["keep_branches"]


def test_resume_keeps_the_fixed_set_and_refuses_a_different_one(tmp_path):
    root, manifest, cfg = _keep_cfg(tmp_path)
    cfg = replace(cfg, fixed_eval_every=1, fixed_eval_batches=2, log_batch_origin=True)
    out = tmp_path / "run"
    R.train(replace(cfg, max_steps=2), manifest, [root], out, log=lambda _: None)
    R.train(replace(cfg, max_steps=4), manifest, [root], out, log=lambda _: None)
    assert [r["step"] for r in _records(out / "fixed_eval.jsonl")] == [0, 1, 2, 3, 4]  # nessuna ripetizione del passo 0 alla ripresa
    path = out / "fixed_eval_set.json"
    ident = json.loads(path.read_text())
    ident["windows"][0][2] += 1
    path.write_text(json.dumps(ident))
    with pytest.raises(ValueError, match="insieme fisso"):
        R.train(replace(cfg, max_steps=5), manifest, [root], out, log=lambda _: None)
    with pytest.raises(ValueError, match="fixed_eval_every"):
        R.train(replace(cfg, fixed_eval_every=-1), manifest, [root], tmp_path / "bad", log=lambda _: None)


def test_ddp_records_concatenate_the_origin_in_rank_order():
    part = {"windows": 2, "skipped_sessions": 0, "virtual": 0, "classes": {"A": 0, "B": 2, "C": 0}, "t_data_s": .1, "t_step_s": .5}
    merged = R._merge_records([{**part, "origin": [[0, None, 5], [1, 2, 9]]}, {**part, "origin": [[3, None, 7], [3, None, 8]]}])
    assert merged["origin"] == [[0, None, 5], [1, 2, 9], [3, None, 7], [3, None, 8]] and merged["windows"] == 4
    assert "origin" not in R._merge_records([part, part])
