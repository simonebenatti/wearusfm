"""Prefetch del batch (Simone, 15/10/2026: «vai con la 1, il prefetch sulla GPU»): il batch successivo preso, preparato e copiato sul
dispositivo in un thread mentre gira il passo (`RunConfig.prefetch`). Non cambia il training: pesi, teacher, generatori e metriche del passo
bit per bit come senza (un processo, accumulo, worker, codici RVQ, ripresa); la verifica `prefetch_audit` passa sugli ingressi giusti e ferma
il run su quelli sbagliati; un errore del thread arriva al ciclo. Lo stream CUDA si prova su GPU (`docs/protocollo_gpu_dati_finestra1.md`);
in DDP il test e' in `test_training_ddp.py`."""

import copy
import gc
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.data import rvq_codes as RC
from wearusfm.training import run as R

from test_pretraining_loader import _tree
from test_stability_diagnostics import _records, _same_training

THREAD = "wearusfm-prefetch"


def _cfg(**kw):
    return replace(R.small_config(datasets=None, max_steps=4), **kw)


def _no_prefetch_thread() -> bool:
    for _ in range(40):  # un thread di un test fallito di proposito si ferma quando il suo prefetcher viene raccolto
        gc.collect()
        if not any(t.name == THREAD for t in threading.enumerate()):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("workers,accum", [(0, 1), (2, 1), (0, 3)])
def test_prefetch_leaves_the_training_unchanged(tmp_path, workers, accum):
    root, mpath = _tree(tmp_path)
    cfg = _cfg(accum_steps=accum)
    R.train(cfg, mpath, [root], tmp_path / "plain", log=lambda _: None, num_workers=workers)
    s = R.train(replace(cfg, prefetch=True), mpath, [root], tmp_path / "pre", log=lambda _: None, num_workers=workers)
    _same_training(tmp_path / "plain", tmp_path / "pre")
    assert s["steps"] == 4 and s["prefetch"] is True and s["prefetch_audit_steps"] is None
    assert _no_prefetch_thread()  # chiuso a fine run


def test_prefetch_with_rvq_codes_on_resume_with_audit(tmp_path):
    from test_rvq_codes import _session, fake_encode

    root, mpath = _tree(tmp_path)
    codes_root = tmp_path / "codes"
    for row in L.ManifestIndex.load(mpath, [root]).rows:
        s = _session(root, row)
        trials, _ = RC.canonical_trials(s.segments, s.fs, s.qc_valid)
        p = RC.code_path(codes_root, row["dataset"], row["subject"], row["session"])
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **RC.session_codes(trials, s.qc_valid, fake_encode))
    cfg = replace(R.with_rvq(_cfg(), codes_root), stream_segment=4, ckpt_every=100)
    R.train(cfg, mpath, [root], tmp_path / "plain", log=lambda _: None)
    R.train(replace(cfg, max_steps=2), mpath, [root], tmp_path / "pre", log=lambda _: None)
    # prefetch e verifica alla ripresa: variazioni operative ammesse dalla ricetta
    s = R.train(replace(cfg, prefetch=True, prefetch_audit=True), mpath, [root], tmp_path / "pre", log=lambda _: None)
    _same_training(tmp_path / "plain", tmp_path / "pre")
    assert s["prefetch_audit_steps"] == 2 and sum(r["rvq_targets"] for r in _records(tmp_path / "pre" / "metrics.jsonl")) > 0


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_of_a_prefetch_run_stopped_with_a_batch_in_flight(tmp_path, monkeypatch, workers):
    """A fine run il prefetch ha gia' preso e preparato il batch successivo: si scarta, e la ripresa riparte dallo stream al passo giusto
    (GPT, 15/10/2026: ripresa anche con un batch in volo). Arresto dentro un segmento dello stream: la ripresa rigioca a secco."""
    root, mpath = _tree(tmp_path)
    staged, stage = [], R._stage

    def recording(batch, accum, device):
        if threading.current_thread().name == THREAD:
            staged.append(batch.index)
        return stage(batch, accum, device)

    monkeypatch.setattr(R, "_stage", recording)
    cfg = _cfg(stream_segment=4, ckpt_every=100)
    R.train(cfg, mpath, [root], tmp_path / "plain", log=lambda _: None, num_workers=workers)
    pre = replace(cfg, prefetch=True)
    R.train(replace(pre, max_steps=3), mpath, [root], tmp_path / "pre", log=lambda _: None, num_workers=workers)
    assert max(staged) >= 3  # il batch del passo 3 era gia' preparato quando il run si e' fermato
    staged.clear()
    R.train(pre, mpath, [root], tmp_path / "pre", log=lambda _: None, num_workers=workers)
    assert staged[:2] == [3, 4]  # la ripresa riparte dal batch 3
    a, b = (torch.load(tmp_path / d / "checkpoint.pt", weights_only=False) for d in ("plain", "pre"))
    for part in ("student", "teacher"):
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part]), part
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    # record del passo uguali; il run interrotto ha in piu' la valutazione al passo dell'arresto (step == max_steps)
    for x, y in zip(_records(tmp_path / "plain" / "metrics.jsonl"), _records(tmp_path / "pre" / "metrics.jsonl"), strict=True):
        common = (set(x) & set(y)) - {"t_data_s", "t_step_s"}
        assert set(x) <= set(y) and {k: x[k] for k in common} == {k: y[k] for k in common}


def test_the_update_does_not_modify_its_inputs(tmp_path, monkeypatch):
    """La premessa della verifica su GPU (ingressi a fine passo = batch): sulla CPU gli ingressi condividono la memoria del batch, qui sono
    copie, quindi la verifica vedrebbe ogni modifica in place durante il passo."""
    root, mpath = _tree(tmp_path)
    prepare = R._prepare
    monkeypatch.setattr(R, "_prepare", lambda batch, device, dtype=None: prepare(copy.deepcopy(batch), device, dtype))
    for accum in (1, 3):
        s = R.train(_cfg(accum_steps=accum, prefetch_audit=True), mpath, [root], tmp_path / f"run{accum}", log=lambda _: None)
        assert s["stopped"] == "max_steps" and s["prefetch_audit_steps"] == 4 and s["prefetch"] is False


def test_the_audit_stops_the_run_on_inputs_changed_in_the_thread(tmp_path, monkeypatch):
    root, mpath = _tree(tmp_path)
    stage = R._stage

    def corrupted(batch, accum, device):
        micros = stage(batch, accum, device)
        if threading.current_thread().name == THREAD and batch.index == 2:
            micros[0].inp.signals[0] = micros[0].inp.signals[0] + 1.0
        return micros

    monkeypatch.setattr(R, "_stage", corrupted)
    out = tmp_path / "run"
    s = R.train(_cfg(prefetch=True, prefetch_audit=True), mpath, [root], out, log=lambda _: None)
    assert s["stopped"] == "verifica del prefetch: ingressi sul dispositivo diversi dal batch al passo 2"
    assert s["steps"] == 2 and s["prefetch_audit_steps"] == 2
    assert (out / "STOP").read_text().startswith("verifica del prefetch")
    assert [r["step"] for r in _records(out / "metrics.jsonl")] == [1, 2]


def test_an_error_in_the_prefetch_thread_reaches_the_loop(tmp_path, monkeypatch):
    root, mpath = _tree(tmp_path)
    stage = R._stage

    def broken(batch, accum, device):
        if threading.current_thread().name == THREAD and batch.index == 1:
            raise OSError("lettura fallita")
        return stage(batch, accum, device)

    monkeypatch.setattr(R, "_stage", broken)
    with pytest.raises(OSError, match="lettura fallita"):
        R.train(_cfg(prefetch=True), mpath, [root], tmp_path / "run", log=lambda _: None)
    assert [r["step"] for r in _records(tmp_path / "run" / "metrics.jsonl")] == [1]


def test_prefetcher_keeps_the_order_and_closes():
    counter = iter(range(1000))
    pre = R._Prefetcher(lambda: next(counter), lambda b: [b * 10], "cpu")
    assert [pre.next() for _ in range(4)] == [(0, [0]), (1, [10]), (2, [20]), (3, [30])]
    pre.close()
    assert not pre._thread.is_alive()
