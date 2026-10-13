"""Equivalenza del trainer col contratto delle loss distribuite (`docs/contratto_loss_distribuite.md`, punto 5; consegna GPT del 12/10/2026):
lo stesso batch globale dato a 1 processo col percorso di sempre (`jepa_losses`), a 1 processo con l'accumulo, a 2 e 4 processi DDP `gloo`
senza e con l'accumulo, da' le stesse loss (ogni componente e le diagnostiche), gli stessi gradienti dopo la riduzione, gli stessi pesi dopo
clipping e AdamW e lo stesso teacher dopo l'EMA. CPU, float64, tolleranza relativa e assoluta 1e-10 (dichiarata nel contratto). Il batch ha
un processo intero senza target RMS (conteggio locale zero) e finestre di lunghezze diverse (scala lenta assente su alcune)."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mp = pytest.importorskip("torch.multiprocessing")

from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.jepa import ema_update, jepa_losses, loss_counts, make_teacher  # noqa: E402

from test_pretraining_loader import _tree  # noqa: E402

TOL = dict(rtol=1e-10, atol=1e-10)
WINDOW_S = {"emg2pose/u1/sessA": 3.9, "ninapro_db2/s01/session1": 3.0, "capgmyo/s01/s": 1.0}


def _config(per_window: bool, multiscale: bool):
    cfg = R.small_config(datasets=None)
    if multiscale:
        cfg = R.with_window1_rules(cfg)  # anche finestre per finestra, ancore multi-scala, montaggi virtuali
    return replace(cfg, jepa=replace(cfg.jepa, loss_per_window=per_window), model=replace(cfg.model, muscle_dropout=0.0))


def _global_batch(tmp_path, cfg, n=4):
    root, mpath = _tree(tmp_path)
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), cfg.loader, {}, WINDOW_S if cfg.loader.time_weighted else None)
    batch = loader.batch(n, np.random.default_rng(3))
    for t in batch.anchor_targets[: n // 2]:  # la prima meta' (il processo 0 con 2 processi) senza target RMS
        t.rms_valid[:] = False
    return batch


def _update(cfg, init, batch, accum, ctx, model_wrap):
    """Un aggiornamento come nel trainer: `_accumulate_terms`, poi un clipping, AdamW ed EMA."""
    student = WearUsFM(cfg.model).double()
    student.load_state_dict(init)
    teacher = make_teacher(student)
    model = model_wrap(student)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    micros = [R._prepare(mb, "cpu", torch.float64) for mb in L.split_batch(batch, accum)]
    student.train()
    opt.zero_grad(set_to_none=True)
    failure, metrics = R._accumulate_terms(model, student, teacher, micros, cfg, ctx, "cpu", nullcontext())
    assert failure is None, failure
    grads = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
    opt.step()
    ema_update(teacher, student, cfg.jepa.ema_momentum)
    return {"metrics": metrics, "grads": grads, "student": student.state_dict(), "teacher": teacher.state_dict()}


def _reference(cfg, init, batch):
    """Un processo, batch intero, il percorso di sempre (`jepa_losses`, la media storica)."""
    student = WearUsFM(cfg.model).double()
    student.load_state_dict(init)
    teacher = make_teacher(student)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    m = R._prepare(batch, "cpu", torch.float64)
    student.train()
    losses = jepa_losses(student, teacher, m.inp, m.visible, cfg.jepa, anchor_targets=m.anchor_targets, kinds=m.kinds)
    losses["total"].backward()
    grads = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
    opt.step()
    ema_update(teacher, student, cfg.jepa.ema_momentum)
    return {"metrics": {k: float(v.detach()) for k, v in losses.items()}, "grads": grads, "student": student.state_dict(),
            "teacher": teacher.state_dict()}


def _worker(rank, world, store, cfg, init, batch, accums, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        mine = L.split_batch(batch, world)[rank]  # campioni contigui: il processo r ha la sua parte del batch globale
        wrap = lambda s: R._ddp_wrap(s, cfg, R.DistContext(rank, world, rank), "cpu")  # come il trainer (anche find_unused_parameters)  # noqa: E731
        res = {k: _update(cfg, init, mine, k, R.DistContext(rank, world, rank), wrap) for k in accums}
        torch.save(res, Path(out) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _ddp(tmp_path, tag, cfg, init, batch, world, accums):
    out = tmp_path / f"out_{tag}"
    out.mkdir()
    mp.spawn(_worker, args=(world, str(tmp_path / f"store_{tag}"), cfg, init, batch, accums, str(out)), nprocs=world, join=True)
    return [torch.load(out / f"rank{r}.pt", weights_only=False) for r in range(world)]


def _same(ref, got, what):
    assert set(got["metrics"]) == set(ref["metrics"]), (what, set(got["metrics"]) ^ set(ref["metrics"]))
    for k, v in ref["metrics"].items():
        # le diagnostiche per tipo di maschera e topologia si calcolano in float32 in jepa.py (GPT, 12/10/2026: «le tolleranze float64 non
        # valgono per [...] diagnostiche che fanno cast float32»): tolleranza float32 per loro, 1e-10 per componenti e totale
        tol = dict(rel=1e-5, abs=1e-7) if k.startswith("jepa_") else dict(rel=1e-10, abs=1e-10)
        assert got["metrics"][k] == pytest.approx(v, **tol), (what, k)
    assert set(got["grads"]) == set(ref["grads"]), what
    for part in ("grads", "student", "teacher"):
        for k, v in ref[part].items():
            torch.testing.assert_close(got[part][k], v, **TOL, msg=f"{what}: {part} {k}")


@pytest.mark.parametrize("per_window,multiscale", [(False, False), (True, True)], ids=["sanity", "finestra1"])
def test_one_process_accumulation_and_ddp_equal_the_global_batch(tmp_path, per_window, multiscale):
    cfg = _config(per_window, multiscale)
    batch = _global_batch(tmp_path, cfg)
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    ref = _reference(cfg, init, batch)
    one = R.DistContext()
    _same(ref, _update(cfg, init, batch, 2, one, lambda s: None), "1 processo x 2 micro-batch")
    _same(ref, _update(cfg, init, batch, 4, one, lambda s: None), "1 processo x 4 micro-batch")
    res = _ddp(tmp_path, "w2", cfg, init, batch, 2, (1, 2))
    for k in (1, 2):
        _same(ref, res[0][k], f"2 processi x {k} micro-batch")
        for part in ("student", "teacher"):  # tutti i processi uguali, bit per bit
            assert all(torch.equal(res[0][k][part][n], res[1][k][part][n]) for n in res[0][k][part]), (k, part)


def test_four_processes_one_window_each(tmp_path):
    cfg = _config(True, True)
    batch = _global_batch(tmp_path, cfg)
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    ref = _reference(cfg, init, batch)
    res = _ddp(tmp_path, "w4", cfg, init, batch, 4, (1,))
    _same(ref, res[0][1], "4 processi x 1")


def test_split_batch_keeps_samples_and_counts(tmp_path):
    cfg = _config(True, True)
    batch = _global_batch(tmp_path, cfg)
    parts = L.split_batch(batch, 2)
    assert [len(p.signals) for p in parts] == [2, 2] and parts[0].index == batch.index
    whole = R._prepare(batch, "cpu")
    model_cfg = cfg.model
    total = loss_counts(whole.inp, whole.visible, cfg.jepa, whole.anchor_targets, model_cfg=model_cfg)
    pieces = [R._prepare(p, "cpu") for p in parts]
    counts = [loss_counts(m.inp, m.visible, cfg.jepa, m.anchor_targets, model_cfg=model_cfg) for m in pieces]
    assert all(torch.equal(sum(c[k] for c in counts), total[k]) for k in total)  # i conteggi si sommano
    assert counts[0]["log_rms"] == 0 < counts[1]["log_rms"]
    off = 0
    for p in parts:  # stessi canali, vicini spostati dentro il micro-batch, patch fino al massimo del micro-batch
        n = sum(p.counts)
        assert np.array_equal(p.qc_valid, batch.qc_valid[off:off + n])
        assert np.array_equal(np.where(p.sets.index >= 0, p.sets.index + off, -1), batch.sets.index[off:off + n])
        assert p.visible.shape == (n, max(p.n_patches)) and np.array_equal(p.visible, batch.visible[off:off + n, :max(p.n_patches)])
        off += n
    with pytest.raises(ValueError, match="divisibili"):
        L.split_batch(batch, 3)


def test_coordinated_stops_on_count_mismatch_and_empty_jepa(tmp_path, monkeypatch):
    cfg = _config(True, True)
    batch = _global_batch(tmp_path, cfg)
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    student = WearUsFM(cfg.model).double()
    student.load_state_dict(init)
    teacher = make_teacher(student)
    micros = [R._prepare(mb, "cpu", torch.float64) for mb in L.split_batch(batch, 2)]
    counts = R.loss_counts

    def off_by_one(*a, **k):  # precalcolo sbagliato: il forward non coincide
        out = counts(*a, **k)
        out["jepa"] = out["jepa"] + 1
        return out

    monkeypatch.setattr(R, "loss_counts", off_by_one)
    failure, _ = R._accumulate_terms(None, student, teacher, micros, cfg, R.DistContext(), "cpu", nullcontext())
    assert failure == "conteggi del forward diversi da quelli precalcolati"
    assert all(p.grad is None for p in student.parameters())  # nessun backward
    monkeypatch.setattr(R, "loss_counts", counts)
    for m in micros:  # tutto visibile: nessuna query JEPA nel batch globale
        m.visible[:] = True
    failure, _ = R._accumulate_terms(None, student, teacher, micros, cfg, R.DistContext(), "cpu", nullcontext())
    assert failure == "nessun target JEPA nel batch globale"


def test_single_process_accumulation_through_train_and_resume(tmp_path):
    root, mpath = _tree(tmp_path)
    cfg = replace(R.small_config(datasets=None, max_steps=4), batch_size=4, accum_steps=2, ckpt_every=100)
    s = R.train(cfg, mpath, [root], tmp_path / "a", log=lambda m: None)
    assert s["accum_steps"] == 2 and s["micro_batch"] == 2 and "world_size" not in s
    rec = [json.loads(x) for x in (tmp_path / "a" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rec] == [1, 2, 3, 4] and all(r["windows"] == 4 and r["total"] > 0 for r in rec)
    R.train(replace(cfg, max_steps=2), mpath, [root], tmp_path / "b", log=lambda m: None)  # interrotto al passo 2
    R.train(cfg, mpath, [root], tmp_path / "b", log=lambda m: None)  # ripreso fino al 4
    a, b = (torch.load(tmp_path / d / "checkpoint.pt", weights_only=False) for d in ("a", "b"))
    for part in ("student", "teacher"):
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part]), part
