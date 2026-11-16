"""Test distribuiti indipendenti del percorso contrastivo (review di Claude del 16/11/2026, consegna GPT db5ac41).

Coprono cio' che i sei test distribuiti della consegna non esercitano: finestre di lunghezza diversa fra i rank (`_pad_time`,
`_global_width`), un rank senza posa che e' anche il piu' largo, update consecutivi con coda e ricostruzione dei bucket DDP, e il trainer
vero `R.train` a 2 processi con checkpoint e ripresa (il protocollo, §3, chiede il test distribuito «comprese rank vuote e ripresa»), e
la calibrazione (`calibration_only`) sotto DDP.
CPU, gloo, modello in float64. `pose` e' ridotta in float32 dentro `pose_terms`: per lei (e per `total`) vale la tolleranza float32.
"""

from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from test_pose_training import _pose_setup, _r3_config
from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime
from wearusfm.training.jepa import ema_update, make_teacher

F32 = ("pose", "total")                       # riduzione float32 in jepa.pose_terms
F32_PREFIX = ("jepa_", "contrast_global_")    # diagnostiche con cast float32


def _config(pose_root, arm, *, total=4, **contrast):
    base = _r3_config(pose_root, total=total, keep=())
    kw = dict(arm=arm, capacity_per_dataset=8, min_negatives=1, gap_seconds=.001, fill_steps=0, ramp_steps=1, rnc_stride=2, grad_every=1)
    kw.update(contrast)
    return replace(base, max_steps=total, batch_size=4, fixed_eval_every=0,
                   model=replace(base.model, contrastive_projection=True, muscle_dropout=0.),
                   loader=replace(base.loader, min_window_s=1., max_window_s=4.),   # lunghezze DIVERSE: 40-160 patch
                   contrastive=asdict(ContrastiveConfig(**kw)), datasets=("emg2pose",))


def _batches(cfg, root, manifest, seed, n, drops=None):
    loader, rng = L.PretrainLoader(R.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader, {}, None), np.random.default_rng(seed)
    out = []
    for i in range(n):
        batch = loader.batch(4, rng)
        for k in (drops or {}).get(i, ()):
            batch.pose[k] = None
        out.append(batch)
    return out


def _updates(cfg, init, batches, ctx):
    """Aggiornamenti consecutivi nell'ordine di `train()`: prepare, _accumulate_terms, clipping, AdamW, EMA, commit."""
    student = WearUsFM(cfg.model).double()
    student.load_state_dict(init)
    teacher = make_teacher(student)
    wrapped = R._ddp_wrap(student, cfg, ctx, "cpu") if ctx.active else None
    engine = ContrastiveRuntime(cfg.contrastive)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    out = []
    for step, batch in enumerate(batches):
        local = L.split_batch(batch, ctx.world_size)[ctx.rank] if ctx.active else batch
        opt.zero_grad(set_to_none=True)
        engine.prepare(local, step)
        micro = R._prepare(local, "cpu", torch.float64)
        student.train()
        failure, metrics = R._accumulate_terms(wrapped, student, teacher, [micro], cfg, ctx, "cpu", nullcontext(), contrastive=engine)
        assert failure is None, failure
        grads = {name: p.grad.clone() for name, p in student.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        if not engine.cfg.calibration_only:  # come train(): in calibrazione ne' optimizer ne' EMA, avanza solo la memoria
            opt.step()
            ema_update(teacher, student, cfg.jepa.ema_momentum)
        assert engine.commit() is None
        out.append({"metrics": metrics, "grads": grads, "student": copy.deepcopy(student.state_dict()),
                    "teacher": copy.deepcopy(teacher.state_dict()), "queue": copy.deepcopy(engine.state_dict())})
    return out


def _worker(rank, world, store, cfg, init, batches, out):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        torch.save(_updates(cfg, init, batches, R.DistContext(rank, world, rank)), Path(out) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _train_worker(rank, world, store, cfg, manifest, root, out, results):
    import torch.distributed as dist
    from wearusfm.training import contrastive_objectives as CO
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        held, init = {}, CO.ContrastiveRuntime.__init__

        def capture(self, config):  # il motore contrastivo di QUESTO rank, per confrontare le code fra rank
            init(self, config)
            held["engine"] = self

        CO.ContrastiveRuntime.__init__ = capture
        summary = R.train(cfg, Path(manifest), [Path(root)], Path(out), log=lambda m: None, dist=R.DistContext(rank, world, rank))
        torch.save({"steps": summary["steps"], "stopped": summary["stopped"], "queue": held["engine"].queue.state_dict(),
                    "coverage": held["engine"].coverage}, Path(results) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _spawn(fn, world, args, deadline=300):
    """Solo processi figli propri; una collettiva bloccata non deve fermare pytest."""
    context = mp.spawn(fn, args=(world, *args), nprocs=world, join=False)
    end = time.monotonic() + deadline
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= end:
                raise TimeoutError(f"test DDP sulla CPU oltre {deadline} secondi")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def _same_nested(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _same_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same_nested(a, b)
    else:
        assert left == right


def _assert_step(ref, ranks, pose_weight):
    for got in ranks:
        assert set(got["metrics"]) == set(ref["metrics"])
        for key, value in ref["metrics"].items():
            tol = dict(rel=5e-6, abs=1e-9) if key in F32 or key.startswith(F32_PREFIX) else dict(rel=1e-9, abs=1e-10)
            assert got["metrics"][key] == pytest.approx(value, **tol), key
        rest = lambda m: m["total"] - pose_weight * m["pose"]  # noqa: E731 -- tutto cio' che non e' la posa, in float64
        assert rest(got["metrics"]) == pytest.approx(rest(ref["metrics"]), rel=1e-9, abs=1e-10)
        assert set(got["grads"]) == set(ref["grads"])
        for part in ("grads", "student", "teacher"):
            for key, value in ref[part].items():
                torch.testing.assert_close(got[part][key], value, rtol=1e-10, atol=1e-12, msg=f"{part} {key}")
        want, have = ref["queue"]["queue"], got["queue"]["queue"]
        assert have["last_step"] == want["last_step"]
        assert [(e["identity"], e["step"], e["global_index"]) for e in have["entries"]] == \
            [(e["identity"], e["step"], e["global_index"]) for e in want["entries"]]
        for a, b in zip(have["entries"], want["entries"]):
            torch.testing.assert_close(a["key"], b["key"], rtol=0, atol=1e-12)
        assert got["queue"]["coverage"] == ref["queue"]["coverage"]
    for other in ranks[1:]:  # i rank sono uguali fra loro bit per bit: pesi, teacher e coda
        for part in ("student", "teacher", "queue"):
            _same_nested(ranks[0][part], other[part])


@pytest.mark.parametrize("arm,world,drop", [("CR", 4, (2,)), ("CN", 4, (2,)), ("CR", 2, (2, 3)), ("CN", 2, (0, 1)), ("CR", 4, (0, 1, 2, 3))])
def test_distributed_variable_window_lengths_match_single_process(tmp_path, arm, world, drop):
    root, manifest, pose_root = _pose_setup(tmp_path)
    cfg = _config(pose_root, arm)
    batch = _batches(cfg, root, manifest, 5, 2)[1]
    assert batch.n_patches == [40, 44, 160, 140]  # il rank piu' largo e quelli stretti: scattano _pad_time e _global_width
    for i in drop:
        batch.pose[i] = None
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    reference = _updates(cfg, init, [batch], R.DistContext())
    output = tmp_path / "ddp"
    output.mkdir()
    _spawn(_worker, world, (str(tmp_path / "store"), cfg, init, [batch], str(output)))
    ranks = [torch.load(output / f"rank{rank}.pt", weights_only=False) for rank in range(world)]
    _assert_step(reference[0], [r[0] for r in ranks], cfg.jepa.pose_weight)
    if len(drop) == 4:
        assert reference[0]["metrics"]["contrast_rnc_windows"] == 0 and reference[0]["metrics"]["contrast_global_grad_rnc"] == 0


@pytest.mark.parametrize("arm,world", [("CR", 2), ("CN", 4)])
def test_distributed_consecutive_updates_match_single_process(tmp_path, arm, world):
    root, manifest, pose_root = _pose_setup(tmp_path)
    cfg = _config(pose_root, arm, max_age=2)
    batches = _batches(cfg, root, manifest, 31, 4, {1: (0, 1), 3: (0, 1, 2, 3)})
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    reference = _updates(cfg, init, batches, R.DistContext())
    output = tmp_path / "ddp"
    output.mkdir()
    _spawn(_worker, world, (str(tmp_path / "store"), cfg, init, batches, str(output)))
    ranks = [torch.load(output / f"rank{rank}.pt", weights_only=False) for rank in range(world)]
    for step in range(len(batches)):
        _assert_step(reference[step], [r[step] for r in ranks], cfg.jepa.pose_weight)


@pytest.mark.parametrize("world", [2, 4])
def test_distributed_calibration_only_keeps_weights_and_measures_global_norms(tmp_path, world):
    """La calibrazione e' il primo job a 4 GPU: pesi e teacher fermi su ogni rank, la memoria avanza, norme globali come a un processo."""
    root, manifest, pose_root = _pose_setup(tmp_path)
    cfg = _config(pose_root, "C0", calibration_only=True, grad_every=0, fill_steps=1, nce_weight=1.0, rnc_weight=1.0)
    batches = _batches(cfg, root, manifest, 21, 3, {1: (0, 1)})
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    reference = _updates(cfg, init, batches, R.DistContext())
    output = tmp_path / "ddp"
    output.mkdir()
    _spawn(_worker, world, (str(tmp_path / "store"), cfg, init, batches, str(output)))
    ranks = [torch.load(output / f"rank{rank}.pt", weights_only=False) for rank in range(world)]
    for step in range(len(batches)):
        _assert_step(reference[step], [r[step] for r in ranks], cfg.jepa.pose_weight)
        for got in [reference] + ranks:
            for part in ("student", "teacher"):
                _same_nested(got[step][part], init)                      # nessun aggiornamento, bit per bit
            assert got[step]["queue"]["queue"]["last_step"] == step      # la memoria avanza
            has = any(key.startswith("contrast_global_grad_") for key in got[step]["metrics"])
            assert has == (step >= cfg.contrastive["fill_steps"])        # diagnostica a ogni passo dopo il riempimento
    assert all(reference[step]["metrics"]["contrast_global_grad_base"] > 0 for step in (1, 2))


def test_distributed_trainer_resume_is_bit_identical_and_rank_queues_match(tmp_path):
    # A DUE processi la somma dei gradienti fra i rank non dipende dall'ordine: continuo = interrotto + ripreso, bit per bit. A quattro
    # processi no, gia' nel percorso senza contrastivo: il primo backward dopo la ripartenza usa i bucket iniziali di DDP, il run continuo
    # quelli ricostruiti, e l'ordine della somma fra quattro rank cambia l'ultimo bit (review di Claude del 16/11/2026: a quattro rank
    # restano uguali identita', ordine e copertura della coda; pesi entro l'arrotondamento).
    root, manifest, pose_root = _pose_setup(tmp_path)
    # Soglie dell'allarme a zero: il modello minuscolo fa scattare l'allarme di collasso alla terza valutazione, e un'interruzione per
    # max_steps fuori da eval_every aggiunge una valutazione (run.py, `step == cfg.max_steps`) che sposterebbe l'allarme nel run ripreso.
    # Qui si prova lo stato contrastivo, non l'allarme.
    cfg = replace(_config(pose_root, "CN", total=6, grad_every=2), pose_grad_every=2,
                  alarm=R.AlarmRule(collapse_ratio_min=0.0, erank_fraction_min=0.0))

    def run(tag, cfg, out):
        results = tmp_path / f"results_{tag}"
        results.mkdir()
        _spawn(_train_worker, 2, (str(tmp_path / f"store_{tag}"), cfg, str(manifest), str(root), str(out), str(results)))
        return [torch.load(results / f"rank{rank}.pt", weights_only=False) for rank in range(2)]

    whole = run("whole", cfg, tmp_path / "whole")
    first = run("first", replace(cfg, max_steps=3), tmp_path / "resumed")
    middle = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
    second = run("second", cfg, tmp_path / "resumed")
    for ranks, steps in ((whole, 6), (first, 3), (second, 6)):
        assert all(r["steps"] == steps and r["stopped"] == "max_steps" for r in ranks)
        _same_nested(ranks[0]["queue"], ranks[1]["queue"])  # la coda del rank 1 e' quella che il rank 0 scrive nel checkpoint
        assert ranks[0]["coverage"] == ranks[1]["coverage"]
    _same_nested(middle["contrastive_state"]["queue"], first[1]["queue"])
    assert middle["world_size"] == 2 and len(middle["rank_rng"]) == 2
    a, b = (torch.load(tmp_path / name / "checkpoint.pt", weights_only=False) for name in ("whole", "resumed"))
    for part in ("student", "teacher", "optimizer", "torch_rng", "rank_rng", "contrastive_state"):
        _same_nested(a[part], b[part])
    _same_nested(a["contrastive_state"]["queue"], whole[1]["queue"])
    skip = lambda k: k.startswith(("t_", "student_", "teacher_"))  # noqa: E731 -- tempi e valutazione periodica (una in piu' al passo 3)
    lines = [[{k: v for k, v in json.loads(x).items() if not skip(k)} for x in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]
             for name in ("whole", "resumed")]
    assert len(lines[0]) == 6 and lines[0] == lines[1]
