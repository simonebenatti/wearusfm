"""Review Claude, dimensione `calibrazione` (16/11/2026, seconda esecuzione): il punto che il controllo interno di GPT lascia «da eseguire»:
`autograd.grad` (norme globali) + activation checkpointing + backward DDP, a 4 rank (gloo, CPU), con un rank senza posa.
Modello minuscolo dei test di GPT in float64, segnali sintetici dei test, `grad_checkpoint=True`.
Per ogni caso: (1) le tre norme e i due coseni a 4 rank = a un processo; (2) con checkpointing = senza; (3) il gradiente APPLICATO
(p.grad dopo la media DDP, prima del clipping) sui parametri condivisi ha norma^2 = base^2 + (lambda c)^2 + 2 lambda base c cos,
cioe' le norme misurate sono quelle del gradiente che il braccio applica; (4) p.grad a 4 rank = p.grad a un processo."""
from dataclasses import replace
from datetime import timedelta
import json
import math
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from test_contrastive_runtime import _config, _global_update
from test_pose_training import _pose_setup
from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R

KEYS = ("contrast_global_grad_base", "contrast_global_grad_nce", "contrast_global_grad_rnc")
COS = ("contrast_global_cos_nce_base", "contrast_global_cos_rnc_base")
LN, LR = 0.37, 0.23
OUT = Path(__import__("tempfile").mkdtemp(prefix="prop_cal4_"))  # rapporto JSON in una cartella temporanea


def _worker(rank, world, store, cfg, init, batch, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world, timeout=timedelta(seconds=900))
    try:
        local = L.split_batch(batch, world)[rank]
        result = _global_update(cfg, init, local, R.DistContext(rank, world, rank))
        torch.save({"metrics": result["metrics"], "grads": result["grads"]}, Path(out) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _spawn(world, store, cfg, init, batch, output, limit=900):
    context = mp.spawn(_worker, args=(world, store, cfg, init, batch, output), nprocs=world, join=False)
    deadline = time.monotonic() + limit
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("limite superato")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


def _shared_norm(grads):
    sq = 0.0
    for name, g in grads.items():
        if name.split(".")[0] in R.SHARED_PREFIXES:
            sq += float(g.double().square().sum())
    return math.sqrt(sq)


@pytest.mark.parametrize("case", ["C0cal", "CN", "CR"])
def test_ddp4_with_activation_checkpointing_measured_equals_applied(tmp_path, case):
    root, manifest, pose_root = _pose_setup(tmp_path)
    cal = case == "C0cal"
    cfg = _config(pose_root, "C0" if cal else case, calibration=cal)
    cfg = replace(cfg, batch_size=8, model=replace(cfg.model, grad_checkpoint=True),
                  contrastive={**cfg.contrastive, "grad_every": 1, "nce_weight": LN, "rnc_weight": LR})
    loader = L.PretrainLoader(R.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader, {}, None)
    batch = loader.batch(8, np.random.default_rng(3))
    for i in range(2):  # il primo rank intero senza posa
        batch.pose[i] = None
    batch.rows = [{**row, "session": f"synthetic_session_{i}"} for i, row in enumerate(batch.rows)]
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    ref = _global_update(cfg, init, batch, R.DistContext())
    plain = _global_update(replace(cfg, model=replace(cfg.model, grad_checkpoint=False)), init, batch, R.DistContext())
    m = ref["metrics"]
    assert all(m[k] > 0 for k in KEYS), m
    for k in KEYS:  # (2) checkpointing non cambia le norme
        assert m[k] == pytest.approx(plain["metrics"][k], rel=1e-6), k
    out = tmp_path / "ddp"; out.mkdir()
    _spawn(4, str(tmp_path / "store"), cfg, init, batch, str(out))
    ranks = [torch.load(out / f"rank{r}.pt", weights_only=False) for r in range(4)]
    for got in ranks:  # (1) e (4)
        for k in KEYS:
            assert got["metrics"][k] == pytest.approx(m[k], rel=1e-5), k
        for k in COS:
            assert got["metrics"][k] == pytest.approx(m[k], abs=1e-5), k
        assert set(got["grads"]) == set(ref["grads"])
        for name, g in ref["grads"].items():
            assert torch.allclose(got["grads"][name], g, rtol=1e-7, atol=1e-10), name
    wn, wr = (0.0, 0.0) if cal else ((LN, 0.0) if case == "CN" else (0.0, LR))
    assert (m["contrast_nce_weight"], m["contrast_rnc_weight"]) == (wn, wr)
    b, n, r = (m[k] for k in KEYS)
    expected = math.sqrt(b * b + (wn * n) ** 2 + (wr * r) ** 2 + 2 * wn * b * n * m[COS[0]] + 2 * wr * b * r * m[COS[1]])
    applied = _shared_norm(ranks[0]["grads"])
    assert applied == pytest.approx(expected, rel=2e-5)  # (3)
    report = {"case": case, "weights": [wn, wr], "norms_one_process": {k: m[k] for k in KEYS + COS},
              "norms_no_checkpoint": {k: plain["metrics"][k] for k in KEYS},
              "max_rel_dev_4rank": max(abs(g["metrics"][k] / m[k] - 1) for g in ranks for k in KEYS),
              "applied_shared_norm_4rank": applied, "expected_from_measured": expected, "rel_dev_applied": abs(applied / expected - 1),
              "max_abs_grad_diff_4rank_vs_one": max(float((g["grads"][name] - t).abs().max()) for g in ranks for name, t in ref["grads"].items())}
    (OUT / f"ddp4_ckpt_{case}.json").write_text(json.dumps(report, indent=1))
