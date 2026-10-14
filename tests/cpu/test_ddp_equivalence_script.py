"""Script della prova (A) del protocollo `w1_gpu_dati_v1` (`scripts/ddp_equivalence.py`) sulla CPU: con `gloo` a 2 processi in float64 le
quattro varianti (riferimento, accumulo, DDP, DDP con accumulo) coincidono col riferimento per 3 aggiornamenti entro ~1e-10, e il verdetto
float32 passa; un cablaggio sbagliato (learning rate diverso) viene visto."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mp = pytest.importorskip("torch.multiprocessing")

from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import WearUsFM  # noqa: E402

from test_ddp_equivalence import WINDOW_S, _config  # noqa: E402
from test_pretraining_loader import _tree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import ddp_equivalence as DE  # noqa: E402


def _batches(tmp_path, cfg, n=3):
    root, mpath = _tree(tmp_path)
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), cfg.loader, {}, WINDOW_S if cfg.loader.time_weighted else None)
    rng = np.random.default_rng(5)
    return [loader.batch(4, rng) for _ in range(n)]


def _worker(rank, world, store, cfg, batches, init, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        report = DE.run_equivalence(cfg, batches, init, rank=rank, world=world, device="cpu", dtype=torch.float64, precisions=("fp32",),
                                    accum=(2, 2))
        if rank == 0:
            torch.save(report, out)
    finally:
        dist.destroy_process_group()


def test_four_variants_agree_on_cpu_and_the_verdict_passes(tmp_path):
    cfg = _config(True, True)  # regole della finestra 1, dropout del muscolo a 0
    batches = _batches(tmp_path, cfg)
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    out = tmp_path / "report.pt"
    mp.spawn(_worker, args=(2, str(tmp_path / "store"), cfg, batches, init, str(out)), nprocs=2, join=True)
    report = torch.load(out, weights_only=False)
    fp32 = report["precisions"]["fp32"]
    assert set(fp32["variants"]) == {"acc", "ddp", "ddp_acc"} and report["updates"] == 3
    for name, rows in fp32["variants"].items():
        assert len(rows) == 3
        for row in rows:
            assert row["loss"] < 1e-10 and row["grad"] < 1e-9 and row["params"] < 1e-6 and row["teacher"] < 1e-6, (name, row)
    assert report["passed"] == {"fp32": True, "all": True}


def test_the_verdict_sees_a_wiring_difference():
    """Un learning rate raddoppiato in una variante sposta i pesi ben oltre la soglia float32 (1e-2 dello spostamento)."""
    ref = [{"metrics": {"total": 1.0}, "grad": torch.ones(4), "params": torch.full((4,), 1.0), "teacher": torch.full((4,), 1.0)}]
    p0 = torch.zeros(4)
    bad = [{"metrics": {"total": 1.0}, "grad": torch.ones(4), "params": torch.full((4,), 2.0), "teacher": torch.full((4,), 1.0)}]
    report = {"precisions": {"fp32": {"variants": {"ddp": DE._compare(bad, ref, p0)}}}}
    assert DE._verdict(report, {"fp32": ref}) == {"fp32": False, "all": False}
    good = {"precisions": {"fp32": {"variants": {"ddp": DE._compare(ref, ref, p0)}}}}
    assert DE._verdict(good, {"fp32": ref})["all"]
