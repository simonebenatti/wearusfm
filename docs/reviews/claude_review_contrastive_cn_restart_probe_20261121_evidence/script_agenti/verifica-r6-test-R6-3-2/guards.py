"""Esegue le guardie vere di instrument_leg sotto DDP Gloo a 2 rank con riscaldamenti sostitutivi difettosi."""
import copy, json, random, sys, tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np, torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
sys.path.insert(0, "tests/cpu")
import test_cn_restart_leg as T
leg = T.leg

def bad_opt(ddp, student, teacher, micros, cfg, ctx, device, amp, *, contrastive):
    r = T.fake_accumulate(ddp, student, teacher, micros, cfg, ctx, device, amp, contrastive=contrastive)
    if contrastive.ids == "WARM":  # solo nel riscaldamento
        opt = OPT[0]; p = next(student.parameters()); opt.state[p]["exp_avg"] = torch.zeros_like(p)
    return r
OPT = []

def case(rank, world, dest, rv, name):
    dist.init_process_group("gloo", init_method=rv, rank=rank, world_size=world, timeout=timedelta(seconds=30))
    out = {}
    try:
        torch.manual_seed(252); student = T.TestModelFixture(); teacher = copy.deepcopy(student)
        ddp = DistributedDataParallel(student, bucket_cap_mb=.001)
        opt = torch.optim.AdamW(student.parameters(), lr=.001); OPT[:] = [opt]
        batch = [torch.randn(3, 4) + rank]; batch[0].anchor_targets = [torch.tensor(2.)]
        runtime = T.FixtureRuntime(); ctx = SimpleNamespace(rank=rank, world_size=world)
        acc = T.fake_accumulate; variant = "ordinary"
        if name == "opt_state":
            acc = bad_opt; variant = "warm"
            orig = leg.copy.deepcopy
            def dc(x, *a, **k):
                y = orig(x, *a, **k)
                if isinstance(x, T.FixtureRuntime): y.ids = "WARM"
                return y
            dcp = patch.object(leg.copy, "deepcopy", dc)
        else:
            dcp = patch.object(leg, "PARAMETERS", 270)
        if name == "residual":
            student(batch[0]).backward()
        if name == "first_step":
            runtime.step = 545
        d = Path(dest) / name / str(rank); d.mkdir(parents=True, exist_ok=True)
        try:
            with (patch.object(leg.trainer, "_accumulate_terms", acc), patch.object(leg.trainer, "_micro_digest", leg.state_sha256),
                  patch.object(leg, "device_identity", lambda _: {"host": "g", "device_index": rank, "device_name": "cpu", "gpu_uuid": "t"}),
                  dcp, leg.instrument_leg(variant, d)):
                opt.load_state_dict(copy.deepcopy(opt.state_dict()))
                leg.trainer._accumulate_terms(ddp, student, teacher, batch, None, ctx, "cpu", None, contrastive=runtime)
            out["error"] = None
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        b = d / f"boundary.rank{rank}.json"
        if b.exists():
            bj = json.loads(b.read_text())
            out["diff_keys"] = sorted(k for k in bj["before"] if bj["before"][k] != bj["after_warmup"][k])
        (Path(dest) / f"{name}.{rank}.json").write_text(json.dumps(out))
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    for name in ("opt_state", "residual", "first_step"):
        tmp = tempfile.mkdtemp(dir=sys.argv[1])
        mp.spawn(case, args=(2, tmp, (Path(tmp) / "store").as_uri(), name), nprocs=2, join=True)
        for r in range(2):
            print(name, r, (Path(tmp) / f"{name}.{r}.json").read_text())
