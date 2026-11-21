"""R3: bitwise equivalence of native DDP reduction vs ReductionObservation hook (CPU/Gloo, 4 ranks).

Scenarios (same seeds/data):
  A  no hook, ordinary                         (historical CN_A analogue)
  B  hook, ordinary, observe update 0          (diagnostic 'ordinary' leg)
  D  no hook, null warm backward before upd 0  (reference for warm)
  C  hook, null warm backward, observe upd 0   (diagnostic 'warm' leg)
Model: mixed fp32/fp64, includes subnormal-gradient parameter and odd mantissas.
AdamW + clip_grad_norm_, 5 updates, record pre-clip grads at each update and final params.
Also exercises record_optimizer_step from the leg script.
"""
from datetime import timedelta
import os, sys, json, tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

CLONE = sys.argv[1]
OUT = Path(sys.argv[2])
sys.path[:0] = [CLONE + "/scripts/diagnostics", CLONE + "/scripts", CLONE + "/src"]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 96)
        self.b = torch.nn.Linear(96, 80)
        self.c = torch.nn.Linear(80, 7)
        self.tiny = torch.nn.Parameter(torch.randn(33))            # subnormal grads
        self.f64 = torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float64))

    def forward(self, x):
        h = torch.tanh(self.a(x)) * 1.000123
        h = torch.relu(self.b(h))
        y = self.c(h).sum() / 3.0
        y = y + (self.tiny * torch.linspace(1, 3, 33)).sum() * 3e-39   # grads ~1e-39 (fp32 subnormal)
        z = (self.f64 @ torch.ones(3, 2, dtype=torch.float64) * x[:5, None].double() / 7).square().sum()
        return y + z.float()


def run(rank, world, variant, folder):
    from contrastive_cn_restart_observe import ReductionObservation, load_observation
    from contrastive_cn_restart_leg import record_optimizer_step
    torch.manual_seed(5)
    model = Net()
    ddp = DistributedDataParallel(model, bucket_cap_mb=0.01)  # several buckets after rebuild
    hook = variant in ("B", "C")
    obs = ReductionObservation(ddp, model, expected_parameters=len(list(model.parameters()))) if hook else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    gen = torch.Generator().manual_seed(100 + rank)
    xs = [torch.randn(64, generator=gen) * (1 + rank) for _ in range(6)]
    if variant in ("C", "D"):
        opt.zero_grad(set_to_none=True)
        ddp(xs[0]).backward()   # null warm backward on the same input
        opt.zero_grad(set_to_none=True)
    grads, hooks = [], []
    for u in range(5):
        opt.zero_grad(set_to_none=True)
        if hook and u == 0:
            obs.start()
        ddp(xs[u]).backward()
        grads.append([p.grad.detach().clone() for p in model.parameters()])
        if hook and u == 0:
            obs.finish(folder, rank)
            hooks.extend(record_optimizer_step(opt, model, Path(folder), rank,
                                               expected_count=len(list(model.parameters()))))
        elif hooks:
            while hooks:
                hooks.pop().remove()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        opt.step()
    sizes = ddp._get_ddp_logging_data().get("rebuilt_bucket_sizes")
    return grads, [p.detach().clone() for p in model.parameters()], sizes


def worker(rank, world, root, addr):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=addr, rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        res = {}
        for v in "ABDC":
            folder = Path(root) / v
            res[v] = run(rank, world, v, str(folder))
            dist.barrier()
        torch.save({k: (g, p, s) for k, (g, p, s) in res.items()}, Path(root) / f"res{rank}.pt")
    finally:
        dist.destroy_process_group()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(4, str(OUT), (OUT / "store").as_uri()), nprocs=4, join=True)
    from contrastive_cn_restart_observe import load_observation
    report = {}
    for rank in range(4):
        r = torch.load(OUT / f"res{rank}.pt")
        def same(x, y):
            return all(torch.equal(a, b) for ua, ub in zip(x, y) for a, b in zip(ua, ub))
        report[rank] = {
            "A==B grads all updates": same(r["A"][0], r["B"][0]),
            "A==B final params": all(torch.equal(a, b) for a, b in zip(r["A"][1], r["B"][1])),
            "D==C grads all updates": same(r["D"][0], r["C"][0]),
            "D==C final params": all(torch.equal(a, b) for a, b in zip(r["D"][1], r["C"][1])),
            "A==D grads upd0 (construction vs rebuilt)": all(torch.equal(a, b) for a, b in zip(r["A"][0][0], r["D"][0][0])),
            "A==D final": all(torch.equal(a, b) for a, b in zip(r["A"][1], r["D"][1])),
            "rebuilt sizes": {k: r[k][2] for k in "ABCD"},
            "tiny grad subnormal upd0": bool(((r["A"][0][0][0].abs() < 1.17e-38) & (r["A"][0][0][0] != 0)).all()),
        }
        for v in "BC":
            rec, vec = load_observation(OUT / v, rank, expected_parameters=8)
            report[rank][v + " buckets"] = [(b["dtype"], b["nbytes"], b["parameter_ids"]) for b in rec["buckets"]]
            names = [p["name"] for p in rec["parameters"]]
            ref = r["A" if v == "B" else "D"][0][0]
            report[rank][v + " reduced==native grad"] = all(torch.equal(vec["reduced"][i], ref[i]) for i in range(8))
            post = json.loads((OUT / v / f"post_update.rank{rank}.json").read_text())
            report[rank][v + " post_update rows"] = len(post["rows"])
    # local average check against reduced (rank-local unscaled)
    for v in "BC":
        vecs = [load_observation(OUT / v, r_, expected_parameters=8)[1] for r_ in range(4)]
        ok = []
        for i in range(8):
            loc = torch.stack([vv["local"][i] for vv in vecs])
            ok.append(bool(torch.allclose(loc.mean(0), vecs[0]["reduced"][i], rtol=1e-6, atol=1e-45)))
        report["local mean ~ reduced " + v] = ok
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
