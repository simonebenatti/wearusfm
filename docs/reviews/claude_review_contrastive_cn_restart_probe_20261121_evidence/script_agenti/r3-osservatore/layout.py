import sys, json, tempfile
from datetime import timedelta
from pathlib import Path
import torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
sys.path[:0]=[sys.argv[1]+"/scripts/diagnostics", sys.argv[1]+"/scripts", sys.argv[1]+"/src"]

from equiv import Net
def w(rank, world, root, addr):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=addr, rank=rank, world_size=world, timeout=timedelta(seconds=60))
    from contrastive_cn_restart_observe import ReductionObservation, load_observation
    out = {}
    for v in ("cold", "hot", "plain"):
        torch.manual_seed(5); m = Net(); ddp = DistributedDataParallel(m, bucket_cap_mb=0.01)
        obs = ReductionObservation(ddp, m, expected_parameters=8) if v != "plain" else None
        x = torch.randn(64)
        if v in ("hot", "plain"):
            ddp(x).backward(); m.zero_grad(set_to_none=True)
        if obs: obs.start()
        ddp(x).backward()
        if obs: obs.finish(Path(root)/v, rank)
        ddp(x).backward()  # one more so logger lag is irrelevant
        out[v] = {k: ddp._get_ddp_logging_data().get(k) for k in ("bucket_sizes","rebuilt_bucket_sizes","rebuilt_per_bucket_param_indices","has_rebuilt_buckets")}
    if rank == 0:
        import contrastive_cn_restart_compare as c
        cold = json.loads((Path(root)/"cold"/"observation.rank0.json").read_text())
        hot = json.loads((Path(root)/"hot"/"observation.rank0.json").read_text())
        print(json.dumps(out["plain"]))
        print("layouts() ->", c.layouts(cold, hot, out["plain"]))
    dist.destroy_process_group()
if __name__ == "__main__":
    root = Path(tempfile.mkdtemp(dir=sys.argv[2]))
    mp.spawn(w, args=(4, str(root), (root/"store").as_uri()), nprocs=4, join=True)
