"""CPU/gloo probe: in which iteration does the DDP Reducer actually use rebuilt buckets,
and what does _get_ddp_logging_data report when sampled after backward (as the trace does)."""
import os, sys, json, torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def run(rank, ws, use_hook, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    torch.manual_seed(0)
    # forward order != registration order so the rebuilt layout differs from the initial one
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.a = torch.nn.Linear(64, 64); s.b = torch.nn.Linear(64, 64); s.c = torch.nn.Linear(64, 8)
        def forward(s, x):
            return s.c(torch.relu(s.b(torch.relu(s.a(x)))))
    m = DDP(M(), bucket_cap_mb=0.01)
    log = []
    if use_hook:
        def hook(state, bucket):
            log.append((bucket.index(), int(bucket.buffer().numel())))
            fut = dist.all_reduce(bucket.buffer().div_(ws), async_op=True).get_future()
            return fut.then(lambda f: f.value()[0])
        m.register_comm_hook(None, hook)
    for it in range(4):
        log.clear()
        out = m(torch.randn(4, 64)).sum()
        py_flag = getattr(m, "_has_rebuilt_buckets", None)
        out.backward()
        d = m._get_ddp_logging_data()
        if rank == 0:
            print(json.dumps({"hook": use_hook, "loop_iter": it, "py_has_rebuilt_after_fwd": py_flag,
                              "log_iteration": d.get("iteration"), "log_has_rebuilt": d.get("has_rebuilt_buckets"),
                              "log_bucket_sizes": d.get("bucket_sizes"), "log_rebuilt_sizes": d.get("rebuilt_bucket_sizes"),
                              "buckets_reduced_this_iter": list(log) if use_hook else None}), flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    print("torch", torch.__version__)
    mp.spawn(run, args=(2, False, 29611), nprocs=2)
    mp.spawn(run, args=(2, True, 29612), nprocs=2)
