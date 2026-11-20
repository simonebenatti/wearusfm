"""DDP all'init sovrascrive i parametri dei rank >0 con quelli del rank 0? (gloo, CPU, 2 processi)."""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def work(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(100 + rank)  # parametri diversi per rank
    m = torch.nn.Linear(3, 2)
    before = m.weight.detach().clone()
    ddp = torch.nn.parallel.DistributedDataParallel(m)
    after = m.weight.detach().clone()
    gathered = [None] * world
    dist.all_gather_object(gathered, (rank, before.tolist(), after.tolist()))
    if rank == 0:
        r0_before = gathered[0][1]
        res = {r: {"changed_by_wrap": b != a, "equals_rank0_before": a == r0_before} for r, b, a in gathered}
        with open(out, "w") as f:
            f.write(repr({"torch": torch.__version__, **res}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    out = sys.argv[1]
    mp.spawn(work, args=(2, 29517, out), nprocs=2, join=True)
