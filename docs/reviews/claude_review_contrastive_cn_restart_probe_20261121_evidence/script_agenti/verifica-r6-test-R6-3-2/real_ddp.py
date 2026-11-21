"""null_backward con il vero _accumulate_terms sotto DDP Gloo a 2 rank (stesso montaggio del test test_null_shadow)."""
import sys, tempfile
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.multiprocessing as mp
sys.path.insert(0, "tests/cpu")

def worker(rank, world, base, rv):
    dist.init_process_group("gloo", init_method=rv, rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        torch.set_num_threads(1)
        import contrastive_cn_restart_leg as leg
        from test_contrastive_runtime import _config, _pose_setup
        from wearusfm.data.pretraining_loader import PretrainLoader
        from wearusfm.model.fm import WearUsFM
        from wearusfm.training.contrastive_objectives import ContrastiveRuntime, attach_contrastive_head
        from wearusfm.training.contrastive_memory import WindowIdentity
        from wearusfm.training.jepa import make_teacher
        tmp = Path(base) / f"r{rank}"; tmp.mkdir()
        root, manifest, pose = _pose_setup(tmp)
        cfg = _config(pose, arm="CN")
        loader = PretrainLoader(leg.trainer.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
        batch = loader.batch(4, np.random.default_rng(42 + rank))
        micros = [leg.trainer._prepare(batch, "cpu")]
        torch.manual_seed(0)
        student = attach_contrastive_head(WearUsFM(cfg.model)); teacher = make_teacher(student)
        ctx = leg.trainer.DistContext(rank=rank, world_size=world, local_rank=rank)
        ddp = leg.trainer._ddp_wrap(student, cfg, ctx, "cpu")
        opt = torch.optim.AdamW(student.parameters())
        runtime = ContrastiveRuntime(cfg.contrastive); runtime.expected_datasets = ("emg2pose",)
        keydim = student.contrastive_head.layers[-1].out_features
        key = torch.nn.functional.normalize(torch.ones(1, keydim), dim=-1)
        runtime.queue.commit(key, [WindowIdentity("emg2pose", "other", "other_recording", 0, 0, 50, 2000.)], step=0)
        st = runtime.state_dict(); st["queue"]["last_step"] = st["queue"]["entries"][0]["step"] = 543
        runtime.load_state_dict(st, completed_steps=544); runtime.prepare(batch, 544)
        before = leg.snapshot(student, teacher, opt, runtime, micros, "cpu")
        def warm(shadow):
            return leg.trainer._accumulate_terms(ddp, student, teacher, micros, cfg, ctx, "cpu", nullcontext(), contrastive=shadow)
        leg.null_backward(warm, student, teacher, opt, runtime, micros, "cpu")
        after = leg.snapshot(student, teacher, opt, runtime, micros, "cpu")
        diff = sorted(k for k in before if before[k] != after[k])
        fail, _ = leg.trainer._accumulate_terms(ddp, student, teacher, micros, cfg, ctx, "cpu", nullcontext(), contrastive=runtime)
        grads = sum(p.grad is not None for p in student.parameters())
        print(f"rank{rank} diff={diff} real_failure={fail} grads={grads} pending={runtime.pending is not None}", flush=True)
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    base = tempfile.mkdtemp(dir=sys.argv[1])
    mp.spawn(worker, args=(2, base, (Path(base) / "store").as_uri()), nprocs=2, join=True)
