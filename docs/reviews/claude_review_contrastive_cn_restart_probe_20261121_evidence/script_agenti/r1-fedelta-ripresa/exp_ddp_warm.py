"""R1 experiment: real WearUsFM + contrastive head + real ContrastiveRuntime + real
trainer._accumulate_terms under a real Gloo DDP (2 ranks, CPU). Three variants per rank:
  nohook   : plain trainer path (no instrumentation) = reference
  ordinary : instrument_leg('ordinary')
  warm     : instrument_leg('warm')  (null warm-up backward before the observed one)
Two optimizer updates (544->545, 545->546) with muscle dropout 0.4 and global-gradient
diagnostics active at 545, so RNG, collectives and autograd.grad are exercised.
Simulation notes: CPU/Gloo, not CUDA/NCCL; device_identity is patched.
"""
import copy, json, random, sys, os
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CLONE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path[:0] = [str(CLONE / "tests/cpu"), str(CLONE / "src"), str(CLONE / "scripts"), str(CLONE / "scripts/diagnostics")]


def run(rank, world, out, store):
    import contrastive_cn_restart_leg as leg
    from test_contrastive_runtime import _config, _pose_setup
    from wearusfm.data.pretraining_loader import PretrainLoader
    from wearusfm.model.fm import WearUsFM
    from wearusfm.training.contrastive_objectives import ContrastiveRuntime, attach_contrastive_head, ContrastiveConfig
    from wearusfm.training.contrastive_memory import WindowIdentity
    from wearusfm.training.jepa import make_teacher
    R = leg.trainer
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=world, timeout=timedelta(seconds=120))
    out = Path(out)
    data = out / "data"
    root, manifest, pose = _pose_setup(data) if False else (data / "root", None, None)
    root, manifest, pose = json.loads((out / "setup.json").read_text()).values()
    root, manifest, pose = Path(root), Path(manifest), Path(pose)
    cfg = _config(pose, arm="CN")
    c = dict(cfg.contrastive); c.update(grad_every=545)
    cfg = replace(cfg, contrastive=asdict(ContrastiveConfig(**c)), model=replace(cfg.model, muscle_dropout=.4))
    ctx = R.DistContext(rank, world, rank)
    loader = PretrainLoader(R.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader)
    batches = [loader.batch(4, np.random.default_rng([42, rank, s])) for s in (544, 545)]
    results = {}
    for variant in ("nohook", "ordinary", "warm"):
        folder = out / variant
        if rank == 0:
            folder.mkdir()
        dist.barrier()
        torch.manual_seed(cfg.seed)
        student = attach_contrastive_head(WearUsFM(cfg.model))
        teacher = make_teacher(student)
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
        random.seed(7 + rank); np.random.seed(7 + rank); torch.manual_seed(1000 + rank)
        model = R._ddp_wrap(student, cfg, ctx, "cpu")
        runtime = ContrastiveRuntime(cfg.contrastive)
        runtime.expected_datasets = ("emg2pose",)
        keydim = student.contrastive_head.layers[-1].out_features
        key = torch.nn.functional.normalize(torch.ones(1, keydim), dim=-1)
        ident = WindowIdentity("emg2pose", "other", "other_recording", 0, 0, 50, 2000.)
        runtime.queue.commit(key, [ident], step=0)
        st = runtime.state_dict()
        st["queue"]["last_step"] = st["queue"]["entries"][0]["step"] = 543
        runtime.load_state_dict(st, completed_steps=544)
        n_params = sum(1 for p in student.parameters() if p.requires_grad)
        patches = [patch.object(leg, "device_identity", lambda d: {"host": "gloo", "device_index": rank,
                                                                     "device_name": "cpu", "gpu_uuid": f"u{rank}"}),
                   patch.object(leg, "PARAMETERS", n_params)]
        import contrastive_cn_restart_observe as obs
        orig_init = obs.ReductionObservation.__init__
        def init(self, ddp, student_, *, expected_parameters=270):
            return orig_init(self, ddp, student_, expected_parameters=n_params)
        patches.append(patch.object(obs.ReductionObservation, "__init__", init))
        orig_rec = leg.record_optimizer_step
        patches.append(patch.object(leg, "record_optimizer_step", lambda o, s_, f, r: orig_rec(o, s_, f, r, expected_count=n_params)))
        rec = {}
        with (patches[0], patches[1], patches[2], patches[3],
              leg.instrument_leg(variant, folder) if variant != "nohook" else nullcontext()):
            opt.load_state_dict(copy.deepcopy(opt.state_dict()))
            for step, batch in zip((544, 545), batches):
                for g in opt.param_groups:
                    g["lr"] = 1e-3
                student.train()
                opt.zero_grad(set_to_none=True)
                micros = [R._prepare(batch, "cpu")]
                runtime.prepare(batch, step)
                failure, values = R._accumulate_terms(model, student, teacher, micros, cfg, ctx, "cpu", nullcontext(),
                                                      grad_diag=False, contrastive=runtime)
                assert failure is None, failure
                rec[f"local_grad_{step}"] = leg.state_sha256([p.grad for p in student.parameters() if p.requires_grad])
                rec[f"values_{step}"] = {k: v for k, v in sorted(values.items())}
                assert runtime.before_update() is None
                gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                rec[f"gnorm_{step}"] = float(gn)
                opt.step()
                R.ema_update(teacher, student, .99) if hasattr(R, "ema_update") else None
                runtime.commit()
        rec["student"] = leg.state_sha256(student.state_dict())
        rec["teacher"] = leg.state_sha256(teacher.state_dict())
        rec["opt"] = leg.state_sha256(opt.state_dict())
        rec["runtime"] = leg.state_sha256(runtime.state_dict())
        rec["rng"] = leg.state_sha256(leg.rng_states("cpu"))
        rec["ddp_iteration"] = model._get_ddp_logging_data().get("iteration")
        rec["ddp_buckets"] = model._get_ddp_logging_data().get("rebuilt_bucket_sizes")
        results[variant] = rec
        del model
        dist.barrier()
    (out / f"result.rank{rank}.json").write_text(json.dumps(results, indent=1, default=str))
    dist.destroy_process_group()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    from test_contrastive_runtime import _pose_setup
    root, manifest, pose = _pose_setup(OUT / "data")
    (OUT / "setup.json").write_text(json.dumps({"root": str(root), "manifest": str(manifest), "pose": str(pose)}))
    mp.spawn(run, args=(2, str(OUT), (OUT / "store").as_uri()), nprocs=2, join=True)
    res = [json.loads((OUT / f"result.rank{r}.json").read_text()) for r in range(2)]
    for r, rr in enumerate(res):
        keys = sorted(rr["nohook"])
        for k in keys:
            same = [rr[v][k] == rr["nohook"][k] for v in ("ordinary", "warm")]
            print(f"rank{r} {k:22s} ordinary==nohook {same[0]}  warm==nohook {same[1]}")
        print("ddp", {v: (rr[v]["ddp_iteration"], rr[v]["ddp_buckets"]) for v in rr})
    for r in range(2):
        for v in ("ordinary", "warm"):
            b = json.loads((OUT / v / f"boundary.rank{r}.json").read_text())
            o = json.loads((OUT / v / f"observation.rank{r}.json").read_text())
            print(v, r, "boundary before==after", b["before"] == b["after_warmup"], "buckets", len(o["buckets"]))
        a = json.loads((OUT / "ordinary" / f"observation.rank{r}.json").read_text())["parameters"]
        w = json.loads((OUT / "warm" / f"observation.rank{r}.json").read_text())["parameters"]
        print("rank", r, "local equal all", all(x["local_sha256"] == y["local_sha256"] for x, y in zip(a, w)),
              "reduced equal all", all(x["reduced_sha256"] == y["reduced_sha256"] for x, y in zip(a, w)))
        bo = json.loads((OUT / "ordinary" / f"boundary.rank{r}.json").read_text())["before"]
        bw = json.loads((OUT / "warm" / f"boundary.rank{r}.json").read_text())["before"]
        print("rank", r, "boundary ordinary==warm (excl py/np)", {k for k in bo if bo[k] != bw[k]})
