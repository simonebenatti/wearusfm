"""CPU/gloo multi-rank common-fork experiment with the real trainer (review r2, read-only on the package)."""
import json, os, sys, time, copy, shutil
from dataclasses import replace
from datetime import timedelta
from pathlib import Path


def apply_mutation(mutation, rank, phase, R, torch):
    if phase != "reload_b" or mutation == "none":
        return
    if mutation == "rng_rank2" and rank == 2:
        original = torch.set_rng_state
        target = {}
        def skip_restore(state):
            if "fork" not in target:
                import glob
                fork = torch.load(sorted(glob.glob(os.environ["R2_FORK_GLOB"]))[0], map_location="cpu", weights_only=False)
                target["fork"] = fork["rank_rng"][2]["torch_rng"]
            if torch.equal(state.cpu(), target["fork"]):
                return None  # resume bug: restored CPU RNG of rank 2 not applied
            return original(state)
        torch.set_rng_state = skip_restore
    elif mutation == "adam_zero":
        load = torch.optim.AdamW.load_state_dict
        def zeroing(opt, state):
            value = load(opt, state)
            for s in opt.state.values():
                s["exp_avg"].zero_()
            return value
        torch.optim.AdamW.load_state_dict = zeroing
    elif mutation == "lr_shift":
        lr_at = R._lr_at
        R._lr_at = lambda step, cfg: lr_at(step + 1, cfg)
    elif mutation == "queue_age":
        from wearusfm.training.contrastive_objectives import ContrastiveRuntime
        load = ContrastiveRuntime.load_state_dict
        def drop_oldest(self, state, completed):
            state = copy.deepcopy(state)
            state["queue"]["entries"] = state["queue"]["entries"][1:]
            return load(self, state, completed)
        ContrastiveRuntime.load_state_dict = drop_oldest
    elif mutation == "teacher_noise":
        make = R.make_teacher
        def noisy(student):
            teacher = make(student)
            return teacher
        # perturb after the checkpoint load: patch load_state_dict of the teacher module class used by make_teacher
        import torch.nn as nn
        load = nn.Module.load_state_dict
        flag = {"n": 0}
        def perturbed(self, state_dict, *a, **k):
            value = load(self, state_dict, *a, **k)
            flag["n"] += 1
            if flag["n"] == 2:  # second load in train(): the teacher
                with torch.no_grad():
                    for p in self.parameters():
                        p.mul_(1.0 + 1e-6)
            return value
        nn.Module.load_state_dict = perturbed
    elif mutation == "batch_skip" and rank == 1:
        pass


def worker(rank, world, store, cfg, manifest, data, out, phase, initial_sha, provenance, fork, end, mutation, num_workers):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    import contrastive_target_resume as C
    from wearusfm.training import run as R
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        apply_mutation(mutation, rank, phase, R, torch)
        with C.trace_resume_updates(Path(out) / f"trace-{rank}.jsonl", phase, fork_step=fork, end_step=end):
            summary = R.train(cfg, manifest, [data], Path(out), log=lambda _: None, dist=R.DistContext(rank, world, rank),
                              num_workers=num_workers, expected_contrastive_initial_sha256=initial_sha,
                              contrastive_provenance=provenance)
        if summary["steps"] != end or summary["stopped"] != "max_steps":
            raise RuntimeError(str(summary))
    finally:
        dist.destroy_process_group()


def spawn(world, args, limit=300):
    import torch.multiprocessing as mp
    store = Path(args[3]).parent / f".store-{time.time_ns()}"
    context = mp.spawn(worker, args=(world, str(store), *args), nprocs=world, join=False)
    deadline = time.monotonic() + limit
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("spawn exceeded limit")
    finally:
        for p in context.processes:
            if p.is_alive():
                p.terminate()
        for p in context.processes:
            p.join(timeout=5)


def main():
    T, out, arm, world = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    mutation = sys.argv[5] if len(sys.argv) > 5 else "none"
    num_workers = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    fixed_every = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    reuse = Path(sys.argv[8]) if len(sys.argv) > 8 else None
    sys.path[:0] = [str(T / 'src'), str(T / 'scripts'), str(T / 'tests/cpu')]
    import torch
    torch.set_num_threads(1)
    import contrastive_target_resume as C
    from wearusfm.training import run as R
    from wearusfm.training.pairing import state_sha256
    from test_contrastive_runtime import _config
    from test_pose_training import _pose_setup
    FORK, END = 4, 8
    out.mkdir(parents=True, exist_ok=False)
    data, manifest, pose = _pose_setup(out / "data") if reuse is None else (None, None, None)
    if reuse is not None:
        meta = json.loads((reuse / "meta.json").read_text())
        data, manifest, pose = Path(meta["data"]), Path(meta["manifest"]), Path(meta["pose"])
    base = _config(pose, arm)
    cfg = replace(base, max_steps=END, batch_size=8, model=replace(base.model, muscle_dropout=.4), log_batch_origin=True,
                  schedule=replace(base.schedule, total_steps=20, keep_checkpoints=(FORK,)), prefetch=True,
                  fixed_eval_every=fixed_every, fixed_eval_batches=1, eval_every=1000)
    if os.environ.get("R2_WARMUP"):
        cfg = replace(cfg, warmup_steps=int(os.environ["R2_WARMUP"]), schedule=replace(cfg.schedule, total_steps=40))
    provenance = {"source_sha256": "test", "input_sha256": {}}
    paths = {phase: out / phase for phase in C.PHASES}
    t0 = time.time()
    if reuse is None:
        calibration = replace(cfg, contrastive={**cfg.contrastive, "calibration_only": True})
        R.train(calibration, manifest, [data], out / "calibration", log=lambda _: None, contrastive_provenance=provenance)
        initial = C.checkpoint(out / "calibration/checkpoint.pt")["student"]
        initial_sha = state_sha256(initial)
        spawn(world, (cfg, manifest, data, str(paths["continuous"]), "continuous", initial_sha, provenance, FORK, END, mutation, num_workers))
        (out / "meta.json").write_text(json.dumps({"data": str(data), "manifest": str(manifest), "pose": str(pose)}))
    else:
        shutil.copytree(reuse / "calibration", out / "calibration")
        shutil.copytree(reuse / "continuous", paths["continuous"])
        initial = C.checkpoint(out / "calibration/checkpoint.pt")["student"]
        initial_sha = state_sha256(initial)
    fork = paths["continuous"] / f"checkpoint_step{FORK:07d}.pt"
    fork_sha = C.file_sha(fork)
    t1 = time.time()
    for phase in C.PHASES[1:]:
        C.prepare_reload(fork, fork_sha, paths[phase], cfg, provenance, initial_sha, phase=phase,
                         fork_step=FORK, end_step=END, ranks=world)
        spawn(world, (cfg, manifest, data, str(paths[phase]), phase, initial_sha, provenance, FORK, END, mutation, num_workers))
    t2 = time.time()
    result = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=world, fork_step=FORK, end_step=END)
    slim = {k: v for k, v in result.items() if k != "pairs"}
    slim["pairs"] = {name: {k: v for k, v in pair.items() if k not in ("floating_tensor_relative_to_left", "by_block_relative_to_left_displacement_from_fork")}
                     for name, pair in result["pairs"].items()}
    for name, pair in result["pairs"].items():
        slim["pairs"][name]["n_floating_tensors"] = len(pair.get("floating_tensor_relative_to_left", {}))
        diffs = pair.get("input_rng_counter_order_differences", [])
        slim["pairs"][name]["input_rng_counter_order_differences"] = diffs[:12] + ([{"more": len(diffs) - 12}] if len(diffs) > 12 else [])
    traces = {f"{phase}/rank{r}": len((paths[phase] / f"trace-{r}.jsonl").read_text().splitlines()) for phase in C.PHASES for r in range(world)}
    ddp = {f"{phase}/rank{r}": (paths[phase] / f"trace-{r}.ddp.jsonl").exists() for phase in C.PHASES for r in range(world)}
    report = {"arm": arm, "world": world, "mutation": mutation, "num_workers": num_workers, "fixed_eval_every": fixed_every,
              "seconds": {"calibration+continuous": t1 - t0, "two_reloads": t2 - t1}, "trace_rows": traces,
              "ddp_trace_files": ddp, "result": slim}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"mutation": mutation, "status": result["status"], "passed": result["passed"],
                      "boundary_mismatches": result["boundary_mismatches"][:8],
                      "pairs": {n: (p["passed"], p.get("failure_reasons", [])[:3], len(p["input_rng_counter_order_differences"]),
                                    p.get("relative_to_left_displacement_from_initial"))
                                for n, p in result["pairs"].items()},
                      "trace_rows": sorted(set(traces.values())), "seconds": report["seconds"]}, default=str))


if __name__ == "__main__":
    main()
