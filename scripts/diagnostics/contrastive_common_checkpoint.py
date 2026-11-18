"""Four-GPU restart diagnostic on an authenticated production checkpoint per arm.

One live process saves at update 580 and continues. Two fresh torchrun launches
load byte-identical copies of this snapshot. Instrumentation observes production
trainer calls, without changing their arguments or their return values.
"""
import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from contrastive_prepare import config_from_dict, source_digest
from contrastive_benchmark_readout import _tensor_distance
from wearusfm.training import run as trainer
from wearusfm.training.contrastive_objectives import ContrastiveRuntime
from wearusfm.training.pairing import state_sha256, batch_sha256

EVIDENCE = ROOT / "results/infra/contrastive_comparison_20261116"
CHECKPOINTS = {
    "CN": "668bb2119d728126c7250903882376099e5be607bfe253a0552c44dedc7eddd5",
    "CR": "574220fa015a341a5caa9786a65085fb960f40ba144f8b4b39ac5dad347b31c8",
}
SOURCE = "46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320"
START, FORK, END = 576, 580, 588
PHASES = ("continuous", "reload_a", "reload_b")
FLOAT_EVIDENCE = ("total", "grad_norm", "contrast_projector_grad_norm", "grad_clip_factor")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint(path):
    return torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)


def diagnostic_seed(source, target, expected_sha, fork_step):
    """Only request an additional named save in a separate, verified seed copy."""
    if file_sha(source) != expected_sha or target.exists():
        raise ValueError("checkpoint source pin or fresh destination")
    original = checkpoint(source)
    keep = original["config"]["schedule"]["keep_checkpoints"]
    if fork_step in keep or fork_step <= original["step"]:
        raise ValueError("fork must be a new future save point")
    original_digest = state_sha256(original)
    original["config"]["schedule"]["keep_checkpoints"] = sorted([*keep, fork_step])
    torch.save(original, target)
    disk = checkpoint(target)
    if disk["config"]["schedule"]["keep_checkpoints"] != sorted([*keep, fork_step]):
        raise ValueError("fork save point missing on disk")
    disk["config"]["schedule"]["keep_checkpoints"] = keep
    if state_sha256(disk) != original_digest:
        raise ValueError("diagnostic copy changed a field outside the save list")
    return {"source_sha256": expected_sha, "seed_sha256": file_sha(target),
            "source_payload_sha256": original_digest, "extra_keep_checkpoint": fork_step}


def runtime_rng(device):
    return {"torch_rng": torch.get_rng_state(), "cuda_rng": (
        torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)}


def sha_fields(values):
    return {name: state_sha256(value) for name, value in values.items()}


@contextmanager
def trace_updates(path, fork_step):
    """Record real consumed batches, actual device inputs and RNG on each rank."""
    prepare = ContrastiveRuntime.prepare
    accumulate = trainer._accumulate_terms
    load_optimizer = torch.optim.AdamW.load_state_dict
    partial = {}
    loaded = []

    def observe_load(opt, state):
        value = load_optimizer(opt, state)
        loaded.append(opt)
        return value

    def observe_batch(runtime, batch, step):
        if step >= fork_step:
            partial.update(update=step + 1, batch_sha256=batch_sha256(batch))
        return prepare(runtime, batch, step)

    def observe_backward(model, student, teacher, micros, cfg, ctx, device, amp, **kwargs):
        runtime = kwargs["contrastive"]
        if runtime.step >= fork_step:
            if partial["update"] != runtime.step + 1:
                raise ValueError("trace batch/backward order")
            partial["input_sha256"] = trainer._micro_digest(micros)
            partial["rng_before_sha256"] = state_sha256(runtime_rng(device))
            if runtime.step == fork_step:
                opt, = loaded
                partial["boundary"] = sha_fields({
                    "student": student.state_dict(), "teacher": teacher.state_dict(),
                    "optimizer_moments": opt.state_dict()["state"],
                    "optimizer_groups_at_next_lr": opt.state_dict()["param_groups"],
                    "contrastive_state": runtime.state_dict(),
                })
            stream.write(json.dumps(partial, allow_nan=False) + "\n")
            stream.flush()
            partial.clear()
        result = accumulate(model, student, teacher, micros, cfg, ctx, device, amp, **kwargs)
        if runtime.step >= fork_step and hasattr(model, "_get_ddp_logging_data"):
            # A new group may build reduction buckets in a different order. These
            # optional backend observations are separate from the input/RNG gate.
            logging_data = model._get_ddp_logging_data()
            fields = ("iteration", "has_rebuilt_buckets", "bucket_sizes",
                      "rebuilt_bucket_sizes", "rebuilt_per_bucket_param_indices")
            bucket_log.write(json.dumps({"update": runtime.step + 1, **{
                key: logging_data[key] for key in fields if key in logging_data}}) + "\n")
            bucket_log.flush()
        return result

    with path.open("x") as stream, ExitStack() as stack:
        bucket_log = stack.enter_context(path.with_suffix(".ddp.jsonl").open("x"))
        stack.enter_context(patch.object(ContrastiveRuntime, "prepare", observe_batch))
        stack.enter_context(patch.object(torch.optim.AdamW, "load_state_dict", observe_load))
        stack.enter_context(patch.object(trainer, "_accumulate_terms", observe_backward))
        yield


def job_for(arm):
    return read(EVIDENCE / "releases/bf16_20261118" / f"full_{arm}.job.json")


def rank_phase(arm, folder, phase):
    job = job_for(arm)
    seed_path = folder / phase / "checkpoint.pt"
    pin = read(folder / phase / "start.json")["sha256"]
    if file_sha(seed_path) != pin:
        raise ValueError("phase starts from wrong checkpoint bytes")
    seed = checkpoint(seed_path)
    if seed["step"] != (START if phase == "continuous" else FORK):
        raise ValueError("wrong checkpoint step")
    cfg = replace(config_from_dict(seed["config"]), max_steps=END)
    provenance = seed["contrastive_provenance"]
    initial = seed["contrastive_initial_sha256"]
    del seed
    paths = {name: Path(row["path"]) for name, row in job["inputs"].items()}
    ctx = trainer.dist_from_env()
    if ctx.world_size != 4 or not torch.cuda.is_available():
        raise RuntimeError("requires four CUDA ranks")
    torch.cuda.set_device(ctx.local_rank)
    torch.distributed.init_process_group("nccl")
    try:
        with trace_updates(folder / phase / f"trace-{ctx.rank}.jsonl", FORK):
            summary = trainer.train(
                cfg, paths["manifest"], [Path(s) for s in job["roots"]], folder / phase,
                scales=read(paths["scales"])["scales"], window_s=read(paths["window_seconds"])["window_s"],
                device=f"cuda:{ctx.local_rank}", dist=ctx, num_workers=6,
                expected_contrastive_initial_sha256=initial, contrastive_provenance=provenance)
        if summary["steps"] != END or summary["stopped"] != "max_steps":
            raise RuntimeError(f"{phase} stopped early: {summary['stopped']}")
    finally:
        torch.distributed.destroy_process_group()


def suffix(path, step_field, start, end, allow_prefix=False):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if allow_prefix:
        rows = [row for row in rows if row[step_field] > start]
    if [row[step_field] for row in rows] != list(range(start + 1, end + 1)):
        raise ValueError(f"incomplete or duplicate trace in {path}")
    return rows


def cpu_analysis(folder, fork, end, ranks):
    boundary = checkpoint(folder / "continuous" / f"checkpoint_step{fork:07d}.pt")
    groups = copy.deepcopy(boundary["optimizer"]["param_groups"])
    for group in groups:
        group["lr"] = trainer._lr_at(fork, config_from_dict(boundary["config"]))
    expected = sha_fields({"optimizer_groups_at_next_lr": groups, "student": boundary["student"], "teacher": boundary["teacher"],
                           "optimizer_moments": boundary["optimizer"]["state"],
                           "contrastive_state": boundary["contrastive_state"]})
    traces = {}
    mismatches = []
    for phase in PHASES:
        for rank in range(ranks):
            rows = suffix(folder / phase / f"trace-{rank}.jsonl", "update", fork, end)
            traces[phase, rank] = rows
            for key, digest in expected.items():
                if rows[0]["boundary"][key] != digest:
                    mismatches.append([phase, rank, key])
            expected_rng = (boundary["rank_rng"][rank] if ranks > 1 else {
                "torch_rng": boundary["torch_rng"], "cuda_rng": None})
            if rows[0]["rng_before_sha256"] != state_sha256(expected_rng):
                mismatches.append([phase, rank, "rng_before_sha256"])
    checkpoints = {phase: checkpoint(folder / phase / "checkpoint.pt") for phase in PHASES}
    metrics = {phase: suffix(folder / phase / "metrics.jsonl", "step", fork, end, True) for phase in PHASES}
    pairs = {}
    for left, right in itertools.combinations(PHASES, 2):
        trace_diff = []
        for rank in range(ranks):
            for a, b in zip(traces[left, rank], traces[right, rank]):
                for key in a.keys() | b.keys():
                    if a.get(key) != b.get(key):
                        trace_diff.append({"update": a["update"], "rank": rank, "field": key})
        a, b = checkpoints[left], checkpoints[right]
        if a["step"] != end or b["step"] != end:
            raise ValueError("incomplete final checkpoint")
        model_distance = {key: _tensor_distance(a[key], b[key], boundary[key]) for key in ("student", "teacher")}
        checks = ["optimizer", "contrastive_state", "config", "data_signature", "stream", "alarm_counts",
                  "contrastive_initial_sha256", "contrastive_provenance", "pose_stats_sha256",
                  "rank_rng" if ranks > 1 else "torch_rng"]
        exact = {key: state_sha256(a[key]) == state_sha256(b[key]) for key in checks}
        def nonnumeric_contract(state):
            optimizer = state["optimizer"]
            contrastive = state["contrastive_state"]
            return {
                "optimizer_groups": optimizer["param_groups"],
                "optimizer_steps": {pid: values["step"] for pid, values in optimizer["state"].items()},
                "contrastive_metadata": {k: v for k, v in contrastive.items() if k != "queue"},
                "queue_metadata": {k: v for k, v in contrastive["queue"].items() if k != "entries"},
            }
        left_contract = sha_fields(nonnumeric_contract(a))
        right_contract = sha_fields(nonnumeric_contract(b))
        exact.update({key: value == right_contract[key] for key, value in left_contract.items()})
        optimizer_errors = {}
        for key in ("exp_avg", "exp_avg_sq"):
            def moments(state):
                return {str(pid): value[key] for pid, value in state["optimizer"]["state"].items()}
            optimizer_errors[key] = _tensor_distance(moments(a), moments(b))
        queue_a = a["contrastive_state"]["queue"]["entries"]
        queue_b = b["contrastive_state"]["queue"]["entries"]
        def queue_ids(entries):
            return [{k: v for k, v in entry.items() if k != "key"} for entry in entries]
        queue_ids_equal = queue_ids(queue_a) == queue_ids(queue_b)
        queue_error = (_tensor_distance({str(i): e["key"] for i, e in enumerate(queue_a)},
                                        {str(i): e["key"] for i, e in enumerate(queue_b)})
                       if queue_ids_equal and queue_a else None)
        metric_rows = [{k: [x[k], y[k]] for k in ("step", *FLOAT_EVIDENCE)}
                       for x, y in zip(metrics[left], metrics[right])]
        first_metric_difference = next(({
            "step": row["step"][0],
            "fields": [key for key in FLOAT_EVIDENCE if row[key][0] != row[key][1]]
        } for row in metric_rows if any(row[key][0] != row[key][1] for key in FLOAT_EVIDENCE)), None)
        pairs[f"{left}__{right}"] = {
            "input_rng_boundary_differences": trace_diff, "final_exact": exact,
            "model_relative_to_left_displacement_from_fork": model_distance,
            "optimizer_moments_relative_to_left": optimizer_errors,
            "queue_identities_in_order_equal": queue_ids_equal, "queue_keys_relative_to_left": queue_error,
            "first_exact_metric_difference": first_metric_difference, "step_metrics": metric_rows}
    return {"steps": [fork, end], "boundary_state_or_rng_mismatches": mismatches, "pairs": pairs,
            "final_checkpoint_sha256": {phase: file_sha(folder / phase / "checkpoint.pt") for phase in PHASES}}


def run(arm, out):
    job = job_for(arm)
    if shutil.disk_usage(out).free < 7 * 1024**3:
        raise ValueError("diagnostic requires 7 GiB of free space")
    if source_digest() != SOURCE or job["source_sha256"] != SOURCE:
        raise ValueError("frozen production source digest mismatch")
    for row in job["inputs"].values():
        if file_sha(row["path"]) != row["sha256"]:
            raise ValueError("input pin differs: " + row["path"])
    source = Path(job["out"]) / "checkpoint.pt"
    if file_sha(source) != CHECKPOINTS[arm]:
        raise ValueError("historical endpoint pin differs")
    seed = checkpoint(source)
    raw = read(ROOT / job["config"]["path"])
    if (seed["step"] != START or seed["config"] != trainer.config_to_dict(config_from_dict(raw))
            or seed["world_size"] != 4 or seed["contrastive_state"]["queue"]["last_step"] != START-1
            or seed["contrastive_provenance"]["source_sha256"] != SOURCE):
        raise ValueError("historical checkpoint provenance or recipe differs")
    del seed
    fork = out / "continuous" / f"checkpoint_step{FORK:07d}.pt"
    seed_receipt = None
    fork_sha = None
    for phase in PHASES:
        directory = out / phase
        directory.mkdir()
        start = directory / "checkpoint.pt"
        if phase == "continuous":
            seed_receipt = diagnostic_seed(source, start, CHECKPOINTS[arm], FORK)
            write(out / "seed.json", seed_receipt)
            pin = seed_receipt["seed_sha256"]
        else:
            if file_sha(fork) != fork_sha:
                raise ValueError("common fork file changed")
            shutil.copyfile(fork, start)
            pin = fork_sha
        write(directory / "start.json", {"sha256": pin})
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                        str(Path(__file__).resolve()), "--arm", arm, "--out-dir", str(out),
                        "--rank-phase", phase], check=True)
        if phase == "continuous":
            fork_sha = file_sha(fork)
    report = cpu_analysis(out, FORK, END, 4)
    paired = report["pairs"].values()
    controls_match = not report["boundary_state_or_rng_mismatches"] and all(
        not row["input_rng_boundary_differences"] for row in paired)
    final_structure_matches = all(
        row["queue_identities_in_order_equal"] and
        all(value for key, value in row["final_exact"].items() if key not in ("optimizer", "contrastive_state"))
        for row in paired)
    numeric_difference = any(
        any(row["model_relative_to_left_displacement_from_fork"].values())
        or not row["final_exact"]["optimizer"] or not row["final_exact"]["contrastive_state"]
        or row["first_exact_metric_difference"] is not None for row in paired)
    write(out / "report.json", {"arm": arm, "source_sha256": SOURCE, "fork_sha256": fork_sha,
                                "seed": seed_receipt, "training_approved": False,
                                "status": ("CONTROL_INPUTS_VERIFIED" if controls_match and final_structure_matches
                                           else "CONTROL_MISMATCH"),
                                "final_rng_metadata_queue_identity_exact": final_structure_matches,
                                "any_numeric_or_metric_difference": numeric_difference,
                                "model_denominator": "L2 displacement of left state from common step 580, floor 1e-12",
                                "calculation": "reload_a__reload_b measures descriptive fresh-process repeatability",
                                "historical_one_percent_gate_applied": False,
                                **report})
    return 0 if controls_match and final_structure_matches else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=CHECKPOINTS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank-phase", choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rank_phase:
        rank_phase(args.arm, args.out_dir, args.rank_phase)
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        return run(args.arm, args.out_dir)
    except Exception as exc:
        write(args.out_dir / "failure.json", {"stage": "common_checkpoint", "arm": args.arm,
                                              "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
