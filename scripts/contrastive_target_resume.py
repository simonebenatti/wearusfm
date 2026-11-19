"""Prospective RMS100/fast125 restart comparison from one live step-544 fork.

No job submission and no independent training prefix. The continuous benchmark
saves its configured named checkpoint and stays in the same process to 576.
Two fresh distributed invocations load identical bytes; this module observes
production calls and applies the historical one-percent rule to all three pairs.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from contrastive_prepare import config_from_dict
from wearusfm.training import run as trainer
from wearusfm.training.contrastive_objectives import ContrastiveRuntime, attach_contrastive_head
from wearusfm.training.pairing import state_sha256, batch_sha256
from wearusfm.model.fm import WearUsFM

FORK, END = 544, 576
PHASES = ("continuous", "reload_a", "reload_b")
PROTOCOL = "contrastive-target-common-fork-20261119-v1"
RELATIVE_LIMIT = .01
TRACE_EQUAL = ("batch_sha256", "input_sha256", "rng_cpu_before_sha256", "rng_cuda_before_sha256",
               "rng_cpu_after_sha256", "rng_cuda_after_sha256", "queue_order_before_sha256",
               "queue_last_step_before", "queue_length_before", "coverage_before_sha256",
               "global_windows", "contrast_nce_weight", "contrast_rnc_weight")
SUPPORT_FIELDS = ("contrast_nce_eligible", "contrast_rnc_windows", "contrast_rnc_pairs")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checkpoint(path):
    return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)


def validate_fork(source_checkpoint, expected_sha, expected_cfg, expected_provenance,
                  initial_sha, *, fork_step=FORK, end_step=END, ranks=4):
    """Read-only check of the exact live prefix; no config mutation on reload."""
    source = Path(source_checkpoint)
    if file_sha(source) != expected_sha:
        raise ValueError("common fork checkpoint SHA256 differs")
    state = checkpoint(source)
    raw = trainer.config_to_dict(expected_cfg) if not isinstance(expected_cfg, dict) else expected_cfg
    if (state.get("step") != fork_step or state.get("world_size", 1) != ranks
            or state.get("config") != raw or raw.get("max_steps") != end_step
            or fork_step not in raw.get("schedule", {}).get("keep_checkpoints", ())
            or raw.get("contrastive", {}).get("arm") not in ("CN", "CR")
            or state.get("contrastive_initial_sha256") != initial_sha
            or state.get("contrastive_provenance") != expected_provenance
            or state.get("contrastive_state", {}).get("queue", {}).get("last_step") != fork_step - 1):
        raise ValueError("common fork state/config/provenance differs")
    if ranks > 1:
        rank_rng = state.get("rank_rng")
        if (not isinstance(rank_rng, list) or len(rank_rng) != ranks
                or any(not isinstance(r.get("torch_rng"), torch.Tensor)
                       or not isinstance(r.get("cuda_rng"), torch.Tensor) for r in rank_rng)):
            raise ValueError("common fork lacks separate CPU/CUDA RNG for all ranks")
    if file_sha(source) != expected_sha:
        raise ValueError("common fork changed while validating")
    return {"path": str(source.resolve()), "sha256": expected_sha, "step": fork_step,
            "end_step": end_step, "world_size": ranks,
            "payload_sha256": state_sha256(state)}


def prepare_reload(source_checkpoint, expected_sha, out_dir, expected_cfg,
                   expected_provenance, initial_sha, *, phase,
                   fork_step=FORK, end_step=END, ranks=4):
    """Copy one pinned fork, exclusively, without changing any saved field."""
    if phase not in PHASES[1:]:
        raise ValueError("reload phase must be reload_a or reload_b")
    evidence = validate_fork(source_checkpoint, expected_sha, expected_cfg,
                             expected_provenance, initial_sha,
                             fork_step=fork_step, end_step=end_step, ranks=ranks)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    target = out / "checkpoint.pt"
    with Path(source_checkpoint).open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if file_sha(target) != expected_sha or file_sha(source_checkpoint) != expected_sha:
        raise ValueError("reload copy is not byte-identical to common fork")
    raw = trainer.config_to_dict(expected_cfg) if not isinstance(expected_cfg, dict) else expected_cfg
    origin_pin = None
    if raw.get("log_batch_origin"):
        origin = Path(source_checkpoint).parent / "origin_sessions.json"
        with origin.open("rb") as src, (out / "origin_sessions.json").open("xb") as dst:
            shutil.copyfileobj(src, dst)
        origin_pin = file_sha(origin)
        if file_sha(out / "origin_sessions.json") != origin_pin:
            raise ValueError("reload origin session mapping differs")
    record = {"protocol": PROTOCOL, "phase": phase, "source": evidence,
              "origin_sessions_sha256": origin_pin,
              "copied_checkpoint_sha256": file_sha(target)}
    write(out / "start.json", record)
    return record


def rng_hashes(device, when):
    cuda = torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None
    return {f"rng_cpu_{when}_sha256": state_sha256(torch.get_rng_state()),
            f"rng_cuda_{when}_sha256": None if cuda is None else state_sha256(cuda)}


def queue_order(queue):
    return [{k: value for k, value in entry.items() if k != "key"}
            for entry in queue["entries"]]


def fields_sha(values):
    return {name: state_sha256(value) for name, value in values.items()}


@contextmanager
def trace_resume_updates(path, phase, *, fork_step=FORK, end_step=END):
    """Observe actual trainer calls and separate RNG domains, without new draws."""
    if phase not in PHASES:
        raise ValueError("unknown common-fork phase")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare = ContrastiveRuntime.prepare
    accumulate = trainer._accumulate_terms
    adam_init = torch.optim.AdamW.__init__
    optimizers, pending = [], {}

    def observe_adam(opt, *args, **kwargs):
        value = adam_init(opt, *args, **kwargs)
        optimizers.append(opt)
        return value

    def observe_batch(runtime, batch, step):
        value = prepare(runtime, batch, step)
        if fork_step <= step < end_step:
            if pending:
                raise ValueError("unconsumed restart trace batch")
            queue = runtime.queue.state_dict()
            pending.update(update=step + 1, batch_sha256=batch_sha256(batch),
                           queue_order_before_sha256=state_sha256(queue_order(queue)),
                           queue_last_step_before=queue["last_step"],
                           queue_length_before=len(queue["entries"]),
                           coverage_before_sha256=state_sha256(runtime.coverage),
                           global_windows=len(runtime.global_ids))
        return value

    def observe_backward(model, student, teacher, micros, cfg, ctx, device, amp, **kwargs):
        runtime = kwargs.get("contrastive")
        observing = runtime is not None and fork_step <= runtime.step < end_step
        if not observing:
            return accumulate(model, student, teacher, micros, cfg, ctx, device, amp, **kwargs)
        if pending.get("update") != runtime.step + 1:
            raise ValueError("restart trace batch/backward order differs")
        if len(optimizers) != 1:
            raise ValueError("restart trace requires the unique production AdamW")
        record = dict(pending)
        pending.clear()
        record.update(rank=ctx.rank, phase=phase, input_sha256=trainer._micro_digest(micros),
                      **rng_hashes(device, "before"))
        if runtime.step == fork_step:
            opt = optimizers[0]
            record["boundary"] = fields_sha({
                "student": student.state_dict(), "teacher": teacher.state_dict(),
                "optimizer_moments": opt.state_dict()["state"],
                "optimizer_groups_at_next_lr": opt.state_dict()["param_groups"],
                "contrastive_state": runtime.state_dict(),
            })
        failure, metrics = accumulate(model, student, teacher, micros, cfg, ctx, device, amp, **kwargs)
        record.update(**rng_hashes(device, "after"), backward_failure=failure,
                      **{name: metrics.get(name) for name in (*SUPPORT_FIELDS, "contrast_nce_weight", "contrast_rnc_weight")})
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        if hasattr(model, "_get_ddp_logging_data"):
            data = model._get_ddp_logging_data()
            fields = ("iteration", "has_rebuilt_buckets", "bucket_sizes",
                      "rebuilt_bucket_sizes", "rebuilt_per_bucket_param_indices")
            buckets.write(json.dumps({"update": record["update"], "rank": ctx.rank,
                                      **{key: data[key] for key in fields if key in data}}) + "\n")
            buckets.flush()
        return failure, metrics

    with path.open("x") as stream, ExitStack() as stack:
        buckets = stack.enter_context(path.with_suffix(".ddp.jsonl").open("x"))
        stack.enter_context(patch.object(torch.optim.AdamW, "__init__", observe_adam))
        stack.enter_context(patch.object(ContrastiveRuntime, "prepare", observe_batch))
        stack.enter_context(patch.object(trainer, "_accumulate_terms", observe_backward))
        yield
    if pending:
        raise ValueError("restart trace stopped with an unconsumed batch")


def rows(path, start, end, *, allow_prefix=False):
    values = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if allow_prefix:
        values = [row for row in values if row.get("update", row.get("step", -1)) > start]
    if [row.get("update", row.get("step")) for row in values] != list(range(start + 1, end + 1)):
        raise ValueError("incomplete, duplicated or unordered restart trace: " + str(path))
    return values


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _floating_distances(left, right, path="state"):
    """Every floating tensor, including tiny Adam moments; no aggregate waiver."""
    from contrastive_benchmark_readout import _tensor_distance
    result = {}
    if isinstance(left, torch.Tensor):
        if isinstance(right, torch.Tensor) and left.is_floating_point() and not path.endswith(".step"):
            result[path] = _tensor_distance({"x": left}, {"x": right})
    elif isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        for key in left:
            result.update(_floating_distances(left[key], right[key], path + "." + str(key)))
    elif isinstance(left, (tuple, list)) and type(left) is type(right) and len(left) == len(right):
        for index, (a, b) in enumerate(zip(left, right)):
            result.update(_floating_distances(a, b, path + "." + str(index)))
    return result


def historical_numeric_pair(left, right, initial, parameters, *, fork=None):
    """Original denominators: parameters/init displacement; moments/left tensor."""
    from contrastive_benchmark_readout import _nested_equal, _tensor_distance
    failures, relative, floating_errors, all_float = [], {}, {}, {}
    for part in ("student", "teacher"):
        if set(left[part]) != set(right[part]) or set(left[part]) != set(initial):
            raise ValueError("model state schema differs from authenticated initialization")
        _nested_equal({k: v for k, v in left[part].items() if k not in parameters},
                      {k: v for k, v in right[part].items() if k not in parameters}, part + ".buffers")
        relative[part] = _tensor_distance({k: v for k, v in left[part].items() if k in parameters},
                                         {k: v for k, v in right[part].items() if k in parameters},
                                         {k: v for k, v in initial.items() if k in parameters})
        if relative[part] > RELATIVE_LIMIT:
            failures.append(f"{part}: relative displacement exceeds historical 1%")
    for part in ("optimizer", "contrastive_state"):
        _nested_equal(left[part], right[part], part, RELATIVE_LIMIT, floating_errors)
        all_float.update(_floating_distances(left[part], right[part], part))
    for part in ("stream", "config", "data_signature", "pose_stats_sha256", "contrastive_provenance",
                 "contrastive_initial_sha256", "alarm_counts"):
        _nested_equal(left[part], right[part], part)
    if "rank_rng" in left or "rank_rng" in right:
        _nested_equal(left["rank_rng"], right["rank_rng"], "rank_rng")
    else:
        _nested_equal(left["torch_rng"], right["torch_rng"], "torch_rng")
    failures.extend(f"{name}: floating tensor exceeds historical 1%" for name in floating_errors)
    report = {"passed": not failures, "bound": RELATIVE_LIMIT,
              "relative_to_left_displacement_from_initial": relative,
              "floating_state_errors": floating_errors,
              "floating_tensor_relative_to_left": all_float,
              "failure_reasons": failures,
              "denominators": {"model": "L2 parameter displacement of the left final model from authenticated initialization; floor 1e-12",
                               "floating_state": "L2 norm of each left final tensor separately; floor 1e-12"}}
    if fork is not None:
        report["relative_to_left_displacement_from_544"] = {
            part: _tensor_distance({k: v for k, v in left[part].items() if k in parameters},
                                   {k: v for k, v in right[part].items() if k in parameters},
                                   {k: v for k, v in fork[part].items() if k in parameters})
            for part in ("student", "teacher")}
        report["by_block_relative_to_left_displacement_from_fork"] = {
            part: {block: _tensor_distance(
                {k: v for k, v in left[part].items() if k in parameters and k.split(".")[0] == block},
                {k: v for k, v in right[part].items() if k in parameters and k.split(".")[0] == block},
                {k: v for k, v in fork[part].items() if k in parameters and k.split(".")[0] == block})
                   for block in sorted({k.split(".")[0] for k in parameters})}
            for part in ("student", "teacher")}
    return report


def compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, *, ranks=4,
                        fork_step=FORK, end_step=END):
    """Authenticate one fork, trace all 32 updates, apply all three pairwise gates."""
    if set(paths) != set(PHASES):
        raise ValueError("restart comparison requires continuous and both fresh reloads")
    paths = {key: Path(value) for key, value in paths.items()}
    if len({p.resolve() for p in paths.values()}) != len(PHASES):
        raise ValueError("three restart paths must be distinct")
    if state_sha256(initial) != initial_sha:
        raise ValueError("authenticated calibration initialization differs")
    raw = trainer.config_to_dict(cfg)
    arm = raw.get("contrastive", {}).get("arm")
    if arm not in ("CN", "CR"):
        raise ValueError("restart comparison requires active CN or CR")
    fork_path = paths["continuous"] / f"checkpoint_step{fork_step:07d}.pt"
    fork = checkpoint(fork_path)
    validate_fork(fork_path, fork_sha, raw, fork["contrastive_provenance"], initial_sha,
                  fork_step=fork_step, end_step=end_step, ranks=ranks)
    with torch.random.fork_rng(devices=[]):
        model = attach_contrastive_head(WearUsFM(cfg.model))
    parameters = {name for name, _ in model.named_parameters()}
    if set(model.state_dict()) != set(initial):
        raise ValueError("initial schema differs from production architecture")
    del model
    groups = copy.deepcopy(fork["optimizer"]["param_groups"])
    for group in groups:
        group["lr"] = trainer._lr_at(fork_step, cfg)
    expected_boundary = fields_sha({"student": fork["student"], "teacher": fork["teacher"],
        "optimizer_moments": fork["optimizer"]["state"], "optimizer_groups_at_next_lr": groups,
        "contrastive_state": fork["contrastive_state"]})
    boundary_errors, traces, finals, metrics, support_missing = [], {}, {}, {}, []
    evidence = {}
    for phase, directory in paths.items():
        if phase != "continuous":
            start = read(directory / "start.json")
            if (start.get("protocol") != PROTOCOL or start.get("phase") != phase
                    or start.get("source", {}).get("sha256") != fork_sha
                    or start.get("copied_checkpoint_sha256") != fork_sha
                    or start.get("source", {}).get("payload_sha256") != state_sha256(fork)):
                raise ValueError("reload did not start from the same immutable fork bytes")
        summary = read(directory / "summary.json")
        final_path = directory / "checkpoint.pt"
        final = checkpoint(final_path)
        if (summary.get("steps") != end_step or summary.get("stopped") != "max_steps"
                or summary.get("world_size", 1) != ranks or final.get("step") != end_step
                or final.get("world_size", 1) != ranks or final.get("config") != raw
                or read(directory / "config.json") != raw or (directory / "STOP").exists()
                or final.get("contrastive_provenance") != fork["contrastive_provenance"]
                or final.get("contrastive_initial_sha256") != initial_sha
                or final.get("contrastive_state", {}).get("queue", {}).get("last_step") != end_step - 1):
            raise ValueError("incomplete or mismatched common-fork endpoint")
        c = summary.get("contrastive", {})
        if (c.get("optimizer_updates") != end_step or c.get("queue_last_step") != end_step - 1
                or c.get("initial_sha256") != initial_sha or c.get("provenance") != fork["contrastive_provenance"]):
            raise ValueError("common-fork summary counters or provenance differ")
        metrics[phase] = rows(directory / "metrics.jsonl", fork_step, end_step, allow_prefix=phase == "continuous")
        finals[phase] = final
        evidence[phase] = {name: file_sha(directory / name) for name in
                           ("checkpoint.pt", "summary.json", "config.json", "metrics.jsonl")}
        if phase != "continuous":
            evidence[phase]["start.json"] = file_sha(directory / "start.json")
        if cfg.log_batch_origin:
            origin_sha = file_sha(directory / "origin_sessions.json")
            if origin_sha != file_sha(paths["continuous"] / "origin_sessions.json"):
                raise ValueError("restart origin session mapping differs")
            if phase != "continuous" and start.get("origin_sessions_sha256") != origin_sha:
                raise ValueError("restart origin session mapping pin differs")
            evidence[phase]["origin_sessions.json"] = origin_sha
        for rank in range(ranks):
            trace = rows(directory / f"trace-{rank}.jsonl", fork_step, end_step)
            traces[phase, rank] = trace
            evidence[phase][f"trace-{rank}.jsonl"] = file_sha(directory / f"trace-{rank}.jsonl")
            expected_rng = fork["rank_rng"][rank] if ranks > 1 else {
                "torch_rng": fork["torch_rng"], "cuda_rng": None}
            for field, value in expected_boundary.items():
                if trace[0].get("boundary", {}).get(field) != value:
                    boundary_errors.append({"phase": phase, "rank": rank, "field": field})
            for domain, field in (("cpu", "torch_rng"), ("cuda", "cuda_rng")):
                value = expected_rng.get(field)
                digest = None if value is None else state_sha256(value)
                if trace[0].get(f"rng_{domain}_before_sha256") != digest:
                    boundary_errors.append({"phase": phase, "rank": rank, "field": f"rng_{domain}_before_sha256"})
            for row, metric in zip(trace, metrics[phase]):
                if (row.get("rank") != rank or row.get("phase") != phase
                        or row.get("queue_last_step_before") != row["update"] - 2
                        or row.get("global_windows") != cfg.batch_size
                        or row.get("backward_failure") is not None
                        or any(field not in row for field in (*TRACE_EQUAL, *SUPPORT_FIELDS))
                        or (ranks > 1 and any(row.get(f"rng_{domain}_{when}_sha256") is None
                            for domain in ("cpu", "cuda") for when in ("before", "after")))):
                    raise ValueError("invalid runtime restart trace invariants")
                weights = (row["contrast_nce_weight"], row["contrast_rnc_weight"])
                from wearusfm.training.contrastive_objectives import ContrastiveConfig
                expected_weights = ContrastiveConfig(**cfg.contrastive).weights(row["update"] - 1)
                if weights != expected_weights or not any(v > 0 for v in weights):
                    raise ValueError("restart contrastive support has inactive or changed weights")
                if cfg.log_batch_origin and (not isinstance(metric.get("origin"), list)
                                               or len(metric["origin"]) != cfg.batch_size):
                    raise ValueError("restart origin trace is incomplete")
                for name in (*SUPPORT_FIELDS, "contrast_nce_weight", "contrast_rnc_weight"):
                    if not _finite_number(row[name]) or row[name] != metric.get(name):
                        raise ValueError("restart support trace disagrees with production metrics")
                active_support = row["contrast_nce_eligible"] if arm == "CN" else min(row["contrast_rnc_windows"], row["contrast_rnc_pairs"])
                if active_support <= 0:
                    support_missing.append({"phase": phase, "rank": rank, "update": row["update"], "arm": arm})
    pairs = {}
    for left, right in itertools.combinations(PHASES, 2):
        differences = []
        for rank in range(ranks):
            for a, b in zip(traces[left, rank], traces[right, rank]):
                for field in (*TRACE_EQUAL, *SUPPORT_FIELDS):
                    if a[field] != b[field]:
                        differences.append({"rank": rank, "update": a["update"], "field": field})
        if cfg.log_batch_origin:
            for a, b in zip(metrics[left], metrics[right]):
                if a.get("origin") != b.get("origin"):
                    differences.append({"rank": "global", "update": a["step"], "field": "origin"})
        try:
            pair = historical_numeric_pair(finals[left], finals[right], initial, parameters, fork=fork)
        except (ValueError, KeyError) as exc:
            pair = {"passed": False, "failure_reasons": [str(exc)], "bound": RELATIVE_LIMIT}
        pair["input_rng_counter_order_differences"] = differences
        pair["passed"] = pair["passed"] and not differences and not boundary_errors and not support_missing
        pair["final_checkpoint_sha256"] = {left: evidence[left]["checkpoint.pt"], right: evidence[right]["checkpoint.pt"]}
        pairs[left + "__" + right] = pair
    passed = all(pair["passed"] for pair in pairs.values())
    # Missing active support is a missing comparison, not a successful numerical test.
    status = "PASSED" if passed else ("INCONCLUSIVE" if support_missing and not boundary_errors
             and all(not p["input_rng_counter_order_differences"] and not p["failure_reasons"] for p in pairs.values())
             else "FAILED_REQUIRES_REVIEW")
    return {"protocol": PROTOCOL, "arm": arm, "steps": [fork_step, end_step], "passed": passed,
            "status": status, "training_approved": False, "fork_sha256": fork_sha,
            "historical_one_percent_gate_applied": True, "boundary_mismatches": boundary_errors,
            "missing_active_support": support_missing, "pairs": pairs, "evidence_sha256": evidence,
            "fork_distances_descriptive_only": True}
