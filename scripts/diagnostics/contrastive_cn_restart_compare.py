"""Prospective CPU readout of the two-leg, common-fork CN reducer intervention."""
import argparse
from collections.abc import Mapping
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from contrastive_target_resume import (FORK, END, TRACE_EQUAL, SUPPORT_FIELDS, checkpoint,
                                       file_sha, read, rows, write)

PACKAGE = ROOT / "results/infra/contrastive_targets_20261118"
ARCHIVE = {"ordinary": PACKAGE / "reloads_execution_20261121/CN_A",
           "warm": PACKAGE / "benchmark_execution_20261121/CN"}
REFERENCE_REPORT = "d4300a2e31c85b39f1f36b8f3146886811d70fe9e24c2d0a9d07e683e84cd4b8"
FORK_SHA = "aa5952f0c4b7a8f07b5965b096877d568f3b5662bf08e6d8a4fc56f2a9e37ce6"
ENDPOINT = {"ordinary": "bb39f33998df27c2a0b7b6ca7554c77afccf08bb7405831a17a28c4ef6594301",
            "warm": "4743abaff631d43c0f9267b886087fdcbefdfc156332bf34d43b668297668dec"}
LIMIT = 1e-6
DENOM_FLOOR = 1e-12
RANKS = 4
PARAMETERS = 270


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identical(a, b):
    """Strict IEEE bytes, including signed zero, after dtype/shape checks."""
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def checkpoint_differences(reference, measured):
    """Compare all payload fields and leaves. File container metadata is outside this comparison."""
    changed, invalid = [], []

    def visit(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                invalid.append(path)
            elif a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                invalid.append(path)
            elif not identical(a, b):
                kind = changed if a.is_floating_point() and not path.endswith(".step") else invalid
                kind.append(path)
        elif isinstance(a, Mapping):
            if not isinstance(b, Mapping) or a.keys() != b.keys():
                invalid.append(path)
                return
            for k in a:
                visit(a[k], b[k], f"{path}.{k}" if path else str(k))
        elif isinstance(a, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                invalid.append(path)
                return
            for i, value in enumerate(a):
                visit(value, b[i], f"{path}.{i}")
        elif type(a) is not type(b) or a != b:
            invalid.append(path)

    visit(reference, measured, "")
    return {"floating_tensors_different": changed, "metadata_or_schema_different": invalid,
            "exact_payload": not changed and not invalid}


def compare_vector(a, b):
    require(a.shape == b.shape and a.dtype == b.dtype and a.numel() > 0, "gradient schema differs")
    require(a.is_floating_point() and torch.isfinite(a).all() and torch.isfinite(b).all(), "invalid gradient")
    x, y = a.reshape(-1).double(), b.reshape(-1).double()
    dx = x - y
    nx, ny, nd = [float(torch.linalg.vector_norm(t)) for t in (x, y, dx)]
    # Represent signed-zero bit differences separately from real norm differences.
    raw = (a.contiguous().reshape(-1).view(torch.uint8) != b.contiguous().reshape(-1).view(torch.uint8))
    count = int(raw.reshape(a.numel(), a.element_size()).any(1).sum())
    return {"different_elements": count, "elements": a.numel(),
            "ordinary_l2": nx, "warm_l2": ny, "difference_l2": nd,
            "max_abs": float(dx.abs().max()), "relative_l2": nd / max(nx, ny, DENOM_FLOOR)}


def parameter_identity(parameters):
    return [(p["id"], p["name"], tuple(p["shape"]), p["dtype"]) for p in parameters]


def layouts(ordinary, warm, logged):
    """Use hook order. Cold archival logger has only byte sizes, no per-ID ordering."""
    cold = [(b["nbytes"], tuple(b["parameter_ids"])) for b in ordinary["buckets"]]
    hot = [(b["nbytes"], tuple(b["parameter_ids"])) for b in warm["buckets"]]
    sizes = [int(s) for s in logged["rebuilt_bucket_sizes"].split(",")]
    ids = [tuple(map(int, group.split())) for group in logged["rebuilt_per_bucket_param_indices"].split(",")]
    initial_bytes = list(map(int, logged["bucket_sizes"].split(",")))
    require(len(sizes) == len(ids), "historical warm bucket sizes/IDs differ")
    by_dtype = {}
    for row in ordinary["parameters"]:
        by_dtype.setdefault(row["dtype"], []).append(row["id"])
    require(sorted(ln for ln, _ in cold) == sorted(initial_bytes), "ordinary initial bucket bytes differ")
    require(sorted(ids_ for _, ids_ in cold) == sorted(tuple(ids_) for ids_ in by_dtype.values()),
            "ordinary initial dtype-specific parameter membership/order differs")
    require(hot == list(zip(sizes, ids)), "warm actual bucket order/size/parameter IDs differ from continuous")
    require(cold != hot, "intervention did not change reducer layout")
    return {"ordinary": ordinary["buckets"], "warm": warm["buckets"]}


def classify(local_changed, reduced_changed, excessive, endpoint_exact, warm_first_update_exact):
    if local_changed:
        return "HYPOTHESIS_UNSUPPORTED", "step 545 local gradients already differ"
    if excessive:
        return "HYPOTHESIS_UNSUPPORTED", "step 545 reduced difference exceeds the prospective rounding screen"
    if not reduced_changed:
        return "HYPOTHESIS_UNSUPPORTED", "no nonzero post-reduction discrepancy at step 545"
    if not warm_first_update_exact:
        return "PARTIAL_SUPPORT", "difference localized to reduction; warm gradients do not match the historical first update"
    if not endpoint_exact:
        return "PARTIAL_SUPPORT", "difference localized to reduction; warm trajectory does not recover continuous"
    return "SUPPORTED", "localized post-reduction difference and exact warm continuous endpoint"


def history(ordinary, continuous):
    # The passing historical context and the negative historical gate cannot be replaced by this result.
    raw = gzip.decompress((PACKAGE / "readout_execution_20261121/preflight_readout.json.gz").read_bytes())
    require(hashlib.sha256(raw).hexdigest() == REFERENCE_REPORT, "historical CPU readout SHA mismatch")
    r = json.loads(raw)
    require(r["status"] == "FAILED_REQUIRES_REVIEW" and r["training_approved"] is False
            and r["zero_weight_parity"]["assessment"]["status"] == "PASSED"
            and r["numeric_context_across_runs"]["status"] == "PASSED", "historical control differs")
    for mode, folder in [("ordinary", ordinary), ("warm", continuous)]:
        require(file_sha(folder / "checkpoint.pt") == ENDPOINT[mode], f"historical {mode} endpoint SHA mismatch")
    require(file_sha(continuous / "checkpoint_step0000544.pt") == FORK_SHA, "historical fork SHA mismatch")
    return {"report_sha256": REFERENCE_REPORT, "status": r["status"], "zero_weight_parity": "PASSED",
            "numeric_context": "PASSED", "fork_sha256": FORK_SHA, "reference_endpoint_sha256": ENDPOINT}


def branch_traces(folder, reference, mode, rank):
    archive = ARCHIVE[mode]
    attrs = ("batch_sha256", "input_sha256", "rng_cpu_before_sha256", "rng_cuda_before_sha256",
             "queue_order_before_sha256", "coverage_before_sha256")
    name = f"trace-{rank}.jsonl"
    require(file_sha(reference / name) == file_sha(archive / name), f"historical trace mismatch: {mode}/{name}")
    base, run = [rows(d / name, FORK, END) for d in (reference, folder)]
    require(len(base) == len(run) == END - FORK, "incomplete trace")
    for x, y in zip(base, run):
        require(x["rank"] == y["rank"] == rank and y["backward_failure"] is None and y["phase"] == "reload_a",
                "rank, phase or backward failed")
        require(all(x[k] == y[k] for k in (*TRACE_EQUAL, *SUPPORT_FIELDS)), f"stream/RNG/support differs at {y['update']}")
        require(all(isinstance(y[k], str) and len(y[k]) == 64 for k in attrs), "missing trace fingerprint")
    require(base[0]["boundary"] == run[0]["boundary"], "historical boundary mismatch")
    profile = f"numeric_profile.rank{rank}.json"
    require(file_sha(reference / profile) == file_sha(archive / profile), "historical profile SHA mismatch")
    old, new = [read(d / profile) for d in (reference, folder)]
    require(new["rank"] == rank and new["world_size"] == RANKS and new["flags"] == old["flags"]
            and new["source_sha256"] == old["source_sha256"]
            and new["config_canonical_sha256"] == old["config_canonical_sha256"], "numeric/source/config mismatch")
    return run[0]


def check_boundary(folder, mode, rank, trace):
    b = read(folder / f"boundary.rank{rank}.json")
    require(b["schema"] == "cn-ddp-boundary-v1" and b["rank"] == rank and b["variant"] == mode
            and b["world_size"] == RANKS and b["update"] == FORK + 1
            and b["null_warmups"] == int(mode == "warm"), "intervention receipt differs")
    before = b["before"]
    require(before == b["after_warmup"] and before["grad_none"] is True, "warmup changed state/input or RNG")
    required = {"student", "teacher", "optimizer", "contrastive_state", "prepared_runtime_sha256", "anchor_targets_sha256", "input_sha256",
                "rng_torch_cpu", "rng_torch_device", "numpy_rng", "python_rng",
                "student_buffers_sha256", "teacher_buffers_sha256"}
    require(required <= before.keys() and all(isinstance(before[k], str) and len(before[k]) == 64 for k in required),
            "incomplete boundary digests")
    for k in ("student", "teacher", "contrastive_state"):
        require(before[k] == trace["boundary"][k], "boundary receipt disagrees with production trace: " + k)
    for k, t in [("input_sha256", "input_sha256"), ("rng_torch_cpu", "rng_cpu_before_sha256"),
                 ("rng_torch_device", "rng_cuda_before_sha256")]:
        require(before[k] == trace[t], "batch/torch RNG disagrees with production trace")
    return b


def post_update_hashes(folder, rank, identity):
    data = read(folder / f"post_update.rank{rank}.json")
    keys = ("post_clip_gradient_sha256", "exp_avg_sha256", "exp_avg_sq_sha256", "step_sha256", "parameter_sha256")
    require(data["schema"] == "cn-ddp-optimizer-update-v1" and data["rank"] == rank and data["update"] == FORK + 1,
            "wrong post-update trace")
    values = data["rows"]
    require(len(values) == len(identity), "incomplete post-update trace")
    for row, (pid, name, _, _) in zip(values, identity):
        require(row["id"] == pid and row["name"] == name, "wrong post-update parameter ordering")
        require(all(isinstance(row[k], str) and len(row[k]) == 64
                    and all(c in "0123456789abcdef" for c in row[k]) for k in keys), "invalid post-update SHA")
    return [{k: row[k] for k in keys} for row in values]


def metrics_evidence(folder, reference, mode):
    # The floating logging-only losses can vary even between bit-identical historical reloads.
    require(file_sha(reference / "metrics.jsonl") == file_sha(ARCHIVE[mode] / "metrics.jsonl"),
            "historical metrics mismatch")
    x, y = [rows(d / "metrics.jsonl", FORK, END, allow_prefix=d == reference and mode == "warm")
            for d in (reference, folder)]
    require(all(a.get("origin") and a["origin"] == b["origin"] for a, b in zip(x, y)),
            "origin sequence differs from production")
    names = ("grad_norm", "grad_clip_factor", "contrast_projector_grad_norm", "contrast_global_grad_base",
             "contrast_global_grad_nce", "contrast_global_grad_rnc", "contrast_global_cos_nce_base",
             "contrast_global_cos_rnc_base")
    differences = [{"step": a["step"], "field": k, "actual": b[k], "reference": a[k]}
                   for a, b in zip(x, y) for k in names if a[k] != b[k]]
    return {"steps": [FORK + 1, END], "fields": list(names), "differences": differences}


def compare(ordinary, warm, historical_ordinary, historical_continuous):
    from contrastive_cn_restart_observe import load_observation
    from contrastive_target_resume import WearUsFM, attach_contrastive_head, config_from_dict

    legs = {"ordinary": ordinary, "warm": warm}
    previous = {"ordinary": historical_ordinary, "warm": historical_continuous}
    require(len({p.resolve() for p in (*legs.values(), *previous.values())}) == 4, "four distinct directories required")
    prior = history(historical_ordinary, historical_continuous)
    cfg = read(ARCHIVE["ordinary"] / "config.json")
    with torch.random.fork_rng(devices=[]):
        model = attach_contrastive_head(WearUsFM(config_from_dict(cfg).model))
    identity = [(i, name, tuple(p.shape), str(p.dtype)) for i, (name, p) in
                enumerate((name, p) for name, p in model.named_parameters() if p.requires_grad)]
    require(len(identity) == PARAMETERS, "production parameter count differs")
    del model

    endpoints, endpoint_hashes, metrics = {}, {}, {}
    for mode, folder in legs.items():
        actual, ref = folder / "checkpoint.pt", previous[mode] / "checkpoint.pt"
        start, summary = read(folder / "start.json"), read(folder / "summary.json")
        require(start["phase"] == "reload_a" and start["source"]["sha256"] == start["copied_checkpoint_sha256"] == FORK_SHA
                and start["source"]["payload_sha256"] == read(ARCHIVE["ordinary"] / "start.json")["source"]["payload_sha256"]
                and summary["steps"] == END and summary["stopped"] == "max_steps" and not (folder / "STOP").exists()
                and read(folder / "config.json") == cfg, "invalid reload fork/config/completion")
        state = checkpoint(actual)
        require(state["step"] == END and state["world_size"] == RANKS and state["config"] == cfg,
                "endpoint step/world size/config mismatch")
        endpoints[mode] = checkpoint_differences(checkpoint(ref), state)
        endpoint_hashes[mode] = file_sha(actual)
        del state
        metrics[mode] = metrics_evidence(folder, previous[mode], mode)

    require(endpoint_hashes["ordinary"] == ENDPOINT["ordinary"] and endpoints["ordinary"]["exact_payload"]
            and not metrics["ordinary"]["differences"], "instrumented ordinary reload did not reproduce historical ordinary")
    require(not endpoints["warm"]["metadata_or_schema_different"], "warm endpoint metadata/schema or nonfloat state differs")
    details, first_layout, reduced_by_leg, placements = [], None, {}, []
    after_adam = {}
    any_local = any_reduced = excessive = False
    for rank in range(RANKS):
        rec, tensors, boundary = {}, {}, {}
        for mode, folder in legs.items():
            trace = branch_traces(folder, previous[mode], mode, rank)
            boundary[mode] = check_boundary(folder, mode, rank, trace)
            rec[mode], tensors[mode] = load_observation(folder, rank, expected_parameters=PARAMETERS)
            require(rec[mode]["world_size"] == RANKS and parameter_identity(rec[mode]["parameters"]) == identity,
                    "observation maps to wrong model parameter")
            current = post_update_hashes(folder, rank, identity)
            if rank == 0:
                after_adam[mode] = current
            require(current == after_adam[mode], "post-update evidence differs between ranks in the same leg")
            hashes = [p["reduced_sha256"] for p in rec[mode]["parameters"]]
            if rank == 0:
                reduced_by_leg[mode] = hashes
            require(reduced_by_leg[mode] == hashes, "reduced gradients differ between ranks in the same leg")
        left, right = boundary["ordinary"], boundary["warm"]
        # The archive has no Python/NumPy global RNG: verify within each leg, not between fresh processes.
        for k in set(left["before"]) | set(right["before"]):
            if k not in ("python_rng", "numpy_rng"):
                require(left["before"][k] == right["before"][k], "intervention state already differs: " + k)
        for k in ("host", "device_index", "device_name", "gpu_uuid"):
            require(left[k] == right[k], "diagnostic legs used different rank placements")
        placements.append({k: left[k] for k in ("rank", "host", "device_index", "device_name", "gpu_uuid")})
        name = f"trace-{rank}.ddp.jsonl"
        require(file_sha(historical_continuous / name) == file_sha(ARCHIVE["warm"] / name), "continuous DDP trace differs")
        logged = rows(historical_continuous / name, FORK, END)[0]
        require(logged["rank"] == rank and logged["has_rebuilt_buckets"] == 1, "invalid historical steady-state layout")
        layout = layouts(rec["ordinary"], rec["warm"], logged)
        if rank == 0:
            first_layout = layout
        require(first_layout == layout, "communication bucket layouts differ between ranks")
        for pid, p_name, _, _ in identity:
            stages = {s: compare_vector(tensors["ordinary"][s][pid], tensors["warm"][s][pid])
                      for s in ("local", "reduced")}
            any_local |= bool(stages["local"]["different_elements"])
            any_reduced |= stages["reduced"]["difference_l2"] > 0
            excessive |= stages["reduced"]["relative_l2"] > LIMIT
            details.append({"rank": rank, "id": pid, "name": p_name, **stages,
                            "post_update_fields_different": [k for k in after_adam["ordinary"][pid]
                                                             if after_adam["ordinary"][pid][k] != after_adam["warm"][pid][k]],
                            "bucket_membership": {m: {k: rec[m]["parameters"][pid][k]
                                                      for k in ("bucket_index", "bucket_position", "bucket_element_offset")}
                                                  for m in legs}})
        del tensors
    require(len({p["host"] for p in placements}) == 1 and len({p["device_index"] for p in placements}) == RANKS
            and all(p["gpu_uuid"] for p in placements) and len({p["gpu_uuid"] for p in placements}) == RANKS,
            "diagnostic requires a single four-device node")
    first_exact = not any(d["step"] == FORK + 1 for d in metrics["warm"]["differences"])
    status, reason = classify(any_local, any_reduced, excessive, endpoints["warm"]["exact_payload"], first_exact)
    return {"status": status, "reason": reason, "historical": prior, "endpoint": endpoints,
            "actual_endpoint_sha256": endpoint_hashes, "historical_gradient_metrics": metrics, "placements": placements,
            "actual_bucket_order": first_layout, "per_parameter_and_rank": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("ordinary", "warm", "historical-ordinary", "historical-continuous", "report"):
        parser.add_argument("--" + k, type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "cn-restart-intervention-v1", "training_approved": False,
              "historical_status": "FAILED_REQUIRES_REVIEW", "prospective_relative_l2_limit": LIMIT,
              "relative_l2_denominator": "max(ordinary L2, warm L2, 1e-12)", "update_observed": FORK + 1}
    try:
        report.update(compare(args.ordinary, args.warm, args.historical_ordinary, args.historical_continuous))
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, IndexError, EOFError) as exc:
        report.update(status="INCONCLUSIVE", reason=f"{type(exc).__name__}: {exc}")
    write(args.report, report)
    print(json.dumps({k: report[k] for k in ("status", "reason", "training_approved")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
