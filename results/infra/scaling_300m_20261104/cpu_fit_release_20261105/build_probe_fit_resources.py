"""Reproduce the CPU packing from 15 accepted train-only benchmark reports.

This offline release artifact does not import or modify numerical controllers,
load feature caches, select model hyperparameters, contact SLURM or submit jobs.
"""
import functools
import hashlib
import itertools
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
INFRA = OUT.parent
PLAN = INFRA / "cpu_readers_20261105/probe_plan.json"
ACCEPTED = INFRA / "cpu_recovery_20261105/accepted_benchmarks.json"
CODE = "aef665ba7ea66fc80291846487907bb3ef9107df"
CELLS = ("T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched")
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
NUMERIC = "1ac8481eb8b53fbdd423404a927cca02e908197f16ed6494268b56e5ddc24b1c"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jhash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def file_pin(path):
    content = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(content), "bytes": len(content)}


def combinations(items):
    for size in range(1, len(items) + 1):
        yield from itertools.combinations(items, size)


def solve(rectangles, bit_count):
    """Exact cover: minimum job count, then total rounded seconds, then key."""
    by_bit = {i: [] for i in range(bit_count)}
    for index, rect in enumerate(rectangles):
        for bit in by_bit:
            if rect["mask"] & (1 << bit):
                by_bit[bit].append(index)

    def objective(indices):
        return (len(indices), sum(rectangles[i]["walltime_seconds"] for i in indices), indices)

    @functools.lru_cache(None)
    def choose(uncovered):
        if not uncovered:
            return ()
        anchor = (uncovered & -uncovered).bit_length() - 1
        best = None
        for index in by_bit[anchor]:
            mask = rectangles[index]["mask"]
            if mask & uncovered != mask:
                continue
            tail = choose(uncovered ^ mask)
            if tail is None:
                continue
            candidate = tuple(sorted((index,) + tail))
            if best is None or objective(candidate) < objective(best):
                best = candidate
        return best

    solution = choose((1 << bit_count) - 1)
    assert solution is not None, "no exact cover within 10800 seconds per shard"
    return [rectangles[i] for i in solution], choose.cache_info().currsize


def build():
    plan = json.loads(PLAN.read_text())
    accepted = json.loads(ACCEPTED.read_text())
    assert accepted["matrix_complete"] is True and accepted["known_complete_reports"] == 15
    assert len(accepted["reports"]) == 15
    reports, pins, supports = {}, {c: {} for c in CELLS}, {}
    for entry in accepted["reports"]:
        local = entry["local_report"]
        blob = (ROOT / local["path"]).read_bytes()
        assert digest(blob) == local["sha256"] and len(blob) == local["bytes"]
        r = json.loads(blob)
        cell, ds = r["cell"], r["dataset"]
        assert (cell, ds) not in reports and (cell, ds) == (entry["cell"], entry["dataset"])
        assert r["complete"] is True and r["kind"] == "benchmark" and r["train_only"] is True
        assert r["selection_performed"] is False and r["fold"] == 0 and r["step"] == 5000
        assert r["code_sha"] == CODE and r["plan_sha256"] == digest(PLAN.read_bytes())
        assert r["plan_content_sha256"] == jhash(plan) and r["recipe"] == plan["recipe"]
        assert r["numeric_sha256"] == NUMERIC == jhash(r["numeric_context"])
        assert r["n_features"] == plan["recipe"]["widths"][cell]
        fits = r["fits"]
        assert [f["C"] for f in fits] == plan["recipe"]["C_grid"]
        assert all(f["converged_by_criterion"] and not f["reached_limit"] and f["exception"] is None
                   and not f["warnings"] and math.isfinite(f["elapsed_s"]) and f["elapsed_s"] > 0
                   and f["fit_phase"] == "benchmark_train_only" and f["selected"] is None for f in fits)
        counts = r["fold_train_counts"]
        assert len(counts) == 4 and counts[0] == r["n_train"] and all(c > 0 for c in counts)
        support = (counts, r["roles_sha256"], fits[0]["classes"], fits[0]["class_counts"])
        assert supports.setdefault(ds, support) == support
        assert r["peak_rss_bytes"] > 0 and r["peak_rss_bytes"] * 1.3 <= 30 * 2**30
        remote = entry["report"]
        assert remote["sha256"] == digest(blob) and remote["bytes"] == len(blob)
        pins[cell][ds] = {"path": remote["remote_path"], "sha256": remote["sha256"], "bytes": remote["bytes"]}
        reports[cell, ds] = r

    all_datasets = sum(TASKS.values(), ())
    assert set(reports) == set(itertools.product(CELLS, all_datasets))
    shards, groups, coverage = [], [], []
    for cell in CELLS:
        for task, datasets in TASKS.items():
            rects = []
            for selected_ds in combinations(datasets):
                for folds in combinations(tuple(range(4))):
                    needed, rss = 120.0, 0
                    for ds in selected_ds:
                        r = reports[cell, ds]
                        times = [f["elapsed_s"] for f in r["fits"]]
                        counts = r["fold_train_counts"]
                        needed += 1.5 * (sum(times) + max(times)) * sum(max(1.0, counts[f]/counts[0]) for f in folds)
                        rss = max(rss, r["peak_rss_bytes"])
                    seconds = math.ceil(needed/60) * 60
                    if seconds <= 10800:
                        mask = sum(1 << (datasets.index(ds)*4 + f) for ds in selected_ds for f in folds)
                        rects.append(dict(datasets=list(selected_ds), folds=list(folds), mask=mask,
                                          estimated_seconds_with_margin=needed, walltime_seconds=seconds,
                                          observed_peak_rss_bytes=rss))
            # Stable lexicographic order, independent of source file iteration order.
            rects.sort(key=lambda r: (r["datasets"], r["folds"]))
            chosen, states = solve(rects, len(datasets)*4)
            groups.append(dict(cell=cell, task=task, candidate_rectangles=len(rects),
                               dynamic_program_states=states, minimum_shards=len(chosen),
                               rounded_seconds=sum(r["walltime_seconds"] for r in chosen)))
            for index, rectangle in enumerate(chosen, 1):
                shard = {k: v for k, v in rectangle.items() if k != "mask"}
                shard.update(cell=cell, task=task, cpus=8, memory_gib=30,
                             shard_name=f"{cell}-{task}-{index:02d}")
                shards.append(shard)
                coverage.extend((cell, ds, f) for ds in shard["datasets"] for f in shard["folds"])
    assert len(coverage) == len(set(coverage)) == 60
    assert set(coverage) == set(itertools.product(CELLS, all_datasets, range(4)))
    assert len(shards) <= 15
    seconds = sum(s["walltime_seconds"] for s in shards)
    assert seconds * 8 <= 240 * 3600
    numeric_contexts = [r["numeric_context"] for r in reports.values()]
    assert all(x == numeric_contexts[0] for x in numeric_contexts)

    budget = dict(protocol="scaling-probe-resources-20261104-v1", complete=True,
                  code_sha=CODE, plan_content_sha256=jhash(plan), plan_sha256=digest(PLAN.read_bytes()),
                  caps=dict(cpus=8, memory_gib=30, walltime_seconds=10800, maximum_shards=15,
                            maximum_allocated_core_hours=240),
                  benchmarks=pins, accepted_benchmark_manifest=file_pin(ACCEPTED),
                  numeric_sha256=NUMERIC, shards=shards)
    audit = dict(protocol="scaling-probe-packing-audit-20261105-v1", complete=True,
                 method="Exact-cover dynamic programming separately per cell/task; minimum shards, then minimum sum rounded seconds, then lexicographic rectangle indices",
                 formula="120 + 1.5 * sum_dataset((sum(C_times)+max(C_times))*sum_fold(max(1,n_train_fold/n_train_fold0)))",
                 allocation_rounding_seconds=60, benchmarks_complete=15, covered_cells_dataset_folds=60,
                 source_plan=file_pin(PLAN), source_acceptance_manifest=file_pin(ACCEPTED),
                 builder=file_pin(Path(__file__).resolve()), numeric_context=numeric_contexts[0],
                 numerical_runtime_code_sha=CODE, no_scientific_scores_computed=True,
                 groups=groups, minimum_shards=len(shards), total_requested_seconds=seconds,
                 requested_core_hours=seconds*8/3600, sequential_walltime_hours=seconds/3600,
                 formula_seconds_before_rounding=sum(s["estimated_seconds_with_margin"] for s in shards),
                 formula_core_hours_before_rounding=sum(s["estimated_seconds_with_margin"] for s in shards)*8/3600,
                 scheduler_overtime_allowance_seconds_per_job=180,
                 core_hours_with_scheduler_allowance=(seconds+len(shards)*180)*8/3600,
                 remaining_to_240_requested_core_hours=240-seconds*8/3600,
                 limitations=["Runtime/cache guards must still execute on Leonardo; no live cache loading here.",
                              "Walltime is the frozen conservative allocation formula, not an actual duration forecast.",
                              "Publication, checkout, user approval record and live queue/ledger checks remain root responsibilities."])
    return budget, audit


if __name__ == "__main__":
    budget, audit = build()
    (OUT / "probe_fit_resources.json").write_text(json.dumps(budget, indent=2, allow_nan=False) + "\n")
    (OUT / "probe_fit_packing_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: audit[k] for k in ("minimum_shards", "requested_core_hours", "sequential_walltime_hours", "core_hours_with_scheduler_allowance")}))
