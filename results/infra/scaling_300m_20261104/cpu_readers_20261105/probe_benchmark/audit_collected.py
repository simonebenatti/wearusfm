"""Offline evidence audit, not a runtime gate or authorization to submit jobs.

Run with Python 3 from this repository. It reads only collected benchmark
metadata, logs, accounting and the published plan; it does not load caches,
calculate downstream scores, or contact Leonardo.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jhash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read(path):
    return json.loads(path.read_text())


def main():
    plan_path = BASE.parent / "probe_plan.json"
    plan = read(plan_path)
    plan_sha = sha(plan_path.read_bytes())
    assert plan_sha == "d2cc7da1cf08473067d96b52bff532463453ff5ce633814743b1a53dda5bbb63"
    manifest = read(BASE / "collection_manifest.json")
    for relative, pin in manifest.items():
        data = (BASE / relative).read_bytes()
        assert len(data) == pin["bytes"] and sha(data) == pin["sha256"], relative
    for relative, digest in plan["sources"].items():
        assert sha((ROOT / relative).read_bytes()) == digest, relative

    contexts, rows, completed_fits = [], [], 0
    support = {}
    for path in sorted(BASE.glob("*/*/benchmark.json")):
        r = read(path)
        cell, ds = path.parent.parent.name, path.parent.name
        arm = plan["recipe"]["cells"][cell]["arm"]
        assert (r["cell"], r["dataset"], r["arm"]) == (cell, ds, arm)
        assert r["protocol"] == plan["protocol"] and r["kind"] == "benchmark" and r["complete"] is True
        assert r["code_sha"] == "aef665ba7ea66fc80291846487907bb3ef9107df"
        assert r["plan_sha256"] == plan_sha and r["plan_content_sha256"] == jhash(plan)
        assert r["recipe"] == plan["recipe"] and r["step"] == plan["step"] == 5000
        assert r["train_only"] is True and r["selection_performed"] is False and r["fold"] == 0
        assert r["n_features"] == plan["recipe"]["widths"][cell]
        assert r["numeric_sha256"] == jhash(r["numeric_context"])
        assert all(t["num_threads"] == 8 for t in r["numeric_context"]["threadpools"])
        contexts.append(r["numeric_context"])
        audit = [json.loads(line) for line in path.with_name("audit_progress.jsonl").read_text().splitlines()]
        assert audit == r["fits"] and [f["C"] for f in audit] == plan["recipe"]["C_grid"]
        assert len(audit) == 6 and r["n_train"] == r["fold_train_counts"][0]
        for f in audit:
            check_fit(f, cell, ds, arm, r["n_features"])
            assert f["n_train"] == r["n_train"] and f["roles_sha256"] == r["roles_sha256"]
        assert len({f["scaler_sha256"] for f in audit}) == 1
        assert r["audit_summary"] == dict(fits=6, convergence_warnings=0, reached_limit=0,
                                          converged_by_criterion=True, unresolved_fits=[])
        # Cross-view support agreement; raw labels/subjects are not reloaded here.
        signature = [r["n_train"], r["fold_train_counts"], r["roles_sha256"],
                     audit[0]["classes"], audit[0]["class_counts"]]
        assert support.setdefault(ds, signature) == signature
        times = [f["elapsed_s"] for f in audit]
        assert r["elapsed_seconds"] >= sum(times)
        terms = [1.5 * (sum(times) + max(times)) * max(1., n/r["n_train"])
                 for n in r["fold_train_counts"]]
        rows.append(dict(cell=cell, dataset=ds, report_sha256=sha(path.read_bytes()),
                         elapsed_seconds=r["elapsed_seconds"], fit_seconds=sum(times),
                         max_fit_seconds=max(times), peak_rss_bytes=r["peak_rss_bytes"],
                         n_iterations=[f["n_iter"] for f in audit],
                         budget_terms_seconds_by_fold=terms,
                         budget_all_four_folds_seconds=120 + sum(terms)))
        completed_fits += len(audit)
    assert len(rows) == 13 and all(c == contexts[0] for c in contexts)

    partial_path = BASE / "T100-G-300M-natural/epn612/audit_progress.jsonl"
    partial = [json.loads(line) for line in partial_path.read_text().splitlines()]
    assert [f["C"] for f in partial] == plan["recipe"]["C_grid"][:5]
    for f in partial:
        check_fit(f, "T100-G-300M-natural", "epn612", "T100-G-300M", 1792)
        assert f["roles_sha256"] == support["epn612"][2]
        assert [f["classes"], f["class_counts"]] == support["epn612"][3:]
    completed_fits += len(partial)

    accounts = []
    for path in sorted(BASE.glob("*/accounting.json")):
        a = read(path)
        raw = path.with_name("sacct.txt").read_bytes()
        assert sha(raw) == a["raw_sacct"]["sha256"]
        sacct = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="|"))
        parent = next(r for r in sacct if r["JobIDRaw"] == str(a["JobIDRaw"]))
        for k in ("State", "ExitCode", "NodeList", "Partition", "Start", "End", "Timelimit"):
            assert a[k] == parent[k]
        assert a["ElapsedRaw"] == int(parent["ElapsedRaw"])
        assert a["AllocCPUS"] == int(parent["AllocCPUS"]) == 8
        cost = int(parent["ElapsedRaw"]) * 8 / 3600
        assert a["actual_core_hours"] == cost
        rss = max(int(r["MaxRSS"][:-1])*1024 for r in sacct if r["MaxRSS"].endswith("K"))
        assert rss == a["MaxRSS_bytes"]
        accounts.append(dict(cell=path.parent.name, job_id=int(a["JobIDRaw"]), state=a["State"],
                             elapsed_seconds=a["ElapsedRaw"], actual_core_hours=cost,
                             max_rss_gib=rss/2**30))
    assert len(accounts) == 3
    known_fit_lower_bound = sum(sum(r["budget_terms_seconds_by_fold"]) for r in rows)*8/3600
    summary = dict(
        status="HALTED_TIMEOUT_AND_INSUFFICIENT_FIT_CAP", fits_released=False,
        collected_files_hash_verified=len(manifest), complete_reports=len(rows), expected_reports=15,
        completed_benchmark_fits=completed_fits, completed_fits_converged=completed_fits,
        missing_reports=["T100-G-300M-natural/epn612", "T100-G-300M-natural/uci_emg"],
        numeric_context_identical=True, numeric_sha256=jhash(contexts[0]),
        plan_sha256=plan_sha, scientific_scores_computed=False,
        limitations=["Offline check does not rehash live remote caches or reconstruct fold subjects/labels.",
                     "Three complete reports belong to the TIMEOUT parent; no release or implicit reuse.",
                     "Partial audit is not a complete benchmark and does not identify the interrupted instruction.",
                     "Lower bound is on the frozen budget formula, not on future actual runtime."],
        accounts=accounts, probe_actual_core_hours=sum(a["actual_core_hours"] for a in accounts),
        known_fit_budget_lower_bound_core_hours=known_fit_lower_bound,
        known_fit_budget_excludes="120 seconds per shard plus both missing benchmark datasets",
        fit_cap_core_hours=136, fit_cap_impossible_from_known_reports=known_fit_lower_bound > 136,
        partial_epn_completed_C=[f["C"] for f in partial],
        partial_epn_completed_fit_seconds=sum(f["elapsed_s"] for f in partial), reports=rows)
    return summary


def check_fit(f, cell, ds, arm, width):
    assert (f["cell"], f["dataset"], f["arm"], f["n_features"]) == (cell, ds, arm, width)
    assert f["fit_phase"] == "benchmark_train_only" and f["fold"] == 0 and f["selected"] is None
    assert f["dtype"] == "float32" and sum(f["class_counts"]) == f["n_train"]
    assert f["converged_by_criterion"] is True and f["reached_limit"] is False
    assert f["exception"] is None and f["warnings"] == [] and f["convergence_warnings"] == 0
    assert math.isfinite(f["elapsed_s"]) and f["elapsed_s"] > 0 and 0 < f["n_iter"] < 12000
    assert f["max_iter"] == 12000
    assert f["solver_params"] == dict(C=f["C"], class_weight="balanced", dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=12000, multi_class="auto", n_jobs=None,
        penalty="l2", random_state=0, solver="lbfgs", tol=1e-4, verbose=0, warm_start=False)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
