"""Review-gated fresh fifteen-epoch EPN fine-tuning and one nested development fold.

No sbatch, no automatic retry, no held predictions until LR/epoch selection.
The benchmark writes timing only and uses fold-0 training subjects exclusively.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import torch
import calibration_probe as CP
import tokenizer_train as T
import consolidation_finetune as FIVE
from wearusfm.harness import benchmarks, fm_features as FF, finetuning as F
from wearusfm.harness.downstream_ab import fingerprint

PROTOCOL = "consolidation-finetune15-epn-20261105-v1"
READING_RULE15 = {
    "epochs": 15,
    "primary_contrasts": ["finetune_minus_scratch", "finetune_minus_frozen"],
    "secondary_contrasts": ["frozen15_minus_frozen5", "finetune15_minus_finetune5", "scratch15_minus_scratch5"],
    "bootstrap": {"replicates": 2000, "seed": 20261102, "unit": "subject", "paired": True},
    "validation_tail": {
        "regimes": ["frozen", "finetune", "scratch"], "folds": 4, "best_epoch15_folds_at_least": 3,
        "or_median_validation_bacc_epoch15_minus14_at_least": .005,
        "missing_or_diverged": "not_evaluable", "false_does_not_establish_convergence": True,
        "automatic_extension": False,
    },
    "interpretation": "development; fixed15epoch recipe including new cosine horizon; not isolated dose effect; historical logistic reference descriptive only; no V-R1 adoption gate",
}
FIT_ALLOCATION_BY_REGIME = {
    regime: {"gpus": 1, "cpus": 8, "memory": "100G", "seconds": seconds}
    for regime, seconds in (("frozen", 21600), ("finetune", 57600), ("scratch", 57600))
}
FIT_PACKAGING_POLICY = {
    "mode": "split_only", "seconds_reserved_per_job": 300,
    "otherwise": "blocked; no automatic dose/batch/grid change",
    "selection": "second split job loads first candidate receipt; held only after both results; no held-based fallback",
}
RESOURCE_PROJECTION_POLICY = {
    "reference_epochs": 5, "target_epochs": 15, "elapsed_multiplier": 1.25, "extra_seconds": 600,
    "throughput": "seconds_per_update_steady", "slowdown_floor": 1, "load_multiplier": 1.5,
    "formula": "ceil(3*M*1.25*q+600+D)",
}
REFERENCE_SOURCE_SHA256 = "aa857fb321bae83fd4b6463092c89dd2dba27e248cbe063f25a30032bf529b98"
SOURCE_EXTRA_PATHS = (
    "scripts/consolidation_finetune15.py", "scripts/consolidation_finetune.py",
    "scripts/calibration_probe.py", "scripts/tokenizer_train.py", "scripts/tokenizer_preflight.py",
    "scripts/slurm/consolidation_finetune15.sbatch", "scripts/slurm/lib_code_dir.sh",
)


def recipe_identity(plan):
    keys = ("protocol", "checkpoint", "epn_roots", "input_sha256", "batch", "epochs", "max_updates", "seed", "lr_grid",
            "head_lr_multiplier", "reading_rule15")
    return hashlib.sha256(json.dumps({k: plan[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def source_identity():
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / path for path in SOURCE_EXTRA_PATHS]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)): T.R.sha256_file(p) for p in files}, sort_keys=True).encode()).hexdigest()


def load_inputs(plan):
    source = T.P.pinned_file(plan["checkpoint"])
    state = torch.load(source["path"], map_location="cpu", weights_only=False)
    T.require(state["step"] == 19618, "wrong downstream initialization dose")
    expected = json.loads((ROOT / "results/infra/tokenizer_execution_20261103/19618/T100-G/config.json").read_text())
    T.require(state["config"] == expected, "wrong checkpoint recipe")
    w, y, s, fs = benchmarks.load_epn612([Path(x) for x in plan["epn_roots"]])
    s = np.asarray(s).astype(str)
    internal = CP.p2_internal_subjects(sorted(set(s)))
    keep = np.isin(s, internal)
    w, y, s = w[keep], y[keep], s[keep]
    montage = FF.myo8_montage("epn612", fs)
    digest = fingerprint(w, y, s, fs, montage)
    T.require(digest == plan["input_sha256"], "EPN inputs differ from historical frozen probes")
    return state, w, y, s, fs, montage, CP.fold_roles(internal), source


def fit_resource_policy(plan):
    T.require(plan.get("fit_allocation_by_regime") == FIT_ALLOCATION_BY_REGIME, "fit allocation policy changed")
    T.require(plan.get("fit_packaging_policy") == FIT_PACKAGING_POLICY, "fit packaging policy changed")
    T.require(plan.get("resource_projection_policy") == RESOURCE_PROJECTION_POLICY, "resource projection policy changed")
    return {regime: allocation["seconds"] - FIT_PACKAGING_POLICY["seconds_reserved_per_job"]
            for regime, allocation in FIT_ALLOCATION_BY_REGIME.items()}


def reference_five(plan):
    """Verify all prior elapsed-time evidence without loading held predictions."""
    T.require(plan["reference_source_sha256"] == REFERENCE_SOURCE_SHA256 == FIVE.source_identity(),
              "five-epoch source equivalence lost")
    T.require(set(plan["reference_five"]) == {"accounting", "manifest", "benchmark", "plan"}, "incomplete reference pins")
    pins = {name: T.P.pinned_file(pin) for name, pin in plan["reference_five"].items()}
    ref = {name: json.loads(Path(pin["path"]).read_text()) for name, pin in pins.items()}
    old, manifest, bench, accounting = (ref[k] for k in ("plan", "manifest", "benchmark", "accounting"))
    T.require(old["epochs"] == 5 and old["recipe_sha256"] == FIVE.recipe_identity(old), "wrong five-epoch recipe")
    for key in ("checkpoint", "epn_roots", "input_sha256", "batch", "max_updates", "seed", "lr_grid", "head_lr_multiplier"):
        T.require(old[key] == plan[key], "five-epoch recipe differs: " + key)
    T.require(manifest["source_sha256"] == REFERENCE_SOURCE_SHA256 and
              manifest["plan_sha256"] == pins["plan"]["sha256"], "five-epoch manifest provenance mismatch")
    T.require(bench["complete"] is True and bench["mode"] == "benchmark" and bench["protocol"] == FIVE.PROTOCOL and
              bench["source_sha256"] == REFERENCE_SOURCE_SHA256 and bench["recipe_sha256"] == old["recipe_sha256"] and
              bench["checkpoint_sha256"] == old["checkpoint"]["sha256"] and bench["input_sha256"] == old["input_sha256"],
              "five-epoch benchmark provenance mismatch")
    T.require(accounting["all_29_completed_zero"] is True, "five-epoch accounting not complete")
    listed = {item["relative_path"]: item for item in manifest["files"]}
    T.require(len(listed) == len(manifest["files"]), "duplicate reference paths")
    reports_root = ROOT / plan["reference_report_root"]
    T.require(reports_root.resolve() == (Path(pins["manifest"]["path"]).parent / "fit_reports").resolve(),
              "unexpected reference reports root")
    jobs = {item["id"]: item for item in accounting["jobs"] if item["id"].startswith("FT-EPN-")}
    T.require(len(jobs) == 24, "exactly 24 reference fit jobs required")
    max_seconds = {regime: 0 for regime in F.REGIMES}
    for regime in F.REGIMES:
        for fold in range(4):
            for number in (0, 1):
                prefix = f"fold{fold}/{regime}/candidate{number}"
                docs = {}
                for name in ("report.json", f"candidate_{number}.json"):
                    relative = prefix + "/" + name
                    item = listed[relative]
                    pin = T.P.file_identity(reports_root / relative)
                    T.require(pin["sha256"] == item["sha256"] and pin["bytes"] == item["bytes"], "reference report changed")
                    docs[name] = json.loads(Path(pin["path"]).read_text())
                report, candidate = docs["report.json"], docs[f"candidate_{number}.json"]
                T.require(report["complete"] is True and report["protocol"] == FIVE.PROTOCOL and report["mode"] == "fit" and
                          report["code_sha"] == manifest["code_sha"] and report["source_sha256"] == REFERENCE_SOURCE_SHA256 and
                          report["plan_sha256"] == pins["plan"]["sha256"] and report["recipe_sha256"] == old["recipe_sha256"] and
                          report["checkpoint_sha256"] == old["checkpoint"]["sha256"] and report["input_sha256"] == old["input_sha256"] and
                          report["fold"] == fold and report["regime"] == regime, "reference fit provenance mismatch")
                T.require(report["candidate_file_sha256"][str(number)] == listed[prefix + f"/candidate_{number}.json"]["sha256"] and
                          candidate["complete"] is True and candidate["candidate"] == number and candidate["stopped"] == "fixed_budget" and
                          candidate["provenance"] == report["candidate_provenance"] and candidate["lr"] == F.LR_GRID[regime][number] and
                          candidate["head_lr"] == F.LR_GRID[regime][number] * F.HEAD_LR_MULTIPLIER[regime],
                          "reference candidate mismatch or divergence")
                history, dose = candidate["history"], candidate["planned_updates"]
                T.require(dose > 0 and dose % 5 == 0 and candidate["updates"] == dose and
                          [r["epoch"] for r in history] == list(range(1, 6)) and
                          [r["step"] for r in history] == [dose // 5 * i for i in range(1, 6)] and
                          all(np.isfinite(r["val_bacc"]) and np.isfinite(r["loss"]) for r in history) and
                          np.isfinite(candidate["val_bacc"]) and candidate["best_epoch"] in range(1, 6),
                          "reference candidate incomplete or nonfinite")
                job = jobs[f"FT-EPN-{regime}-F{fold}-20261104-C{number}"]
                T.require(job["gpus"] == 1 and job["cpus"] == 8 and type(job["seconds"]) is int and job["seconds"] > 0,
                          "invalid reference elapsed time")
                max_seconds[regime] = max(max_seconds[regime], job["seconds"])
    return {"pins": plan["reference_five"], "source_sha256": REFERENCE_SOURCE_SHA256,
            "max_elapsed_seconds_by_regime": max_seconds, "benchmark": bench}


def resource_projection(measurements, reference):
    """Pre-registered complete-job scaling, with a floor on measured slowdown."""
    old = reference["benchmark"]
    def fixed_cost(report):
        values = (report["load_seconds"], report["projected_full_preprocessing_seconds"])
        T.require(all(np.isfinite(v) and v >= 0 for v in values), "invalid measured fixed cost")
        return 1.5 * values[0] + values[1]
    delta = max(0., fixed_cost(measurements) - fixed_cost(old))
    ratios, projected = {}, {}
    for regime in F.REGIMES:
        for report in (old, measurements):
            timing = report["timings"][regime]
            T.require(timing["updates"] == 32 and timing["warmup_updates"] == 4 and
                      np.isfinite(timing["seconds_per_update_steady"]) and timing["seconds_per_update_steady"] > 0 and
                      np.isclose(timing["seconds_per_update_steady"], timing["seconds"] / 32, rtol=1e-12, atol=0),
                      "incompatible benchmark timing")
        ratios[regime] = max(1., measurements["timings"][regime]["seconds_per_update_steady"] /
                            old["timings"][regime]["seconds_per_update_steady"])
        maximum = reference["max_elapsed_seconds_by_regime"][regime]
        T.require(type(maximum) is int and maximum > 0, "invalid reference maximum")
        projected[regime] = math.ceil(3 * maximum * 1.25 * ratios[regime] + 600 + delta)
    return {"max_elapsed_seconds_by_regime": reference["max_elapsed_seconds_by_regime"],
            "slowdown_by_regime": ratios, "additional_fixed_seconds": delta,
            "projected_one_candidate_seconds": projected}


def benchmark_packaging(gate, plan, science_sha, reference):
    limits = fit_resource_policy(plan)
    T.require(gate["complete"] is True and gate["protocol"] == PROTOCOL and gate["source_sha256"] == science_sha and
              gate["mode"] == "benchmark" and gate["recipe_sha256"] == plan["recipe_sha256"], "benchmark provenance mismatch")
    T.require(gate.get("fit_allocation_by_regime") == plan["fit_allocation_by_regime"] and
              gate.get("fit_packaging_policy") == plan["fit_packaging_policy"] and
              gate.get("resource_projection_policy") == plan["resource_projection_policy"] and
              gate.get("reference_five") == plan["reference_five"] and
              gate.get("reference_source_sha256") == plan["reference_source_sha256"] and
              gate.get("fit_limit_seconds_by_regime") == limits, "benchmark resource policy mismatch")
    projection = resource_projection(gate, reference)
    T.require(gate["resource_projection"] == projection, "benchmark projection differs from frozen rule")
    packaging = {regime: "split" if projection["projected_one_candidate_seconds"][regime] <= limits[regime] else "blocked"
                 for regime in F.REGIMES}
    T.require(gate["fit_packaging"] == packaging, "benchmark packaging differs from measured projections")
    T.require(gate["resource_gate_passed"] is True, "benchmark memory gate failed")
    return packaging


def select_candidate(results):
    """Numerical divergence is an outcome; no invented score or silent fallback."""
    finite = [r for r in results if r["stopped"] != "nonfinite" and r["val_bacc"] is not None and
              np.isfinite(r["val_bacc"])]
    return min(finite, key=lambda r: (-r["val_bacc"], r["candidate"])) if finite else None


def require_full_dose(result, roles, plan):
    expected = plan["epochs"] * int(np.ceil(len(roles["train"]) / plan["batch"]))
    T.require(result["stopped"] == "fixed_budget" and result["updates"] == expected and
              result["planned_updates"] == expected, "candidate did not complete the fixed fifteen-epoch dose")
    history = result["history"]
    T.require([row["epoch"] for row in history] == list(range(1, 16)) and history[-1]["step"] == expected,
              "candidate lacks fifteen complete epochs")
    T.require(result["best_epoch"] in range(1, 16), "selected epoch outside fifteen-epoch fit")


def run_candidate(state, data, labels, roles, weights, classes, regime, number, plan, device, out, provenance):
    lr = F.LR_GRID[regime][number]
    model = F.build_classifier(state, regime, len(classes), seed=plan["seed"]).to(device)
    def log(row):
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"candidate": number, "lr": lr, **row}, allow_nan=False) + "\n")
    try:
        result = F.fit_candidate(model, data, labels, roles, weights, regime, lr, device,
                                 seed=plan["seed"], batch=plan["batch"], epochs=plan["epochs"], callback=log)
    except F.NumericalDivergence as exc:
        result = {"stopped": "nonfinite", "val_bacc": None, "best_epoch": None, "history": exc.history,
                  "numeric_failure": exc.as_dict(), "convergence_claim": False}
    else:
        require_full_dose(result, roles, plan)
        path = out / f"candidate_{number}.pt"
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "classes": classes.tolist(), "candidate": number, "lr": lr}, path)
        result.update(checkpoint=str(path.resolve()), checkpoint_sha256=T.R.sha256_file(path))
    result.update(candidate=number, lr=lr, head_lr=lr * F.HEAD_LR_MULTIPLIER[regime],
                  provenance=provenance, complete=True)
    # Persist immediately, including divergence; OOM/programming errors propagate.
    T.P.write_json(out / f"candidate_{number}.json", result)
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def prior_candidate(out, provenance):
    """Second split job consumes its own verified first-job receipt read-only."""
    prior_dir = out.parent / "candidate0"
    receipt_pin = T.P.file_identity(prior_dir / "report.json")
    receipt = json.loads(Path(receipt_pin["path"]).read_text())
    T.require(receipt["complete"] is True and receipt.get("candidate_only") == 0 and
              receipt["candidate_provenance"] == provenance, "wrong first-candidate receipt")
    path = prior_dir / "candidate_0.json"
    T.require(T.R.sha256_file(path) == receipt["candidate_file_sha256"]["0"], "first-candidate report changed")
    result = json.loads(path.read_text())
    T.require(result["complete"] is True and result["candidate"] == 0 and result["provenance"] == provenance,
              "wrong first candidate")
    if result["stopped"] != "nonfinite":
        T.require(Path(result["checkpoint"]).resolve() == (prior_dir / "candidate_0.pt").resolve(), "foreign checkpoint path")
        T.require(T.R.sha256_file(result["checkpoint"]) == result["checkpoint_sha256"], "first-candidate checkpoint changed")
    T.require(T.P.file_identity(prior_dir / "report.json") == receipt_pin, "receipt changed during read")
    result["source_receipt"] = {"path": receipt_pin["path"], "sha256": receipt_pin["sha256"]}
    return result


def execute(args):
    code = T.P.code_identity(args.expected_code)
    pin = T.P.file_identity(args.plan)
    plan = json.loads(args.plan.read_text())
    T.require(plan["protocol"] == PROTOCOL and plan["status"] == "claude_review_converged", "Claude review required before execution")
    T.require(torch.cuda.device_count() == 1, "one visible GPU required")
    T.require(not args.out.exists() and args.out.is_absolute() and "wearusfm_runs" in args.out.parts and
              "wearusfm_campaigns" not in args.out.parts and not args.out.is_symlink(), "new external output required")
    output_root = Path("/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/finetune15_20261105")
    T.require(plan["retention"]["root"] == str(output_root) and plan["fit_output_root"] == str(output_root / "epn") and
              args.out.is_relative_to(output_root) and args.out != output_root and
              not any(path.is_symlink() for path in args.out.parents), "outputs must use the persistent fifteen-epoch namespace")
    T.require(plan["batch"] == 8 and plan["epochs"] == 15 and plan["max_updates"] is None and plan["seed"] == 0,
              "training recipe changed")
    T.require(plan["lr_grid"] == {k: list(v) for k, v in F.LR_GRID.items()}, "LR grid changed")
    T.require(plan["head_lr_multiplier"] == F.HEAD_LR_MULTIPLIER, "head LR multiplier changed")
    T.require(plan["reading_rule15"] == READING_RULE15, "fifteen-epoch reading rule changed")
    T.require(plan["recipe_sha256"] == recipe_identity(plan), "recipe hash mismatch")
    fit_limits = fit_resource_policy(plan)
    science_sha = source_identity()
    reference = reference_five(plan)
    if args.mode == "fit":
        T.require(plan.get("fits_status") == "claude_review_converged", "corrected fit protocol review still required")
        T.require(args.fold in range(4) and args.regime in F.REGIMES, "fold/regime required")
        T.require(isinstance(plan["benchmark_receipt"], dict), "pin benchmark receipt before fits")
        gate_pin = T.P.pinned_file(plan["benchmark_receipt"])
        gate = json.loads(Path(gate_pin["path"]).read_text())
        packaging = benchmark_packaging(gate, plan, science_sha, reference)[args.regime]
        T.require(packaging == "split" and args.candidate in (0, 1), "fit packaging differs from benchmark policy or is blocked")
        suffix = f"candidate{args.candidate}"
        expected_out = Path(plan["fit_output_root"]) / f"fold{args.fold}" / args.regime / suffix
        T.require(args.out == expected_out, "fit output differs from the frozen per-fold layout")
        print(json.dumps({"event": "fit_gate_accepted", "protocol": PROTOCOL,
                          "benchmark_receipt": {"path": gate_pin["path"], "sha256": gate_pin["sha256"]},
                          "source_sha256": science_sha, "recipe_sha256": plan["recipe_sha256"],
                          "fold": args.fold, "regime": args.regime, "candidate": args.candidate,
                          "allocation": plan["fit_allocation_by_regime"][args.regime],
                          "limit_seconds": fit_limits[args.regime], "packaging": packaging,
                          "memory_gate_passed": gate["resource_gate_passed"]}, sort_keys=True), flush=True)
    started = time.monotonic()
    state, w, y, s, fs, montage, fold_roles, source = load_inputs(plan)
    load_seconds = time.monotonic() - started
    args.out.mkdir(parents=True)
    device = "cuda"
    report = {"protocol": PROTOCOL, "complete": False, "mode": args.mode, "code_sha": code,
              "plan_sha256": pin["sha256"], "recipe_sha256": plan["recipe_sha256"], "source_sha256": science_sha,
              "checkpoint_sha256": source["sha256"], "input_sha256": plan["input_sha256"],
              "fit_allocation_by_regime": plan["fit_allocation_by_regime"],
              "fit_packaging_policy": plan["fit_packaging_policy"], "fit_limit_seconds_by_regime": fit_limits,
              "resource_projection_policy": plan["resource_projection_policy"],
              "reference_five": plan["reference_five"], "reference_source_sha256": plan["reference_source_sha256"],
              "initialization": "fresh from teacher19618 or random seed0; new 15-epoch cosine; no resume from five epochs"}
    if args.mode == "benchmark":
        roles = F.role_indices(s, fold_roles[0])
        # No held or validation subjects enter the benchmark forward pass.
        train = roles["train"]
        idx = np.linspace(0, len(train) - 1, min(128, len(train)), dtype=int)
        prep_started = time.monotonic()
        data = F.PreparedWindows(w[train], s[train], fs, montage)
        preprocessing_seconds = time.monotonic() - prep_started
        vocabulary, targets = np.unique(y[train].astype(str), return_inverse=True)
        bench_roles = {"train": idx[:-8], "val": idx[-8:], "held": np.array([], dtype=int)}
        weights = torch.ones(len(vocabulary))
        timings = {}
        for regime in F.REGIMES:
            warm_model = F.build_classifier(state, regime, len(vocabulary)).to(device)
            torch.cuda.synchronize()
            warm_started = time.monotonic()
            warm_result = F.fit_candidate(warm_model, data, targets, bench_roles, weights, regime,
                                          F.LR_GRID[regime][0], device, epochs=1, max_updates=4)
            torch.cuda.synchronize()
            warm_seconds = time.monotonic() - warm_started
            del warm_model
            gc.collect()  # preserve the warm CUDA allocator; no weights reused
            model = F.build_classifier(state, regime, len(vocabulary)).to(device)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.monotonic()
            result = F.fit_candidate(model, data, targets, bench_roles, weights, regime, F.LR_GRID[regime][0],
                                     device, epochs=3, max_updates=32)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            # Conservative: all measured seconds (construction excluded) per
            # update, including tiny validation and CPU best-state copies.
            timings[regime] = {"seconds": elapsed, "updates": result["updates"], "warmup_seconds": warm_seconds,
                "warmup_updates": warm_result["updates"],
                "seconds_per_update_all": (warm_seconds + elapsed) / (warm_result["updates"] + result["updates"]),
                "seconds_per_update_steady": elapsed / result["updates"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
            del model
            gc.collect()
            torch.cuda.empty_cache()
        worst_train = max(len(F.role_indices(s, r)["train"]) for r in fold_roles)
        worst_val = max(len(F.role_indices(s, r)["val"]) for r in fold_roles)
        worst_held = max(len(F.role_indices(s, r)["held"]) for r in fold_roles)
        updates = 15 * int(np.ceil(worst_train / 8))
        # Validation/held are charged at *training* update cost, with 50% margin.
        units = updates + 15 * int(np.ceil(worst_val / 8)) + int(np.ceil(worst_held / 8))
        preprocessing_projection = preprocessing_seconds * len(w) / len(train) * 1.5
        projected = {r: 2 * v["seconds_per_update_steady"] * units * 1.5 +
                     load_seconds * 1.5 + preprocessing_projection + 600 for r, v in timings.items()}
        projected_one = {r: v["seconds_per_update_steady"] * units * 1.5 +
                         load_seconds * 1.5 + preprocessing_projection + 600 for r, v in timings.items()}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1)
        projected_rss = rss * len(w) / len(train) * 1.3
        memory_ok = projected_rss <= .85 * 100 * 1024**3 and all(
            v["peak_reserved_bytes"] <= .85 * v["device_memory_bytes"] for v in timings.values())
        report.update(timings=timings, diagnostic_projected_two_candidates_seconds=projected,
                      diagnostic_projected_one_candidate_seconds=projected_one,
                      load_seconds=load_seconds, train_preprocessing_seconds=preprocessing_seconds,
                      projected_full_preprocessing_seconds=preprocessing_projection, peak_rss_bytes=rss,
                      projected_full_peak_rss_bytes=projected_rss, resource_gate_passed=memory_ok,
                      forward_and_preprocessing_train_subjects_only=True, scientific_checkpoint_written=False)
        projection = resource_projection(report, reference)
        report.update(resource_projection=projection,
                      fit_packaging={r: "split" if projection["projected_one_candidate_seconds"][r] <= fit_limits[r] else "blocked"
                                     for r in F.REGIMES})
        T.P.write_json(args.out / "benchmark_measurements.json", report)
        T.require(memory_ok, "benchmark memory projection exceeds 85% GPU/host allocation")
    else:
        data = F.PreparedWindows(w, s, fs, montage)
        roles = F.role_indices(s, fold_roles[args.fold])
        labels, classes, weights = F.encode_labels(y, roles["train"])
        provenance = {"protocol": PROTOCOL, "recipe_sha256": plan["recipe_sha256"], "source_sha256": science_sha,
                      "checkpoint_sha256": source["sha256"], "input_sha256": plan["input_sha256"],
                      "fold": args.fold, "regime": args.regime, "roles": fold_roles[args.fold], "classes": classes.tolist()}
        results = [prior_candidate(args.out, provenance)] if args.candidate == 1 else []
        numbers = (args.candidate,)
        for number in numbers:
            results.append(run_candidate(state, data, labels, roles, weights, classes, args.regime, number,
                                         plan, device, args.out, provenance))
        report.update(fold=args.fold, regime=args.regime, candidates=results, candidate_provenance=provenance,
                      roles=fold_roles[args.fold], convergence_claim=False,
                      candidate_file_sha256={str(n): T.R.sha256_file(args.out / f"candidate_{n}.json") for n in numbers})
        selected_result = select_candidate(results)
        if args.candidate == 0:
            report.update(candidate_only=0, scientific_status="first_candidate_recorded", held_evaluated=False)
        elif selected_result is None:
            report.update(scientific_status="all_candidates_diverged", held_evaluated=False, selected_lr=None)
        else:
            best_path = Path(selected_result["checkpoint"])
            T.require(T.R.sha256_file(best_path) == selected_result["checkpoint_sha256"], "selected state changed")
            model = F.build_classifier(state, args.regime, len(classes), seed=plan["seed"]).to(device)
            selected = torch.load(best_path, map_location="cpu", weights_only=False)
            T.require(selected["classes"] == classes.tolist(), "selected class vocabulary mismatch")
            model.load_state_dict(selected["model"])
            try:
                prediction = F.predict(model, data, roles["held"], device)
            except F.NumericalDivergence as exc:
                # Do not select the other LR using a held failure.
                report.update(scientific_status="selected_model_nonfinite_on_held", held_evaluated=False,
                              numeric_failure=exc.as_dict(), selected_lr=selected_result["lr"])
            else:
                rows = roles["held"]
                np.savez(args.out / "held.npz", rows=rows, y=y[rows], pred=classes[prediction], s=s[rows],
                         fold=np.full(len(rows), args.fold), input_sha256=plan["input_sha256"])
                report.update(scientific_status="held_evaluated", held_evaluated=True,
                              selected_lr=selected_result["lr"], selected_candidate=selected_result["candidate"],
                              selected_best_epoch=selected_result["best_epoch"], n_held=len(rows),
                              held_bacc=F.balanced_accuracy(labels[rows], prediction, np.arange(len(classes))),
                              held_sha256=T.R.sha256_file(args.out / "held.npz"))
    T.require(T.P.file_identity(source["path"])["sha256"] == source["sha256"], "source checkpoint modified")
    T.require(T.P.file_identity(args.plan) == pin and T.P.code_identity(code) == code, "code or plan changed")
    report.update(complete=True, elapsed_seconds=time.monotonic() - started)
    T.P.write_json(args.out / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--expected-code", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("benchmark", "fit"), required=True)
    p.add_argument("--fold", type=int)
    p.add_argument("--regime", choices=F.REGIMES)
    p.add_argument("--candidate", type=int, choices=(0, 1), help="one candidate per fit job; candidate1 reads candidate0 receipt")
    execute(p.parse_args())
