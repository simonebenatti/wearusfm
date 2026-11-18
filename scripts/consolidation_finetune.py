"""Review-gated EPN fine-tuning benchmark and one nested development fold.

No sbatch, no automatic retry, no held predictions until LR/epoch selection.
The benchmark writes timing only and uses fold-0 training subjects exclusively.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from wearusfm.harness import benchmarks, fm_features as FF, finetuning as F
from wearusfm.harness.downstream_ab import fingerprint

PROTOCOL = "consolidation-finetune-epn-20261104-v1"
SCRATCH_RULE = {
    "folds": 4, "best_epoch5_folds_at_least": 3,
    "or_median_validation_bacc_epoch5_minus4_at_least": .005,
    "followup_epochs": 15, "followup_regimes": ["finetune", "scratch"],
    "followup_initialization": "fresh from same original teacher19618/random seed0; new 15-epoch horizon, not resume from selected5epoch",
    "followup_gate": "preregistered direction; publish measured resources and validate implementation before any extension",
}
FIT_ALLOCATION_BY_REGIME = {
    regime: {"gpus": 1, "cpus": 8, "memory": "100G", "seconds": seconds}
    for regime, seconds in (("frozen", 14400), ("finetune", 25200), ("scratch", 25200))
}
FIT_PACKAGING_POLICY = {
    "seconds_reserved_per_job": 300,
    "joint_if_two_candidates_seconds_lte": {"frozen": 14100, "finetune": 24900, "scratch": 24900},
    "otherwise_split_if_one_candidate_seconds_lte": {"frozen": 14100, "finetune": 24900, "scratch": 24900},
    "otherwise": "blocked; no automatic dose/batch/grid change",
    "selection": "second split job loads first candidate receipt; held only after both results; no held-based fallback",
}


def recipe_identity(plan):
    keys = ("protocol", "checkpoint", "epn_roots", "input_sha256", "batch", "epochs", "max_updates", "seed", "lr_grid",
            "head_lr_multiplier", "scratch_learning_rule")
    return hashlib.sha256(json.dumps({k: plan[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def source_identity():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), ROOT / "scripts/calibration_probe.py"]
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


def fit_packaging(two_seconds, one_seconds, limit_seconds=14100):
    """Prespecified budget rule: regroup jobs, never change scientific dose."""
    T.require(np.isfinite(limit_seconds) and limit_seconds > 0, "invalid per-job time limit")
    T.require(np.isfinite(two_seconds) and np.isfinite(one_seconds) and
              0 < one_seconds <= two_seconds, "invalid benchmark projection")
    if two_seconds <= limit_seconds:
        return "joint"
    if one_seconds <= limit_seconds:
        return "split"
    return "blocked"


def fit_resource_policy(plan):
    """The reviewed operational amendment is explicit, not an implicit fallback."""
    T.require("fit_allocation_each_fold_regime" not in plan, "ambiguous legacy fit allocation")
    T.require(plan.get("fit_allocation_by_regime") == FIT_ALLOCATION_BY_REGIME, "fit allocation policy changed")
    T.require(plan.get("fit_packaging_policy") == FIT_PACKAGING_POLICY, "fit packaging policy changed")
    return {regime: allocation["seconds"] - FIT_PACKAGING_POLICY["seconds_reserved_per_job"]
            for regime, allocation in FIT_ALLOCATION_BY_REGIME.items()}


def benchmark_packaging(gate, plan, science_sha):
    """A previous-source benchmark cannot authorize fits under a new controller."""
    limits = fit_resource_policy(plan)
    T.require(gate["complete"] is True and gate["protocol"] == PROTOCOL and gate["source_sha256"] == science_sha and
              gate["mode"] == "benchmark" and gate["recipe_sha256"] == plan["recipe_sha256"], "benchmark provenance mismatch")
    T.require(gate.get("fit_allocation_by_regime") == plan["fit_allocation_by_regime"] and
              gate.get("fit_packaging_policy") == plan["fit_packaging_policy"] and
              gate.get("fit_limit_seconds_by_regime") == limits, "benchmark resource policy mismatch")
    packaging = {regime: fit_packaging(gate["projected_two_candidates_seconds"][regime],
                                     gate["projected_one_candidate_seconds"][regime], limits[regime])
                 for regime in F.REGIMES}
    T.require(gate["fit_packaging"] == packaging, "benchmark packaging differs from measured projections")
    T.require(gate["resource_gate_passed"] is True, "benchmark memory gate failed")
    return packaging


def select_candidate(results):
    """Numerical divergence is an outcome; no invented score or silent fallback."""
    finite = [r for r in results if r["stopped"] != "nonfinite" and r["val_bacc"] is not None and
              np.isfinite(r["val_bacc"])]
    return min(finite, key=lambda r: (-r["val_bacc"], r["candidate"])) if finite else None


def scratch_still_improving(selected_reports):
    """Development-only flag, from validation, on four completed scratch folds."""
    T.require(len(selected_reports) == 4, "four scratch folds required")
    deltas, at_five = [], 0
    for report in selected_reports:
        T.require(report["stopped"] != "nonfinite" and report["best_epoch"] in range(1, 6), "scratch candidate incomplete")
        history = {row["epoch"]: row["val_bacc"] for row in report["history"]}
        T.require(set(history) == set(range(1, 6)), "five full epochs required")
        deltas.append(history[5] - history[4])
        at_five += report["best_epoch"] == 5
    return {"flag": at_five >= 3 or float(np.median(deltas)) >= .005,
            "folds_best_at_epoch5": at_five, "median_val_bacc_epoch5_minus4": float(np.median(deltas))}


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
    T.require(plan["batch"] == 8 and plan["epochs"] == 5 and plan["max_updates"] is None and plan["seed"] == 0,
              "training recipe changed")
    T.require(plan["lr_grid"] == {k: list(v) for k, v in F.LR_GRID.items()}, "LR grid changed")
    T.require(plan["head_lr_multiplier"] == F.HEAD_LR_MULTIPLIER, "head LR multiplier changed")
    T.require(plan["scratch_learning_rule"] == SCRATCH_RULE, "scratch reading rule changed")
    T.require(plan["recipe_sha256"] == recipe_identity(plan), "recipe hash mismatch")
    fit_limits = fit_resource_policy(plan)
    science_sha = source_identity()
    if args.mode == "fit":
        T.require(plan.get("fits_status") == "claude_review_converged", "corrected fit protocol review still required")
        T.require(args.fold in range(4) and args.regime in F.REGIMES, "fold/regime required")
        T.require(isinstance(plan["benchmark_receipt"], dict), "pin benchmark receipt before fits")
        gate_pin = T.P.pinned_file(plan["benchmark_receipt"])
        gate = json.loads(Path(gate_pin["path"]).read_text())
        packaging = benchmark_packaging(gate, plan, science_sha)[args.regime]
        T.require((packaging == "joint" and args.candidate is None) or
                  (packaging == "split" and args.candidate in (0, 1)), "fit packaging differs from benchmark policy or is blocked")
        suffix = "joint" if args.candidate is None else f"candidate{args.candidate}"
        expected_out = Path(plan["fit_output_root"]) / f"fold{args.fold}" / args.regime / suffix
        T.require(args.out == expected_out, "fit output differs from the frozen per-fold layout")
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
              "initialization": "teacher19618 or fresh random architecture; no EMA downstream"}
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
        updates = 5 * int(np.ceil(worst_train / 8))
        # Validation/held are charged at *training* update cost, with 50% margin.
        units = updates + 5 * int(np.ceil(worst_val / 8)) + int(np.ceil(worst_held / 8))
        preprocessing_projection = preprocessing_seconds * len(w) / len(train) * 1.5
        projected = {r: 2 * v["seconds_per_update_steady"] * units * 1.5 +
                     load_seconds * 1.5 + preprocessing_projection + 600 for r, v in timings.items()}
        projected_one = {r: v["seconds_per_update_steady"] * units * 1.5 +
                         load_seconds * 1.5 + preprocessing_projection + 600 for r, v in timings.items()}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1)
        projected_rss = rss * len(w) / len(train) * 1.3
        memory_ok = projected_rss <= .85 * 100 * 1024**3 and all(
            v["peak_reserved_bytes"] <= .85 * v["device_memory_bytes"] for v in timings.values())
        report.update(timings=timings, projected_two_candidates_seconds=projected,
                      projected_one_candidate_seconds=projected_one,
                      fit_packaging={r: fit_packaging(projected[r], projected_one[r], fit_limits[r]) for r in F.REGIMES},
                      load_seconds=load_seconds, train_preprocessing_seconds=preprocessing_seconds,
                      projected_full_preprocessing_seconds=preprocessing_projection, peak_rss_bytes=rss,
                      projected_full_peak_rss_bytes=projected_rss, resource_gate_passed=memory_ok,
                      forward_and_preprocessing_train_subjects_only=True, scientific_checkpoint_written=False)
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
        numbers = (0, 1) if args.candidate is None else (args.candidate,)
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
    p.add_argument("--candidate", type=int, choices=(0, 1), help="pre-registered split fallback; candidate1 reads candidate0 receipt")
    execute(p.parse_args())
