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


def recipe_identity(plan):
    keys = ("protocol", "checkpoint", "epn_roots", "input_sha256", "batch", "epochs", "max_updates", "seed", "lr_grid")
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
    T.require(plan["recipe_sha256"] == recipe_identity(plan), "recipe hash mismatch")
    science_sha = source_identity()
    if args.mode == "fit":
        T.require(args.fold in range(4) and args.regime in F.REGIMES, "fold/regime required")
        T.require(isinstance(plan["benchmark_receipt"], dict), "pin benchmark receipt before fits")
        gate_pin = T.P.pinned_file(plan["benchmark_receipt"])
        gate = json.loads(Path(gate_pin["path"]).read_text())
        T.require(gate["complete"] is True and gate["protocol"] == PROTOCOL and gate["source_sha256"] == science_sha and
                  gate["mode"] == "benchmark" and gate["recipe_sha256"] == plan["recipe_sha256"], "benchmark provenance mismatch")
        T.require(gate["projected_two_candidates_seconds"][args.regime] <= 14100, "fit exceeds four-hour allocation")
        T.require(gate["resource_gate_passed"] is True, "benchmark memory gate failed")
    started = time.monotonic()
    state, w, y, s, fs, montage, fold_roles, source = load_inputs(plan)
    load_seconds = time.monotonic() - started
    args.out.mkdir(parents=True)
    device = "cuda"
    report = {"protocol": PROTOCOL, "complete": False, "mode": args.mode, "code_sha": code,
              "plan_sha256": pin["sha256"], "recipe_sha256": plan["recipe_sha256"], "source_sha256": science_sha,
              "checkpoint_sha256": source["sha256"], "input_sha256": plan["input_sha256"],
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
            timings[regime] = {"seconds": elapsed, "updates": result["updates"],
                "seconds_per_update_including_small_val": elapsed / result["updates"],
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
        projected = {r: 2 * v["seconds_per_update_including_small_val"] * units * 1.5 +
                     load_seconds * 1.5 + preprocessing_projection + 600 for r, v in timings.items()}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1)
        projected_rss = rss * len(w) / len(train) * 1.3
        memory_ok = projected_rss <= .85 * 100 * 1024**3 and all(
            v["peak_reserved_bytes"] <= .85 * v["device_memory_bytes"] for v in timings.values())
        report.update(timings=timings, projected_two_candidates_seconds=projected,
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
        results, best_score, best_path, chosen = [], -np.inf, None, None
        for number, lr in enumerate(F.LR_GRID[args.regime]):
            model = F.build_classifier(state, args.regime, len(classes), seed=plan["seed"]).to(device)
            def log(row):
                with (args.out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"candidate": number, "lr": lr, **row}) + "\n")
            result = F.fit_candidate(model, data, labels, roles, weights, args.regime, lr, device, callback=log)
            results.append({"lr": lr, **result})
            if result["val_bacc"] > best_score:  # LR tie: first, hence smaller LR
                best_score, chosen = result["val_bacc"], lr
                best_path = args.out / "selected.pt"
                torch.save({"model": model.state_dict(), "classes": classes.tolist(), "lr": lr}, best_path)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        model = F.build_classifier(state, args.regime, len(classes)).to(device)
        selected = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(selected["model"])
        prediction = F.predict(model, data, roles["held"], device)
        rows = roles["held"]
        np.savez(args.out / "held.npz", rows=rows, y=y[rows], pred=classes[prediction], s=s[rows],
                 fold=np.full(len(rows), args.fold), input_sha256=plan["input_sha256"])
        report.update(fold=args.fold, regime=args.regime, candidates=results, selected_lr=chosen,
                      roles=fold_roles[args.fold], n_held=len(rows),
                      held_bacc=F.balanced_accuracy(labels[rows], prediction, np.arange(len(classes))),
                      held_sha256=T.R.sha256_file(args.out / "held.npz"), convergence_claim=False)
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
    execute(p.parse_args())
