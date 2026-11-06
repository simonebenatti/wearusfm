"""Candidate-only pose ridge and four-cell LQ comparisons; never submits jobs.

All scientific fitting, prediction and sufficient statistics use the frozen
historical numerical functions. Imported cells keep their original provenance.
Plans with future cache or fit-release pins cannot execute those stages.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import tokenizer_pose_readout as Q

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "tokenizer-logsquare-pose-readout-20261104-v1"
EXTRACTION_PROTOCOL = "tokenizer-logsquare-evaluation-20261104-v1"
ARM = "T100-LQ"
CONTROLS = ("T100-L", "T25-L", "T100-G")
ARMS = (ARM,) + CONTROLS
require = Q.require
CAPS = {"validate": {"cpus": 2, "memory_mib": 30720, "seconds": 1800},
        "benchmark": {"cpus": 8, "memory_mib": 30720, "seconds": 900},
        "ridge": {"cpus": 8, "memory_mib": 30720, "seconds": 1800},
        "compare": {"cpus": 2, "memory_mib": 30720, "seconds": 1800}}


def source_hashes():
    return {**Q.source_hashes(), str(Path(__file__).relative_to(ROOT)): Q.D.file_sha256(__file__)}


def numeric_context(runtime):
    return {k: runtime[k] for k in ("versions", "threads", "torch_threads", "threadpools")}


def runtime_identity(expected_code_sha):
    result = Q.runtime_identity(expected_code_sha)
    result["sources"] = source_hashes()
    return result


def resolve_pin(pin):
    require(isinstance(pin, dict) and set(pin) == {"path", "sha256"} and
            isinstance(pin["path"], str) and Q._digest(pin["sha256"]), "missing/invalid artifact pin")
    path = Path(pin["path"])
    if not path.is_absolute():
        path = (ROOT/path).resolve()
        require(path.is_relative_to(ROOT), "relative pin escapes code")
    return path


def pinned_json(pin, states):
    path = resolve_pin(pin)
    states.update(Q.verify_file(path, pin["sha256"]))
    return Q.read_json(path)


def evaluation_identity(plan):
    # Receipts and published launch readiness are filled after the train-only
    # benchmark; none may silently change the scientific or execution inputs.
    return Q.json_hash({k: v for k, v in plan.items() if k not in ("status", "fit_release")})


def validate_plan(plan, runtime=False):
    require(plan.get("protocol") == PROTOCOL, "plan protocol mismatch")
    require(plan.get("status") in ("REVIEW_REQUIRED", "APPROVED_FOR_EVALUATION"), "invalid plan status")
    require(plan.get("step") in (5000, 19618) and type(plan["step"]) is int, "invalid dose")
    require(plan.get("recipe") == Q.recipe() and plan.get("stats_sha256") == Q.E.STATS_SHA, "recipe/stats differ")
    require(plan.get("sources") == source_hashes(), "numerical/adapter source pins differ")
    require(plan.get("resources") == CAPS, "resource caps changed")
    require(plan.get("contrasts") == {"primary": ARM+"_minus_T100-L", "secondary": [ARM+"_minus_T25-L", ARM+"_minus_T100-G"]}, "contrasts changed")
    history = plan.get("historical", {})
    require(set(history) == {"plan", "report", "code_sha", "numeric_context"}, "historical provenance missing")
    resolve_pin(history["plan"])
    resolve_pin(history["report"])
    require(Q._digest(history["code_sha"], 40), "historical code pin missing")
    old_plan = Q.read_json(resolve_pin(history["plan"]))
    require(Q.D.file_sha256(resolve_pin(history["plan"])) == history["plan"]["sha256"], "historical plan pin mismatch")
    Q.validate_plan(old_plan)
    for key in ("step", "recipe", "stats_sha256", "fold_users", "references"):
        require(plan.get(key) == old_plan[key], "historical design mismatch: "+key)
    caches = plan.get("caches")
    require(isinstance(caches, dict) and set(caches) == {"train", "val"}, "candidate split missing")
    for split, pin in caches.items():
        require(isinstance(pin, dict) and set(pin) == Q.CACHE_FIELDS, "candidate pin fields differ")
        if all(v is None for v in pin.values()) and not runtime:
            continue
        require(all(isinstance(pin[k], str) and Path(pin[k]).is_absolute() for k in ("path", "provenance_path", "receipt_path")), "unresolved candidate cache path")
        require(all(Q._digest(pin[k]) for k in ("sha256", "provenance_sha256", "receipt_sha256", "checkpoint_sha256")) and Q._digest(pin["extraction_code_sha"], 40), "unresolved candidate cache digest")
        require(type(pin["bytes"]) is int and pin["bytes"] > 0, "invalid candidate cache size")
    if runtime:
        require(plan["status"] == "APPROVED_FOR_EVALUATION", "evaluation not released")
        require(caches["train"]["checkpoint_sha256"] == caches["val"]["checkpoint_sha256"], "train/val checkpoint mismatch")
        paths = [p["path"] for p in caches.values()] + [p["path"] for p in plan["references"].values()]
        require(len(paths) == len(set(paths)), "duplicate candidate/reference paths")
        require(caches["train"]["checkpoint_sha256"] not in {old_plan["caches"][a]["train"]["checkpoint_sha256"] for a in Q.ARMS}, "candidate checkpoint is a historical control")
    return old_plan


def validate_artifact(pin, split, plan, reference):
    # Generic Q guard admits the actual LQ name (100 ms / identity); no renaming.
    inspection = Q.validate_artifact(pin, ARM, split, plan, reference)
    receipt, prov = Q.read_json(pin["receipt_path"]), Q.read_json(pin["provenance_path"])
    require(receipt.get("protocol") == prov.get("protocol") == EXTRACTION_PROTOCOL, "candidate extraction protocol mismatch")
    cfg = prov.get("model_config", {})
    require(receipt.get("model_config") == cfg, "receipt/provenance model configuration mismatch")
    require(cfg.get("tokenizer_logsquare") is True and cfg.get("tokenizer_logsquare_eps") == .001,
            "candidate LQ configuration mismatch")
    return inspection


def load_log_weight_diagnostics(plan, states):
    """Carry R1-b checkpoint norms from pinned extraction receipts, never from caches.

    The arrays are descriptive only. They do not affect alpha, timing or promotion.
    Both split receipts must identify the same endpoint and the same 64 columns.
    """
    diagnostics, receipts = {}, {}
    for split in ("train", "val"):
        pin = plan["caches"][split]
        receipt_pin = {"path": pin["receipt_path"], "sha256": pin["receipt_sha256"]}
        receipt = pinned_json(receipt_pin, states)
        require(receipt.get("protocol") == EXTRACTION_PROTOCOL and receipt.get("complete") is True and
                receipt.get("arm") == ARM and receipt.get("which") == "teacher" and
                receipt.get("checkpoint_step") == plan["step"] and
                receipt.get("checkpoint_sha256") == pin["checkpoint_sha256"], "log-weight receipt endpoint mismatch")
        value = receipt.get("log_weight_diagnostics")
        require(isinstance(value, dict) and set(value) == {"student", "teacher"}, "missing checkpoint log-weight diagnostics")
        for kind, record in value.items():
            norms = np.asarray(record.get("column_l2_norms"), dtype=np.float64)
            require(norms.shape == (64,) and np.isfinite(norms).all() and (norms >= 0).all() and
                    record.get("descriptive_only") is True, "invalid descriptive log-weight columns")
            families = {"fourier": norms[:24], "spline": norms[24:48], "mlp": norms[48:], "non_mlp": norms[:48]}
            require(set(record.get("families", {})) == set(families), "log-weight family coverage differs")
            for family, columns in families.items():
                saved = record["families"][family]
                require(saved.get("n") == len(columns), "log-weight family count differs")
                expected = {"mean": float(columns.mean()), "median": float(np.median(columns)),
                            "min": float(columns.min()), "max": float(columns.max()),
                            "sum_squared_norms": float(columns @ columns)}
                require(all(isinstance(saved.get(k), (int, float)) and np.isfinite(saved[k]) and
                            np.isclose(saved[k], v, rtol=1e-12, atol=1e-14) for k, v in expected.items()),
                        "log-weight family summaries differ from columns")
        diagnostics[split], receipts[split] = value, receipt_pin
    require(diagnostics["train"] == diagnostics["val"] and
            plan["caches"]["train"]["checkpoint_sha256"] == plan["caches"]["val"]["checkpoint_sha256"],
            "train/val log-weight endpoint diagnostics differ")
    return {"checkpoint_step": plan["step"], "checkpoint_sha256": plan["caches"]["train"]["checkpoint_sha256"],
            "source_receipts": receipts, **copy.deepcopy(diagnostics["train"]), "descriptive_only": True}


def validate_inputs(plan, train_only):
    splits = ("train",) if train_only else ("train", "val")
    refs = {s: Q.validate_reference(plan["references"][s]) for s in splits}
    require(refs["train"]["users"] == sorted(sum(plan["fold_users"], [])), "train user folds changed")
    if not train_only:
        require(set(refs["train"]["users"]).isdisjoint(refs["val"]["users"]), "train/val leakage")
    states, inspections = {}, {}
    for split in splits:
        states.update(refs[split]["states"])
        result = validate_artifact(plan["caches"][split], split, plan, refs[split])
        states.update(result["states"])
        inspections[ARM+"/"+split] = result
    return states, inspections


def validate_cell(cell, arm, step):
    require(cell.get("complete") is True and cell.get("arm") == arm and cell.get("checkpoint_step") == step and
            cell.get("which") == "teacher", "incomplete/mixed-dose cell")
    require(cell.get("fit", {}).get("selection", {}).get("boundary_unresolved") is False, "unresolved ridge selection")


def paired_comparison(cells, step, n_boot=Q.N_BOOT, seed=Q.SEED):
    """Same original per-user draws and ratios, with three prespecified LQ contrasts."""
    require(set(cells) == set(ARMS), "need exactly four cells; no intersection comparison")
    reference = cells[CONTROLS[0]]["analysis"]
    for arm, cell in cells.items():
        validate_cell(cell, arm, step)
        require(cell["analysis"]["support_sha256"] == reference["support_sha256"] and
                cell["analysis"]["targets_sha256"] == reference["targets_sha256"], "paired target/support differs")
    output = {}
    for group in ("dynamic", "all", "static"):
        rows = {a: cells[a]["analysis"]["groups"][group]["centered"]["per_user"] for a in ARMS}
        users = sorted(rows[CONTROLS[0]])
        require(len(users) >= 2 and all(sorted(r) == users for r in rows.values()), "paired users missing")
        base = rows[CONTROLS[0]]
        for arm in ARMS:
            require(cells[arm]["analysis"]["groups"][group]["n_selected_windows"] == reference["groups"][group]["n_selected_windows"], "paired selected windows differ")
            for user in users:
                require(all(rows[arm][user][k] == base[user][k] for k in ("n_windows", "n_window_joints", "n_valid_elements")), "paired support counts differ")
                require(rows[arm][user]["sums"]["vary"] == base[user]["sums"]["vary"], "paired denominator differs")
        denominator = np.asarray([base[u]["sums"]["vary"] for u in users])
        draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
        den = denominator[draws].sum(1)
        require(np.isfinite(den).all() and (den > 0).all(), "invalid bootstrap denominator; no replicas discarded")
        errors = {a: np.asarray([rows[a][u]["sums"]["centered"] for u in users]) for a in ARMS}
        result = {"users": users, "n_bootstrap": n_boot, "seed": seed, "draws_sha256": Q.arrays_hash(draws), "contrasts": {}, "cell_ratio_ci95": {}}
        for arm in ARMS:
            ratios = errors[arm][draws].sum(1)/den
            require(np.isfinite(ratios).all(), "nonfinite bootstrap ratio")
            result["cell_ratio_ci95"][arm] = np.percentile(ratios, [2.5, 97.5]).tolist()
        for control in CONTROLS:
            numerator = errors[ARM]-errors[control]
            values = numerator[draws].sum(1)/den
            require(np.isfinite(values).all(), "nonfinite bootstrap contrast")
            ci = np.percentile(values, [2.5, 97.5]).tolist()
            relative = float(-numerator.sum()/errors[control].sum()) if errors[control].sum() > 0 else None
            result["contrasts"][ARM+"_minus_"+control] = {
                "candidate": ARM, "control": control, "role": "primary" if control == "T100-L" else "secondary",
                "delta_ratio": float(numerator.sum()/denominator.sum()), "ci95": ci,
                "relative_error_reduction": relative,
                "signal_2percent_and_ci_below_zero": relative is not None and relative >= .02 and ci[1] < 0,
                "per_user": {u: {"delta_centered_sse": float(v), "target_variance_sum": float(d)} for u, v, d in zip(users, numerator, denominator)}}
        output[group] = result
    return output


def validate_fit(cell, plan, arm, pin, inspections):
    validate_cell(cell, arm, plan["step"])
    require(cell.get("checkpoint_sha256") == pin["train"]["checkpoint_sha256"], "cell checkpoint mismatch")
    selection = cell["fit"]["selection"]
    require(selection["fold_users"] == plan["fold_users"], "cell folds differ")
    require(selection["alpha"] == min(selection["grid"], key=lambda x: (x["mse"], x["alpha"]))["alpha"] and
            all(0 <= v <= 3 for v in selection["extensions"].values()), "invalid alpha selection")
    train, val = inspections[arm+"/train"], inspections[arm+"/val"]
    require(train["users"] == sorted(sum(plan["fold_users"], [])) and set(train["users"]).isdisjoint(val["users"]), "cell train/val users differ")
    require(cell["fit"]["fit_support_sha256"] == train["targets"]["central_mask"] and
            cell["analysis"]["support_sha256"] == val["targets"]["central_mask"] and
            cell["analysis"]["targets_sha256"] == val["targets"], "cell target/support signature mismatch")
    for split in ("train", "val"):
        require(inspections[arm+"/"+split]["cache_sha256"] == pin[split]["sha256"], "cell cache inspection pin differs")


def load_historical(plan, states):
    history = plan["historical"]
    original = pinned_json(history["plan"], states)
    report = pinned_json(history["report"], states)
    require(report.get("protocol") == Q.PROTOCOL and report.get("complete") is True and report.get("status") == "completed", "incomplete historical report")
    require(report.get("step") == plan["step"] and report.get("recipe") == Q.recipe() and report.get("plan_sha256") == history["plan"]["sha256"], "historical dose/recipe/plan mismatch")
    require(report.get("input_pins") == original["caches"] and report.get("reference_pins") == plan["references"], "historical input provenance differs")
    require(report["runtime"]["code_sha"] == history["code_sha"] and report["runtime"]["sources"] == original["sources"] == Q.source_hashes(), "historical source provenance differs")
    require(numeric_context(report["runtime"]) == history["numeric_context"], "historical numeric context differs")
    require(set(report.get("cells", {})) == set(Q.ARMS), "historical cell grid incomplete")
    for arm in Q.ARMS:
        validate_fit(report["cells"][arm], original, arm, original["caches"][arm], report["inspections"])
    require(report["paired"] == Q.paired_comparison(report["cells"], plan["step"]), "historical paired analysis changed")
    return report


def verify_scientific_artifacts(cell, targets, states):
    for key in ("coefficients", "predictions"):
        pin = cell.get(key)
        require(isinstance(pin, dict), "missing scientific artifact")
        states.update(Q.verify_file(resolve_pin(pin), pin["sha256"]))
    with np.load(resolve_pin(cell["coefficients"]), allow_pickle=False) as z:
        weights, bias = z["weights"], z["bias"]
    require(weights.shape == (384, 20) and bias.shape == (20,) and np.isfinite(weights).all() and np.isfinite(bias).all(), "invalid ridge coefficients")
    require(Q.arrays_hash(weights, bias) == cell["fit"]["coefficients_sha256"], "coefficient array digest differs")
    prediction = np.load(resolve_pin(cell["predictions"]), allow_pickle=False, mmap_mode="r")
    require(prediction.dtype == np.float64 and np.isfinite(prediction).all(), "invalid prediction precision/values")
    require(Q.summarize(targets, prediction) == cell["analysis"], "prediction/support/target statistics differ")


def compare(plan, candidate_pin, states):
    historical = load_historical(plan, states)
    candidate = pinned_json(candidate_pin, states)
    require(candidate.get("protocol") == PROTOCOL and candidate.get("complete") is True and candidate.get("status") == "completed" and
            candidate.get("step") == plan["step"] and candidate.get("recipe") == Q.recipe(), "incomplete candidate report")
    require(candidate.get("evaluation_sha256") == evaluation_identity(plan) and candidate.get("input_pins") == plan["caches"] and
            candidate.get("reference_pins") == plan["references"], "candidate plan/cache identity differs")
    require(candidate["runtime"]["sources"] == source_hashes(), "candidate report source differs")
    require(numeric_context(candidate["runtime"]) == plan["historical"]["numeric_context"], "candidate/historical numeric context differs")
    require(set(candidate.get("cells", {})) == {ARM}, "candidate report must contain only LQ")
    validate_fit(candidate["cells"][ARM], plan, ARM, plan["caches"], candidate["inspections"])
    diagnostics = load_log_weight_diagnostics(plan, states)
    require(candidate.get("log_weight_diagnostics") == diagnostics, "candidate log-weight readout differs from extraction receipt")
    reference = Q.validate_reference(plan["references"]["val"])
    states.update(reference["states"])
    targets = Q.read_targets(plan["references"]["val"]["path"])
    cells = {**copy.deepcopy(historical["cells"]), **copy.deepcopy(candidate["cells"])}
    for arm, cell in cells.items():
        require(cell["analysis"]["targets_sha256"] == reference["targets"], "imported cell target reference differs")
        verify_scientific_artifacts(cell, targets, states)
    return {"status": "compared", "complete": True, "cells": cells, "paired": paired_comparison(cells, plan["step"]),
            "log_weight_diagnostics": diagnostics,
            "provenance": {"historical": {"report": plan["historical"]["report"], "plan": plan["historical"]["plan"], "runtime": historical["runtime"], "input_pins": historical["input_pins"]},
                           "candidate": {"report": candidate_pin, "plan_sha256": candidate["plan_sha256"], "runtime": candidate["runtime"], "input_pins": candidate["input_pins"]}}}


def required_ridge_seconds(validate_elapsed, statistics, eigensystems):
    require(all(isinstance(v, (int, float)) and np.isfinite(v) and v > 0 for v in (validate_elapsed, statistics, eigensystems)), "invalid timing observations")
    # Original historical proxy, with exactly one candidate instead of 3.
    return int(math.ceil((1.5*(2*validate_elapsed+11*statistics+13*eigensystems)+120)/60)*60)


def fit_gate(plan, states, identity):
    release = plan.get("fit_release")
    require(isinstance(release, dict) and release.get("status") == "BENCHMARK_READ_AND_ACCEPTED", "candidate benchmark not read/released")
    reports = {s: pinned_json(release[s]["report"], states) for s in ("validate", "benchmark")}
    for stage, report in reports.items():
        require(report.get("complete") is True and report.get("protocol") == PROTOCOL and report.get("stage") == stage and
                report.get("evaluation_sha256") == evaluation_identity(plan), "benchmark/validation identity differs")
        require(report.get("status") == ("validated" if stage == "validate" else "train_only_timing"), "benchmark/validation incomplete")
        require(report["runtime"]["sources"] == identity["sources"], "benchmark source mismatch")
        require(report.get("step") == plan["step"] and report.get("recipe") == Q.recipe() and
                report.get("input_pins") == plan["caches"] and report.get("reference_pins") == plan["references"], "benchmark input/recipe mismatch")
        acct = release[stage]["accounting"]
        require(acct.get("State") == "COMPLETED" and acct.get("ExitCode") == "0:0" and str(acct.get("JobIDRaw")) == report.get("slurm_job_id"), "benchmark/validation job not completed")
        require(type(acct.get("ElapsedRaw")) is int and 0 < acct["ElapsedRaw"] <= CAPS[stage]["seconds"] and acct.get("AllocCPUS") == CAPS[stage]["cpus"], "benchmark/validation accounting outside cap")
    bench = reports["benchmark"]
    require(numeric_context(bench["runtime"]) == numeric_context(identity) == plan["historical"]["numeric_context"], "fit numeric context differs")
    seconds = required_ridge_seconds(release["validate"]["accounting"]["ElapsedRaw"], bench["timings"]["statistics_seconds"], bench["timings"]["three_eigensystems_seconds"])
    require(release.get("required_seconds") == seconds and seconds <= CAPS["ridge"]["seconds"], "candidate ridge exceeds timing ceiling")
    return {"required_seconds": seconds, "maximum_seconds": CAPS["ridge"]["seconds"], "release": release}


def run_candidate(plan, out):
    train = Q.read_data(plan["caches"]["train"])
    weights, bias, fit = Q.fit_reader(train, plan["fold_users"])
    train_users = {m["user"] for m in train["meta"]}
    del train
    # No validation payload is opened by the fit/alpha selection itself.
    val = Q.read_data(plan["caches"]["val"])
    require(train_users.isdisjoint({m["user"] for m in val["meta"]}), "train/val user leakage")
    start = perf_counter()
    prediction = Q.predict(val["lat"], val["lengths"], weights, bias)
    seconds = perf_counter()-start
    coeff, pred = out/(ARM+".coefficients.npz"), out/(ARM+".predictions.npy")
    with coeff.open("xb") as stream:
        np.savez_compressed(stream, weights=weights, bias=bias)
    with pred.open("xb") as stream:
        np.save(stream, prediction, allow_pickle=False)
    cell = {"complete": True, "arm": ARM, "which": "teacher", "checkpoint_step": plan["step"],
            "checkpoint_sha256": plan["caches"]["train"]["checkpoint_sha256"], "fit": fit, "prediction_seconds": seconds,
            "analysis": Q.summarize(val, prediction), "coefficients": {"path": str(coeff), "sha256": Q.D.file_sha256(coeff)},
            "predictions": {"path": str(pred), "sha256": Q.D.file_sha256(pred)}}
    return {"status": "completed", "complete": True, "cells": {ARM: cell}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=CAPS)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--candidate-report-sha256")
    args = parser.parse_args(argv)
    Q.D.fresh_destination(args.out)
    states = Q.verify_file(args.plan, args.expected_plan_sha256)
    plan_bytes = args.plan.read_bytes()
    plan = Q.read_json(args.plan)
    validate_plan(plan, runtime=True)
    identity = runtime_identity(args.expected_code_sha)
    require((args.candidate_report is not None) == (args.stage == "compare") and
            (args.candidate_report_sha256 is not None) == (args.stage == "compare"), "candidate report belongs only to compare")
    inspections = {}
    if args.stage == "compare":
        result = compare(plan, {"path": str(args.candidate_report), "sha256": args.candidate_report_sha256}, states)
    else:
        if args.stage in ("ridge", "benchmark"):
            require(numeric_context(identity) == plan["historical"]["numeric_context"], "numerical environment differs from historical fit")
        gate = fit_gate(plan, states, identity) if args.stage == "ridge" else None
        observed, inspections = validate_inputs(plan, train_only=args.stage == "benchmark")
        states.update(observed)
        if args.stage == "validate":
            load_historical(plan, states)
            result = {"status": "validated", "complete": True, "scientific_metrics_computed": False}
        elif args.stage == "benchmark":
            result = {"status": "train_only_timing", "complete": True, "arm": ARM,
                      "timings": Q.benchmark_train(Q.read_data(plan["caches"]["train"]))}
        else:
            diagnostics = load_log_weight_diagnostics(plan, states)
            args.out.mkdir(parents=True, exist_ok=False)
            result = run_candidate(plan, args.out)
            result["log_weight_diagnostics"] = diagnostics
            result["fit_gate"] = gate
    require(args.plan.read_bytes() == plan_bytes and runtime_identity(args.expected_code_sha) == identity, "plan/source/runtime changed")
    require(states == {p: Q.T.file_state(p) for p in states}, "input artifact changed during execution")
    result.update(protocol=PROTOCOL, stage=args.stage, step=plan["step"], recipe=Q.recipe(), runtime=identity,
                  plan_sha256=args.expected_plan_sha256, evaluation_sha256=evaluation_identity(plan),
                  input_pins=plan["caches"], reference_pins=plan["references"], inspections=inspections,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  scope="Single seed and reused development validation. No between-training uncertainty or general transfer claim.")
    if args.stage != "ridge":
        args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/"report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
