"""Candidate-only T100-LQ pool768 probes with original historical provenance.

Numerical fits and metrics are imported from the frozen tokenizer reader.
No submissions, model training, recipe search extension or historical refit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tokenizer_probe as T

PROTOCOL = "logsquare-probe-20261104-v1"
CANDIDATE = "T100-LQ"
HISTORICAL_ARMS = T.ARMS
ARMS = (*HISTORICAL_ARMS, CANDIDATE)
TASKS, DATASETS, WEIGHTS = T.TASKS, T.DATASETS, T.WEIGHTS
C_GRID, MAX_ITER, SEED, N_BOOT = T.C_GRID, T.MAX_ITER, T.SEED, T.N_BOOT
CP, P1, DA, CR = T.CP, T.P1, T.DA, T.CR
require, sha, digest, jhash, write_json, fresh = T.require, T.sha, T.digest, T.jhash, T.write_json, T.fresh
recipe, fit_fold, AuditLog, audit_summary = T.recipe, T.fit_fold, T.AuditLog, T.audit_summary
numeric_context, validate_oof, scored = T.numeric_context, T.validate_oof, T.scored
CONTRASTS = {"T100-LQ_minus_T100-L": (0., -1., 0., 1.),
             "T100-LQ_minus_T25-L": (-1., 0., 0., 1.),
             "T100-LQ_minus_T100-G": (0., 0., -1., 1.)}
SHARDS = {"db2-01": ("p1", ["ninapro_db2"], [0, 1]),
          "db2-23": ("p1", ["ninapro_db2"], [2, 3]),
          "db3": ("p1", ["ninapro_db3"], [0, 1, 2, 3]),
          "db6": ("p1", ["ninapro_db6"], [0, 1, 2, 3]),
          "p2": ("p2", ["epn612", "uci_emg"], [0, 1, 2, 3])}


def load_cache(plan, arm, ds, features=True):
    require(arm == CANDIDATE, "only candidate features may be loaded by LQ reader")
    pin = plan["caches"][arm][ds]
    parsed = []
    for path_key, hash_key in (("receipt_path", "receipt_sha256"), ("provenance_path", "provenance_sha256")):
        require(sha(pin[path_key]) == pin[hash_key], "LQ provenance hash mismatch")
        obj = json.loads(Path(pin[path_key]).read_text())
        require(obj.get("protocol") == "tokenizer-logsquare-evaluation-20261104-v1", "wrong LQ extraction protocol")
        cfg = obj.get("model_config", {})
        expected = {"tokenizer_logsquare": True, "tokenizer_logsquare_eps": .001,
                    "window_ms": 100., "patch_ms": 25., "tokenizer_activation": "identity",
                    "dim": 384, "k_latents": 64}
        require(isinstance(cfg, dict) and all(cfg.get(k) == v for k, v in expected.items()), "wrong LQ loaded model configuration")
        parsed.append(obj)
    require(parsed[0]["model_config"] == parsed[1]["model_config"], "LQ receipt/provenance model configurations differ")
    return T.load_cache(plan, arm, ds, features=features)


def source_hashes():
    return {**T.source_hashes(), str(Path(__file__).relative_to(ROOT)): sha(__file__)}


def pinned_json(pin):
    require(isinstance(pin, dict) and digest(pin.get("sha256")), "missing/invalid JSON pin")
    path = Path(pin.get("path", ""))
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and sha(path) == pin["sha256"], "pinned JSON missing/hash mismatch: " + str(path))
    return json.loads(path.read_text())


def historical_plan(plan):
    old = pinned_json(plan["historical_plan"])
    T.validate_plan(old)
    require(old["datasets"] == plan["datasets"] and old["recipe"] == recipe(), "historical dataset/recipe mismatch")
    require(old["sources"] == T.source_hashes(), "historical numerical sources differ")
    return old


def validate_plan(plan):
    require(plan.get("protocol") == PROTOCOL and plan.get("step") == 19618, "wrong protocol/endpoint")
    require(plan.get("status") == "APPROVED_FOR_PROBES", "probe plan not released")
    require(plan.get("recipe") == recipe() and plan.get("sources") == source_hashes(), "recipe/source pins differ")
    old = historical_plan(plan)
    require(set(plan.get("caches", {})) == {CANDIDATE}, "only candidate caches permitted")
    require(isinstance(plan["caches"][CANDIDATE], dict) and set(plan["caches"][CANDIDATE]) == set(DATASETS), "missing candidate cache pins")
    paths, checkpoints = [], set()
    for ds, pin in plan["caches"][CANDIDATE].items():
        require(isinstance(pin, dict), "missing candidate cache pin")
        require(all(Path(pin.get(k, "")).is_absolute() for k in ("path", "receipt_path", "provenance_path")), "absolute cache paths required")
        require(all(digest(pin.get(k)) for k in ("sha256", "receipt_sha256", "provenance_sha256", "checkpoint_sha256", "extraction_plan_sha256")), "invalid candidate cache pins")
        require(digest(pin.get("extraction_code_sha"), 40), "invalid extraction source SHA")
        require(all(pin.get(k) == old["datasets"][ds][k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")), "candidate dataset identity mismatch")
        checkpoints.add(pin["checkpoint_sha256"])
        paths.append(pin["path"])
    require(len(checkpoints) == 1 and len(set(paths)) == 5, "mixed candidate checkpoints or duplicate caches")
    require(not checkpoints.intersection({old["caches"][a][d]["checkpoint_sha256"] for a in HISTORICAL_ARMS for d in DATASETS}), "candidate reuses historical checkpoint")
    references = plan.get("historical_reports", {})
    require(set(references) == {a+"/"+t for a in HISTORICAL_ARMS for t in TASKS}, "need six original historical report pins")
    for key, pin in references.items():
        require(Path(pin.get("path", "")).is_absolute() and digest(pin.get("sha256")), "invalid historical report pin")
        require(digest(pin.get("code_sha"), 40) and digest(pin.get("plan_sha256")) and digest(pin.get("numeric_sha256")), "missing historical provenance")
        require(pin.get("plan_content_sha256") == jhash(old), "historical content pin mismatch")
    require(len({p["numeric_sha256"] for p in references.values()}) == 1, "historical numeric contexts differ")
    require(plan.get("expected_numeric_sha256") == next(iter(references.values()))["numeric_sha256"], "numeric context pin mismatch")
    policy = pinned_json(plan["resources"])
    require(policy.get("status") == "APPROVED_FOR_PROBES" and policy.get("method") == "historical_per_shard_gate_reuse", "CPU resource policy not released")
    resources = pinned_json(policy["historical_resources"])
    require(resources.get("all_resource_limits_passed") is True and resources.get("gpu_hours") == 0, "historical CPU resource gate failed")
    require(policy.get("recipe_sha256") == jhash(recipe()) and policy.get("numeric_sha256") == plan["expected_numeric_sha256"], "resource recipe/numeric mismatch")
    require(set(policy.get("jobs", {})) == set(SHARDS) | {"aggregate-p1", "aggregate-p2", "compare"}, "CPU shard policy incomplete")
    for name, (task, datasets, folds) in SHARDS.items():
        row = policy["jobs"][name]
        require(row["cpus"] == 8 and row["memory_gib"] == 30 and row["seconds_cap"] == (1800 if name in ("db3", "db6") else 3600), "unregistered CPU fit cap")
        evidence = [r for r in resources["jobs"] if r["stage"] == "fit-shard" and r["logical_id"].endswith("-"+name+"-20261104")]
        require(len(evidence) == 3 and all(r["state"] == "COMPLETED" and r["exit_code"] == "0:0" and r["cpus"] == 8 and 0 < r["elapsed_seconds"] <= row["seconds_cap"] and 0 < r["batch_max_rss_KiB"] <= 30*1024**2 for r in evidence), "historical per-shard CPU gate failed")
    for name in ("aggregate-p1", "aggregate-p2", "compare"):
        r = policy["jobs"][name]
        require(r == {"cpus": 2, "memory_gib": 30, "seconds_cap": 1800 if name == "compare" else 1200}, "unregistered CPU report cap")
    return policy


def fit_gate(plan, task, datasets, folds):
    require((task, datasets, folds) in SHARDS.values(), "unregistered candidate shard")
    require(jhash(numeric_context()) == plan["expected_numeric_sha256"], "candidate/historical numeric contexts differ")
    # Counts from the immutable historical reports, independent of candidate scores.
    for ds in datasets:
        pin = plan["historical_reports"]["T100-L/"+task]
        reference = pinned_json(pin)
        _, y, subjects = load_cache(plan, CANDIDATE, ds, features=False)
        roles = CP.fold_roles(plan["datasets"][ds]["subjects"])
        for fr in reference["datasets"][ds]["folds"]:
            f = fr["fold"]
            mt, mv, mh = (np.isin(subjects, roles[f][k]) for k in ("train", "val", "held"))
            kv, kh = P1.keep_train_classes(y[mt], y[mv], y[mh])
            require((int(mt.sum()), int(kv.sum()), int(kh.sum())) == (fr["n_train"], fr["n_val"], fr["n_held"]), "candidate cache shape differs from CPU gate")

def run_shard(plan, arm, task, datasets, folds, out, identity):
    validate_plan(plan)
    require(arm == CANDIDATE and task in TASKS and len(set(datasets)) == len(datasets) and datasets
            and set(datasets) <= set(TASKS[task]), "invalid shard arm/task/datasets")
    require(folds and folds == sorted(set(folds)) and all(type(f) is int and f in range(4) for f in folds), "invalid shard folds")
    fit_gate(plan, task, datasets, folds)
    out = Path(out)
    fresh(out)
    out.mkdir(parents=True)
    audit, arrays, reports = AuditLog(out / "audit_progress.jsonl"), {}, {}
    numeric = numeric_context()
    try:
        for ds in datasets:
            x, y, subjects = load_cache(plan, arm, ds)
            roles = CP.fold_roles(plan["datasets"][ds]["subjects"])
            results = [fit_fold(x, y, subjects, roles, f, audit, {"arm": arm, "dataset": ds, "task": task,
                       "checkpoint_sha256": plan["caches"][arm][ds]["checkpoint_sha256"],
                       "cache_sha256": plan["caches"][arm][ds]["sha256"], **identity}) for f in folds]
            for key in ("row", "y", "pred", "s", "fold"):
                arrays[f"{ds}_{key}"] = np.concatenate([r[key] for r in results])
            reports[ds] = {"folds": [r["report"] for r in results]}
            del x
            load_cache(plan, arm, ds, features=False)  # full hashes, after fitting
        require(numeric_context() == numeric, "numeric runtime changed during fits")
        summary = audit_summary(audit)
        require(len(audit) == 7*len(folds)*len(datasets), "wrong fit count")
        completion_identity(plan, identity)
        np.savez(out / "shard.oof.npz", **arrays)
        with (out / "audit.jsonl").open("x") as f:
            for record in audit:
                f.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
        result = {"protocol": PROTOCOL, "kind": "shard", "complete": True, "arm": arm, "task": task,
                  "step": plan["step"], "recipe": recipe(), "plan_content_sha256": jhash(plan), **identity,
                  "numeric_context": numeric, "numeric_sha256": jhash(numeric), "datasets": reports,
                  "audit_summary": summary, "audit": str(out / "audit.jsonl"), "audit_sha256": sha(out / "audit.jsonl"),
                  "oof": str(out / "shard.oof.npz"), "oof_sha256": sha(out / "shard.oof.npz")}
        write_json(out / "shard.json", result)
        return result
    except BaseException as exc:
        write_json(out / "failed.json", {"complete": False, "exception": f"{type(exc).__name__}: {exc}", "completed_datasets": list(reports), **identity})
        raise


def read_completed(path, kind, plan):
    r = json.loads(Path(path).read_text())
    require(r.get("complete") is True and r.get("kind") == kind and r.get("protocol") == PROTOCOL, "incomplete/wrong report")
    require(r.get("plan_content_sha256") == jhash(plan) and r.get("recipe") == recipe() and r.get("step") == plan["step"], "report recipe/plan mismatch")
    require(sha(r["oof"]) == r["oof_sha256"] and sha(r["audit"]) == r["audit_sha256"], "report output hash mismatch")
    require(jhash(r["numeric_context"]) == r["numeric_sha256"] == plan["expected_numeric_sha256"], "numeric signature mismatch")
    records = [json.loads(line) for line in Path(r["audit"]).read_text().splitlines()]
    require(audit_summary(records) == r["audit_summary"], "audit summary mismatch")
    with np.load(r["oof"], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return r, arrays, records


def completion_identity(plan, identity):
    """Before publishing complete outputs, recheck source and (for CLI) plan/code."""
    validate_plan(plan)
    if "plan_path" in identity:
        require(sha(identity["plan_path"]) == identity["plan_sha256"], "plan file changed during execution")
        code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        require(code == identity["code_sha"], "code HEAD changed during execution")
        require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(), "tracked checkout changed during execution")


def aggregate(plan, arm, task, shards, out, identity):
    validate_plan(plan)
    require(arm == CANDIDATE and task in TASKS and len(set(map(str, shards))) == len(shards) and shards, "invalid aggregation")
    out = Path(out)
    for p in (out, out.with_suffix(".oof.npz"), out.with_suffix(".audit.jsonl")):
        fresh(p)
    loaded = [read_completed(p, "shard", plan) for p in shards]
    require(all(r["arm"] == arm and r["task"] == task and r["code_sha"] == identity["code_sha"] for r, _, _ in loaded), "shard identity mismatch")
    require(len({r["numeric_sha256"] for r, _, _ in loaded}) == 1, "mixed numeric fit contexts")
    records = sum((audit for _, _, audit in loaded), [])
    dataset_reports, arrays = {}, {}
    for ds in TASKS[task]:
        cells = {}
        for report, values, audit in loaded:
            require(set(report["datasets"]) <= set(TASKS[task]), "foreign dataset in shard")
            if ds not in report["datasets"]:
                continue
            row, y, pred, subjects, fold = (values[f"{ds}_{k}"] for k in ("row", "y", "pred", "s", "fold"))
            validate_oof(row, y, pred, subjects, fold)
            for fr in report["datasets"][ds]["folds"]:
                f = fr["fold"]
                require(f in range(4) and f not in cells, "duplicate fold")
                pick = fold == f
                require(pick.any(), "missing fold predictions")
                cells[f] = (fr, {k: values[f"{ds}_{k}"][pick] for k in ("row", "y", "pred", "s", "fold")})
            require(set(fold.tolist()) == {fr["fold"] for fr in report["datasets"][ds]["folds"]}, "unreported OOF fold")
        require(set(cells) == set(range(4)), "missing dataset/fold")
        _, original_y, original_s = load_cache(plan, arm, ds, features=False)
        roles = CP.fold_roles(plan["datasets"][ds]["subjects"])
        for f, (fr, a) in sorted(cells.items()):
            mt, mv, mh = (np.isin(original_s, roles[f][k]) for k in ("train", "val", "held"))
            kv, kh = P1.keep_train_classes(original_y[mt], original_y[mv], original_y[mh])
            expected_rows = np.flatnonzero(mh)[kh]
            order = np.argsort(a["row"])
            require(np.array_equal(a["row"][order], expected_rows)
                    and np.array_equal(a["y"][order], original_y[expected_rows])
                    and np.array_equal(a["s"][order].astype(str), original_s[expected_rows]), "OOF rows/targets/subjects do not match fold")
            require(all(fr[k] == roles[f][k] for k in roles[f]) and fr["roles_sha256"] == jhash(roles[f]), "reported roles differ")
            require(fr["n_train"] == int(mt.sum()) and fr["n_val"] == int(kv.sum()) and fr["n_held"] == len(expected_rows), "fold counts mismatch")
            fits = [r for r in records if r["dataset"] == ds and r["fold"] == f]
            require(len(fits) == 7 and sorted(r["C"] for r in fits if r["fit_phase"] == "candidate") == list(C_GRID)
                    and len([r for r in fits if r["fit_phase"] == "refit"]) == 1, "audit candidate/refit coverage differs")
            require(all(r["max_iter"] == MAX_ITER and r["arm"] == arm and r["roles_sha256"] == jhash(roles[f]) for r in fits), "fit audit identity differs")
            for r in fits:
                expected_solver = LogisticRegression(C=r["C"], max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4).get_params()
                require(r["solver_params"] == expected_solver and r["n_features"] == 768 and r["dtype"] == "float32"
                        and r["scaler_sha256"] == fr["scaler_sha256"] and r["n_train"] == fr["n_train"]
                        and r["checkpoint_sha256"] == plan["caches"][arm][ds]["checkpoint_sha256"]
                        and r["cache_sha256"] == plan["caches"][arm][ds]["sha256"] and r["task"] == task
                        and r["code_sha"] == identity["code_sha"] and r["plan_sha256"] == identity["plan_sha256"], "solver/feature identity differs")
            candidates = {r["C"]: r for r in fits if r["fit_phase"] == "candidate"}
            chosen = max(C_GRID, key=lambda c: (candidates[c]["val_bacc"], -c))
            require(fr["c"] == chosen and fr["val_bacc"] == candidates[chosen]["val_bacc"]
                    and fr["val_scores"] == {str(c): candidates[c]["val_bacc"] for c in C_GRID}
                    and all(r["selected"] == (r["C"] == chosen) for r in candidates.values())
                    and next(r["C"] for r in fits if r["fit_phase"] == "refit") == chosen, "C selection/refit differs")
        joined = {k: np.concatenate([cells[f][1][k] for f in range(4)]) for k in ("row", "y", "pred", "s", "fold")}
        validate_oof(*(joined[k] for k in ("row", "y", "pred", "s", "fold")))
        arrays.update({f"{ds}_{k}": v for k, v in joined.items()})
        score, metric_warnings = scored(joined["y"], joined["pred"])
        dataset_reports[ds] = {"balanced_accuracy": score, "n_oof_windows": len(joined["row"]),
                              "n_oof_subjects": len(set(joined["s"].tolist())), "folds": [cells[f][0] for f in range(4)],
                              "identity_sha256": DA.fingerprint(*(joined[k] for k in ("row", "y", "s", "fold"))),
                              "scoring_warnings": metric_warnings}
    require(len(records) == 28*len(TASKS[task]), "extra audit fits")
    completion_identity(plan, identity)
    for shard in shards:
        read_completed(shard, "shard", plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".oof.npz"), **arrays)
    with out.with_suffix(".audit.jsonl").open("x") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
    result = {"protocol": PROTOCOL, "kind": "aggregate", "complete": True, "arm": arm, "task": task, "step": plan["step"],
              "plan_content_sha256": jhash(plan), "recipe": recipe(), **identity,
              "numeric_context": loaded[0][0]["numeric_context"], "numeric_sha256": loaded[0][0]["numeric_sha256"],
              "datasets": dataset_reports, "weights": dict(zip(TASKS[task], WEIGHTS[task])),
              "balanced_accuracy": float(np.dot(WEIGHTS[task], [dataset_reports[d]["balanced_accuracy"] for d in TASKS[task]])),
              "audit_summary": audit_summary(records), "audit": str(out.with_suffix(".audit.jsonl")), "audit_sha256": sha(out.with_suffix(".audit.jsonl")),
              "oof": str(out.with_suffix(".oof.npz")), "oof_sha256": sha(out.with_suffix(".oof.npz")),
              "shards": [{"path": str(p), "sha256": sha(p)} for p in shards]}
    write_json(out, result)
    return result


def _summary(points, draws, noninferiority=False):
    out = {"cells": {arm: {"balanced_accuracy": float(points[i]), "ci95_percent": (100*np.quantile(draws[:, i], [.025, .975])).tolist()}
                     for i, arm in enumerate(ARMS)}, "contrasts": {}}
    for name, coefficients in CONTRASTS.items():
        co = np.asarray(coefficients)
        ci = (100*np.quantile(draws @ co, [.025, .975])).tolist()
        out["contrasts"][name] = {"delta_pp": float(100*(points @ co)), "ci95_pp": ci}
        if noninferiority:
            out["contrasts"][name]["noninferiority_1pp"] = "supported" if ci[0] > -1 else "loss_beyond_margin" if ci[1] < -1 else "unresolved"
    return out


def compare_arrays(cells, n_boot=N_BOOT, seed=SEED):
    """Pure paired comparison. Each cell maps dataset to row/y/pred/s/fold arrays."""
    require(set(cells) == set(ARMS) and all(set(c) == set(DATASETS) for c in cells.values()), "need complete four-arm/five-dataset grid")
    require(type(n_boot) is int and n_boot >= 2, "invalid bootstrap count")
    rng, datasets, points, samples = np.random.default_rng(seed), {}, {}, {}
    for ds in sorted(DATASETS):
        aligned = []
        for arm in ARMS:
            a = cells[arm][ds]
            validate_oof(*(a[k] for k in ("row", "y", "pred", "s", "fold")))
            ix = np.argsort(a["row"])
            aligned.append({k: a[k][ix] for k in ("row", "y", "pred", "s", "fold")})
        base = aligned[0]
        require(all(all(np.array_equal(a[k], base[k]) for k in ("row", "y", "s", "fold")) for a in aligned), "OOF pairing differs")
        stats = [DA.subject_counts(a["y"], a["pred"], a["s"]) for a in aligned]
        users, denominator, _ = DA._arrays(stats[0])
        counts = [DA._arrays(s)[2] for s in stats]
        draw = rng.integers(0, len(users), size=(n_boot, len(users)))
        den = denominator[draw].sum(1)
        present = den > 0
        classes_present = present.sum(1)
        require((classes_present > 0).all(), "empty bootstrap targets")
        values = np.column_stack([np.divide(hits[draw].sum(1), den, out=np.zeros_like(den), where=present).sum(1)/classes_present for hits in counts])
        total = denominator.sum(0)
        point = np.asarray([np.mean(h.sum(0)[total > 0]/total[total > 0]) for h in counts])
        require(np.isfinite(values).all(), "invalid bootstrap metric")
        points[ds], samples[ds] = point, values
        datasets[ds] = {**_summary(point, values), "subjects": users, "classes": stats[0]["classes"],
                        "draws_sha256": DA.fingerprint(draw), "paired_counts": dict(zip(ARMS, stats)),
                        "replicates_missing_classes": int((classes_present < len(total)).sum()),
                        "single_class_replicates": int((classes_present == 1).sum()),
                        "missing_class_rule": "macro recall over target classes present in each draw; all draws retained",
                        "role": "primary_classification" if ds == "epn612" else "secondary_dataset"}
    tasks = {task: _summary(np.sum([w*points[d] for w, d in zip(WEIGHTS[task], TASKS[task])], axis=0),
                           np.sum([w*samples[d] for w, d in zip(WEIGHTS[task], TASKS[task])], axis=0), noninferiority=True) for task in TASKS}
    return {"datasets": datasets, "tasks": tasks, "n_bootstrap": n_boot, "seed": seed,
            "uncertainty": "paired subject bootstrap conditional on one pretraining seed, fixed checkpoints and fitted probes; development validation; no multiplicity adjustment"}


def apply_solver_reading(result):
    """Unresolved fits retain descriptive estimates, but cannot yield NI decisions."""
    converged = all(s["converged_by_criterion"] for s in result["fit_audit"].values())
    result["reading_status"] = "solver_converged_by_criterion" if converged else "solver_unresolved"
    result["decision_valid"] = bool(converged)
    result["decision_scope"] = "conditional descriptive criteria only; no automatic FM adoption or continuation"
    if not converged:
        for group in ("datasets", "tasks"):
            for block in result[group].values():
                for contrast in block["contrasts"].values():
                    if "noninferiority_1pp" in contrast:
                        contrast["noninferiority_1pp"] = "unresolved_solver"
        result["decision_scope"] = "numerical estimates retained; NI/promotion interpretation withheld because configured solver convergence is unresolved; no automatic retry or candidate exclusion"
    return result


def verify_aggregate(report, values, records, source_plan):
    """Recompute OOF scores and audit identity without relabelling provenance."""
    arm, task = report["arm"], report["task"]
    require(task in TASKS and set(report["datasets"]) == set(TASKS[task]), "aggregate dataset grid mismatch")
    require(report["weights"] == dict(zip(TASKS[task], WEIGHTS[task])), "aggregate weights differ")
    require(len(records) == 28*len(TASKS[task]), "extra/missing aggregate audit fits")
    cells = {}
    for ds in TASKS[task]:
        a = {k: values[f"{ds}_{k}"] for k in ("row", "y", "pred", "s", "fold")}
        validate_oof(*(a[k] for k in ("row", "y", "pred", "s", "fold")))
        frs = report["datasets"][ds]["folds"]
        require(len(frs) == 4 and {f["fold"] for f in frs} == set(range(4)), "aggregate missing/duplicate fold")
        roles = CP.fold_roles(source_plan["datasets"][ds]["subjects"])
        for fr in frs:
            f = fr["fold"]
            require(all(fr[k] == roles[f][k] for k in roles[f]) and fr["roles_sha256"] == jhash(roles[f]), "aggregate fold roles mismatch")
            selected = a["fold"] == f
            require(int(selected.sum()) == fr["n_held"] and set(a["s"][selected]) <= set(roles[f]["held"]), "OOF held coverage mismatch")
            score, _ = scored(a["y"][selected], a["pred"][selected])
            require(score == fr["held_bacc"], "held score mismatch")
            fits = [r for r in records if r["dataset"] == ds and r["fold"] == f]
            require(len(fits) == 7 and sorted(r["C"] for r in fits if r["fit_phase"] == "candidate") == list(C_GRID)
                    and len([r for r in fits if r["fit_phase"] == "refit"]) == 1, "audit grid/refit mismatch")
            for r in fits:
                expected = LogisticRegression(C=r["C"], max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4).get_params()
                require(r["solver_params"] == expected and r["max_iter"] == MAX_ITER and r["n_features"] == 768 and r["dtype"] == "float32", "audit solver recipe mismatch")
                require(r["arm"] == arm and r["task"] == task and r["roles_sha256"] == fr["roles_sha256"]
                        and r["scaler_sha256"] == fr["scaler_sha256"] and r["n_train"] == fr["n_train"], "audit fold identity mismatch")
                require(r["code_sha"] == report["code_sha"] and r["plan_sha256"] == report["plan_sha256"], "audit source identity mismatch")
                require(r["checkpoint_sha256"] == source_plan["caches"][arm][ds]["checkpoint_sha256"]
                        and r["cache_sha256"] == source_plan["caches"][arm][ds]["sha256"], "audit cache identity mismatch")
            candidates = {r["C"]: r for r in fits if r["fit_phase"] == "candidate"}
            best = max(C_GRID, key=lambda c: (candidates[c]["val_bacc"], -c))
            require(fr["c"] == best and fr["val_bacc"] == candidates[best]["val_bacc"]
                    and fr["val_scores"] == {str(c): candidates[c]["val_bacc"] for c in C_GRID}
                    and all(r["selected"] == (r["C"] == best) for r in candidates.values())
                    and next(r["C"] for r in fits if r["fit_phase"] == "refit") == best, "aggregate C selection mismatch")
        score, _ = scored(a["y"], a["pred"])
        dr = report["datasets"][ds]
        require(score == dr["balanced_accuracy"] and len(a["row"]) == dr["n_oof_windows"]
                and len(set(a["s"])) == dr["n_oof_subjects"]
                and DA.fingerprint(*(a[k] for k in ("row", "y", "s", "fold"))) == dr["identity_sha256"], "aggregate OOF score/identity mismatch")
        cells[ds] = a
    require(report["balanced_accuracy"] == float(np.dot(WEIGHTS[task], [report["datasets"][d]["balanced_accuracy"] for d in TASKS[task]])), "aggregate score mismatch")
    return cells


def read_historical(plan, key):
    pin = plan["historical_reports"][key]
    require(sha(pin["path"]) == pin["sha256"], "historical report hash mismatch")
    original_plan = historical_plan(plan)
    r, arrays, records = T.read_completed(pin["path"], "aggregate", original_plan)
    require(key == r["arm"]+"/"+r["task"], "historical cell mismatch")
    for field in ("code_sha", "plan_sha256", "plan_path", "plan_content_sha256", "numeric_sha256"):
        require(r.get(field) == pin.get(field), "historical provenance mismatch: " + field)
    for shard in r["shards"]:
        require(sha(shard["path"]) == shard["sha256"], "historical shard hash mismatch")
    return r, arrays, records, original_plan


def compare(plan, paths, out, identity):
    validate_plan(plan)
    require(len(paths) == 2 and len(set(map(str, paths))) == 2, "need two distinct candidate aggregates")
    candidate = [read_completed(p, "aggregate", plan) for p in paths]
    require({(r["arm"], r["task"]) for r, _, _ in candidate} == {(CANDIDATE, t) for t in TASKS}, "candidate aggregate grid mismatch")
    require(all(r["code_sha"] == identity["code_sha"] and r["plan_sha256"] == identity["plan_sha256"] for r, _, _ in candidate), "candidate source identity mismatch")
    loaded = [(r, values, audit, plan) for r, values, audit in candidate]
    loaded += [read_historical(plan, key) for key in sorted(plan["historical_reports"])]
    require({r["numeric_sha256"] for r, _, _, _ in loaded} == {plan["expected_numeric_sha256"]}, "mixed numeric contexts")
    cells = {a: {} for a in ARMS}
    for report, values, records, original_plan in loaded:
        cells[report["arm"]].update(verify_aggregate(report, values, records, original_plan))
    result = {"protocol": PROTOCOL, "kind": "comparison", "complete": True, "step": plan["step"], "recipe": recipe(),
              "plan_content_sha256": jhash(plan), **identity, **compare_arrays(cells),
              "fit_audit": {r["arm"]+"/"+r["task"]: r["audit_summary"] for r, _, _, _ in loaded},
              "candidate_reports": [{"path": str(p), "sha256": sha(p)} for p in paths],
              "historical_reports": plan["historical_reports"],
              "original_provenance": {r["arm"]+"/"+r["task"]: {k: r[k] for k in ("protocol", "code_sha", "plan_sha256", "plan_content_sha256", "plan_path", "numeric_sha256", "oof_sha256", "audit_sha256")} for r, _, _, _ in loaded},
              "primary_contrast": "T100-LQ_minus_T100-L", "secondary_contrasts": ["T100-LQ_minus_T25-L", "T100-LQ_minus_T100-G"]}
    apply_solver_reading(result)
    completion_identity(plan, identity)
    # Re-read all hashes after computation before publication.
    for key in sorted(plan["historical_reports"]):
        read_historical(plan, key)
    for path in paths:
        read_completed(path, "aggregate", plan)
    fresh(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit-shard", "aggregate", "compare"))
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--expected-plan-sha256", required=True)
    p.add_argument("--expected-code-sha", required=True)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--shard", choices=SHARDS)
    p.add_argument("--shards", nargs="+", type=Path)
    p.add_argument("--reports", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    require(sha(a.plan) == a.expected_plan_sha256, "plan file hash mismatch")
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(code == a.expected_code_sha and digest(code, 40), "execution code SHA mismatch")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(), "tracked checkout is dirty")
    plan = json.loads(a.plan.read_text())
    identity = {"code_sha": code, "plan_sha256": a.expected_plan_sha256, "plan_path": str(a.plan.resolve())}
    if a.stage == "fit-shard":
        require(a.shard is not None, "fit-shard requires registered shard")
        task, datasets, folds = SHARDS[a.shard]
        result = run_shard(plan, CANDIDATE, task, datasets, folds, a.out, identity)
    elif a.stage == "aggregate":
        require(a.task and a.shards, "aggregate requires task/shards")
        result = aggregate(plan, CANDIDATE, a.task, a.shards, a.out, identity)
    else:
        require(a.reports, "compare requires two candidate reports")
        result = compare(plan, a.reports, a.out, identity)
    require(sha(a.plan) == a.expected_plan_sha256 and plan["sources"] == source_hashes(), "plan/source changed during execution")
    print(json.dumps({"complete": result["complete"], "kind": result["kind"], "out": str(a.out)}))


if __name__ == "__main__":
    main()
