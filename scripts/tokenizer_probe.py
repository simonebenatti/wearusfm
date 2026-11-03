"""Frozen teacher pool768 probes for the three tokenizer arms; never submits jobs.

Minimal audited-fit port from Claude a59079a44b482b85c9ffa26efb2b4851ac6e68a6:
src/wearusfm/harness/probe.py SHA256
f17d338db6fdf031160a122fca455b75bdfedae4af3935b5d31d8d1e40152659;
scripts/channel_readout.py SHA256
9d46e1a17c4eb682b788ec7ca52b694ffdbb0693abe6facf4c56056b9538c39b.
Preserves scaler, candidate order, validation selection and train-only refit.
Old modules/campaigns are not modified. Per-fold bootstrap SE is omitted: the
scientific intervals are the shared subject bootstrap of the final OOF arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import calibration_probe as CP
import channel_readout as CR
from wearusfm.harness import downstream_ab as DA
from wearusfm.harness import p1_cross_subject as P1

PROTOCOL = "tokenizer-probe-20261103-v1"
ARMS = ("T25-L", "T100-L", "T100-G")
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
DATASETS = sum(TASKS.values(), ())
WEIGHTS = {"p1": (8/13, 3/13, 2/13), "p2": (.5, .5)}
C_GRID = (1e-3, 1e-2, 1e-1, 1., 10., 100.)
MAX_ITER, SEED, N_BOOT = 12000, 20261102, 2000
SPLITS_PATH = ROOT / "splits/v1/splits_v1.json"
CONTRASTS = {"T100-L_minus_T25-L": (-1., 1., 0.), "T100-G_minus_T100-L": (0., -1., 1.),
             "T100-G_minus_T25-L": (-1., 0., 1.)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value, n=64):
    return isinstance(value, str) and len(value) == n and all(c in "0123456789abcdef" for c in value)


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def fresh(path):
    p = Path(path)
    require(p.is_absolute() and not p.exists() and not p.is_symlink(), "output must be new and absolute")
    require(ROOT != p.resolve() and ROOT not in p.resolve().parents, "outputs must be outside code")


def recipe():
    return {"encoder": "teacher", "layer": "pool", "width": 768, "features_dtype": "float32", "pca": False,
            "scaler": "StandardScaler fitted only on fold train", "solver": "lbfgs", "penalty": "l2",
            "class_weight": "balanced", "max_iter": MAX_ITER, "tol": 1e-4, "seed": 0, "C_grid": list(C_GRID),
            "selection": "max validation balanced accuracy; ties smallest C; refit train only",
            "k_folds": 4, "fold_seed": CP.FOLD_SEED, "val_frac": CP.VAL_FRAC,
            "bootstrap": {"unit": "subject within dataset", "seed": SEED, "replicates": N_BOOT,
                          "missing_classes": "macro recall over present target classes; no discarded draws"}}


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), Path(CP.__file__), Path(CR.__file__), Path(DA.__file__), Path(P1.__file__))}


def make_plan(caches, datasets, step=19618):
    plan = {"protocol": PROTOCOL, "step": step, "recipe": recipe(), "sources": source_hashes(),
            "datasets": datasets, "caches": caches}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    require(plan.get("protocol") == PROTOCOL and plan.get("step") == 19618, "wrong protocol/endpoint")
    require(plan.get("recipe") == recipe() and plan.get("sources") == source_hashes(), "recipe/source pins differ")
    require(set(plan.get("datasets", {})) == set(DATASETS), "need all five datasets")
    require(set(plan.get("caches", {})) == set(ARMS), "need all three arms")
    for ds, entry in plan["datasets"].items():
        subs = entry.get("subjects", [])
        require(isinstance(subs, list) and len(subs) >= 4 and all(isinstance(s, str) and s for s in subs)
                and subs == sorted(set(subs)), "invalid subjects")
        require(entry.get("roles_sha256") == jhash(CP.fold_roles(subs)), "fold roles mismatch")
        task = next(t for t in TASKS if ds in TASKS[t])
        require(entry.get("task") == task and entry.get("extraction_recipe") == CR.recipe(task), "dataset recipe/task mismatch")
        require(all(digest(entry.get(k)) for k in ("input_sha256", "splits_sha256")), "invalid input/split pins")
        if task == "p1":
            require(sha(SPLITS_PATH) == entry["splits_sha256"], "P1 split file hash mismatch")
            require(CP.pretraining_subjects(json.loads(SPLITS_PATH.read_text()), ds) == subs, "official P1 test subjects not excluded")
    paths, checkpoints = [], []
    for arm in ARMS:
        require(set(plan["caches"][arm]) == set(DATASETS), "incomplete cache grid")
        cps = set()
        for ds, pin in plan["caches"][arm].items():
            require(all(Path(pin.get(k, "")).is_absolute() for k in ("path", "receipt_path", "provenance_path")), "absolute input paths required")
            require(all(digest(pin.get(k)) for k in ("sha256", "receipt_sha256", "provenance_sha256", "checkpoint_sha256", "extraction_plan_sha256")), "invalid cache pins")
            require(digest(pin.get("extraction_code_sha"), 40), "invalid extraction code SHA")
            for k in ("input_sha256", "splits_sha256", "extraction_recipe"):
                require(pin.get(k) == plan["datasets"][ds][k], "cache dataset identity differs")
            paths.append(str(Path(pin["path"]).resolve()))
            cps.add(pin["checkpoint_sha256"])
        require(len(cps) == 1, "mixed checkpoints within arm")
        checkpoints.extend(cps)
    require(len(set(paths)) == len(paths) and len(set(checkpoints)) == len(ARMS), "duplicate cache/checkpoint across arms")


def labels(values, name):
    a = np.asarray(values)
    require(a.ndim == 1 and len(a) > 0 and a.dtype.kind in "biufUS", "invalid " + name)
    require(a.dtype.kind not in "f" or np.isfinite(a).all(), "nonfinite " + name)
    return a


def load_cache(plan, arm, ds, features=True):
    pin, expected = plan["caches"][arm][ds], plan["datasets"][ds]
    for p, key in ((pin["path"], "sha256"), (pin["provenance_path"], "provenance_sha256"), (pin["receipt_path"], "receipt_sha256")):
        require(sha(p) == pin[key], "input hash mismatch: " + str(p))
    prov = json.loads(Path(pin["provenance_path"]).read_text())
    receipt = json.loads(Path(pin["receipt_path"]).read_text())
    identity = {"arm": arm, "which": "teacher", "checkpoint_step": plan["step"],
                "checkpoint_sha256": pin["checkpoint_sha256"], "postload_recipe_passed": True}
    for k, v in identity.items():
        require(receipt.get(k) == v and prov.get(k) == v, "teacher/loaded model identity mismatch: " + k)
    require(receipt.get("complete") is True and receipt.get("checkpoint_unchanged") is True, "incomplete extraction receipt")
    require(receipt.get("code_sha") == pin["extraction_code_sha"] and receipt.get("plan_sha256") == pin["extraction_plan_sha256"], "extraction source mismatch")
    output = receipt.get("outputs", {}).get(ds, {})
    require(output.get("path") == pin["path"] and output.get("sha256") == pin["sha256"]
            and output.get("provenance_sha256") == pin["provenance_sha256"], "receipt output mismatch")
    require(prov.get("cache_sha256") == pin["sha256"], "provenance cache mismatch")
    for key in ("input_sha256", "splits_sha256", "extraction_recipe"):
        require(prov.get(key) == pin[key], "provenance dataset mismatch: " + key)
    with np.load(pin["path"], allow_pickle=False) as z:
        require(str(z["protocol"]) == CR.PROTOCOL and str(z["feature_set"]) == "channel_readout", "wrong cache format")
        require(str(z["task"]) == expected["task"] and int(z["encoder_dim"]) == 384, "wrong cache task/width")
        for key in ("input_sha256", "splits_sha256", "checkpoint_sha256"):
            require(str(z[key]) == pin[key], "embedded cache pin mismatch: " + key)
        require(str(z["code_sha"]) == pin["extraction_code_sha"] and json.loads(str(z["extraction_recipe"])) == pin["extraction_recipe"], "embedded recipe mismatch")
        y, s = labels(z["y"], "targets"), labels(z["s"], "subjects").astype(str)
        require(len(y) == len(s) and sorted(set(s)) == expected["subjects"]
                and z["subjects"].astype(str).tolist() == expected["subjects"], "subject coverage mismatch")
        if expected["task"] == "p2":
            require(CP.p2_internal_subjects(z["all_subjects"].tolist()) == expected["subjects"], "official P2 test subjects not excluded")
            require(pin["splits_sha256"] == CP.P2_SPLITS_SHA256, "P2 split definition differs")
        x = z["x_pool"] if features else None
    if features:
        require(x.shape == (len(y), 768) and x.dtype == np.float32 and np.isfinite(x).all(), "pool must be finite float32 width768")
    return x, y, s


def numeric_context():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "threadpools": sorted([{k: p.get(k) for k in ("internal_api", "prefix", "version", "num_threads", "architecture", "threading_layer")}
                                   for p in threadpool_info()], key=lambda p: json.dumps(p, sort_keys=True)),
            "features_dtype": "float32"}


def described(caught):
    return [{"category": w.category.__name__, "message": str(w.message), "file": str(w.filename), "line": w.lineno} for w in caught]


def reemit(caught):
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


class AuditLog(list):
    def __init__(self, path=None):
        super().__init__()
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.touch(exist_ok=False)

    def persist(self, record):
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())


def scored(y, p):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = float(balanced_accuracy_score(y, p))
    reemit(caught)
    require(np.isfinite(value), "nonfinite balanced accuracy")
    return value, described(caught)


def fit_fold(x, y, subjects, roles, fold, audit, context=None):
    """Minimal Claude numerical path; labels remain native, scaler sees train only."""
    role = roles[fold]
    mt, mv, mh = (np.isin(subjects, role[k]) for k in ("train", "val", "held"))
    kv, kh = P1.keep_train_classes(y[mt], y[mv], y[mh])
    train, val, held = x[mt], x[mv][kv], x[mh][kh]
    yt, yv, yh = y[mt], y[mv][kv], y[mh][kh]
    require(all(len(a) for a in (train, val, held)), "empty fold after train-class filtering")
    scaler = StandardScaler().fit(train)
    xt, xv, xh = scaler.transform(train), scaler.transform(val), scaler.transform(held)
    classes, counts = np.unique(yt, return_counts=True)
    base = {**(context or {}), "fold": fold, "roles_sha256": jhash(role), "n_train": len(xt), "n_features": xt.shape[1],
            "dtype": str(xt.dtype), "classes": classes.tolist(), "class_counts": counts.tolist(), "max_iter": MAX_ITER,
            "scale": True, "scaler_sha256": DA.fingerprint(np.asarray(scaler.mean_, dtype=np.float64), np.asarray(scaler.scale_, dtype=np.float64))}

    def fit(c, phase):
        model = LogisticRegression(C=c, max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4)
        record = {**base, "fit_phase": phase, "C": float(c), "solver_params": model.get_params(), "selected": None}
        audit.append(record)
        start, caught = time.perf_counter(), []
        try:
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                try:
                    model.fit(xt, yt)
                finally:
                    caught = list(messages)
        except BaseException as exc:
            record.update(elapsed_s=time.perf_counter()-start, warnings=described(caught), exception=f"{type(exc).__name__}: {exc}")
            audit.persist(record)
            reemit(caught)
            raise
        n_iter = np.atleast_1d(model.n_iter_)
        count = sum(w.category.__name__ == "ConvergenceWarning" for w in caught)
        limit = bool(int(n_iter.max()) >= MAX_ITER)
        record.update(elapsed_s=time.perf_counter()-start, warnings=described(caught), exception=None,
                      n_iter=int(n_iter.max()), n_iter_all=n_iter.astype(int).tolist(), convergence_warnings=count,
                      reached_limit=limit, converged_by_criterion=not limit and count == 0, anomalous_stop=count > 0 and not limit)
        reemit(caught)
        return model, record

    scores, mine = {}, []
    for c in C_GRID:
        model, record = fit(c, "candidate")
        # Persist the completed fit before scoring as well: a scoring exception must not lose its diagnostics.
        try:
            scores[c], record["scoring_warnings"] = scored(yv, model.predict(xv))
            record["val_bacc"] = scores[c]
        except BaseException:
            audit.persist(record)
            raise
        audit.persist(record)
        mine.append(record)
    best = max(C_GRID, key=lambda c: (scores[c], -c))
    for record in mine:
        record["selected"] = record["C"] == best
    model, record = fit(best, "refit")
    audit.persist(record)
    pred = model.predict(xh)
    held_score, held_warnings = scored(yh, pred)
    train_score, train_warnings = scored(yt, model.predict(xt))
    return {"row": np.flatnonzero(mh)[kh], "y": yh, "pred": pred, "s": subjects[mh][kh],
            "fold": np.full(len(yh), fold, dtype=np.int64), "report": {"fold": fold, **role, "c": best,
            "val_bacc": scores[best], "held_bacc": held_score, "train_bacc": train_score,
            "val_scores": {str(c): v for c, v in scores.items()}, "n_train": len(xt), "n_val": len(xv), "n_held": len(xh),
            "dropped_val_windows": int((~kv).sum()), "dropped_held_windows": int((~kh).sum()),
            "scoring_warnings": {"held": held_warnings, "train": train_warnings}, "roles_sha256": jhash(role),
            "scaler_sha256": base["scaler_sha256"]}}


def audit_summary(records):
    require(all(r.get("exception") is None for r in records), "failed fit in completed output")
    return {"fits": len(records), "convergence_warnings": sum(r["convergence_warnings"] for r in records),
            "reached_limit": sum(r["reached_limit"] for r in records),
            "converged_by_criterion": all(r["converged_by_criterion"] for r in records),
            "unresolved_fits": [{k: r[k] for k in ("arm", "dataset", "fold", "fit_phase", "C", "n_iter")}
                                for r in records if not r["converged_by_criterion"]]}


def run_shard(plan, arm, task, datasets, folds, out, identity):
    validate_plan(plan)
    require(arm in ARMS and task in TASKS and len(set(datasets)) == len(datasets) and datasets
            and set(datasets) <= set(TASKS[task]), "invalid shard arm/task/datasets")
    require(folds and folds == sorted(set(folds)) and all(type(f) is int and f in range(4) for f in folds), "invalid shard folds")
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
    require(jhash(r["numeric_context"]) == r["numeric_sha256"], "numeric signature mismatch")
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


def validate_oof(row, y, pred, subjects, fold):
    require(row.ndim == 1 and row.dtype.kind in "iu" and len(row) == len(np.unique(row)) and (row >= 0).all(), "duplicate/invalid OOF rows")
    for value, name in ((y, "targets"), (pred, "predictions"), (subjects, "subjects")):
        require(len(labels(value, name)) == len(row), "OOF length mismatch")
    require(fold.shape == row.shape and fold.dtype.kind in "iu" and np.isin(fold, range(4)).all(), "invalid OOF folds")


def aggregate(plan, arm, task, shards, out, identity):
    validate_plan(plan)
    require(arm in ARMS and task in TASKS and len(set(map(str, shards))) == len(shards) and shards, "invalid aggregation")
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
    require(set(cells) == set(ARMS) and all(set(c) == set(DATASETS) for c in cells.values()), "need complete three-arm/five-dataset grid")
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


def compare(plan, paths, out, identity):
    validate_plan(plan)
    require(len(paths) == 6 and len(set(map(str, paths))) == 6, "need six distinct aggregates")
    loaded = [read_completed(p, "aggregate", plan) for p in paths]
    require({(r["arm"], r["task"]) for r, _, _ in loaded} == {(a, t) for a in ARMS for t in TASKS}, "incomplete/duplicate aggregate grid")
    require(all(r["code_sha"] == identity["code_sha"] for r, _, _ in loaded) and len({r["numeric_sha256"] for r, _, _ in loaded}) == 1, "mixed code/numeric recipes")
    cells = {a: {} for a in ARMS}
    for report, values, _ in loaded:
        task = report["task"]
        require(set(report["datasets"]) == set(TASKS[task]) and report["weights"] == dict(zip(TASKS[task], WEIGHTS[task])), "aggregate dataset weights differ")
        for ds in TASKS[task]:
            a = {k: values[f"{ds}_{k}"] for k in ("row", "y", "pred", "s", "fold")}
            score, _ = scored(a["y"], a["pred"])
            require(score == report["datasets"][ds]["balanced_accuracy"], "reported dataset score differs from OOF")
            cells[report["arm"]][ds] = a
        require(report["balanced_accuracy"] == float(np.dot(WEIGHTS[task], [report["datasets"][d]["balanced_accuracy"] for d in TASKS[task]])), "reported aggregate differs")
    result = {"protocol": PROTOCOL, "kind": "comparison", "complete": True, "step": plan["step"], "recipe": recipe(),
              "plan_content_sha256": jhash(plan), **identity, **compare_arrays(cells),
              "fit_audit": {r["arm"]+"/"+r["task"]: r["audit_summary"] for r, _, _ in loaded},
              "reports": [{"path": str(p), "sha256": sha(p)} for p in paths]}
    apply_solver_reading(result)
    completion_identity(plan, identity)
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
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--folds", nargs="+", type=int)
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
        require(a.arm and a.task and a.datasets and a.folds is not None, "fit-shard requires arm/task/datasets/folds")
        result = run_shard(plan, a.arm, a.task, a.datasets, a.folds, a.out, identity)
    elif a.stage == "aggregate":
        require(a.arm and a.task and a.shards, "aggregate requires arm/task/shards")
        result = aggregate(plan, a.arm, a.task, a.shards, a.out, identity)
    else:
        require(a.reports, "compare requires reports")
        result = compare(plan, a.reports, a.out, identity)
    require(sha(a.plan) == a.expected_plan_sha256 and plan["sources"] == source_hashes(), "plan/source changed during execution")
    print(json.dumps({"complete": result["complete"], "kind": result["kind"], "out": str(a.out)}))


if __name__ == "__main__":
    main()
