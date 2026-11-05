"""D-independent classification adapters for the approved long T100-G matrix.

No submissions or checkpoint selection occur here. The controller attests the
teacher and source pins; this module preserves historical inputs, fits the
12000-iteration probe recipe, and retains paired OOF and numerical diagnostics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch  # Load torch before sklearn in the supported local environment.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import calibration_probe as CP
import channel_readout as CR
import tokenizer_probe as TP
from wearusfm.harness import backbone_depth as BD
from wearusfm.harness import downstream_ab as DA
from wearusfm.harness import p1_cross_subject as P1
from wearusfm.harness import psd3_inputs as I

PROTOCOL = "consolidation-long-classification-20261105-v1"
DOSES = (19618, 39237, 49046)
VIEWS = ("pre", "post", "pool")
TASKS, DATASETS, WEIGHTS = TP.TASKS, TP.DATASETS, TP.WEIGHTS
C_GRID, MAX_ITER, N_BOOT, SEED = TP.C_GRID, TP.MAX_ITER, TP.N_BOOT, TP.SEED
OOF_KEYS = ("row", "y", "pred", "s", "fold")
require, sha, jhash = TP.require, TP.sha, TP.jhash


def extraction_recipe(task):
    require(task in TASKS, "unknown classification task")
    return {"base": CP.RECIPES[task], "extractor": "harness.backbone_depth.extract_windows",
            "views": {"pre": "D, Perceiver slots averaged over valid time",
                      "post": "D, final normalized backbone slots averaged over valid time",
                      "pool": "2D, historical post plus local readout"},
            "dtype": "float32", "crop_s": 4.0, "k_neighbors": 8,
            "scale_windows": 10, "encoder": "teacher", "same_forward": True,
            "crop_reduction": "mean after independent crop pooling"}


def probe_recipe(view, dim):
    require(view in VIEWS and type(dim) is int and dim > 0, "invalid view/dimension")
    return {**TP.recipe(), "layer": view, "width": dim * (2 if view == "pool" else 1)}


def _task(dataset):
    require(dataset in DATASETS, "unregistered dataset")
    return next(t for t, datasets in TASKS.items() if dataset in datasets)


def _check_arrays(arrays, dim):
    y, s, subjects = (arrays[k] for k in ("y", "s", "subjects"))
    for name in ("y", "s", "subjects"):
        I._labels(arrays[name], name)
    require(s.dtype.kind in "US" and subjects.dtype.kind in "US", "subjects must be strings")
    require(len(y) == len(s) and sorted(set(s.tolist())) == subjects.tolist(), "subject coverage/order differs")
    require(type(dim) is int and dim > 0, "invalid encoder width")
    for view in VIEWS:
        x = arrays["x_" + view]
        require(x.shape == (len(y), dim * (2 if view == "pool" else 1))
                and x.dtype == np.float32 and np.isfinite(x).all(), "invalid/nonfinite view: " + view)
    require(np.allclose(arrays["x_post"], arrays["x_pool"][:, :dim], rtol=1e-5, atol=1e-6),
            "post view differs from historical pool first half")
    I.validate_roles(CP.fold_roles(subjects.tolist()), subjects)


def extract_dataset(model, dataset, *, roots=(), epn_roots=(), uci_root=None,
                    splits=None, splits_sha256=None, batch=64, device="cpu"):
    """Extract all three views in one teacher forward per crop/batch.

    P1 uses the historical session order and session scale keys; P2 filters the
    union of official tests before extraction. Return arrays and metadata for
    ``save_cache``; model loading/teacher attestation belongs to the controller.
    """
    from wearusfm.harness import benchmarks as B, fm_features as FF

    task = _task(dataset)
    model.eval()
    if task == "p1":
        require(splits is not None and TP.digest(splits_sha256), "P1 requires pinned official splits")
        subjects = CP.pretraining_subjects(splits, dataset)
        digest, chunks, ys, ss, signature = hashlib.sha256(), {v: [] for v in VIEWS}, [], [], None
        for subject in subjects:
            for session in P1.sessions_of(CR._root_of(list(map(Path, roots)), dataset, subject), dataset, subject):
                sw = P1.session_windows(session, dataset, window_s=CP.WINDOW_S)
                if not len(sw.windows):
                    continue
                scale_keys = np.full(len(sw.windows), f"{subject}/{sw.session}")
                digest.update(DA.fingerprint("pretraining", subject, sw.session, sw.windows,
                                            sw.labels, sw.fs, sw.montage, scale_keys).encode())
                signature = CR._same_montage(signature, sw.montage, f"{dataset}/{subject}/{sw.session}")
                views = BD.extract_windows(model, sw.windows, scale_keys, sw.fs, sw.montage,
                                           batch=batch, device=device)
                for view in VIEWS:
                    chunks[view].append(views[view])
                ys.append(sw.labels)
                ss.append(np.full(len(sw.windows), subject))
        require(bool(ys), "empty P1 extraction")
        arrays = {"x_" + v: np.concatenate(chunks[v]) for v in VIEWS}
        arrays.update(y=np.concatenate(ys), s=np.concatenate(ss), subjects=np.asarray(subjects))
        input_sha = digest.hexdigest()
    else:
        w, y, s, fs = B.load_epn612(list(map(Path, epn_roots))) if dataset == "epn612" else B.load_uci_emg(Path(uci_root))
        s = np.asarray([str(value) for value in s])
        all_subjects = sorted(set(s.tolist()))
        subjects = CP.p2_internal_subjects(all_subjects)
        keep = np.isin(s, subjects)
        w, y, s = w[keep], y[keep], s[keep]
        montage = FF.myo8_montage(dataset, fs)
        input_sha, signature = DA.fingerprint(w, y, s, fs, montage), CR.montage_signature(montage)
        views = BD.extract_windows(model, w, s, fs, montage, batch=batch, device=device)
        arrays = {"x_" + v: views[v] for v in VIEWS}
        arrays.update(y=y, s=s, subjects=np.asarray(subjects), all_subjects=np.asarray(all_subjects))
        splits_sha256 = CP.P2_SPLITS_SHA256
    _check_arrays(arrays, int(model.cfg.dim))
    require(arrays["subjects"].tolist() == subjects, "extraction subjects differ")
    return {"arrays": arrays, "metadata": {
        "dataset": dataset, "task": task, "protocol": PROTOCOL, "calibration_protocol": CP.PROTOCOL,
        "feature_set": "backbone_depth", "encoder_dim": int(model.cfg.dim), "encoder": "teacher",
        "input_sha256": input_sha, "splits_sha256": splits_sha256,
        "extraction_recipe": extraction_recipe(task), "historical_extraction_recipe": CR.recipe(task),
        "montage_signature": signature, "channels": len(signature),
        "roles_sha256": jhash(CP.fold_roles(subjects)),
        "ordered_rows_sha256": DA.fingerprint(np.arange(len(arrays["y"]), dtype=np.int64), arrays["y"], arrays["s"])}}


def verify_historical_inputs(extraction, reference, *, compare_pool=False):
    """Require the pinned historical cache's exact raw fingerprint and row order."""
    path = Path(reference["path"])
    before = I._signature(path)
    require(sha(path) == reference["sha256"], "historical cache hash differs")
    arrays, meta = extraction["arrays"], extraction["metadata"]
    exact, max_error = None, None
    with np.load(path, allow_pickle=False) as old:
        for key in ("y", "s", "subjects") + (("all_subjects",) if meta["task"] == "p2" else ()):
            require(arrays[key].dtype == old[key].dtype and np.array_equal(arrays[key], old[key]),
                    "historical ordered rows differ: " + key)
        for key in ("input_sha256", "splits_sha256", "task", "channels"):
            require(old[key].item() == meta[key], "historical input identity differs: " + key)
        require(json.loads(str(old["extraction_recipe"])) == CR.recipe(meta["task"]), "historical recipe differs")
        require(json.loads(str(old["montage_signature"])) == meta["montage_signature"], "historical montage differs")
        if compare_pool:
            require(str(old["checkpoint_sha256"]) == meta["checkpoint_sha256"], "pool parity needs same checkpoint")
            require(old["x_pool"].shape == arrays["x_pool"].shape and
                    np.allclose(old["x_pool"], arrays["x_pool"], rtol=1e-5, atol=1e-6),
                    "same-checkpoint historical pool parity failed")
            exact = bool(np.array_equal(old["x_pool"], arrays["x_pool"]))
            max_error = float(np.max(np.abs(old["x_pool"] - arrays["x_pool"])))
    require(I._signature(path) == before, "historical cache changed during comparison")
    return {"reference_cache_sha256": reference["sha256"], "historical_pool_parity": bool(compare_pool),
            "historical_pool_exact_equal": exact, "historical_pool_max_abs_error": max_error,
            "historical_pool_tolerance": {"rtol": 1e-5, "atol": 1e-6}}


def save_cache(path, extraction, identity, reference):
    """Write a new cache after historical input checks; never overwrite artifacts.

    identity requires step, checkpoint, checkpoint_sha256, code_sha and
    plan_sha256. The controller supplies its attested frozen-teacher identity.
    """
    require(identity.get("step") in DOSES and TP.digest(identity.get("checkpoint_sha256"))
            and TP.digest(identity.get("code_sha"), 40) and TP.digest(identity.get("plan_sha256"))
            and Path(identity.get("checkpoint", "")).is_absolute(), "invalid extraction identity")
    require(not set(identity) & set(extraction["metadata"]), "identity cannot overwrite extraction metadata")
    extraction = {"arrays": extraction["arrays"], "metadata": {**extraction["metadata"], **identity}}
    _check_arrays(extraction["arrays"], extraction["metadata"]["encoder_dim"])
    parity = verify_historical_inputs(extraction, reference, compare_pool=identity["step"] == DOSES[0])
    meta = {**extraction["metadata"], **parity}
    TP.fresh(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encoded = {k: np.asarray(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) or v is None else v) for k, v in meta.items()}
    with Path(path).open("xb") as stream:
        np.savez(stream, **extraction["arrays"], **encoded)
    return {"path": str(path), "sha256": sha(path), "bytes": Path(path).stat().st_size, **meta}


def load_cache(path, view, *, expected):
    """Load a pinned D/2D cache and validate official-test exclusion and roles.

    expected requires sha256; optional scalar/dictionary metadata pins are
    checked. P1 additionally requires ``subjects`` from the pinned split plan.
    """
    require(view in VIEWS and TP.digest(expected.get("sha256")), "pinned cache/view required")
    before = I._signature(Path(path))
    require(sha(path) == expected["sha256"], "cache hash differs")
    if "bytes" in expected:
        require(before[2] == expected["bytes"], "cache byte count differs")
    with np.load(path, allow_pickle=False) as z:
        meta = {k: z[k].item() for k in z.files
                if not k.startswith("x_") and k not in ("y", "s", "subjects", "all_subjects")}
        for key in ("extraction_recipe", "historical_extraction_recipe", "montage_signature", "historical_pool_tolerance"):
            meta[key] = json.loads(meta[key])
        for key in ("historical_pool_exact_equal", "historical_pool_max_abs_error"):
            if meta[key] == "null":
                meta[key] = None
        for key, value in expected.items():
            if key in meta:
                require(meta[key] == value, "cache metadata pin differs: " + key)
        require(meta.get("protocol") == PROTOCOL and meta.get("feature_set") == "backbone_depth"
                and meta.get("encoder") == "teacher" and meta.get("step") in DOSES, "wrong cache protocol/teacher/dose")
        task = _task(meta["dataset"])
        require(meta["task"] == task and meta["extraction_recipe"] == extraction_recipe(task), "cache recipe/task differs")
        require(TP.digest(meta.get("checkpoint_sha256")) and TP.digest(meta.get("code_sha"), 40)
                and TP.digest(meta.get("plan_sha256")), "invalid embedded source/checkpoint identity")
        x, y, s, subjects = (z[k] for k in ("x_" + view, "y", "s", "subjects"))
        for name, value in (("y", y), ("s", s), ("subjects", subjects)):
            I._labels(value, name)
        require(s.dtype.kind in "US" and subjects.dtype.kind in "US" and len(y) == len(s)
                and sorted(set(s.tolist())) == subjects.tolist(), "invalid subject coverage")
        if task == "p2":
            require(subjects.tolist() == CP.p2_internal_subjects(z["all_subjects"].tolist())
                    and meta["splits_sha256"] == CP.P2_SPLITS_SHA256, "official P2 test subjects leaked")
        else:
            require(subjects.tolist() == expected.get("subjects"), "P1 subjects must match pinned official split")
    dim = meta["encoder_dim"]
    require(type(dim) is int and dim > 0 and x.shape == (len(y), dim * (2 if view == "pool" else 1))
            and x.dtype == np.float32 and np.isfinite(x).all(), "invalid/nonfinite feature matrix")
    roles = CP.fold_roles(subjects.tolist())
    I.validate_roles(roles, subjects)
    require(meta["roles_sha256"] == jhash(roles) and meta["ordered_rows_sha256"] ==
            DA.fingerprint(np.arange(len(y), dtype=np.int64), y, s), "roles/ordered row identity differs")
    require(I._signature(Path(path)) == before, "cache changed while loading")
    return x, y, s, {**meta, "cache_sha256": expected["sha256"], "roles": roles, "view": view}


def _fit(train, y, c, audit, base, phase):
    model = LogisticRegression(C=c, max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4)
    record = {**base, "C": c, "fit_phase": phase, "solver_params": model.get_params(), "selected": None}
    audit.append(record)
    start, caught = time.perf_counter(), []
    try:
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            try:
                model.fit(train, y)
            finally:
                caught = list(messages)
        require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), "nonfinite fitted coefficients")
        n_iter = np.atleast_1d(model.n_iter_)
        limit = bool(int(n_iter.max()) >= MAX_ITER)
        count = sum(w.category.__name__ == "ConvergenceWarning" for w in caught)
        record.update(n_iter=int(n_iter.max()), n_iter_all=n_iter.astype(int).tolist(),
                      convergence_warnings=count, reached_limit=limit,
                      converged_by_criterion=not limit and count == 0, anomalous_stop=count > 0 and not limit,
                      exception=None)
    except BaseException as exc:
        record["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.update(elapsed_s=time.perf_counter()-start, warnings=TP.described(caught))
        audit.persist(record)
        TP.reemit(caught)
    return model, record


def _predict(model, x):
    require(np.isfinite(model.decision_function(x)).all(), "nonfinite classification decision scores")
    pred = model.predict(x)
    I._labels(pred, "predictions")
    return pred


def fit_fold(x, y, subjects, roles, fold, audit=None, context=None):
    """Unchanged scientific fit, with finite checks and delayed held prediction."""
    I.validate_roles(roles, np.asarray(sorted(set(subjects))))
    require(x.ndim == 2 and x.dtype == np.float32 and len(x) == len(y) == len(subjects)
            and np.isfinite(x).all(), "invalid/nonfinite input features")
    audit = TP.AuditLog() if audit is None else audit
    role = roles[fold]
    mt, mv, mh = (np.isin(subjects, role[k]) for k in ("train", "val", "held"))
    kv, kh = P1.keep_train_classes(y[mt], y[mv], y[mh])
    require(mt.any() and kv.any() and kh.any() and len(np.unique(y[mt])) >= 2, "empty/single-class fold")
    scaler = StandardScaler().fit(x[mt])
    xt, xv = scaler.transform(x[mt]), scaler.transform(x[mv][kv])
    require(np.isfinite(xt).all() and np.isfinite(xv).all(), "nonfinite scaled train/validation")
    classes, counts = np.unique(y[mt], return_counts=True)
    base = {**(context or {}), "fold": fold, "roles_sha256": jhash(role), "max_iter": MAX_ITER,
            "n_train": len(xt), "n_features": x.shape[1], "dtype": str(xt.dtype), "scale": True,
            "classes": classes.tolist(), "class_counts": counts.tolist(),
            "scaler_sha256": DA.fingerprint(np.asarray(scaler.mean_, dtype=np.float64), np.asarray(scaler.scale_, dtype=np.float64))}
    scores, records = {}, []
    for c in C_GRID:
        model, record = _fit(xt, y[mt], c, audit, base, "candidate")
        try:
            scores[c], record["scoring_warnings"] = TP.scored(y[mv][kv], _predict(model, xv))
        except BaseException as exc:
            record["scoring_exception"] = f"{type(exc).__name__}: {exc}"
            audit.persist(record)
            raise
        record["val_bacc"] = scores[c]
        audit.persist(record)
        records.append(record)
    best = max(C_GRID, key=lambda c: (scores[c], -c))
    for record in records:
        record["selected"] = record["C"] == best
        audit.persist(record)
    model, record = _fit(xt, y[mt], best, audit, base, "refit")
    record["selected"] = True
    audit.persist(record)
    xh = scaler.transform(x[mh][kh])
    require(np.isfinite(xh).all(), "nonfinite scaled held features")
    pred = _predict(model, xh)
    held_score, hw = TP.scored(y[mh][kh], pred)
    train_score, tw = TP.scored(y[mt], _predict(model, xt))
    return {"row": np.flatnonzero(mh)[kh], "y": y[mh][kh], "pred": pred, "s": subjects[mh][kh],
            "fold": np.full(int(kh.sum()), fold, dtype=np.int64), "report": {
                "fold": fold, **role, "c": best, "val_bacc": scores[best], "held_bacc": held_score,
                "train_bacc": train_score, "val_scores": {str(c): score for c, score in scores.items()},
                "n_train": len(xt), "n_val": len(xv), "n_held": len(xh),
                "dropped_val_windows": int((~kv).sum()), "dropped_held_windows": int((~kh).sum()),
                "roles_sha256": base["roles_sha256"], "scaler_sha256": base["scaler_sha256"],
                "scoring_warnings": {"held": hw, "train": tw}}}


def audit_summary(records):
    require(records and all(r.get("exception") is None for r in records), "incomplete/failed fit audit")
    return {"fits": len(records), "convergence_warnings": sum(r["convergence_warnings"] for r in records),
            "reached_limit": sum(r["reached_limit"] for r in records),
            "converged_by_criterion": all(r["converged_by_criterion"] for r in records),
            "unresolved_fits": [{k: r[k] for k in ("fold", "fit_phase", "C", "n_iter")}
                                for r in records if not r["converged_by_criterion"]]}


def fit_cache(path, view, out, *, expected, identity, folds=None):
    """Fit one dataset/view, optionally a registered subset of its four folds."""
    x, y, s, meta = load_cache(path, view, expected=expected)
    folds = list(range(4)) if folds is None else list(folds)
    require(folds and folds == sorted(set(folds)) and all(type(f) is int and f in range(4) for f in folds), "invalid fold subset")
    require(TP.digest(identity.get("code_sha"), 40) and TP.digest(identity.get("plan_sha256")), "fit identity required")
    out = Path(out)
    for target in (out, out.with_suffix(".oof.npz"), out.with_suffix(".audit.jsonl")):
        TP.fresh(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = TP.AuditLog(out.with_suffix(".audit.jsonl"))
    numeric = TP.numeric_context()
    context = {**identity, **{k: meta[k] for k in ("dataset", "task", "step", "checkpoint_sha256", "cache_sha256", "view")}}
    results = [fit_fold(x, y, s, meta["roles"], f, audit, context) for f in folds]
    joined = {k: np.concatenate([r[k] for r in results]) for k in OOF_KEYS}
    TP.validate_oof(*(joined[k] for k in OOF_KEYS))
    with out.with_suffix(".oof.npz").open("xb") as stream:
        np.savez(stream, **joined)
    summary = audit_summary(audit)
    require(TP.numeric_context() == numeric, "numeric runtime changed during fits")
    result = {"protocol": PROTOCOL, "kind": "probe", "complete": True, **context,
              "cache_path": str(path), "input_sha256": meta["input_sha256"], "splits_sha256": meta["splits_sha256"],
              "roles_sha256": meta["roles_sha256"], "ordered_rows_sha256": meta["ordered_rows_sha256"],
              "encoder_dim": meta["encoder_dim"], "recipe": probe_recipe(view, meta["encoder_dim"]),
              "folds": [r["report"] for r in results], "fold_indices": folds,
              "balanced_accuracy": TP.scored(joined["y"], joined["pred"])[0],
              "oof_identity_sha256": DA.fingerprint(*(joined[k] for k in ("row", "y", "s", "fold"))),
              "oof": str(out.with_suffix(".oof.npz")), "oof_sha256": sha(out.with_suffix(".oof.npz")),
              "audit": str(out.with_suffix(".audit.jsonl")), "audit_sha256": sha(out.with_suffix(".audit.jsonl")),
              "fit_audit": list(audit), "audit_summary": summary,
              "numeric_context": numeric, "numeric_sha256": jhash(numeric),
              "decision_valid": summary["converged_by_criterion"]}
    require(sha(path) == expected["sha256"], "input cache changed during fitting")
    TP.write_json(out, result)
    return result


def cell_name(step, view):
    require(step in DOSES and view in VIEWS, "unregistered classification cell")
    return f"{step}/{view}"


def contrast_pairs():
    """All preregistered dose contrasts per view and matched post-minus-pre."""
    contrasts = {}
    for view in VIEWS:
        for new, old in ((49046, 19618), (39237, 19618), (49046, 39237)):
            contrasts[f"{view}:{new}-{old}"] = (cell_name(new, view), cell_name(old, view))
    for step in DOSES:
        contrasts[f"{step}:post-pre"] = (cell_name(step, "post"), cell_name(step, "pre"))
    return contrasts


def _comparison_summary(points, draws, names, resolved):
    out = {"cells": {name: {"balanced_accuracy": float(points[i]),
                            "ci95_percent": (100*np.quantile(draws[:, i], [.025, .975])).tolist()}
                     for i, name in enumerate(names)}, "contrasts": {}}
    for label, (new, old) in contrast_pairs().items():
        i, j = names.index(new), names.index(old)
        delta = float(100*(points[i]-points[j]))
        ci = (100*np.quantile(draws[:, i]-draws[:, j], [.025, .975])).tolist()
        valid = resolved.get(new, False) and resolved.get(old, False)
        reading = ("practical_benefit" if delta >= 1 and ci[0] > 0 else
                   "practical_deterioration" if delta <= -1 and ci[1] < 0 else
                   "below_practical_margin" if abs(delta) < 1 and (ci[0] > 0 or ci[1] < 0) else "inconclusive")
        out["contrasts"][label] = {"new": new, "reference": old, "delta_pp": delta, "ci95_pp": ci,
            "role": "primary_dose" if label == "pool:49046-19618" else "depth_diagnostic" if ":post-pre" in label else "secondary_dose",
            "decision_valid": bool(valid), "reading": reading if valid else "unresolved_solver",
            "noninferiority_1pp": ("supported" if ci[0] > -1 else "loss_beyond_margin" if ci[1] < -1 else "unresolved")
                                    if valid else "unresolved_solver"}
    return out


def classification_reading(tasks):
    """V-R1 two-sided decisions from task-level intervals, never dataset picking."""
    evaluated = {}
    for name in contrast_pairs():
        p1, p2 = (tasks[task]["contrasts"][name] for task in ("p1", "p2"))
        valid = p1["decision_valid"] and p2["decision_valid"]
        gains = [x["delta_pp"] >= 1 and x["ci95_pp"][0] > 0 for x in (p1, p2)]
        losses = [x["delta_pp"] <= -1 and x["ci95_pp"][1] < 0 for x in (p1, p2)]
        benefit = (gains[0] and p2["ci95_pp"][0] > -1) or (gains[1] and p1["ci95_pp"][0] > -1)
        status = ("tradeoff" if any(gains) and any(losses) else "practical_deterioration" if any(losses)
                  else "practical_benefit" if benefit else "small_or_inconclusive")
        evaluated[name] = {"status": status if valid else "unresolved_solver",
                           "practical_benefit": bool(benefit) if valid else None,
                           "practical_deterioration": bool(any(losses)) if valid else None,
                           "decision_valid": bool(valid)}
    a, b, increment = (evaluated["pool:"+suffix] for suffix in ("39237-19618", "49046-19618", "49046-39237"))
    if not all(x["decision_valid"] for x in (a, b, increment)):
        candidate = None
    elif a["practical_benefit"]:
        candidate = 49046 if b["practical_benefit"] and increment["practical_benefit"] else 39237
    else:
        candidate = 49046 if b["practical_benefit"] else 19618
    return {"rule": "V-R1", "contrasts": evaluated, "classification_reference_candidate": candidate,
            "reference_scope": "exploratory development choice only; primary remains pool 49046 minus 19618",
            "depth_scope": "D-matched reader accessibility diagnostic; no backbone-information destruction claim or architecture change",
            "historical_replica_context": {"contrast": "T25-L minus B19618", "n_bootstrap": 2000,
                "dataset_delta_pp": {"epn612": -.236, "uci_emg": .049, "ninapro_db2": .444,
                                     "ninapro_db3": .453, "ninapro_db6": .863},
                "p1": {"delta_pp": .510, "ci95_pp": [.090, .926]},
                "p2": {"delta_pp": -.093, "ci95_pp": [-1.382, 1.065]},
                "use": "context from preregistered V-R1; neither a universal noise floor nor a correction to the new contrasts"}}


def compare_arrays(cells, *, convergence=None, n_boot=N_BOOT, seed=SEED):
    """Paired subject bootstrap for nine cells × five datasets, on exact support.

    ``cells['19618/pre'][dataset]`` holds row/y/pred/s/fold arrays. Convergence
    evidence has the same cell/dataset keys and boolean values; absent evidence
    retains descriptive numbers and withholds V-R1 decisions.
    """
    names = [cell_name(step, view) for step in DOSES for view in VIEWS]
    require(set(cells) == set(names) and all(set(cells[name]) == set(DATASETS) for name in names),
            "incomplete nine-cell/five-dataset matrix")
    require(type(n_boot) is int and n_boot >= 2, "invalid bootstrap count")
    convergence = convergence or {}
    rng, datasets, points, samples = np.random.default_rng(seed), {}, {}, {}
    for ds in sorted(DATASETS):
        aligned = []
        for name in names:
            a = cells[name][ds]
            TP.validate_oof(*(a[k] for k in OOF_KEYS))
            require(set(a["fold"].tolist()) == set(range(4)), "incomplete OOF folds")
            ix = np.argsort(a["row"])
            aligned.append({k: a[k][ix] for k in OOF_KEYS})
        base = aligned[0]
        require(all(all(np.array_equal(a[k], base[k]) for k in ("row", "y", "s", "fold")) for a in aligned),
                "OOF pairing differs: " + ds)
        stats = [DA.subject_counts(a["y"], a["pred"], a["s"]) for a in aligned]
        users, denominator, _ = DA._arrays(stats[0])
        counts = [DA._arrays(stat)[2] for stat in stats]
        draw = rng.integers(0, len(users), size=(n_boot, len(users)))
        den = denominator[draw].sum(1)
        present = den > 0
        classes_present = present.sum(1)
        require((classes_present > 0).all(), "empty bootstrap targets")
        values = np.column_stack([np.divide(hits[draw].sum(1), den, out=np.zeros_like(den), where=present).sum(1)/classes_present for hits in counts])
        total = denominator.sum(0)
        point = np.asarray([np.mean(h.sum(0)[total > 0]/total[total > 0]) for h in counts])
        require(np.isfinite(values).all(), "nonfinite bootstrap estimates")
        points[ds], samples[ds] = point, values
        resolved = {name: convergence.get(name, {}).get(ds) is True for name in names}
        datasets[ds] = {**_comparison_summary(point, values, names, resolved), "subjects": users,
            "classes": stats[0]["classes"], "draws_sha256": DA.fingerprint(draw),
            "paired_counts": dict(zip(names, stats)),
            "replicates_missing_classes": int((classes_present < len(total)).sum()),
            "single_class_replicates": int((classes_present == 1).sum()),
            "missing_class_rule": "macro recall over target classes present in each draw; all draws retained"}
    tasks = {}
    for task in TASKS:
        point = np.sum([w*points[d] for w, d in zip(WEIGHTS[task], TASKS[task])], axis=0)
        sample = np.sum([w*samples[d] for w, d in zip(WEIGHTS[task], TASKS[task])], axis=0)
        resolved = {name: all(convergence.get(name, {}).get(ds) is True for ds in TASKS[task]) for name in names}
        tasks[task] = {**_comparison_summary(point, sample, names, resolved), "weights": dict(zip(TASKS[task], WEIGHTS[task]))}
    return {"datasets": datasets, "tasks": tasks, "reading": classification_reading(tasks),
            "n_bootstrap": n_boot, "seed": seed,
            "uncertainty": "paired subject bootstrap conditional on one pretraining seed, checkpoints and fitted probes; development data; no multiplicity adjustment"}


def _read_pinned_report(pin):
    require(TP.digest(pin.get("sha256")) and sha(pin["path"]) == pin["sha256"], "report pin differs")
    if "bytes" in pin:
        require(Path(pin["path"]).stat().st_size == pin["bytes"], "report size differs")
    result = json.loads(Path(pin["path"]).read_text())
    require(result.get("complete") is True and sha(result["oof"]) == result["oof_sha256"]
            and sha(result["audit"]) == result["audit_sha256"], "report incomplete or OOF/audit pin differs")
    require(jhash(result["numeric_context"]) == result["numeric_sha256"], "numeric signature differs")
    if "fit_audit" in result:
        # The durable log contains successive snapshots. Its last snapshot for
        # every fit must be the record embedded in the completed report.
        latest = {}
        for line in Path(result["audit"]).read_text().splitlines():
            record = json.loads(line)
            latest[(record["dataset"], record["fold"], record["fit_phase"], record["C"])] = record
        require(list(latest.values()) == result["fit_audit"], "persisted/embedded fit audit differs")
    with np.load(result["oof"], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return result, arrays


def read_historical_pool(report_pin, cache_pins, *, expected_plan=None):
    """Reuse pinned T100-G19618 tokenizer aggregate or complete shards, no refit.

    ``cache_pins[dataset]`` supplies path, sha256, subjects and input_sha256 of
    the historical channel_readout cache. Check roles and support against it.
    """
    pins = report_pin if isinstance(report_pin, list) else [report_pin]
    require(pins and len({p["sha256"] for p in pins}) == len(pins), "duplicate/empty historical reports")
    loaded, records = [], []
    for pin in pins:
        report, arrays = _read_pinned_report(pin)
        task = report.get("task")
        require(report.get("protocol") == TP.PROTOCOL and report.get("kind") in ("aggregate", "shard")
                and report.get("arm") == "T100-G" and report.get("step") == 19618
                and task in TASKS and report.get("recipe") == TP.recipe(), "historical probe identity/recipe differs")
        if report["kind"] == "aggregate":
            require(report["weights"] == dict(zip(TASKS[task], WEIGHTS[task])), "historical task weights differ")
        if expected_plan is not None:
            require(report["plan_content_sha256"] == jhash(expected_plan), "historical report plan differs")
        audit = [json.loads(line) for line in Path(report["audit"]).read_text().splitlines()]
        require(report["audit_summary"] == TP.audit_summary(audit), "historical convergence summary differs")
        require(set(report["datasets"]) <= set(TASKS[task]), "foreign historical dataset")
        records.extend(audit)
        loaded.append((report, arrays))
    require(len({(r["task"], r["numeric_sha256"], r["code_sha"], r["plan_sha256"]) for r, _ in loaded}) == 1,
            "mixed historical task/numeric/source contexts")
    task = loaded[0][0]["task"]
    cells, convergence, identities = {}, {}, {}
    for ds in TASKS[task]:
        pin = cache_pins[ds]
        require(sha(pin["path"]) == pin["sha256"], "historical feature pin differs")
        with np.load(pin["path"], allow_pickle=False) as z:
            y, s, subjects = (z[k] for k in ("y", "s", "subjects"))
            require(str(z["input_sha256"]) == pin["input_sha256"] and subjects.tolist() == pin["subjects"],
                    "historical source identity differs")
            cp_sha = str(z["checkpoint_sha256"])
            dim = int(z["encoder_dim"])
            require(dim > 0 and TP.recipe()["width"] == 2*dim, "historical encoder/readout width differs")
            roles = CP.fold_roles(subjects.tolist())
            identities[ds] = {"input_sha256": str(z["input_sha256"]), "splits_sha256": str(z["splits_sha256"]),
                              "roles_sha256": jhash(roles), "checkpoint_sha256": cp_sha,
                              "encoder_dim": dim,
                              "ordered_rows_sha256": DA.fingerprint(np.arange(len(y), dtype=np.int64), y, s)}
        parts = [(r, a) for r, a in loaded if ds in r["datasets"]]
        folds = [fr for r, _ in parts for fr in r["datasets"][ds]["folds"]]
        require(sorted(fr["fold"] for fr in folds) == list(range(4)), "duplicate/missing historical folds")
        a = {k: np.concatenate([a[f"{ds}_{k}"] for _, a in parts]) for k in OOF_KEYS}
        TP.validate_oof(*(a[k] for k in OOF_KEYS))
        _verify_support(a, y, s, roles, range(4))
        fits = [r for r in records if r["dataset"] == ds]
        _verify_fits(fits, folds, roles, pin["sha256"], cp_sha, 2*dim, legacy_refit_marker=True)
        for r, values in parts:
            require(all(f["code_sha"] == r["code_sha"] and f["plan_sha256"] == r["plan_sha256"]
                        and f["dataset"] == ds and f["task"] == task and f["arm"] == "T100-G"
                        for f in fits if f["fold"] in [fr["fold"] for fr in r["datasets"][ds]["folds"]]),
                    "historical fit source/task differs")
            if r["kind"] == "aggregate":
                require(TP.scored(a["y"], a["pred"])[0] == r["datasets"][ds]["balanced_accuracy"], "historical OOF score differs")
        cells[ds], convergence[ds] = a, all(r["converged_by_criterion"] for r in fits)
    require(len(records) == 28*len(TASKS[task]), "extra/missing historical audit fits")
    return {"cells": cells, "convergence": convergence, "identities": identities,
            "reports": [r for r, _ in loaded], "records": records}


def _verify_support(a, y, s, roles, folds):
    expected_all = []
    for fold in folds:
        mt, mv, mh = (np.isin(s, roles[fold][key]) for key in ("train", "val", "held"))
        _, kh = P1.keep_train_classes(y[mt], y[mv], y[mh])
        rows = np.flatnonzero(mh)[kh]
        actual = np.sort(a["row"][a["fold"] == fold])
        require(np.array_equal(actual, rows), "OOF support differs from historical train-class policy")
        expected_all.extend(rows.tolist())
    require(sorted(a["row"].tolist()) == sorted(expected_all), "extra/missing OOF rows")
    require(np.array_equal(a["y"], y[a["row"]]) and np.array_equal(a["s"], s[a["row"]]), "OOF target/subject identity differs")


def _verify_fits(records, fold_reports, roles, cache_sha, checkpoint_sha, width, *, legacy_refit_marker=False):
    require(len(records) == 7*len(fold_reports), "extra/missing audit fits")
    for fr in fold_reports:
        f = fr["fold"]
        require(all(fr[k] == roles[f][k] for k in roles[f]) and fr["roles_sha256"] == jhash(roles[f]),
                "reported subject roles differ")
        fits = [r for r in records if r["fold"] == f]
        candidates = {r["C"]: r for r in fits if r["fit_phase"] == "candidate"}
        refits = [r for r in fits if r["fit_phase"] == "refit"]
        require(len(fits) == 7 and set(candidates) == set(C_GRID) and len(refits) == 1, "candidate grid/refit differs")
        for r in fits:
            solver = LogisticRegression(C=r["C"], max_iter=MAX_ITER, class_weight="balanced", random_state=0, solver="lbfgs", tol=1e-4)
            require(r["solver_params"] == solver.get_params() and r["n_features"] == width and r["dtype"] == "float32"
                    and r["cache_sha256"] == cache_sha and r["checkpoint_sha256"] == checkpoint_sha
                    and r["roles_sha256"] == jhash(roles[f]) and r["scaler_sha256"] == fr["scaler_sha256"]
                    and r["n_train"] == fr["n_train"] and r["max_iter"] == MAX_ITER and r.get("exception") is None,
                    "solver/scaler/source identity differs")
            require(r["converged_by_criterion"] == (r["n_iter"] < MAX_ITER and r["convergence_warnings"] == 0)
                    and r["reached_limit"] == (r["n_iter"] >= MAX_ITER), "convergence evidence differs")
        chosen = max(C_GRID, key=lambda c: (candidates[c]["val_bacc"], -c))
        require(fr["c"] == chosen and refits[0]["C"] == chosen
                and fr["val_bacc"] == candidates[chosen]["val_bacc"]
                and (refits[0]["selected"] is None if legacy_refit_marker else refits[0]["selected"] is True)
                and fr["val_scores"] == {str(c): candidates[c]["val_bacc"] for c in C_GRID}
                and all(candidates[c]["selected"] == (c == chosen) for c in C_GRID), "validation selection differs")


def compare_reports(report_pins, *, cache_pins, historical_report_pins=(), historical_cache_pins=None,
                    historical_baseline_policy="exact", historical_plan=None, out=None, identity=None):
    """Validate and join per-cell fold shards, then run the paired matrix analysis.

    cache_pins maps ``'step/view' -> dataset -> {path,sha256,subjects,...}``;
    new pre/post/pool views may point to the same all-view extraction cache.
    Historical aggregate reports replace only the19618/pool cell.
    """
    require(historical_baseline_policy in ("exact", "declared_historical_cache"), "unregistered historical baseline policy")
    cells, convergence, identities, pending = {}, {}, {}, {}
    for pin in report_pins:
        report, arrays = _read_pinned_report(pin)
        require(report.get("protocol") == PROTOCOL and report.get("kind") == "probe", "wrong probe report")
        name, ds = cell_name(report["step"], report["view"]), report["dataset"]
        require(ds in DATASETS and report["task"] == _task(ds), "report dataset/task differs")
        pending.setdefault((name, ds), []).append((report, arrays))
    for (name, ds), parts in pending.items():
        pin = cache_pins[name][ds]
        _, y, s, meta = load_cache(pin["path"], name.split("/")[1], expected=pin)
        fold_ids, joined, records = [], {k: [] for k in OOF_KEYS}, []
        require(len({p[0]["numeric_sha256"] for p in parts}) == 1, "mixed numeric contexts across fold shards")
        for report, arrays in parts:
            require(report["recipe"] == probe_recipe(meta["view"], meta["encoder_dim"]), "probe recipe differs")
            for key in ("cache_sha256", "checkpoint_sha256", "input_sha256", "splits_sha256", "roles_sha256", "ordered_rows_sha256", "step", "dataset", "task"):
                require(report[key] == meta[key], "report/cache identity differs: " + key)
            folds = report["fold_indices"]
            require(folds == sorted(set(folds)) and set(folds) <= set(range(4)) and len(report["folds"]) == len(folds)
                    and [f["fold"] for f in report["folds"]] == folds, "invalid fold shard coverage")
            TP.validate_oof(*(arrays[k] for k in OOF_KEYS))
            _verify_support(arrays, y, s, meta["roles"], folds)
            _verify_fits(report["fit_audit"], report["folds"], meta["roles"], meta["cache_sha256"], meta["checkpoint_sha256"], report["recipe"]["width"])
            require(report["audit_summary"] == audit_summary(report["fit_audit"]), "convergence summary differs")
            require(report["decision_valid"] == report["audit_summary"]["converged_by_criterion"]
                    and all(all(r[k] == report[k] for k in ("code_sha", "plan_sha256", "dataset", "task", "view", "step"))
                            for r in report["fit_audit"]), "fit source/decision identity differs")
            require(TP.scored(arrays["y"], arrays["pred"])[0] == report["balanced_accuracy"], "reported score differs")
            fold_ids.extend(folds)
            records.extend(report["fit_audit"])
            for k in OOF_KEYS:
                joined[k].append(arrays[k])
        require(sorted(fold_ids) == list(range(4)), "duplicate/missing cell folds")
        cells.setdefault(name, {})[ds] = {k: np.concatenate(v) for k, v in joined.items()}
        convergence.setdefault(name, {})[ds] = all(r["converged_by_criterion"] for r in records)
        identities.setdefault(name, {})[ds] = meta
    for pin in historical_report_pins:
        item = read_historical_pool(pin, historical_cache_pins, expected_plan=historical_plan)
        name = cell_name(19618, "pool")
        for ds in item["cells"]:
            require(ds not in cells.get(name, {}), "duplicate historical pool dataset")
            cells.setdefault(name, {})[ds] = item["cells"][ds]
            convergence.setdefault(name, {})[ds] = item["convergence"][ds]
            identities.setdefault(name, {})[ds] = item["identities"][ds]
    require(set(cells) == {cell_name(step, view) for step in DOSES for view in VIEWS}, "incomplete matrix")
    for ds in DATASETS:
        for key in ("input_sha256", "splits_sha256", "roles_sha256", "ordered_rows_sha256"):
            require(len({identities[name][ds][key] for name in cells}) == 1, "cross-cell input identity differs: " + key)
        if historical_report_pins:
            for view in ("pre", "post"):
                meta = identities[cell_name(19618, view)][ds]
                require(meta["historical_pool_parity"] is True and meta["reference_cache_sha256"] == historical_cache_pins[ds]["sha256"],
                        "historical pool baseline lacks same-checkpoint parity evidence")
                require(meta["historical_pool_exact_equal"] is True or historical_baseline_policy == "declared_historical_cache",
                        "tolerance-only parity cannot reuse OOF as identical new features; reviewed declared historical baseline required")
    for step in DOSES:
        require(len({identities[cell_name(step, view)][ds]["checkpoint_sha256"] for view in VIEWS for ds in DATASETS}) == 1,
                "mixed checkpoints within dose")
        require(len({identities[cell_name(step, view)][ds]["encoder_dim"] for view in VIEWS for ds in DATASETS}) == 1,
                "unmatched representation width within dose")
    result = {"protocol": PROTOCOL, "kind": "comparison", "complete": True, **(identity or {}),
              **compare_arrays(cells, convergence=convergence), "convergence": convergence,
              "reports": list(report_pins), "historical_reports": list(historical_report_pins),
              "historical_baseline_policy": historical_baseline_policy,
              "historical_baseline_scope": "pool19618 is the pinned historical feature cache and its fitted OOF; tolerance parity alone does not establish identical new probe predictions"}
    if out is not None:
        TP.fresh(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        TP.write_json(out, result)
    return result


def benchmark_cache(path, view, *, expected, max_rows=None):
    """Time all C candidates on fold0 TRAIN only; no validation/held scoring.

    By default use the complete fold0 train. An explicit smaller deterministic
    sample is diagnostic only and cannot satisfy the runtime execution gate.
    """
    start = time.perf_counter()
    x, y, s, meta = load_cache(path, view, expected=expected)
    load_s = time.perf_counter()-start
    require(max_rows is None or type(max_rows) is int and max_rows >= 2, "invalid benchmark row cap")
    train_rows = np.flatnonzero(np.isin(s, meta["roles"][0]["train"]))
    rows = train_rows if max_rows is None else train_rows[np.linspace(0, len(train_rows)-1, min(max_rows, len(train_rows)), dtype=np.int64)]
    require(len(np.unique(y[rows])) >= 2, "benchmark train sample has fewer than two classes")
    start = time.perf_counter()
    train = StandardScaler().fit_transform(x[rows])
    scale_s = time.perf_counter()-start
    audit = TP.AuditLog()
    base = {"fold": 0, "rows_sha256": DA.fingerprint(rows), "n_train": len(rows),
            "n_features": x.shape[1], "dtype": str(train.dtype), "max_iter": MAX_ITER}
    for c in C_GRID:
        _fit(train, y[rows], c, audit, base, "benchmark_train_only")
    return {"protocol": PROTOCOL, "kind": "benchmark", "dataset": meta["dataset"], "view": view,
            "step": meta["step"], "cache_sha256": meta["cache_sha256"], "load_s": load_s, "scale_s": scale_s,
            "sample_rows": len(rows), "sample_classes": len(np.unique(y[rows])), "width": x.shape[1],
            "full_train_rows": len(train_rows), "full_train_classes": len(np.unique(y[train_rows])),
            "fold_train_rows": [int(np.isin(s, r["train"]).sum()) for r in meta["roles"]],
            "fold_train_classes": [len(np.unique(y[np.isin(s, r["train"])])) for r in meta["roles"]],
            "n_rows": len(y), "encoder_dim": meta["encoder_dim"], "input_sha256": meta["input_sha256"],
            "splits_sha256": meta["splits_sha256"], "roles_sha256": meta["roles_sha256"],
            "ordered_rows_sha256": meta["ordered_rows_sha256"], "checkpoint_sha256": meta["checkpoint_sha256"],
            "full_fold0_train": bool(np.array_equal(rows, train_rows)),
            "rows_sha256": DA.fingerprint(rows), "subjects": sorted(set(s[rows].tolist())),
            "fold0_train_subjects": meta["roles"][0]["train"], "fits": list(audit),
            "numeric_context": TP.numeric_context(), "max_iter": MAX_ITER,
            "timing_only": True, "validation_or_held_scored": False}


def project_full_matrix(benchmarks, cells, *, safety_factor=2.0, fixed_seconds=600.0):
    """Explicit conservative planning estimate; not a walltime upper bound.

    Each cell specifies dataset, view, train_rows (four folds), classes (four
    folds), and width. Scale the worst measured seconds/iteration to every
    fold's rows × width × classes, charge all seven fits at MAX_ITER, double
    that cost for data/scaling/predictions, then apply >=2 safety. Solver and
    hardware differences still require review; this helper never submits.
    """
    require(benchmarks and cells and safety_factor >= 2 and fixed_seconds >= 0, "invalid conservative projection")
    rates = []
    for bench in benchmarks:
        require(bench["timing_only"] is True and bench["validation_or_held_scored"] is False
                and bench["max_iter"] == MAX_ITER and len(bench["fits"]) == len(C_GRID), "invalid train-only benchmark")
        for fit in bench["fits"]:
            require(fit["exception"] is None and fit["elapsed_s"] > 0 and fit["n_iter"] > 0, "invalid measured timing")
            rates.append(fit["elapsed_s"] / fit["n_iter"] / (bench["sample_rows"]*bench["width"]*bench["sample_classes"]))
    rate = max(rates)
    estimates = []
    for cell in cells:
        require(cell["dataset"] in DATASETS and cell["view"] in VIEWS and len(cell["train_rows"]) == len(cell["classes"]) == 4
                and cell["width"] > 0 and min(cell["train_rows"]) > 0 and min(cell["classes"]) >= 2, "invalid projection cell")
        estimate = rate * sum(n*cell["width"]*k for n, k in zip(cell["train_rows"], cell["classes"])) * MAX_ITER * 7 * 2 * safety_factor
        estimates.append({**cell, "projected_seconds": float(estimate)})
    total = sum(c["projected_seconds"] for c in estimates) + fixed_seconds
    return {"cells": estimates, "projected_seconds": total, "projected_core_hours_at_8cpu": total*8/3600,
            "safety_factor": safety_factor, "fixed_seconds": fixed_seconds, "max_iter_charged_per_fit": MAX_ITER,
            "fits_per_fold": 7, "data_and_prediction_multiplier": 2,
            "method": "worst measured seconds/iteration/(rows*width*classes), every candidate/refit charged12000iterations",
            "limitation": "conservative planning extrapolation, not a runtime guarantee or permission to launch; source, numeric environment and resource review required"}
