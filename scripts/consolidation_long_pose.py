"""Frozen long-dose pose extraction/readers and paired descriptive inference.

The controller owns artifact pins and runtime provenance. This library delegates
all raw preprocessing/selection to the original pose harness, and delegates the
complete ridge policy to the converged tokenizer reader. The trained post head
is a separate reader, never a capacity-matched depth contrast.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import pose_dynamics as D
import tokenizer_pose_readout as R
from wearusfm.harness import backbone_depth as BD, pose_eval as PE

PROTOCOL = "consolidation-long-pose-20261105-v1"
DOSES = (19618, 39237, 49046)
READERS = ("pre_ridge", "post_ridge", "trained_head")
DEPTHS = ("pre", "post")
DOSE_CONTRASTS = ((49046, 19618), (39237, 19618), (49046, 39237))
OFFSET, N_BOOT, SEED = R.OFFSET, R.N_BOOT, R.SEED
require = D.require


def recipe():
    ridge = R.recipe().copy()
    ridge.pop("dim")  # Shape follows the encoder; the campaign controller pins D.
    return {"protocol": PROTOCOL, "doses": list(DOSES), "readers": list(READERS),
            "extraction": "original pose_eval.extract ModelInputs; one unmasked FP32 teacher forward; same-forward pre/post",
            "cache": "pre/post mean_K FP16; trained post-head predictions FP32; original targets/masks unchanged",
            "ridge": ridge, "head": "existing trained post pose head; separately identified, no supervised updates",
            "support": "common crop6 central target mask; original full-window pose_var threshold",
            "e2": "historical full-window pairing and std exclusions; errors on intersection of crop6 masks",
            "bootstrap": {"unit": "user", "replicates": N_BOOT, "seed": SEED,
                          "relative_improvement": "100 * (1 - pooled_E_dyn_candidate / pooled_E_dyn_control) in each draw"},
            "practical_relative_improvement_percent": 2.0}


def select_recordings(csv, processed_root, split):
    require(split in ("train", "val"), "expected train or val split")
    return PE.recordings_for(Path(csv), Path(processed_root), split,
                             n_users=40 if split == "train" else None,
                             n_recordings=200 if split == "train" else None,
                             seed=PE.SEED_WINDOWS)


class _EncoderTap:
    """Observe the original encoding without replacing any model module."""
    def __init__(self, model):
        self.model, self.encoding = model, None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode(self, inp, mask):
        self.encoding = self.model.encode(inp, mask)
        return self.encoding


class _PoseDepthAdapter:
    def __init__(self, model):
        self.model, self.pre, self.valid = model, [], []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def eval(self):
        self.model.eval()
        return self

    def encode(self, inp, mask):
        require(mask is None and not torch.is_grad_enabled(), "pose extraction requires unmasked gradient-free forward")
        tap = _EncoderTap(self.model)
        views = BD.depth_readouts(tap, inp)
        # Returning the actual encoding keeps PE.extract's historical latent
        # mean and trained head path entirely unchanged.
        historical = tap.encoding.z.mean(dim=2).float()
        require(torch.equal(views["sequences"]["post"], historical), "same-forward post mean differs from historical pose readout")
        self.pre.extend(views["sequences"]["pre"].cpu().numpy().astype(np.float16))
        self.valid.extend(views["valid"].cpu().numpy().astype(bool))
        return tap.encoding


@torch.inference_mode()
def extract_pose(model, recordings, pose_store, ev, reference=None, selection=None, log=print):
    """Run the ORIGINAL loader, filtering, scales, windows, target grid and head.

    No classification PreparedWindows or averaging of temporal crops is used.
    Pass a pinned target-only reference and the historical count selection at
    runtime. Explicit FP32 preserves equality with historical PE.extract.
    """
    require(getattr(model, "pose_head", None) is not None, "long pose evaluation requires the historical trained post head")
    require(ev.n_windows == 10 and ev.window_s == 4. and ev.seed == PE.SEED_WINDOWS,
            "pose extraction window selection differs from historical recipe")
    adapter = _PoseDepthAdapter(model)
    with torch.autocast(device_type=torch.device(ev.device).type, enabled=False):
        data = PE.extract(adapter, recordings, pose_store, ev, log=log)
    n, patches, dim = data["lat"].shape
    require(len(adapter.pre) == len(adapter.valid) == n, "same-forward windows differ")
    data["pre"] = np.zeros((n, patches, dim), dtype=np.float16)
    data["key_time_valid"] = np.zeros((n, patches), dtype=bool)
    for i, meta in enumerate(data["meta"]):
        length = meta["n_patches"]
        require(len(adapter.pre[i]) >= length and len(adapter.valid[i]) >= length, "pose grid exceeds encoded sequence")
        data["pre"][i, :length] = adapter.pre[i][:length]
        data["key_time_valid"][i, :length] = adapter.valid[i][:length]
    validate_cache(data, reference, selection)
    return data


def _target_view(data):
    result = {key: data[key] for key in ("target", "mask", "std", "pose_var", "meta", "n_recordings")}
    result["lengths"] = np.asarray([m["n_patches"] for m in data["meta"]], dtype=np.int64)
    result["central_mask"] = R.T.central_mask(result["mask"], result["lengths"], offset=OFFSET)
    return result


def validate_cache(data, reference=None, selection=None):
    y, mask, std = (data[k] for k in ("target", "mask", "std"))
    require(y.ndim == 3 and len(y) > 0 and y.dtype == np.float32 and mask.shape == y.shape and mask.dtype == bool,
            "invalid original target/mask geometry or precision")
    require(len(data["meta"]) == len(y) and all(isinstance(m.get("user"), str) and m["user"] and
            type(m.get("n_patches")) is int and 2*OFFSET < m["n_patches"] <= y.shape[1] for m in data["meta"]),
            "invalid original pose metadata")
    target = _target_view(data)
    available = np.arange(y.shape[1])[None, :] < target["lengths"][:, None]
    valid = data["key_time_valid"]
    require(valid.dtype == bool and valid.shape == available.shape and np.array_equal(valid, available),
            "key_time_valid differs from historical real-patch support; cannot silently change target masks")
    require(not np.any(mask & ~available[..., None]) and np.isfinite(y[mask]).all(), "invalid supported original targets")
    require(std.dtype == np.float32 and std.shape == (len(y), y.shape[2]) and np.isfinite(std).all() and (std > 0).all(),
            "invalid original pose scales")
    pv = data["pose_var"]
    require(pv.dtype == np.float32 and pv.shape == (len(y),) and not np.isinf(pv).any() and (pv[np.isfinite(pv)] >= 0).all(),
            "invalid original pose variance")
    require(type(data["n_recordings"]) is int and data["n_recordings"] > 0, "invalid recording count")
    require(data["has_head"] is True and type(data["dim"]) is int and data["dim"] > 0, "expected trained head and encoder width")
    for key in ("pre", "lat"):
        a = data[key]
        require(a.dtype == np.float16 and a.shape == (*y.shape[:2], data["dim"]) and np.isfinite(a[available]).all(),
                "invalid depth sequence: " + key)
    require(data["pred"].shape == y.shape and data["pred"].dtype == np.float32 and np.isfinite(data["pred"][mask]).all(),
            "invalid historical trained-head prediction")
    if selection is not None:
        R.check_selection(target, selection)
    if reference is not None:
        require(R.target_signature(target) == R.target_signature(_target_view(reference)) and
                data["n_recordings"] == reference["n_recordings"], "historical metadata/target/mask/std/pose_var mismatch")
    return {"targets": R.target_signature(target), "key_time_valid_sha256": R.arrays_hash(valid),
            "n_windows": len(y), "n_users": len({m["user"] for m in data["meta"]}),
            "n_recordings": data["n_recordings"], "dim": data["dim"]}


def save_cache(path, data):
    validate_cache(data)
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **{k: data[k] for k in
                            ("pre", "lat", "key_time_valid", "pred", "target", "mask", "std", "pose_var", "has_head", "dim", "n_recordings")},
                            meta_json=np.asarray(json.dumps(data["meta"])), config_json=np.asarray(json.dumps(data["config"])))


def read_cache(path, reference=None, selection=None):
    before = R.T.file_state(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("pre", "lat", "key_time_valid", "pred", "target", "mask", "std", "pose_var")}
        data.update(meta=json.loads(str(archive["meta_json"])), config=json.loads(str(archive["config_json"])),
                    has_head=bool(archive["has_head"]), dim=int(archive["dim"]), n_recordings=int(archive["n_recordings"]))
    require(R.T.file_state(path) == before, "pose cache changed while reading")
    validate_cache(data, reference, selection)
    return data


def view_data(data, depth):
    require(depth in DEPTHS, "expected pre/post depth")
    return {**_target_view(data), "lat": data["pre" if depth == "pre" else "lat"]}


def fit_reader(train, depth, expected_folds=None):
    """Exact converged train-only group CV, normalization, grid and refit."""
    return R.fit_reader(view_data(train, depth), expected_folds)


def predict_reader(data, depth, weights, bias):
    require(depth in DEPTHS, "expected pre/post depth")
    lengths = np.asarray([m["n_patches"] for m in data["meta"]], dtype=np.int64)
    return R.predict(data["pre" if depth == "pre" else "lat"], lengths, weights, bias)


def benchmark_train(train, depth="pre"):
    """Timing only; caller supplies train, never opens held cache here."""
    return R.benchmark_train(view_data(train, depth))


def project_fit_seconds(benchmark, *, n_readers=6, safety_factor=2., fixed_seconds=120.):
    """Conservative train-timing projection, not authorization to launch fits.

    Historical benchmark measures sufficient statistics and all three cached
    eigensystems. Each of at most 13 alphas has two fold scores and solves; count
    every score/solve pair as another full three-eigensystem timing, then refit.
    Disk I/O and held prediction need their own explicitly measured allowance.
    """
    stats = benchmark["statistics_seconds"]
    eig = benchmark["three_eigensystems_seconds"]
    require(all(np.isfinite(x) and x >= 0 for x in (stats, eig, fixed_seconds)) and
            type(n_readers) is int and n_readers > 0 and safety_factor >= 1., "invalid fit projection inputs")
    return {"fit_seconds": float(safety_factor*n_readers*(stats+27*eig)+fixed_seconds),
            "n_readers": n_readers, "safety_factor": safety_factor, "fixed_seconds": fixed_seconds,
            "formula": "safety_factor * n_readers * (statistics_seconds + 27 * three_eigensystems_seconds) + fixed_seconds",
            "complete_resource_plan": False, "excludes": ["cache validation and I/O", "held prediction and aggregation"]}


def e2_support(data):
    """Freeze historical partner choice before cropping either target mask."""
    partner, info = PE.misaligned_pairs(data["meta"], data["target"], data["mask"])
    ok = partner >= 0
    ix = np.maximum(partner, 0)
    mismatch = ok & ~np.all(np.isclose(data["std"], data["std"][ix]), axis=1)
    partner[mismatch] = -1
    ok, ix = partner >= 0, np.maximum(partner, 0)
    target = _target_view(data)
    common = target["central_mask"] & target["central_mask"][ix] & ok[:, None, None]
    dynamic = np.isfinite(data["pose_var"]) & (data["pose_var"] >= R.THRESHOLD)
    common &= dynamic[:, None, None]
    info.update(n_excluded_std_mismatch=int(mismatch.sum()), n_kept=int(ok.sum()))
    # Strict JSON also when there are no candidate pairs.
    if not np.isfinite(info["distance_threshold"]):
        info["distance_threshold"] = None
    return {"partner": partner, "mask": common, "pairing": info,
            "signature": {"partners_sha256": R.arrays_hash(partner), "common_mask_sha256": R.arrays_hash(common),
                          "dynamic_threshold": R.THRESHOLD, "pairing": info}}


def _e2_summary(data, prediction, support):
    ix = np.maximum(support["partner"], 0)
    mask = support["mask"]
    target = data["target"][:, OFFSET:-OFFSET]
    counts = mask.sum(axis=(1, 2))
    y, ym, p = (np.where(mask, a, 0).astype(np.float64) for a in (target, target[ix], prediction))
    scale = data["std"][:, None, :]
    sums = {"aligned_sum_rad": (np.abs(p-y)*scale).sum(axis=(1, 2)),
            "misaligned_sum_rad": (np.abs(p-ym)*scale).sum(axis=(1, 2)),
            "constant_sum_rad": (np.abs(ym)*scale).sum(axis=(1, 2))}
    users = np.asarray([m["user"] for m in data["meta"]])
    per_user = {}
    for user in sorted(set(users[counts > 0])):
        pick = (users == user) & (counts > 0)
        per_user[user] = {"n_windows": int(pick.sum()), "n_valid_elements": int(counts[pick].sum()),
                          **{k: float(v[pick].sum()) for k, v in sums.items()}}
    count = int(counts.sum())
    total = {k: float(v.sum()) for k, v in sums.items()}
    return {"support": support["signature"], "per_user": per_user,
            "n_windows": int((counts > 0).sum()), "n_users": len(per_user), "n_valid_elements": count,
            "mae_aligned": total["aligned_sum_rad"]/count if count else None,
            "mae_misaligned": total["misaligned_sum_rad"]/count if count else None,
            "gap_mae": (total["misaligned_sum_rad"]-total["aligned_sum_rad"])/count if count else None,
            "misaligned_constant_ratio": total["misaligned_sum_rad"]/total["constant_sum_rad"] if total["constant_sum_rad"] > 0 else None}


def summarize(data, prediction, *, support=None):
    """The converged crop6 dynamics plus historical-pair E2 on common support."""
    target = _target_view(data)
    result = R.summarize(target, prediction)
    support = e2_support(data) if support is None else support
    require(prediction.shape == target["central_mask"].shape and np.isfinite(prediction[target["central_mask"]]).all(),
            "invalid pose prediction on common support")
    result["E2"] = _e2_summary(data, prediction, support)
    return result


def evaluate_reader(train, held, depth, expected_folds=None):
    """Pure convenience API; a controller should fit before opening held data."""
    require({m["user"] for m in train["meta"]}.isdisjoint({m["user"] for m in held["meta"]}), "train/held user leakage")
    weights, bias, fit = fit_reader(train, depth, expected_folds)
    started = perf_counter()
    prediction = predict_reader(held, depth, weights, bias)
    return {"weights": weights, "bias": bias, "fit": fit, "prediction": prediction,
            "prediction_seconds": perf_counter()-started, "analysis": summarize(held, prediction)}


def _interval(point, draws):
    point = float(point) if np.isfinite(point) else None
    if draws is None or not np.isfinite(draws).all():
        return {"point": point, "ci95": None, "evaluable": False,
                "reason": "undefined statistic in at least one subject draw; no replicas discarded"}
    return {"point": point, "ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "evaluable": True}


def _metrics(sums, count):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"E_dyn": sums["centered"]/sums["vary"], "level_mse": sums["mean_error"]/count,
                "total_mse": sums["total"]/count, "predicted_variance": sums["varp"]/count,
                "target_variance": sums["vary"]/count, "variance_ratio": sums["varp"]/sums["vary"],
                "correlation": sums["cov"]/np.sqrt(sums["varp"]*sums["vary"])}


def _bootstrap_group(analyses, group, n_boot, seed):
    rows = {key: value["groups"][group]["centered"]["per_user"] for key, value in analyses.items()}
    base = next(iter(rows.values()))
    users = sorted(base)
    require(all(sorted(row) == users for row in rows.values()), "paired users missing; no intersection comparison")
    for row in rows.values():
        for user in users:
            require(all(row[user][k] == base[user][k] for k in ("n_windows", "n_window_joints", "n_valid_elements")),
                    "paired centered support counts differ")
            require(all(row[user]["sums"][k] == base[user]["sums"][k] for k in ("vary", "baseline_total", "mean_target_squared")),
                    "paired target denominator differs")
    if len(users) < 2:
        return {"evaluable": False, "users": users, "reason": "fewer than two supported users"}, {}, {}
    draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
    count = np.asarray([base[u]["n_valid_elements"] for u in users], dtype=np.float64)
    point, samples = {}, {}
    for cell, row in rows.items():
        sums = {k: np.asarray([row[u]["sums"][k] for u in users]) for k in D.CENTERED_FIELDS}
        point[cell] = _metrics({k: v.sum() for k, v in sums.items()}, count.sum())
        samples[cell] = _metrics({k: v[draws].sum(1) for k, v in sums.items()}, count[draws].sum(1))
    return {"evaluable": True, "users": users, "n_bootstrap": n_boot, "seed": seed,
            "draws_sha256": R.arrays_hash(draws)}, point, samples


def _bootstrap_e2(analyses, n_boot, seed):
    rows = {key: value["E2"]["per_user"] for key, value in analyses.items()}
    base = next(iter(rows.values()))
    users = sorted(base)
    require(all(sorted(row) == users for row in rows.values()), "paired E2 users missing")
    for row in rows.values():
        for user in users:
            require(all(row[user][k] == base[user][k] for k in ("n_windows", "n_valid_elements", "constant_sum_rad")),
                    "paired E2 support/denominator differs")
    if len(users) < 2:
        return {"evaluable": False, "users": users, "reason": "fewer than two users on E2 common support"}, {}, {}
    draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
    point, samples = {}, {}
    def metrics(values):
        with np.errstate(divide="ignore", invalid="ignore"):
            return {"mae_aligned": values["aligned_sum_rad"]/values["n_valid_elements"],
                    "mae_misaligned": values["misaligned_sum_rad"]/values["n_valid_elements"],
                    "gap_mae": (values["misaligned_sum_rad"]-values["aligned_sum_rad"])/values["n_valid_elements"],
                    "misaligned_constant_ratio": values["misaligned_sum_rad"]/values["constant_sum_rad"]}
    for cell, row in rows.items():
        arrays = {k: np.asarray([row[u][k] for u in users]) for k in
                  ("aligned_sum_rad", "misaligned_sum_rad", "constant_sum_rad", "n_valid_elements")}
        point[cell] = metrics({k: v.sum() for k, v in arrays.items()})
        samples[cell] = metrics({k: v[draws].sum(1) for k, v in arrays.items()})
    return {"evaluable": True, "users": users, "n_bootstrap": n_boot, "seed": seed,
            "draws_sha256": R.arrays_hash(draws)}, point, samples


def practical_assessment(relative):
    if not relative["evaluable"] or relative["point"] is None:
        return "inconclusive"
    point, (low, high) = relative["point"], relative["ci95"]
    if point >= 2. and low > 0:
        return "practical_benefit"
    if point <= -2. and high < 0:
        return "practical_worsening"
    return "small" if abs(point) < 2. and (low > 0 or high < 0) else "inconclusive"


def _contrast(candidate, control, point, samples):
    metrics = {k: _interval(point[candidate][k]-point[control][k], samples[candidate][k]-samples[control][k])
               for k in point[candidate]}
    result = {"candidate": candidate, "control": control, "delta": metrics}
    if "E_dyn" in point[candidate]:
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = _interval(100*(1-point[candidate]["E_dyn"]/point[control]["E_dyn"]),
                                 100*(1-samples[candidate]["E_dyn"]/samples[control]["E_dyn"]))
        result.update(relative_improvement_percent=relative, assessment=practical_assessment(relative))
    return result


def _clear_regressions(dynamic, e2):
    flags = []
    for name, direction in (("level_mse", 1), ("correlation", -1)):
        metric = dynamic.get("delta", {}).get(name, {})
        ci = metric.get("ci95")
        if ci is not None and (ci[0] > 0 if direction == 1 else ci[1] < 0):
            flags.append(name)
    # E2's aligned/disaligned gap measures alignment sensitivity on precisely
    # the same support. No new practical cutoff is inferred from results.
    ci = e2.get("delta", {}).get("gap_mae", {}).get("ci95")
    if ci is not None and ci[1] < 0:
        flags.append("E2_gap_mae")
    return flags


def select_reference(contrasts, reader):
    def row(new, old):
        return contrasts[f"{new}/{reader}_minus_{old}/{reader}"]
    mid, end, increment = row(39237, 19618), row(49046, 19618), row(49046, 39237)
    good = lambda x: x["groups"]["dynamic"].get("assessment") == "practical_benefit"
    candidate = 19618
    if good(mid):
        candidate = 49046 if good(end) and good(increment) else 39237
    elif good(end):
        candidate = 49046
    relevant = [row(candidate, 19618)] if candidate != 19618 else []
    if candidate == 49046 and good(mid):
        relevant.append(increment)
    regressions = sorted({f for contrast in relevant for f in contrast["clear_regressions"]})
    return {"reader": reader, "candidate_from_E_dyn": candidate,
            "selected_reference": None if regressions else candidate,
            "status": "defer_tradeoff" if regressions else "retain_reference" if candidate == 19618 else "exploratory_candidate",
            "clear_regressions": regressions, "scope": "single seed, development users; no general FM promotion"}


def aggregate(cells, n_boot=N_BOOT, seed=SEED):
    """All nine cells required. No arm, dose, depth or subject intersections."""
    require(type(n_boot) is int and n_boot > 0 and type(seed) is int, "invalid bootstrap recipe")
    expected = {f"{dose}/{reader}" for dose in DOSES for reader in READERS}
    require(set(cells) == expected, "expected all nine long-pose cells")
    analyses, checkpoints = {}, {}
    for key, cell in cells.items():
        dose_text, reader = key.split("/")
        dose = int(dose_text)
        require(cell.get("complete") is True and cell.get("which") == "teacher" and
                cell.get("checkpoint_step") == dose and cell.get("reader") == reader and D.is_sha(cell.get("checkpoint_sha256")),
                "incomplete or mismatched long-pose cell identity")
        checkpoints.setdefault(dose, set()).add(cell["checkpoint_sha256"])
        if reader.endswith("_ridge"):
            require(cell.get("fit", {}).get("selection", {}).get("boundary_unresolved") is False,
                    "unresolved or absent train-only ridge selection")
        analyses[key] = cell["analysis"]
    require(all(len(v) == 1 for v in checkpoints.values()) and len(set.union(*checkpoints.values())) == len(DOSES),
            "mixed checkpoint within dose or reused checkpoint across doses")
    base = next(iter(analyses.values()))
    for analysis in analyses.values():
        require(all(analysis[k] == base[k] for k in ("support_sha256", "targets_sha256", "dynamic_threshold")),
                "paired original target/support differs")
        require(analysis["E2"]["support"] == base["E2"]["support"], "paired E2 partners/common support differs")
    pairs = [(f"{new}/{reader}", f"{old}/{reader}", "dose") for reader in READERS for new, old in DOSE_CONTRASTS]
    pairs.extend((f"{dose}/post_ridge", f"{dose}/pre_ridge", "depth") for dose in DOSES)
    contrasts = {f"{a}_minus_{b}": {"candidate": a, "control": b, "kind": kind,
                 "primary_dose_contrast": kind == "dose" and a in ("49046/trained_head", "49046/post_ridge") and b.startswith("19618/"),
                 "groups": {}} for a, b, kind in pairs}
    output_groups, output_cells = {}, {k: {"groups": {}} for k in cells}
    for group in ("dynamic", "all", "static"):
        details, point, samples = _bootstrap_group(analyses, group, n_boot, seed)
        output_groups[group] = details
        for cell in cells:
            output_cells[cell]["groups"][group] = ({k: _interval(v, samples[cell][k]) for k, v in point[cell].items()}
                                                  if details["evaluable"] else details.copy())
        for a, b, _ in pairs:
            contrasts[f"{a}_minus_{b}"]["groups"][group] = _contrast(a, b, point, samples) if details["evaluable"] else details.copy()
    details, point, samples = _bootstrap_e2(analyses, n_boot, seed)
    for cell in cells:
        output_cells[cell]["E2"] = ({k: _interval(v, samples[cell][k]) for k, v in point[cell].items()}
                                   if details["evaluable"] else details.copy())
    for a, b, _ in pairs:
        contrast = contrasts[f"{a}_minus_{b}"]
        contrast["E2"] = _contrast(a, b, point, samples) if details["evaluable"] else details.copy()
        contrast["clear_regressions"] = _clear_regressions(contrast["groups"]["dynamic"], contrast["E2"])
    references = {reader: select_reference(contrasts, reader) for reader in READERS}
    return {"protocol": PROTOCOL, "complete": True, "recipe": recipe(), "bootstrap_groups": output_groups,
            "bootstrap_E2": details, "cells": output_cells, "contrasts": contrasts,
            "references": references,
            "reader_policy": "Separate primary-dose contrasts in post_ridge and trained_head panels; pre_ridge doses secondary, post-minus-pre ridge diagnostic. No single primary pose reader/reference selected; protocol clarification requires external Claude review before launch.",
            "scope": "Frozen teacher; same crop6 held-user support. Trained head is historical and separate from equal-width ridge depth diagnostics. Descriptive development IC95, no multiplicity adjustment or between-training uncertainty."}
