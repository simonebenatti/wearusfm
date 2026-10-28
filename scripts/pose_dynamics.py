"""Post-hoc B5000 posture/temporal MSE decomposition from the existing NPZ.

Only numpy and the standard library; no model, trainer, fitting or prediction.
The latent array is never loaded. MSE is standardized; std is used only for the
E1 MAE-in-radians reproduction. All moments use float64 and population ddof=0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SEED, N_BOOT = 20261009, 1000
CACHE_FIELDS = ("pred", "target", "mask", "std", "pose_var", "meta_json", "config_json", "has_head", "n_recordings")
TOTAL_FIELDS = ("total", "baseline_total", "mean_error")
CENTERED_FIELDS = ("total", "baseline_total", "mean_error", "mean_target_squared", "varp", "vary", "cov",
                   "centered", "gain_mean", "gain_centered", "total_gain")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_cache(path):
    """NPZ indexing decompresses only requested fields, never `lat`."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in CACHE_FIELDS}
    data["meta"] = json.loads(str(data.pop("meta_json")))
    data["config"] = json.loads(str(data.pop("config_json")))
    require(data["has_head"].shape == () and data["has_head"].dtype.kind == "b" and bool(data["has_head"]), "cache needs a trained head")
    require(data["n_recordings"].shape == () and data["n_recordings"].dtype.kind in "iu", "invalid recording count")
    data["has_head"], data["n_recordings"] = True, int(data["n_recordings"])
    p, y, mask = data["pred"], data["target"], data["mask"]
    require(p.ndim == 3 and p.shape == y.shape == mask.shape and p.shape[2] == 20 and p.shape[0] > 0,
            "pred/target/mask must have equal nonempty (window,time,20) shapes")
    require(p.dtype.kind == y.dtype.kind == "f" and mask.dtype.kind == "b", "prediction/target must be floating point and mask boolean")
    require(data["std"].shape == (p.shape[0], p.shape[2]) and data["pose_var"].shape == (p.shape[0],), "std/pose_var shape mismatch")
    require(np.isfinite(data["std"]).all() and (data["std"] > 0).all(), "std must be finite and positive")
    require(not np.isinf(data["pose_var"]).any(), "infinite cached pose variance")
    require((data["pose_var"][np.isfinite(data["pose_var"])] >= 0).all(), "negative cached pose variance")
    require(isinstance(data["meta"], list) and len(data["meta"]) == p.shape[0], "metadata/window count mismatch")
    require(all(isinstance(m, dict) and isinstance(m.get("user"), str) and m["user"] for m in data["meta"]), "invalid window user metadata")
    require(isinstance(data["config"], dict), "invalid cache config")
    return data


def check_provenance(data, cache, expected_sha, provenance):
    require(provenance.get("mode") == "report", "expected E1/E2 report preflight provenance")
    require(provenance.get("cache_sha256") == expected_sha, "report provenance/cache SHA mismatch")
    require(Path(provenance["cache"]).resolve() == Path(cache).resolve(), "report provenance/cache path mismatch")
    require(provenance.get("cache_bytes") == Path(cache).stat().st_size, "report provenance/cache size mismatch")
    require(provenance.get("checkpoint_step") == 5000 and is_sha(provenance.get("checkpoint_sha256")), "expected identified B5000 checkpoint")
    require(provenance.get("which") == "student" and provenance.get("split") == "val", "expected student/val provenance")
    cfg = data["config"]
    require(is_sha(provenance.get("pose_stats_sha256")), "missing pose statistics SHA")
    for key, value in {"which": "student", "split": "val", "checkpoint": provenance["checkpoint"],
                       "code_sha": provenance["execution_code_sha"], "pose_stats_sha256": provenance["pose_stats_sha256"],
                       "checkpoint_pose_stats_sha256": provenance["pose_stats_sha256"]}.items():
        require(cfg.get(key) == value, "cache/provenance mismatch: " + key)
    require(data["n_recordings"] == provenance["n_recordings_requested"], "requested recording count mismatch")
    require(len(data["meta"]) == provenance["n_windows"], "window count mismatch")
    valid = data["mask"].any(axis=(1, 2))
    require(int(valid.sum()) == provenance["n_windows_with_valid_pose"], "valid window count mismatch")
    users = {m["user"] for m, ok in zip(data["meta"], valid) if ok}
    require(len(users) == provenance["n_users_with_valid_pose"], "valid user count mismatch")
    records = {m["recording"] for m in data["meta"]}
    require(len(records) == provenance["n_recordings_with_windows"], "observed recording count mismatch")


def window_joint_moments(pred, target, mask):
    """Per window/joint, on precisely the same valid temporal indices.

    Unobserved entries are zeroed BEFORE arithmetic (NaN outside support cannot
    enter a sum). Empty joints return zero placeholders and are excluded later.
    Centering is explicit rather than subtracting two large raw second moments.
    """
    require(pred.shape == target.shape == mask.shape and pred.ndim == 3 and mask.dtype.kind == "b", "moment input shapes/types")
    require(np.isfinite(pred[mask]).all() and np.isfinite(target[mask]).all(), "nonfinite prediction/target on valid support")
    p = np.where(mask, pred, 0).astype(np.float64)
    y = np.where(mask, target, 0).astype(np.float64)
    n = mask.sum(axis=1, dtype=np.int64)
    den = np.maximum(n, 1)
    meanp, meany = p.sum(axis=1) / den, y.sum(axis=1) / den
    cp = np.where(mask, p - meanp[:, None, :], 0)
    cy = np.where(mask, y - meany[:, None, :], 0)
    varp, vary, cov = (a.sum(axis=1) / den for a in (cp * cp, cy * cy, cp * cy))
    total = ((p - y) ** 2).sum(axis=1) / den
    mean_error, meany2 = (meanp - meany) ** 2, meany ** 2
    centered_formula = varp + vary - 2 * cov
    centered_direct = ((cp - cy) ** 2).sum(axis=1) / den
    # Round-off can make exact-zero residual variance slightly negative.
    scale = np.maximum(1.0, varp + vary + 2 * np.abs(cov))
    require((centered_formula >= -1e-10 * scale).all(), "negative centered MSE beyond numerical tolerance")
    centered = np.maximum(centered_formula, 0)
    baseline = (y * y).sum(axis=1) / den
    gain_mean, gain_centered = meany2 - mean_error, vary - centered
    total_gain = baseline - total
    residuals = {"centered_formula": np.abs(centered - centered_direct),
                 "total_decomposition": np.abs(total - mean_error - centered),
                 "baseline_decomposition": np.abs(baseline - meany2 - vary),
                 "gain_decomposition": np.abs(total_gain - gain_mean - gain_centered)}
    for name, residual in residuals.items():
        require((residual <= 1e-9 * np.maximum(1.0, total + baseline + scale)).all(), "identity failed: " + name)
    out = dict(n=n, meanp=meanp, meany=meany, varp=varp, vary=vary, cov=cov, total=total, mean_error=mean_error,
               centered=centered, baseline_total=baseline, mean_target_squared=meany2, gain_mean=gain_mean,
               gain_centered=gain_centered, total_gain=total_gain)
    require(all(np.isfinite(v).all() for v in out.values()), "nonfinite float64 moments")
    out["identity_max_abs_error"] = {k: float(v.max(initial=0)) for k, v in residuals.items()}
    return out


def empty_sums(fields):
    return {"n_windows": 0, "n_window_joints": 0, "n_valid_elements": 0, "sums": {k: 0.0 for k in fields}}


def accumulate(acc, moments, users, window_selection, min_n, fields):
    for u in sorted(set(users[window_selection].tolist())):
        selected = (users == u) & window_selection
        eligible = (moments["n"] >= min_n) & selected[:, None]
        if not eligible.any():
            continue
        row = acc.setdefault(u, empty_sums(fields))
        n = moments["n"][eligible]
        row["n_windows"] += int(eligible.any(axis=1).sum())
        row["n_window_joints"] += int(eligible.sum())
        row["n_valid_elements"] += int(n.sum())
        for key in fields:
            row["sums"][key] += float(np.sum(n * moments[key][eligible], dtype=np.float64))


def pooled_summary(per_user, fields):
    total = empty_sums(fields)
    for row in per_user.values():
        for key in ("n_windows", "n_window_joints", "n_valid_elements"):
            total[key] += row[key]
        for key in fields:
            total["sums"][key] += row["sums"][key]
    count = total["n_valid_elements"]
    means = {k: v / count if count else None for k, v in total["sums"].items()}
    result = {**total, "n_users": len(per_user), "means": means, "per_user": {u: per_user[u] for u in sorted(per_user)}}
    if "centered" in fields:
        v, p, cov = means["vary"], means["varp"], means["cov"]
        result.update(centered_mse_ratio=means["centered"] / v if v is not None and v > 0 else None,
                      variance_ratio=p / v if v is not None and v > 0 else None,
                      pooled_centered_correlation=cov / math.sqrt(p * v) if p is not None and p > 0 and v > 0 else None,
                      centered_gain=means["gain_centered"])
        result["undefined_metrics"] = {}
        if v is None or v <= 0:
            result["undefined_metrics"].update(centered_mse_ratio="no supported target variance", variance_ratio="no supported target variance")
        if p is None or p <= 0 or v <= 0:
            result["undefined_metrics"]["pooled_centered_correlation"] = "no support or zero prediction/target temporal variance"
        gains = [r["sums"]["gain_centered"] for r in per_user.values()]
        result["user_centered_gain_signs"] = {"positive": sum(g > 0 for g in gains), "zero": sum(g == 0 for g in gains),
                                               "negative": sum(g < 0 for g in gains)}
    return result


def bootstrap_centered(per_user, n_boot=N_BOOT, seed=SEED):
    users = sorted(per_user)
    base = {"n_bootstrap": n_boot, "seed": seed, "users": users,
            "estimand": "ratio of pooled centered MSE to pooled target temporal variance; gain = pooled vary minus centered MSE"}
    if len(users) < 2:
        return {**base, "evaluable": False, "reason": "fewer than two users with n>=2 window/joint support"}
    n = np.array([per_user[u]["n_valid_elements"] for u in users], dtype=np.float64)
    y = np.array([per_user[u]["sums"]["vary"] for u in users])
    err = np.array([per_user[u]["sums"]["centered"] for u in users])
    draws = np.random.default_rng(seed).integers(0, len(users), size=(n_boot, len(users)))
    count, den, num = n[draws].sum(axis=1), y[draws].sum(axis=1), err[draws].sum(axis=1)
    valid = np.isfinite(count) & np.isfinite(den) & np.isfinite(num) & (count > 0) & (den > 0)
    if not valid.all():
        return {**base, "evaluable": False, "reason": "zero/nonfinite bootstrap denominator or nonfinite numerator; no replicas discarded",
                "invalid_replicates": int((~valid).sum())}
    ratios, gains = num / den, (den - num) / count
    if not np.isfinite(ratios).all() or not np.isfinite(gains).all():
        return {**base, "evaluable": False, "reason": "nonfinite bootstrap statistic; no replicas discarded"}
    return {**base, "evaluable": True, "ratio_ci95": np.percentile(ratios, [2.5, 97.5]).tolist(),
            "gain_ci95": np.percentile(gains, [2.5, 97.5]).tolist()}


def e1_summary(per_user):
    users = sorted(per_user)
    if not users:
        return {"evaluable": False, "reason": "no dynamic windows with valid pose"}
    errs = np.array([per_user[u]["error_sum_rad"] for u in users])
    const = np.array([per_user[u]["constant_sum_rad"] for u in users])
    n = np.array([per_user[u]["n_valid_elements"] for u in users])
    draws = np.random.default_rng(SEED).integers(0, len(users), size=(N_BOOT, len(users)))
    den = const[draws].sum(axis=1)
    if not np.isfinite(den).all() or (den <= 0).any():
        return {"evaluable": False, "reason": "invalid E1 bootstrap denominator; no replicas discarded", "per_user": per_user}
    ratios = errs[draws].sum(axis=1) / den
    require(np.isfinite(ratios).all(), "nonfinite E1 bootstrap ratio")
    individual = errs / const
    require(np.isfinite(individual).all(), "E1 user ratio undefined")
    return {"evaluable": True, "n_users": len(users), "n_windows": sum(r["n_windows"] for r in per_user.values()),
            "n_valid_elements": int(n.sum()), "mae": float(errs.sum() / n.sum()), "mae_const": float(const.sum() / n.sum()),
            "ratio": float(errs.sum() / const.sum()), "ratio_ci95_users": np.percentile(ratios, [2.5, 97.5]).tolist(),
            "median_user_ratio": float(np.median(individual)), "n_bootstrap": N_BOOT, "seed": SEED,
            "per_user": per_user}


def analyze(data, batch_windows=128):
    require(type(batch_windows) is int and batch_windows > 0, "batch_windows must be positive")
    pv = data["pose_var"]
    has = np.isfinite(pv)
    median = float(np.nanmedian(pv)) if has.any() else None
    dynamic = has & (pv >= median) if median is not None else np.zeros(len(pv), dtype=bool)
    selections = {"all": np.ones(len(pv), dtype=bool), "dynamic": dynamic, "static": has & ~dynamic}
    users = np.asarray([m["user"] for m in data["meta"]])
    groups = {k: {"total": {}, "centered": {}} for k in selections}
    e1, checks = {}, {}
    for start in range(0, len(pv), batch_windows):
        sl = slice(start, start + batch_windows)
        p, y, mask = (data[k][sl] for k in ("pred", "target", "mask"))
        moments = window_joint_moments(p, y, mask)
        for key, value in moments["identity_max_abs_error"].items():
            checks[key] = max(checks.get(key, 0.0), value)
        for key, selection in selections.items():
            accumulate(groups[key]["total"], moments, users[sl], selection[sl], 1, TOTAL_FIELDS)
            accumulate(groups[key]["centered"], moments, users[sl], selection[sl], 2, CENTERED_FIELDS)
        valid = mask.sum(axis=(1, 2), dtype=np.int64)
        rad = data["std"][sl].astype(np.float64)[:, None, :]
        pd, yd = np.where(mask, p, 0).astype(np.float64), np.where(mask, y, 0).astype(np.float64)
        err = (np.abs(pd - yd) * rad).sum(axis=(1, 2), dtype=np.float64)
        const = (np.abs(yd) * rad).sum(axis=(1, 2), dtype=np.float64)
        eligible = dynamic[sl] & (valid > 0)
        for u in sorted(set(users[sl][eligible].tolist())):
            selected = eligible & (users[sl] == u)
            row = e1.setdefault(u, {"n_windows": 0, "n_valid_elements": 0, "error_sum_rad": 0.0, "constant_sum_rad": 0.0})
            row["n_windows"] += int(selected.sum())
            row["n_valid_elements"] += int(valid[selected].sum())
            row["error_sum_rad"] += float(err[selected].sum())
            row["constant_sum_rad"] += float(const[selected].sum())
    output = {}
    for name, accumulators in groups.items():
        output[name] = {"role": "primary descriptive dynamics" if name == "dynamic" else "secondary descriptive",
                        "n_selected_windows": int(selections[name].sum()),
                        "total_support_n_ge_1": pooled_summary(accumulators["total"], TOTAL_FIELDS),
                        "centered_support_n_ge_2": pooled_summary(accumulators["centered"], CENTERED_FIELDS),
                        "bootstrap_centered": bootstrap_centered(accumulators["centered"])}
    return {"schema_version": 1, "scope": "Descriptive decomposition of fixed B5000 student predictions; no pass/fail criterion, fit or new prediction.",
            "units": "standardized squared angles for decomposition; radians for E1 MAE control",
            "aggregation": "Compute population moments per window/joint; pool with valid-element weights. Centered statistics and every component of their decomposition use only n>=2 window/joint cells.",
            "gain_identity": "On the identical n>=2 support: total_gain=(E[y^2]-MSE)=gain_mean+gain_centered; gain_mean=mean(y)^2-(mean(p)-mean(y))^2; gain_centered=var(y)-centered_MSE.",
            "interpretation": "Mean and centered gains are additive error components, not causal mechanisms. Components can be negative; no percentage attribution is imposed. Pooled-centered correlation removes each window/joint mean and contains no between-window covariance.",
            "median_pose_var_cached": median, "n_windows": len(pv), "n_users": len(set(users.tolist())),
            "n_windows_nonfinite_pose_var": int((~has).sum()), "n_dynamic": int(dynamic.sum()),
            "identity_max_abs_error": checks, "groups": output, "e1_dynamic_control": e1_summary(e1)}


def check_reference(result, reference):
    got, expected = result["e1_dynamic_control"], reference["E1"]["dynamic"]
    require(got["evaluable"], "E1 control not evaluable")
    for key in ("n_windows", "n_users"):
        require(got[key] == expected[key], "E1 reference count mismatch: " + key)
    require(result["n_windows"] == reference["n_windows"] and result["n_dynamic"] == reference["n_dynamic"], "reference cache coverage mismatch")
    compared = ("mae", "mae_const", "ratio", "ratio_ci95_users", "median_user_ratio")
    for key in compared:
        require(np.allclose(got[key], expected[key], rtol=1e-6, atol=1e-8, equal_nan=False), "E1 reference value mismatch: " + key)
    require(set(got["per_user"]) == set(expected["per_user"]), "E1 reference users mismatch")
    for u, row in got["per_user"].items():
        old = expected["per_user"][u]
        require(row["n_windows"] == old["n_windows"], "E1 reference user window count mismatch")
        for key, value in {"mae": row["error_sum_rad"] / row["n_valid_elements"],
                           "mae_const": row["constant_sum_rad"] / row["n_valid_elements"],
                           "ratio": row["error_sum_rad"] / row["constant_sum_rad"]}.items():
            require(np.isclose(value, old[key], rtol=1e-6, atol=1e-8), "E1 reference user value mismatch: " + key)
    return {"consistent": True, "fields": list(compared), "per_user_checked": True, "rtol": 1e-6, "atol": 1e-8,
            "note": "Float64 reproduction of the prior float32 E1 computation; same cached dynamic split, masks, element weights and user bootstrap."}


def fresh_destination(out):
    require(out.is_absolute(), "output must be absolute")
    require(not out.exists() and not out.is_symlink(), "output already exists")
    roots = []
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        roots.append(here.parent.parent)
    if os.environ.get("CODE_DIR"):
        roots.append(Path(os.environ["CODE_DIR"]))
    if os.environ.get("WORK"):
        roots.extend(Path(os.environ["WORK"]) / p for p in ("wearusfm", "wearusfm_campaigns"))
    resolved = out.resolve()
    require(not any(resolved == r.resolve() or r.resolve() in resolved.parents for r in roots), "output inside code")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-cache-sha256", required=True)
    ap.add_argument("--report-provenance", type=Path, required=True)
    ap.add_argument("--reference-report", type=Path)
    ap.add_argument("--analysis-source-sha256")
    ap.add_argument("--wrapper-sha256")
    args = ap.parse_args(argv)
    fresh_destination(args.out)
    require(is_sha(args.expected_cache_sha256), "invalid expected cache SHA")
    for key in ("analysis_source_sha256", "wrapper_sha256"):
        value = getattr(args, key)
        require(value is None or is_sha(value), "invalid " + key)
    actual_source = file_sha256(__file__)
    require(args.analysis_source_sha256 is None or args.analysis_source_sha256 == actual_source, "analysis source SHA mismatch")
    actual_cache = file_sha256(args.cache)
    require(actual_cache == args.expected_cache_sha256, "cache SHA mismatch")
    provenance = json.loads(args.report_provenance.read_text())
    require(provenance.get("cache_sha256") == actual_cache, "report provenance cache SHA mismatch")
    data = load_cache(args.cache)
    check_provenance(data, args.cache, actual_cache, provenance)
    result = analyze(data)
    result["provenance"] = {"cache": str(args.cache.resolve()), "cache_sha256": actual_cache,
                            "report_provenance": str(args.report_provenance.resolve()), "report_provenance_sha256": file_sha256(args.report_provenance),
                            "checkpoint_sha256": provenance["checkpoint_sha256"], "checkpoint_step": provenance["checkpoint_step"],
                            "pose_stats_sha256": provenance["pose_stats_sha256"], "cache_config": data["config"],
                            "analysis_source_sha256": actual_source, "wrapper_sha256": args.wrapper_sha256,
                            "numpy_version": np.__version__, "loaded_npz_fields": list(CACHE_FIELDS)}
    if args.reference_report:
        reference = json.loads(args.reference_report.read_text())
        require(reference.get("config") == data["config"], "reference report/cache config mismatch")
        result["reference_e1_check"] = check_reference(result, reference)
        result["provenance"].update(reference_report=str(args.reference_report.resolve()), reference_report_sha256=file_sha256(args.reference_report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out), "n_windows": result["n_windows"], "n_dynamic": result["n_dynamic"],
                      "dynamic_centered": {k: v for k, v in result["groups"]["dynamic"]["centered_support_n_ge_2"].items()
                                           if k not in ("per_user", "sums")}}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
