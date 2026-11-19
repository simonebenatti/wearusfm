"""Window-weighted training MSE and unchanged pooled historical dynamics."""
from __future__ import annotations

import numpy as np
import torch

import pose_dynamics as historical
import tokenizer_pose_readout as reference
from wearusfm.ingest.emg2pose_pose import JOINTS
from .data import CROP, central_support, lengths_for, require

THRESHOLD = reference.THRESHOLD
HISTORICAL_REFERENCE = {
    "post_ridge_E_dyn": 0.7859020004638595,
    "trained_head_E_dyn": 0.7786289843513977,
    "dynamic_selected_windows": 8329,
    "dynamic_centered_windows": 8327,
    "dynamic_centered_elements": 20119920,
    "dynamic_window_joint_pairs": 166540,
    "dynamic_users": 15,
}


def per_window_mse(prediction, target, support):
    """Differentiable mean of within-window masked MSEs, excluding empty windows."""
    if prediction.shape != target.shape or target.shape != support.shape or prediction.ndim != 3:
        raise ValueError("prediction/target/support shapes differ")
    if support.dtype != torch.bool:
        raise ValueError("support must be boolean")
    if not torch.isfinite(prediction[support]).all() or not torch.isfinite(target[support]).all():
        raise ValueError("nonfinite supported prediction or target")
    count = support.sum(dim=(1, 2))
    kept = count > 0
    if not kept.any():
        raise ValueError("minibatch has no supported window")
    p = torch.where(support, prediction, 0)
    y = torch.where(support, target, 0)
    squared = (p-y).square().sum(dim=(1, 2))
    return (squared[kept]/count[kept]).mean()


def dynamic_variance(target, mask):
    """Mean population temporal variance over supported joints, on the full window."""
    target, mask = np.asarray(target), np.asarray(mask)
    require(target.shape == mask.shape and target.ndim == 3 and mask.dtype == bool,
            "invalid variance inputs")
    require(np.isfinite(target[mask]).all(), "nonfinite supported variance target")
    y = np.where(mask, target, 0).astype(np.float64)
    count = mask.sum(axis=1)
    means = y.sum(axis=1)/np.maximum(count, 1)
    centered = np.where(mask, y-means[:, None, :], 0)
    variance = (centered*centered).sum(axis=1)/np.maximum(count, 1)
    eligible = count >= 2
    n = eligible.sum(axis=1)
    result = np.full(len(y), np.nan, dtype=np.float64)
    result[n > 0] = np.where(eligible, variance, 0).sum(axis=1)[n > 0]/n[n > 0]
    return result


def training_threshold(target, mask):
    variances = dynamic_variance(target, mask)
    finite = variances[np.isfinite(variances)]
    require(len(finite) > 0, "fitting targets have no temporal variance support")
    return max(float(np.median(finite)), 1e-12)


def _cropped(prediction, data):
    prediction = np.asarray(prediction)
    y = np.asarray(data["target"])
    if prediction.shape == y.shape:
        prediction = prediction[:, CROP:-CROP]
    support = central_support(data["mask"], lengths_for(data), data.get("key_time_valid"))
    require(prediction.shape == support.shape, "prediction must cover full window or exact crop6")
    require(prediction.dtype.kind == "f", "predictions must be floating point")
    return prediction.astype(np.float64), y[:, CROP:-CROP].astype(np.float64), support


def evaluate(prediction, data, dynamic_threshold=THRESHOLD, *, batch_windows=64, per_joint=False):
    """Return historical-compatible per-user moments plus window-weighted MSE.

    The caller supplies pose_var for the appropriate real or synthetic full
    window. External evaluation keeps cached pose_var and the frozen threshold;
    internal evaluation uses fitting-only thresholds fixed before the curves.
    """
    require(np.isfinite(dynamic_threshold) and dynamic_threshold >= 0, "invalid dynamic threshold")
    require(type(batch_windows) is int and batch_windows > 0, "invalid metric batch size")
    prediction, targets, support = _cropped(prediction, data)
    users = np.asarray([m["user"] for m in data["meta"]])
    pv = np.asarray(data["pose_var"])
    require(pv.shape == (len(users),) and not np.isinf(pv).any(), "invalid full-window variance")
    selected = {"all": np.ones(len(users), dtype=bool),
                "dynamic": np.isfinite(pv) & (pv >= dynamic_threshold),
                "static": np.isfinite(pv) & (pv < dynamic_threshold)}
    acc = {g: {} for g in selected}
    per_window = {g: {} for g in selected}
    mae = {g: {} for g in selected}
    if per_joint:
        require(targets.shape[-1] == len(JOINTS), "per-joint reporting requires all 20 registered joints")
    joint_acc = {g: {j: {} for j in JOINTS} for g in selected} if per_joint else {}
    joint_mae = {g: {j: {} for j in JOINTS} for g in selected} if per_joint else {}
    std = np.asarray(data["std"])
    require(std.shape == (len(users), targets.shape[-1]) and np.isfinite(std).all() and (std > 0).all(),
            "invalid physical target scales")
    for start in range(0, len(users), batch_windows):
        sl = slice(start, start+batch_windows)
        p, y, m = prediction[sl], targets[sl], support[sl]
        moments = historical.window_joint_moments(p, y, m)
        q = m.sum(axis=(1, 2))
        p_safe, y_safe = np.where(m, p, 0), np.where(m, y, 0)
        squared = ((p_safe-y_safe)**2).sum(axis=(1, 2))
        errors = (np.abs(p_safe-y_safe)*std[sl, None, :]).sum(axis=(1, 2))
        for group, keep in selected.items():
            historical.accumulate(acc[group], moments, users[sl], keep[sl], 2, historical.CENTERED_FIELDS)
            for user in sorted(set(users[sl][keep[sl] & (q > 0)])):
                ix = keep[sl] & (users[sl] == user) & (q > 0)
                row = per_window[group].setdefault(user, {"mse_sum": 0., "window_count": 0})
                row["mse_sum"] += float(np.sum(squared[ix]/q[ix], dtype=np.float64))
                row["window_count"] += int(ix.sum())
                row_mae = mae[group].setdefault(user, {"error_sum_rad": 0., "n_valid_elements": 0})
                row_mae["error_sum_rad"] += float(errors[ix].sum())
                row_mae["n_valid_elements"] += int(q[ix].sum())
            if per_joint:
                for j, name in enumerate(JOINTS):
                    one = {key: value[:, j:j+1] for key, value in moments.items()
                           if key != "identity_max_abs_error"}
                    historical.accumulate(joint_acc[group][name], one, users[sl], keep[sl],
                                          2, historical.CENTERED_FIELDS)
                    counts_j = m[:, :, j].sum(axis=1)
                    errors_j = (np.abs(p_safe[:, :, j]-y_safe[:, :, j])*std[sl, j, None]).sum(axis=1)
                    for user in sorted(set(users[sl][keep[sl] & (counts_j > 0)])):
                        ix = keep[sl] & (users[sl] == user) & (counts_j > 0)
                        row_j = joint_mae[group][name].setdefault(user, {"error_sum_rad":0.,"n_valid_elements":0})
                        row_j["error_sum_rad"] += float(errors_j[ix].sum())
                        row_j["n_valid_elements"] += int(counts_j[ix].sum())
    groups = {}
    for group in selected:
        n = sum(r["window_count"] for r in per_window[group].values())
        mse_sum = sum(r["mse_sum"] for r in per_window[group].values())
        elements = sum(r["n_valid_elements"] for r in mae[group].values())
        error = sum(r["error_sum_rad"] for r in mae[group].values())
        groups[group] = {"centered": historical.pooled_summary(acc[group], historical.CENTERED_FIELDS),
                         "per_window_mse": mse_sum/n if n else None,
                         "per_window_mse_sum": mse_sum, "n_supported_windows": n,
                         "per_window_by_user": per_window[group],
                         "mae_rad": error/elements if elements else None, "mae_per_user": mae[group],
                         "n_selected_windows": int(selected[group].sum())}
        if per_joint:
            by_joint = {}
            for name in JOINTS:
                ma = joint_mae[group][name]
                count_j = sum(r["n_valid_elements"] for r in ma.values())
                error_j = sum(r["error_sum_rad"] for r in ma.values())
                by_joint[name] = {"centered":historical.pooled_summary(joint_acc[group][name], historical.CENTERED_FIELDS),
                                  "mae_rad":error_j/count_j if count_j else None,"mae_per_user":ma}
            groups[group]["by_joint"] = by_joint
    return {"groups": groups, "support_sha256": reference.arrays_hash(support),
            "dynamic_threshold": float(dynamic_threshold),
            "prediction_sha256": reference.arrays_hash(prediction)}


def selection_statistics(prediction, data, dynamic_threshold):
    result = evaluate(prediction, data, dynamic_threshold)
    all_group = result["groups"]["all"]
    sums = result["groups"]["dynamic"]["centered"]["sums"]
    return {"mse_sum": all_group["per_window_mse_sum"], "window_count": all_group["n_supported_windows"],
            "dynamic_sse": sums["centered"], "dynamic_target_ss": sums["vary"]}


def historical_reference_gate(data, post_ridge, trained_head):
    """R-1: fixed values and indices, never a choice of new readout settings."""
    shape = (len(data["meta"]), data["target"].shape[1]-2*CROP, data["target"].shape[2])
    require(np.asarray(post_ridge).shape == shape and np.asarray(trained_head).shape == shape,
            "historical prediction support is not crop6")
    support = central_support(data["mask"], lengths_for(data), data["key_time_valid"])
    cached = data["pred"][:, CROP:-CROP].astype(np.float64)
    require(np.array_equal(cached[support], np.asarray(trained_head)[support]),
            "historical trained-head crop indices/values differ")
    outputs = {"post_ridge": evaluate(np.asarray(post_ridge, dtype=np.float64), data),
               "trained_head": evaluate(np.asarray(trained_head, dtype=np.float64), data)}
    observed = {}
    for reader, result in outputs.items():
        dynamic = result["groups"]["dynamic"]
        centered = dynamic["centered"]
        value = centered["centered_mse_ratio"]
        require(value is not None and abs(value-HISTORICAL_REFERENCE[reader+"_E_dyn"]) <= 1e-12,
                "historical E_dyn parity failed: " + reader)
        require(dynamic["n_selected_windows"] == HISTORICAL_REFERENCE["dynamic_selected_windows"] and
                centered["n_windows"] == HISTORICAL_REFERENCE["dynamic_centered_windows"] and
                centered["n_valid_elements"] == HISTORICAL_REFERENCE["dynamic_centered_elements"] and
                centered["n_window_joints"] == HISTORICAL_REFERENCE["dynamic_window_joint_pairs"] and
                centered["n_users"] == HISTORICAL_REFERENCE["dynamic_users"],
                "historical dynamic support counts differ")
        observed[reader] = {"E_dyn": value, "selected_windows": dynamic["n_selected_windows"],
                            "centered_windows": centered["n_windows"],
                            "centered_elements": centered["n_valid_elements"],
                            "window_joint_pairs": centered["n_window_joints"],
                            "users": centered["n_users"]}
    return {"passed": True, "absolute_tolerance": 1e-12, "crop": [6, 154], "observed": observed,
            "reference_policy": HISTORICAL_REFERENCE.copy()}
