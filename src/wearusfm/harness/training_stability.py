"""Audit observed training curves without replacing a preregistered verdict.

The v1 statistic is reproduced exactly. Block summaries and removal of the
largest batch are descriptive attribution checks, never new acceptance gates.
"""
from __future__ import annotations

import numpy as np


def moving_average(values, width):
    x = np.asarray(values, dtype=np.float64)
    if width <= 0 or len(x) < width:
        raise ValueError("moving average requires a positive width and enough records")
    return np.convolve(x, np.ones(width) / width, mode="valid")


def _summary(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def audit_curve(records, *, warmup=100, smooth=50, rise_limit=1.5, block=200, clip=1.0):
    """Reproduce Step2C v1 and expose the mass inside its critical window.

    ``step`` is the completed optimizer update (one-based in the trainer).
    Report window endpoints using those labels, not a convolution index.
    ``block`` controls descriptive, nonoverlapping summaries only.
    """
    if warmup < 0 or smooth < 2 or block < 1 or not np.isfinite(rise_limit) or rise_limit <= 0:
        raise ValueError("invalid audit configuration")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    if len(records) < warmup + smooth:
        raise ValueError("not enough optimizer updates to reproduce v1")
    steps = np.asarray([r["step"] for r in records])
    if not np.issubdtype(steps.dtype, np.integer) or steps[0] < 1 or np.any(np.diff(steps) != 1):
        raise ValueError("metrics must contain consecutive optimizer updates")
    loss = np.asarray([r["keep_branches"] for r in records], dtype=np.float64)
    grad = np.asarray([r["grad_norm"] for r in records], dtype=np.float64)
    finite = np.isfinite(loss) & np.isfinite(grad)
    if not finite.all() or np.any(loss < 0) or np.any(grad < 0):
        bad = ~finite | (loss < 0) | (grad < 0)
        return {"updates": len(records), "finite_nonnegative": False,
                "invalid_updates": steps[bad].tolist(), "v1": {"stable": False},
                "note": "Invalid values are retained as a numerical failure; no curve gate can hide them."}
    ma = moving_average(loss, smooth)
    floor = np.minimum.accumulate(ma)
    # Match collaudo_verdict: convolution start indices >= warmup, while
    # the running minimum includes all preceding windows.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ma / floor
    checked = ratio[warmup:]
    infinite = np.flatnonzero(np.isinf(checked))
    if len(infinite):
        critical = int(infinite[0] + warmup)
        selection = "first infinite rise from a zero reference"
    elif np.isnan(checked).all():
        critical = warmup
        selection = "undefined: all checked ratios are 0/0"
    else:
        critical = int(np.nanargmax(checked) + warmup)
        selection = "largest defined ratio"
    best = int(np.argmin(ma[:critical + 1]))
    window = loss[critical:critical + smooth]
    indices = np.argsort(window)[::-1]
    maximum = int(indices[0])
    without_max = float(np.delete(window, maximum).mean())
    ref = float(floor[critical])
    exceeds = ratio[warmup:] > rise_limit
    runs = np.diff(np.flatnonzero(np.r_[True, exceeds[1:] != exceeds[:-1], True]))
    longest = int(max(runs[::2] if exceeds[0] else runs[1::2], default=0))
    total = float(window.sum())
    out = {
        "updates": len(records), "finite_nonnegative": True,
        "v1": {"smooth": smooth, "warmup": warmup, "rise_limit": rise_limit,
               "max_rise": None if not np.isfinite(ratio[critical]) else float(ratio[critical]),
               "stable": bool(np.max(ratio[warmup:]) <= rise_limit),
               "reference_positive": bool((floor[warmup:] > 0).all()),
               "exceeding_windows": int(exceeds.sum()),
               "checked_windows": len(exceeds), "longest_exceeding_windows": longest,
               "note": "Overlapping windows are not independent; a historical minimum is a selected reference."},
        "first_100": _summary(loss[:100]), "last_100": _summary(loss[-100:]),
        "critical_window": {"selection": selection, "first_update": int(steps[critical]),
                            "last_update": int(steps[critical + smooth - 1]), **_summary(window),
                            "historical_min_mean": ref,
                            "historical_min_first_update": int(steps[best]),
                            "mean_without_largest_batch": without_max,
                            "ratio_without_largest_batch": without_max / ref if ref > 0 else None,
                            "largest_batch_update": int(steps[critical + maximum]),
                            "top5_fraction_of_sum": float(window[indices[:5]].sum() / total) if total > 0 else 0.0,
                            "largest_batches": [{"step": int(steps[critical + i]), "loss": float(window[i]),
                                                 "grad_norm": float(grad[critical + i])} for i in indices[:10]]},
        "gradient": {**_summary(grad), "clip": clip, "clipped_fraction": float((grad > clip).mean())},
        "blocks": [], "heads": {},
        "interpretation": "Descriptive audit, not a new stability verdict or an attribution to session difficulty.",
    }
    for start in range(warmup, len(loss), block):
        end = min(start + block, len(loss))
        out["blocks"].append({"first_update": int(steps[start]), "last_update": int(steps[end - 1]),
                              "n": end - start, **_summary(loss[start:end]),
                              "clipped_fraction": float((grad[start:end] > clip).mean())})
    for key in ("keep_fused", "keep_psd", "keep_wavelet"):
        if all(key in r for r in records):
            x = np.asarray([r[key] for r in records], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"nonfinite head metric {key}")
            out["heads"][key] = {"first_100": _summary(x[:100]), "last_100": _summary(x[-100:])}
    return out
