"""Describe B5000 E1/E2 JSON locally, with the prespecified paired-user E2-gap CI.

Usage: python analyze_e1e2.py /absolute/path/to/report.json [--plot]
Requires only Python stdlib and numpy; --plot additionally requires matplotlib.
Writes only alongside this script. No harness/trainer imports or remote access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEED, REPLICATES = 20261009, 1000


def require(condition, why):
    if not condition:
        raise ValueError(why)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), name + " missing/nonfinite")
    require(value > 0 if positive else value >= 0, name + " invalid")
    return float(value)


def signed(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " missing/nonfinite")
    return float(value)


def count(value, name, *, positive=False):
    require(type(value) is int and (value > 0 if positive else value >= 0), name + " invalid count")
    return value


def agrees(got, expected, name):
    require(math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9), name + " inconsistent with report")


def describe_e1(block):
    mae = number(block.get("mae"), "E1 MAE")
    constant = number(block.get("mae_const"), "E1 constant MAE", positive=True)
    ratio = number(block.get("ratio"), "E1 ratio")
    agrees(mae / constant, ratio, "E1 ratio")
    ci = block.get("ratio_ci95_users")
    require(isinstance(ci, list) and len(ci) == 2, "E1 CI missing")
    ci = [number(x, "E1 CI") for x in ci]
    require(ci[0] <= ci[1], "E1 CI reversed")
    per = block["per_user"]
    require(isinstance(per, dict) and len(per) == count(block.get("n_users"), "E1 users", positive=True), "E1 user count mismatch")
    rows, total_windows = {}, 0
    for u in sorted(per):
        rec = per[u]
        a = number(rec.get("mae"), "E1 user MAE")
        b = number(rec.get("mae_const"), "E1 user constant", positive=True)
        r = number(rec.get("ratio"), "E1 user ratio")
        agrees(a / b, r, "E1 user ratio")
        n = count(rec.get("n_windows"), "E1 user windows", positive=True)
        total_windows += n
        rows[u] = {"mae_rad": a, "mae_const_rad": b, "ratio": r, "advantage_pct": 100 * (1 - r), "n_windows": n}
    require(total_windows == count(block.get("n_windows"), "E1 windows", positive=True), "E1 window count mismatch")
    ratios = np.array([r["ratio"] for r in rows.values()])
    median = float(np.median(ratios))
    agrees(median, number(block.get("median_user_ratio"), "E1 median"), "E1 median user ratio")
    return {"evaluable": True, "mae_rad": mae, "mae_const_rad": constant, "ratio": ratio,
            "ratio_ci95_report": ci, "advantage_pct": 100 * (1 - ratio),
            "advantage_ci95_pct_report": [100 * (1 - ci[1]), 100 * (1 - ci[0])],
            "n_users": len(rows), "n_windows": total_windows,
            "users_ratio_below_one": int((ratios < 1).sum()), "users_ratio_equal_one": int((ratios == 1).sum()),
            "user_ratio_min_median_max": [float(ratios.min()), median, float(ratios.max())], "per_user": rows,
            "checks": "ratio MAE/constant, user ratios/median and window counts agree with JSON",
            "pooling_limit": "Frozen E1 per-user JSON lacks valid-element counts: pooled E1 and its CI cannot be independently rebuilt from this JSON alone. Window counts are not used as substitutes."}


def describe_gap(e2):
    paired = e2["paired_support"]
    per = paired["per_user"]
    require(isinstance(per, dict) and len(per) == count(paired.get("n_users"), "E2 paired users", positive=True), "E2 paired user count mismatch")
    users = sorted(per)
    require(len(users) >= 2, "fewer than two supported users: paired-user CI not evaluable")
    aligned, misaligned, weights, windows, rows = [], [], [], [], {}
    for u in users:
        rec = per[u]
        a = number(rec.get("mae_aligned"), "E2 aligned MAE")
        b = number(rec.get("mae_misaligned"), "E2 misaligned MAE")
        gap = signed(rec.get("gap"), "E2 user gap")
        agrees(b - a, gap, "E2 user gap")
        w = count(rec.get("n_valid_elements"), "E2 valid elements", positive=True)
        n = count(rec.get("n_windows"), "E2 windows", positive=True)
        same = count(rec.get("n_same_recording"), "E2 same recording")
        other = count(rec.get("n_other_session"), "E2 other session")
        require(same + other == n, "E2 user pairing count mismatch")
        aligned.append(a); misaligned.append(b); weights.append(w); windows.append(n)
        rows[u] = {"mae_aligned_rad": a, "mae_misaligned_rad": b, "gap_rad": gap,
                   "n_valid_elements": w, "n_windows": n, "n_same_recording": same, "n_other_session": other}
    a, b, w = map(lambda x: np.asarray(x, dtype=np.float64), (aligned, misaligned, weights))
    require(sum(weights) == count(paired.get("n_valid_elements"), "E2 total valid elements", positive=True), "E2 support count mismatch")
    require(sum(windows) == count(paired.get("n_windows"), "E2 total windows", positive=True), "E2 window count mismatch")
    for key in ("n_same_recording", "n_other_session"):
        require(sum(r[key] for r in rows.values()) == count(paired.get(key), key), key + " aggregate mismatch")
    pooled_a, pooled_b = float(np.sum(a * w) / w.sum()), float(np.sum(b * w) / w.sum())
    pooled_gap = pooled_b - pooled_a
    agrees(pooled_a, number(paired.get("mae_aligned"), "E2 pooled aligned"), "E2 aligned pooling")
    agrees(pooled_b, number(paired.get("mae_misaligned"), "E2 pooled misaligned"), "E2 misaligned pooling")
    agrees(pooled_gap, signed(paired.get("gap_mae"), "E2 pooled gap"), "E2 gap pooling")
    gaps = b - a
    agrees(float(np.median(gaps)), signed(paired.get("median_user_gap"), "E2 median gap"), "E2 median gap")
    # Match the frozen pooled estimand, resampling users with replacement and
    # carrying BOTH outcomes and the same valid-element weight in each draw.
    draws = np.random.default_rng(SEED).integers(0, len(users), size=(REPLICATES, len(users)))
    denominator = w[draws].sum(axis=1)
    require(np.isfinite(denominator).all() and (denominator > 0).all(), "invalid bootstrap support; no replicas discarded")
    samples = ((b[draws] * w[draws]).sum(axis=1) - (a[draws] * w[draws]).sum(axis=1)) / denominator
    require(np.isfinite(samples).all(), "nonfinite bootstrap; no replicas discarded")
    ci = np.percentile(samples, [2.5, 97.5]).tolist()
    # E2's frozen block uses these same users/windows/support. Its per-user
    # constants plus paired valid counts permit an independent ratio/CI check.
    require(set(e2["per_user"]) == set(users), "E2 block and paired users differ")
    require(e2["n_windows"] == sum(windows) and e2["n_users"] == len(users), "E2 block count mismatch")
    constants = []
    for u in users:
        r = e2["per_user"][u]
        require(r["n_windows"] == rows[u]["n_windows"], "E2 block user windows mismatch")
        agrees(number(r["mae"], "E2 block user MAE"), rows[u]["mae_misaligned_rad"], "E2 block/paired MAE")
        constants.append(number(r["mae_const"], "E2 user constant", positive=True))
    c = np.asarray(constants)
    pooled_const = float((c * w).sum() / w.sum())
    e2_ratio = pooled_b / pooled_const
    agrees(pooled_const, number(e2["mae_const"], "E2 constant"), "E2 constant pooling")
    agrees(pooled_b, number(e2["mae"], "E2 MAE"), "E2 MAE pooling")
    agrees(e2_ratio, number(e2["ratio"], "E2 ratio"), "E2 ratio pooling")
    ratio_samples = (b[draws] * w[draws]).sum(axis=1) / (c[draws] * w[draws]).sum(axis=1)
    e2_ci = np.percentile(ratio_samples, [2.5, 97.5]).tolist()
    require(np.allclose(e2_ci, e2["ratio_ci95_users"], rtol=1e-6, atol=1e-9), "E2 ratio CI differs from report")
    return {"evaluable": True, "estimand": "MAE_misaligned minus MAE_aligned, both pooled by common valid elements",
            "n_users": len(users), "n_windows": sum(windows), "n_valid_elements": sum(weights),
            "mae_aligned_rad": pooled_a, "mae_misaligned_rad": pooled_b, "gap_rad": pooled_gap, "gap_ci95_users_rad": ci,
            "n_bootstrap": REPLICATES, "seed": SEED, "user_order": users,
            "gap_positive_users": int((gaps > 0).sum()), "gap_zero_users": int((gaps == 0).sum()),
            "gap_negative_users": int((gaps < 0).sum()),
            "gap_user_min_median_max_rad": [float(gaps.min()), float(np.median(gaps)), float(gaps.max())],
            "unweighted_user_mean_gap_rad_descriptive_only": float(gaps.mean()),
            "e2_ratio_reconstructed": e2_ratio, "e2_ratio_ci95_reconstructed": e2_ci,
            "per_user": rows, "pairing_report": e2.get("pairing"),
            "checks": "paired aggregate MAEs/gap/counts and frozen E2 pooled ratio/CI agree with report",
            "interpretation_limit": "Positive gap favors aligned targets on the retained common support; partner reuse/fallback/filtering make this noncausal. No new gate."}


def safely(function, value):
    try:
        return function(value)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {"evaluable": False, "reason": str(exc)}


def analyze(report, path):
    return {"scope": "Descriptive B student5000 endpoint. No new scientific verdict, thresholds, data filtering or training.",
            "source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config_report": report.get("config"),
            "coverage_report": {k: report.get(k) for k in ("n_windows", "n_users", "n_recordings", "n_dynamic", "has_head")},
            "legacy_booleans_preserved_not_reinterpreted": {"E1_learned": report.get("E1", {}).get("learned"),
                                                            "E2_misaligned_ok": report.get("E2", {}).get("misaligned_ok")},
            "E1_dynamic": safely(describe_e1, report.get("E1", {}).get("dynamic", {})),
            "E1_all": safely(describe_e1, report.get("E1", {}).get("all", {})),
            "E1_static": safely(describe_e1, report.get("E1", {}).get("static", {})),
            "E2_paired_gap": safely(describe_gap, report.get("E2", {}))}


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e1, e2 = result["E1_dynamic"], result["E2_paired_gap"]
    require(e1["evaluable"] and e2["evaluable"], "cannot plot unevaluable endpoint")
    users = sorted(set(e1["per_user"]) | set(e2["per_user"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), layout="constrained")
    for ax in axes:
        ax.set_yticks(range(len(users)), users, fontsize=8)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    for i, u in enumerate(users):
        if u in e1["per_user"]:
            r = e1["per_user"][u]["ratio"]
            axes[0].scatter(r, i, color="#176b87" if r < 1 else "#b34d42")
        if u in e2["per_user"]:
            gap = e2["per_user"][u]["gap_rad"]
            axes[1].scatter(gap, i, color="#176b87" if gap > 0 else "#b34d42")
        else:
            axes[1].text(.001, i, "nessun supporto E2 dinamico", va="center", fontsize=8, color="0.45")
    axes[0].axvline(1, color="0.5", linestyle="--")
    axes[1].axvline(0, color="0.5", linestyle="--")
    axes[0].set(title="E1 · finestre dinamiche", xlabel="MAE testa / MAE costante per utente")
    axes[1].set(title="E2 · stesso supporto valido", xlabel="MAE disallineata − MAE allineata [rad]")
    lo, hi = e2["gap_ci95_users_rad"]
    fig.suptitle(f"B student · 5.000 passi · descrittivi su val\nGap E2 pooled {e2['gap_rad']:.5f} rad, IC95 utenti [{lo:.5f}, {hi:.5f}]", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parent
    report = json.loads(args.report.read_text())
    result = analyze(report, args.report)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(result, output_dir / "per_user_e1_e2.png")
    print(json.dumps({k: {kk: vv for kk, vv in result[k].items() if kk not in ("per_user", "pairing_report")}
                      for k in ("E1_dynamic", "E2_paired_gap")}, indent=2, allow_nan=False))
