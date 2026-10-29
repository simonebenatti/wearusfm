"""Independent local reaggregation of B5000 dynamics sufficient statistics.

Usage: python analyze_dynamics.py /path/to/report.json [--plot]
Only stdlib/numpy; optional figure uses matplotlib. Writes alongside this script.
No scientific code imports, new estimands, fitting, predictions or remote access.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ORDER = ("dynamic", "all", "static")
SEED, N_BOOT = 20261009, 1000


def close(actual, reported, label):
    if not math.isfinite(actual) or not math.isfinite(reported) or not math.isclose(actual, reported, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"{label}: recalculated {actual} differs from report {reported}")


def identities(sums, label):
    """Rebuild centered errors and gains from the independent moment components."""
    centered = math.fsum((sums["varp"], sums["vary"], -2 * sums["cov"]))
    mean_gain = sums["mean_target_squared"] - sums["mean_error"]
    centered_gain = 2 * sums["cov"] - sums["varp"]
    rebuilt = {"centered": centered, "total": sums["mean_error"] + centered,
               "baseline_total": sums["mean_target_squared"] + sums["vary"],
               "gain_mean": mean_gain, "gain_centered": centered_gain, "total_gain": mean_gain + centered_gain}
    for key, value in rebuilt.items():
        close(value, sums[key], label + "/" + key)
    return rebuilt


def analyze_group(name, group):
    saved = group["centered_support_n_ge_2"]
    per = saved["per_user"]
    users = sorted(per)
    if len(users) < 2 or len(users) != saved["n_users"]:
        raise ValueError("insufficient or inconsistent user count")
    rebuilt, user_rows = {}, {}
    for u in users:
        row = per[u]
        if any(type(row[k]) is not int or row[k] <= 0 for k in ("n_valid_elements", "n_windows", "n_window_joints")):
            raise ValueError("invalid user support")
        if not all(math.isfinite(v) for v in row["sums"].values()):
            raise ValueError("nonfinite sufficient statistic")
        r = identities(row["sums"], name + "/" + u)
        rebuilt[u] = r
        n, vary, varp = row["n_valid_elements"], row["sums"]["vary"], row["sums"]["varp"]
        if vary <= 0 or varp < 0:
            raise ValueError("undefined user variance ratio")
        user_rows[u] = {"n_windows": row["n_windows"], "n_valid_elements": n,
                        "centered_mse_ratio": r["centered"] / vary, "variance_ratio": varp / vary,
                        "temporal_rms_ratio": math.sqrt(varp / vary),
                        "gain_mean": r["gain_mean"] / n, "gain_centered": r["gain_centered"] / n,
                        "total_gain": r["total_gain"] / n}
    for key in ("n_valid_elements", "n_windows", "n_window_joints"):
        if sum(per[u][key] for u in users) != saved[key]:
            raise ValueError("pooled support/count mismatch: " + key)
    sums = {key: math.fsum(per[u]["sums"][key] for u in users) for key in saved["sums"]}
    for key, value in sums.items():
        close(value, saved["sums"][key], "pooled sum " + key)
        close(value / saved["n_valid_elements"], saved["means"][key], "pooled mean " + key)
    r = identities(sums, name + "/pooled")
    ratio = r["centered"] / sums["vary"]
    variance_ratio = sums["varp"] / sums["vary"]
    corr = sums["cov"] / math.sqrt(sums["varp"] * sums["vary"])
    for key, value in (("centered_mse_ratio", ratio), ("variance_ratio", variance_ratio), ("pooled_centered_correlation", corr)):
        close(value, saved[key], key)
    # Independent loop implementation: draw sorted users, sum their moment
    # components with fsum, then form the pooled ratio and per-element gain.
    rng = np.random.default_rng(SEED)
    ratios, gains = [], []
    for _ in range(N_BOOT):
        draw = rng.choice(users, size=len(users), replace=True)
        n = sum(per[u]["n_valid_elements"] for u in draw)
        v = math.fsum(per[u]["sums"]["vary"] for u in draw)
        e = math.fsum(rebuilt[u]["centered"] for u in draw)
        if n <= 0 or v <= 0 or not all(math.isfinite(x) for x in (v, e)):
            raise ValueError("invalid bootstrap replica; none discarded")
        ratios.append(e / v)
        gains.append((v - e) / n)
    ratio_ci, gain_ci = (np.percentile(x, [2.5, 97.5]).tolist() for x in (ratios, gains))
    boot = group["bootstrap_centered"]
    if not boot["evaluable"] or boot["seed"] != SEED or boot["n_bootstrap"] != N_BOOT or boot["users"] != users:
        raise ValueError("bootstrap protocol mismatch")
    for got, expected in zip(ratio_ci + gain_ci, boot["ratio_ci95"] + boot["gain_ci95"]):
        close(got, expected, "bootstrap CI")
    sign = {"positive": [], "zero": [], "negative": []}
    for u in users:
        gain = user_rows[u]["gain_centered"]
        sign["positive" if gain > 0 else "negative" if gain < 0 else "zero"].append(u)
    counts = {k: len(v) for k, v in sign.items()}
    if counts != saved["user_centered_gain_signs"]:
        raise ValueError("user gain signs differ")
    n = saved["n_valid_elements"]
    return {"n_users": len(users), "n_windows": saved["n_windows"], "n_valid_elements": n,
            "n_windows_total_support_n_ge_1": group["total_support_n_ge_1"]["n_windows"],
            "centered_mse_ratio": ratio, "ratio_ci95": ratio_ci,
            "centered_mse_reduction_pct": 100 * (1 - ratio),
            "variance_ratio": variance_ratio, "temporal_rms_ratio": math.sqrt(variance_ratio), "pooled_centered_correlation": corr,
            "gain_mean": r["gain_mean"] / n, "gain_centered": r["gain_centered"] / n, "total_gain": r["total_gain"] / n,
            "gain_centered_ci95": gain_ci, "user_centered_gain_signs": counts, "users_by_gain_sign": sign,
            "per_user": user_rows, "checks": "All sufficient sums, counts, decompositions, pooled metrics, signs and bootstrap CIs agree."}


def describe(report, path):
    groups = {name: analyze_group(name, report["groups"][name]) for name in ORDER}
    # Dynamic/static partition is checked on this report's identical n>=2 support.
    for key in report["groups"]["all"]["centered_support_n_ge_2"]["sums"]:
        parts = math.fsum(report["groups"][g]["centered_support_n_ge_2"]["sums"][key] for g in ("dynamic", "static"))
        close(parts, report["groups"]["all"]["centered_support_n_ge_2"]["sums"][key], "group partition " + key)
    return {"scope": "Independent descriptive reaggregation, not a new gate or scientific model.",
            "source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "units": "standardized squared angles for gains; dimensionless MSE, variance and centered temporal RMS ratios",
            "n_bootstrap": N_BOOT, "seed": SEED, "groups": groups,
            "limits": "Centering both trajectories removes each window/joint mean. Gains are additive components of MSE reduction, not causal mechanisms. RMS ratio is sqrt(pooled within-window prediction variance / pooled target variance), not a fitted amplitude coefficient. Static means below cached target-variance median, not motionless."}


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Dinamiche", "Tutte", "Poco dinamiche"]
    rows = [result["groups"][g] for g in ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained")
    x = np.arange(3)
    ratios = np.array([r["centered_mse_ratio"] for r in rows])
    ci = np.array([r["ratio_ci95"] for r in rows])
    axes[0].errorbar(ratios, x, xerr=[ratios - ci[:, 0], ci[:, 1] - ratios], fmt="o", capsize=5, color="#176b87")
    axes[0].axvline(1, color="0.55", linestyle="--")
    axes[0].set(yticks=x, yticklabels=names, xlabel="MSE centrata / varianza temporale del bersaglio",
                title="Ricostruzione delle fluttuazioni · IC95 utenti")
    axes[0].invert_yaxis()
    mean = np.array([r["gain_mean"] for r in rows])
    centered = np.array([r["gain_centered"] for r in rows])
    axes[1].bar(x, mean, color="#a9b3bf", label="Livello medio")
    bottoms = np.where(centered >= 0, np.maximum(mean, 0), np.minimum(mean, 0))
    axes[1].bar(x, centered, bottom=bottoms, color="#176b87", label="Fluttuazioni centrate")
    axes[1].scatter(x, mean + centered, marker="D", color="#24292f", s=35, label="Guadagno totale", zorder=3)
    axes[1].axhline(0, color="0.5", linewidth=1)
    axes[1].set(xticks=x, xticklabels=names, ylabel="Guadagno MSE standardizzata rispetto alla costante",
                title="Contributi additivi sullo stesso supporto")
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("B student · 5.000 passi · 15 utenti val\nSeparazione descrittiva del livello medio e delle fluttuazioni entro finestra", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = describe(json.loads(args.report.read_text()), args.report)
    outdir = Path(__file__).resolve().parent
    (outdir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(result, outdir / "decomposition.png")
    print(json.dumps({g: {k: v for k, v in row.items() if k not in ("per_user", "users_by_gain_sign")}
                      for g, row in result["groups"].items()}, indent=2, allow_nan=False))
