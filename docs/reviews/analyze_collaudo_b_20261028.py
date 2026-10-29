"""Local, post-hoc description of B. Does not change or replace the frozen gate.

Usage from the repository root:
  python docs/reviews/analyze_collaudo_b_20261028.py --plot
Requires numpy; --plot additionally requires matplotlib. No model/data/cluster imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
from pathlib import Path

import numpy as np


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def energy(row):
    """Prediction energy, NOT decomposition of the improvement in MSE.

    All joints must share the temporal support for the saved mean-of-joint
    variance to have the same weights as the saved MSE.
    """
    windows = row["per_window"]
    assert all(len(set(w["n_valid_per_joint"])) == 1 and min(w["n_valid_per_joint"]) >= 2 for w in windows)
    means = np.asarray([w["pred_mean_per_joint"] for w in windows])
    time = st.fmean(w["pred_var"] for w in windows)
    mean_energy = float(np.mean(means**2))
    global_mean = float(np.mean(means.mean(axis=0)**2))
    between = float(np.mean((means - means.mean(axis=0))**2))
    assert np.isclose(global_mean + between, mean_energy)
    return dict(temporal_variance=time, window_means_squared=mean_energy,
                global_joint_means_squared=global_mean, between_window_means_variance=between,
                total_prediction_energy=time + mean_energy,
                temporal_share=time / (time + mean_energy) if time + mean_energy else None)


def describe(run_dir):
    files = ["pose_panel.jsonl", "pose_panel_set.json", "metrics.jsonl", "fixed_eval.jsonl",
             "config.json", "summary.json", "verdetto.json", "accounting.json", "ls_l.txt"]
    panel = read_lines(run_dir / "pose_panel.jsonl")
    metrics = read_lines(run_dir / "metrics.jsonl")
    fixed = read_lines(run_dir / "fixed_eval.jsonl")
    manifest = json.loads((run_dir / "pose_panel_set.json").read_text())
    verdict = json.loads((run_dir / "verdetto.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    users = sorted({w["user"] for w in panel[0]["per_window"]})
    curves = []
    for r in panel:
        by_user = {u: st.fmean(w["mse"] for w in r["per_window"] if w["user"] == u) for u in users}
        assert all(sum(w["user"] == u for w in r["per_window"]) == 8 for u in users)
        assert np.isclose(st.fmean(by_user.values()), r["mse"], atol=1e-12, rtol=0)
        curves.append({**{k: r[k] for k in ("step", "mse", "mae", "mae_rad", "pred_var")},
                       "mse_ratio_to_zero": r["mse"] / panel[0]["mse"],
                       "mse_by_user": by_user, "prediction_energy": energy(r)})
    x = np.array([curves[0]["mse_by_user"][u] for u in users])
    y = np.array([curves[-1]["mse_by_user"][u] for u in users])
    idx = np.random.default_rng(20261027).integers(0, len(users), (1000, len(users)))
    ratio = float(y.mean() / x.mean())
    ci = np.percentile(y[idx].mean(axis=1) / x[idx].mean(axis=1), [2.5, 97.5]).tolist()
    primary = verdict["criteria"]["pannello_ic95_utenti"]["detail"]
    assert np.allclose([ratio, *ci], [primary["rapporto_medie_utente"], *primary["ic95"]], atol=1e-14, rtol=0)
    first, last = panel[0]["per_window"], panel[-1]["per_window"]
    assert [(w["user"], w["session"], w["window"]) for w in first] == [(w["user"], w["session"], w["window"]) for w in last]
    records = {r["recording"]: r for r in manifest["recordings"]}

    def subset(predicate):
        ids = [i for i, w in enumerate(first) if predicate(w)]
        a, b = st.fmean(first[i]["mse"] for i in ids), st.fmean(last[i]["mse"] for i in ids)
        return {"n_windows": len(ids), "n_users": len({first[i]["user"] for i in ids}),
                "mse_initial": a, "mse_final": b, "ratio": b / a,
                "aggregation": "equal window weights; post-hoc, no gate"}

    groups = {
        "full_support": subset(lambda w: w["valid_fraction"] == 1),
        "partial_support": subset(lambda w: w["valid_fraction"] < 1),
        "at_least_10_valid_patches": subset(lambda w: w["n_valid_patches"] >= 10),
        "prompted": subset(lambda w: not records[w["session"]]["unprompted"]),
        "unprompted": subset(lambda w: records[w["session"]]["unprompted"]),
        "left": subset(lambda w: records[w["session"]]["side"] == "left"),
        "right": subset(lambda w: records[w["session"]]["side"] == "right"),
    }
    blocks = []
    for lo, hi in [(2001, 2500), (4501, 5000)]:
        rows = [r for r in metrics if lo <= r["step"] <= hi and r["pose_windows"] > 0]
        blocks.append({"steps": [lo, hi], "records_with_pose": len(rows), "pose_mean": st.fmean(r["pose"] for r in rows)})
    gradients = [{"step": r["step"], "pose_norm": r["shared_grad_pose"], "anchor_norm": r["shared_grad_anchors"],
                  "ratio": r["shared_grad_pose"] / r["shared_grad_anchors"], "cosine": r["shared_grad_cos"]}
                 for r in metrics if "shared_grad_pose" in r]
    clip = config["grad_clip"]
    clip_factors = [min(1, clip / (r["grad_norm"] + 1e-6)) for r in metrics]
    support = [w["n_valid_patches"] for w in first]
    out = {
        "scope": "Post-hoc local description; frozen gate is verdetto.json. No new thresholds, inference or training.",
        "sources_sha256": {f: hashlib.sha256((run_dir / f).read_bytes()).hexdigest() for f in files},
        "gate": {"status": verdict["status"], "ratio": ratio, "ci95_users": ci,
                 "mse_reduction_pct": 100 * (1 - ratio), "mse_reduction_ci95_pct": [100 * (1 - ci[1]), 100 * (1 - ci[0])]},
        "curves": curves,
        "endpoint": {
            "mae_std_reduction_pct": 100 * (1 - panel[-1]["mae"] / panel[0]["mae"]),
            "mae_rad_reduction_pct": 100 * (1 - panel[-1]["mae_rad"] / panel[0]["mae_rad"]),
            "users_improved": int((y < x).sum()), "users_ratio_le_09": int((y / x <= .9).sum()),
            "user_ratio_min_median_max": [float((y / x).min()), float(np.median(y / x)), float((y / x).max())],
            "windows_improved": sum(b["mse"] < a["mse"] for a, b in zip(first, last)),
            "per_user": {u: {"mse_initial": float(a), "mse_final": float(b), "ratio": float(b / a)} for u, a, b in zip(users, x, y)},
            "prediction_energy": energy(panel[-1]),
            "prediction_energy_note": "These shares do not identify the contribution of dynamics, context or bias to MSE improvement.",
            "prediction_temporal_var_min_median_max": [min(w["pred_var"] for w in last), st.median(w["pred_var"] for w in last), max(w["pred_var"] for w in last)],
        },
        "support": {"n_users": len(users), "n_recordings": len(records), "n_windows": len(first),
                    "valid_patch_min_median_max": [min(support), st.median(support), max(support)],
                    "mean_valid_fraction": st.fmean(w["valid_fraction"] for w in first),
                    "windows_below_half_valid": sum(w["valid_fraction"] < .5 for w in first),
                    "windows_below_10_patches": sum(n < 10 for n in support),
                    "selected_recordings_excluded": manifest["excluded"]},
        "subsets_posthoc": groups,
        "training": {"blocks": blocks, "block_ratio": blocks[1]["pose_mean"] / blocks[0]["pose_mean"],
                     "panel_ratio_5000_to_2000": panel[-1]["mse"] / next(r["mse"] for r in panel if r["step"] == 2000),
                     "global_windows": sum(r["windows"] for r in metrics), "global_pose_window_exposures": sum(r["pose_windows"] for r in metrics),
                     "steps_with_pose": sum(r["pose_windows"] > 0 for r in metrics),
                     "lr_sum": sum(r["lr"] for r in metrics), "lr_peak_units": sum(r["lr"] for r in metrics) / .0001,
                     "gradients_local_rank_last_microbatch": gradients,
                     "mean_gradient_ratio": st.fmean(r["ratio"] for r in gradients),
                     "mean_gradient_cosine": st.fmean(r["cosine"] for r in gradients),
                     "clipped_steps": sum(r["grad_norm"] > clip for r in metrics),
                     "clip_factor_min_median_max": [min(clip_factors), st.median(clip_factors), max(clip_factors)],
                     "clipping_note": "Global total gradient before AdamW; factors are not effective learning rates or evidence of a defect."},
        "small_fixed_panel": {"pose_windows": [r["pose_windows"] for r in fixed],
                              "mse_ratio_5000_to_zero": fixed[-1]["pose"] / fixed[0]["pose"],
                              "mae_rad_ratio_5000_to_zero": fixed[-1]["pose_mae"] / fixed[0]["pose_mae"]},
        "accounting": json.loads((run_dir / "accounting.json").read_text()),
    }
    return out


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
    curves = out["curves"]
    steps = [c["step"] for c in curves]
    ax = axes[0, 0]
    ax.plot(steps, [c["mse_ratio_to_zero"] for c in curves], "o-", color="#176b87", label="Pannello train, 32 utenti")
    ax.axhline(1, color="0.6", linewidth=1)
    ax.axhline(.9, color="#aa6633", linestyle="--", label="Soglia sul punto finale: 0,90")
    ax.axvline(1962, color="0.7", linestyle=":", label="Fine warmup")
    ratio, ci = out["gate"]["ratio"], out["gate"]["ci95_users"]
    ax.errorbar([5000], [ratio], yerr=[[ratio-ci[0]], [ci[1]-ratio]], capsize=5, color="#176b87", label="IC95 appaiato per utente, finale")
    ax.set(title="A. B supera il criterio congelato", xlabel="Aggiornamenti", ylabel="MSE / MSE iniziale", ylim=(.81, 1.035))
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[0, 1]
    ax.plot(steps, [c["mae_rad"] for c in curves], "o-", color="#176b87")
    ax.axvline(1962, color="0.7", linestyle=":")
    ax.set(title="B. Errore assoluto sullo stesso pannello", xlabel="Aggiornamenti", ylabel="MAE media [rad]")
    ax = axes[1, 0]
    ratios = sorted(v["ratio"] for v in out["endpoint"]["per_user"].values())
    ax.scatter(range(1, 33), ratios, c=["#176b87" if r < 1 else "#b34d42" for r in ratios])
    ax.axhline(1, color="0.5", linewidth=1)
    ax.axhline(.9, color="#aa6633", linestyle="--")
    ax.set(title="C. Migliorano 23 utenti su 32", xlabel="Utenti ordinati per rapporto", ylabel="MSE finale / MSE iniziale")
    ax = axes[1, 1]
    ax.stackplot(steps, *[[c["prediction_energy"][k] for c in curves] for k in
                         ["global_joint_means_squared", "between_window_means_variance", "temporal_variance"]],
                 labels=["Media globale per giunto", "Medie diverse fra finestre", "Variazione nel tempo"],
                 colors=["#d0d7de", "#79b5c6", "#176b87"])
    ax.set(title="D. Energia delle predizioni (non guadagno MSE)", xlabel="Aggiornamenti", ylabel="Media del quadrato standardizzato")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("WearUsFM · collaudo B · 5.000 passi\nApprendimento sul train; generalizzazione e movimento corretto ancora da misurare", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/infra/pilot_posa/collaudo_b_20261028")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = describe(args.run_dir)
    (args.run_dir / "analisi.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot(result, args.run_dir / "analisi_b.png")
    print(json.dumps({"gate": result["gate"], "users_improved": result["endpoint"]["users_improved"],
                      "pose_exposures": result["training"]["global_pose_window_exposures"]}, indent=2))
