"""Recalculate paired CIs from saved moments and draw the E3 B result.

Run with Python/numpy/matplotlib; no model, cache, training or cluster access.
The source report is immutable. This script imports no experiment code.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
raw = (ROOT / "final/report.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "761276493751bcb959fa0a49efa0a51da825b8730bc6d6125b95eb1164770c94"
report = json.loads(raw)
verified = {}
for group in ("dynamic", "all", "static"):
    h = report["head"]["groups"][group]["centered_support_n_ge_2"]
    r = report["ridge"]["groups"][group]["centered_support_n_ge_2"]
    users = sorted(h["per_user"])
    assert users == sorted(r["per_user"])
    diffs, variances = [], []
    for user in users:
        hu, ru = h["per_user"][user], r["per_user"][user]
        assert hu["n_valid_elements"] == ru["n_valid_elements"]
        assert hu["sums"]["vary"] == ru["sums"]["vary"]
        diffs.append(ru["sums"]["centered"] - hu["sums"]["centered"])
        variances.append(hu["sums"]["vary"])
    diffs, variances = np.asarray(diffs), np.asarray(variances)
    draws = np.random.default_rng(20261009).integers(0, len(users), (1000, len(users)))
    delta = float(diffs.sum() / variances.sum())
    ci = np.percentile(diffs[draws].sum(1) / variances[draws].sum(1), [2.5, 97.5])
    np.testing.assert_allclose(delta, report["paired"][group]["delta_ratio"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ci, report["paired"][group]["delta_ratio_ci95"], rtol=0, atol=1e-12)
    verified[group] = {
        "delta_ratio": delta, "ci95": ci.tolist(),
        "users_ridge_higher_error": int((diffs > 0).sum()),
        "users_ridge_lower_error": int((diffs < 0).sum()),
        "head_ratio": h["centered_mse_ratio"], "ridge_ratio": r["centered_mse_ratio"],
        "mean_mse_delta": r["means"]["mean_error"] - h["means"]["mean_error"],
        "centered_mse_delta": r["means"]["centered"] - h["means"]["centered"],
        "total_mse_delta": r["means"]["total"] - h["means"]["total"],
    }
(ROOT / "independent_recalculation.json").write_text(json.dumps({
    "report_sha256": hashlib.sha256(raw).hexdigest(), "n_bootstrap": 1000,
    "seed": 20261009, "method": "Independent pooled sums and paired resampling of users; no experiment-code import",
    "groups": verified,
}, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
fig.patch.set_facecolor("#fafbfc")
for ax in axes:
    ax.set_facecolor("#fafbfc")
fig.suptitle("E3 B · la ridge migliora il livello medio, non la dinamica", x=.06, ha="left", fontsize=17, fontweight="bold", y=.97)
fig.text(.06, .905, "Student a 5.000 passi · 15 utenti esclusi dal training · lettura esplorativa", color="#526074")
ax = axes[0]
means = [report[k]["groups"]["dynamic"]["centered_support_n_ge_2"]["means"] for k in ("head", "ridge")]
level = np.array([m["mean_error"] for m in means])
motion = np.array([m["centered"] for m in means])
x = np.arange(2)
ax.bar(x, level, .55, color="#466f97", label="Errore del livello medio")
ax.bar(x, motion, .55, bottom=level, color="#e4ac62", label="Errore della dinamica")
for i in range(2):
    ax.text(i, level[i]/2, f"{level[i]:.4f}", ha="center", va="center", color="white", fontweight="bold")
    ax.text(i, level[i]+motion[i]/2, f"{motion[i]:.4f}", ha="center", va="center", color="#263545", fontweight="bold")
    ax.text(i, level[i]+motion[i]+.025, f"Totale {level[i]+motion[i]:.4f}", ha="center", fontsize=10)
ax.set_xticks(x, ["Testa attuale", "Ridge"])
ax.set_ylim(0, 1.35)
ax.set_ylabel("MSE standardizzata · meno è meglio")
ax.set_title("Finestre dinamiche: scomposizione dell'errore", fontsize=11, pad=15)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False, fontsize=9)
ax.yaxis.grid(True, alpha=.15)
ax.set_axisbelow(True)

ax = axes[1]
ax.axvline(0, color="#6c7685", lw=1, ls="--")
for y, group, label in [(2, "dynamic", "Dinamiche · primario"), (1, "all", "Tutte · secondario"), (0, "static", "Poco dinamiche · secondario")]:
    p = verified[group]
    d = p["delta_ratio"] * 100
    lo, hi = np.array(p["ci95"]) * 100
    color = "#ab4b46" if group == "dynamic" else "#466f97"
    ax.errorbar(d, y, xerr=[[d-lo], [hi-d]], fmt="o", color=color, capsize=5, lw=2)
    ax.text(-2.95, y+.29, label, fontsize=10, fontweight="bold" if group == "dynamic" else "normal")
    ax.text(-2.95, y-.25, f"{d:+.3f}  [IC95 {lo:+.3f}; {hi:+.3f}]", fontsize=10, color=color)
ax.set_xlim(-3, 1)
ax.set_ylim(-.65, 2.65)
ax.set_yticks([])
ax.set_xlabel("Differenza ridge − testa, in punti percentuali di Vy\n← meno errore ridge                    più errore ridge →", fontsize=10)
ax.set_title("Dinamica: confronto appaiato con IC95", fontsize=11, pad=15)
ax.xaxis.grid(True, alpha=.15)
fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.23, wspace=.31)
fig.text(.06, .025, "Centratura per finestra/giunto; baseline piatta = Vy. IC: bootstrap utenti, 1.000 repliche.\nLa differenza primaria è piccola e sfavorevole alla ridge; non misura un limite assoluto dell'encoder.", fontsize=9, color="#526074")
fig.savefig(ROOT / "e3_b_results.png", dpi=160, facecolor=fig.get_facecolor())
fig.savefig(ROOT / "e3_b_results.pdf", facecolor=fig.get_facecolor())
print(json.dumps(verified, indent=2))
