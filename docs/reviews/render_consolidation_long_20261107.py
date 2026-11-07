"""Render the completed, prespecified long-evaluation results; no fitting or inference.

Inputs are immutable Git blobs from Claude's delivered snapshot. The plot shows
point estimates; paired contrast intervals are written separately, never inferred
from the overlap of marginal intervals. No subject identifiers are exported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/wearusfm-long-review-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REVISION = "c109bfef5a3f80cc4b2334e3554c2171db938467"
RECEIPTS = "results/infra/consolidation_long_20261105/execution/receipts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output(["git", "show", f"{REVISION}:{RECEIPTS}/LONG-SUMMARY-20261105.json"], cwd=args.repo)
    summary = json.loads(raw)
    classification = summary["classification_result"]
    pose = summary["reports"]["pose"]["comparison"]
    assert summary["complete"] and classification["complete"] and pose["complete"]
    result = {
        "snapshot": REVISION,
        "summary_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Published prespecified results, not a new analysis or model fit; paired subject bootstrap, development data, one pretraining seed.",
        "classification": {"tasks": classification["tasks"], "reading": classification["reading"]},
        "pose": {"cells": pose["cells"], "contrasts": pose["contrasts"]},
    }
    (args.out / "scientific_readout.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def interval(x, digits=2, signed=False, keys=("point", "ci95")):
        point, (low, high) = x[keys[0]], x[keys[1]]
        fmt = ("+" if signed else "") + f".{digits}f"
        return f"{point:{fmt}} [{low:{fmt}}; {high:{fmt}}]"

    lines = ["# Contrasti prespecificati — valutazione lunga", "", "Fonte immutabile: `" + REVISION + "`. IC95% appaiati per soggetto, 2.000 repliche, seed 20261102. Nessun aggiustamento per confronti multipli.", "", "## Classificazione", "", "| Contrasto | P1, pp [IC95%] | P2, pp [IC95%] | V-R1 congiunta |", "|---|---:|---:|---|"]
    for name, reading in classification["reading"]["contrasts"].items():
        vals = [interval(classification["tasks"][task]["contrasts"][name], signed=True, keys=("delta_pp", "ci95_pp")) for task in ("p1", "p2")]
        lines.append(f"| {name} | {vals[0]} | {vals[1]} | {reading['status']} |")
    lines += ["", "## Posa", "", "E_dyn: riduzione relativa positiva = miglioramento; delta E2 positivo = maggior vantaggio dell'allineamento. Gruppo dynamic per E_dyn; E2 sul proprio supporto comune crop6. I due contrasti primari sono post_ridge e trained_head 49.046−19.618.", "", "| Contrasto | Riduzione E_dyn, % [IC95%] | Delta gap E2, rad [IC95%] | Lettura E_dyn |", "|---|---:|---:|---|"]
    for name, contrast in pose["contrasts"].items():
        d = contrast["groups"]["dynamic"]
        lines.append(f"| {name} | {interval(d['relative_improvement_percent'], signed=True)} | {interval(contrast['E2']['delta']['gap_mae'], digits=5, signed=True)} | {d['assessment']} |")
    lines += ["", "## Altri gruppi della posa", "", "| Contrasto | Riduzione E_dyn all, % [IC95%] | Riduzione E_dyn static, % [IC95%] |", "|---|---:|---:|"]
    for name, contrast in pose["contrasts"].items():
        values = [interval(contrast["groups"][g]["relative_improvement_percent"], signed=True) for g in ("all", "static")]
        lines.append(f"| {name} | {values[0]} | {values[1]} |")
    (args.out / "prespecified_contrasts.md").write_text("\n".join(lines) + "\n")

    doses = [19618, 39237, 49046]
    colors = {"pool": "#334155", "post": "#d97706", "post_ridge": "#007c91", "trained_head": "#b44269"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0))
    for ax, task in zip(axes[0], ("p1", "p2")):
        for view, label, style in (("pool", "Pool storico · primario", "-"), ("post", "Uscita backbone · secondario", "--")):
            y = [100 * classification["tasks"][task]["cells"][f"{dose}/{view}"]["balanced_accuracy"] for dose in doses]
            ax.plot(doses, y, marker="o", ls=style, color=colors[view], label=label)
            for x, yy in zip(doses, y):
                offset = -17 if view == "pool" else 8
                if task == "p1" and x == 19618:
                    offset = 8 if view == "pool" else -17
                ax.annotate(f"{yy:.2f}", (x, yy), xytext=(0, offset), textcoords="offset points", ha="center", color=colors[view], fontsize=9)
        ax.set(title=f"Classificazione {task.upper()}", ylabel="Balanced accuracy (%)")
        ax.margins(y=.28)
        ax.legend(loc="upper left", fontsize=8)
    for reader, label in (("post_ridge", "Ridge post-backbone"), ("trained_head", "Testa allenata")):
        edyn = [pose["cells"][f"{d}/{reader}"]["groups"]["dynamic"]["E_dyn"]["point"] for d in doses]
        e2 = [pose["cells"][f"{d}/{reader}"]["E2"]["gap_mae"]["point"] for d in doses]
        for ax, y in ((axes[1, 0], edyn), (axes[1, 1], e2)):
            ax.plot(doses, y, marker="o", color=colors[reader], label=label)
            for x, yy in zip(doses, y):
                offset = 9 if reader == "post_ridge" else -17
                if ax is axes[1, 1]: offset = -17 if reader == "post_ridge" else 9
                ax.annotate(f"{yy:.3f}", (x, yy), xytext=(0, offset), textcoords="offset points", ha="center", color=colors[reader], fontsize=9)
    axes[1, 0].set(title="Posa: errore sul movimento ↓", ylabel="E_dyn · finestre dinamiche")
    axes[1, 1].set(title="Posa: vantaggio dell'allineamento ↑", ylabel="Gap MAE E2 (radianti)")
    for ax in axes[1]:
        ax.legend(fontsize=8, loc="best")
        ax.margins(y=.22)
    for ax in axes.flat:
        ax.set_xticks(doses, ["19.618", "39.237", "49.046"])
        ax.set_xlabel("Passi di pretraining")
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("T100-G · 30M — cosa cambia prolungando il pretraining", fontsize=15, y=.99)
    fig.text(.5, .015, "Stime puntuali. IC95% appaiati dei contrasti nella tabella allegata.\nStesso seme; dati di sviluppo. La fase 39.237→49.046 comprende il decadimento del learning rate.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.savefig(args.out / "long_results.png", dpi=170)
    fig.savefig(args.out / "long_results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
