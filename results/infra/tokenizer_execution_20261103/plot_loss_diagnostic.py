"""Render saved aggregate training diagnostics; no cluster access or fitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / "loss_diagnostic_20261103.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
colors = {"T25-L": "#2878b5", "T100-L": "#e1812c", "T100-G": "#27965e"}
for arm, item in data["arms"].items():
    fixed = [r for r in item["fixed"] if 5000 <= r["step"] <= data["common_step"]]
    panel = [r for r in item["pose_panel"] if 5000 <= r["step"] <= data["common_step"]]
    for ax, ys in zip(axes.flat, (
        [r["total"] for r in fixed],
        [r["total"] - r["pose"] for r in fixed],
        [r["pose"] for r in fixed],
        [r["mse"] for r in panel],
    )):
        rows = panel if ax is axes[1, 1] else fixed
        ax.plot([r["step"] for r in rows], ys, label=arm, color=colors[arm], linewidth=1.8)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
for ax, title in zip(axes.flat, (
    "Loss totale — 8 batch fissi", "Ancore × 0,2 — stessi batch fissi",
    "Loss posa — stessi batch fissi", "MSE posa — pannello fisso di 256 finestre",
)):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Errore (più basso è meglio)")
for ax in axes[1]:
    ax.set_xlabel("Passo di training")
axes[0, 0].legend(frameon=False)
fig.suptitle("Tokenizer: diagnostica sul training a pari passo, fino a 14.500", fontsize=14)
fig.text(0.5, 0.012, "Student in eval · punti ogni 500 passi, senza smoothing · JEPA ha peso zero · non è validazione", ha="center", fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(root / "loss_diagnostic_20261103.png", dpi=160)
