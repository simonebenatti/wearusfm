"""Passo 2 della diagnosi di TEMPONet: sintesi dello screening delle varianti sulla SOLA validazione e regola di scelta congelata.

Regola (`docs/decisioni.md`, 09/11/2026): fra V1 (bersaglio standardizzato), V3 (padding simmetrico) e V13 (entrambi) si sceglie la variante con
la MAE di validazione (fold1 e fold2 con EMA, seme 0, media su 12 soggetti x 2 fold) piu' bassa, solo se migliora la base V0 = 9,906 gradi (run
63988277, stessa media) di almeno 0,20; altrimenti resta V0. Questo script legge solo le chiavi di validazione delle unita': i numeri di test
restano chiusi fino al passo 3.

Uso: python3 scripts/temponet_db8_screen.py <cartella dello screening con sottocartelle V1, V3, V13> [--out <cartella>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

V0_VAL_MAE_DEG = 9.906  # run 63988277, seme 0, fold1/fold2, validazione con EMA, media su 12 soggetti x 2 fold
MIN_GAIN_DEG = 0.20
VARIANTS = ("V1", "V3", "V13")
FOLDS = ("fold1", "fold2")
N_SUBJECTS = 12


def variant_val(run_dir: Path) -> dict:
    units = [json.loads(p.read_text()) for p in sorted((run_dir / "units").glob("S*_seed0.json"))]
    subjects = sorted(u["subject"] for u in units)
    vals = np.array([[u["models"][f]["val_ema"]["mae_deg"] for f in FOLDS] for u in sorted(units, key=lambda u: u["subject"])])
    return {"subjects": subjects, "complete": subjects == list(range(1, N_SUBJECTS + 1)),
            "val_mae_deg": float(vals.mean()) if len(vals) else None,
            "per_fold_val_mae_deg": {f: float(vals[:, i].mean()) for i, f in enumerate(FOLDS)} if len(vals) else {},
            "per_subject_val_mae_deg": {str(s): float(v) for s, v in zip(subjects, vals.mean(axis=1))}}


def choose(results: dict, v0: float = V0_VAL_MAE_DEG, min_gain: float = MIN_GAIN_DEG) -> dict:
    complete = {k: r for k, r in results.items() if r["complete"]}
    if len(complete) != len(results):
        return {"choice": None, "reason": "screening incompleto: nessuna scelta"}
    best = min(complete, key=lambda k: complete[k]["val_mae_deg"])
    gain = v0 - complete[best]["val_mae_deg"]
    if gain >= min_gain:
        return {"choice": best, "gain_deg": gain, "reason": f"{best} migliora V0 di {gain:.3f} gradi (soglia {min_gain:.2f})"}
    return {"choice": "V0", "gain_deg": gain, "reason": f"miglior variante {best}, guadagno {gain:.3f} sotto la soglia {min_gain:.2f}: resta V0"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("screen_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.screen_dir)
    results = {v: variant_val(root / v) for v in VARIANTS if (root / v / "units").is_dir()}
    if set(results) != set(VARIANTS):
        print(f"varianti mancanti: {sorted(set(VARIANTS) - set(results))}", file=sys.stderr)
    s = {"v0_val_mae_deg": V0_VAL_MAE_DEG, "min_gain_deg": MIN_GAIN_DEG, "variants": results,
         **(choose(results) if set(results) == set(VARIANTS) else {"choice": None, "reason": "varianti mancanti"})}
    lines = ["# Screening delle varianti di TEMPONet (passo 2, solo validazione)", "",
             f"Base V0 (run 63988277): {V0_VAL_MAE_DEG:.3f} gradi. Soglia di miglioramento: {MIN_GAIN_DEG:.2f}.", "",
             "| Variante | validazione (fold, EMA) | fold1 | fold2 | soggetti |", "|---|---|---|---|---|"]
    for v, r in results.items():
        if r["val_mae_deg"] is None:
            lines.append(f"| {v} | - | - | - | 0 |")
            continue
        lines.append(f"| {v} | {r['val_mae_deg']:.3f} | {r['per_fold_val_mae_deg']['fold1']:.3f} | {r['per_fold_val_mae_deg']['fold2']:.3f} "
                     f"| {len(r['subjects'])} |")
    lines += ["", f"**Scelta: {s['choice']}** ({s['reason']})."]
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen.json").write_text(json.dumps(s, indent=1))
    (out / "screen.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
