"""Screening delle varianti di TEMPONet sulla SOLA validazione e regole di scelta congelate in `docs/decisioni.md`.

Regola `diagnosis` (passo 2 della diagnosi, 09/11/2026): fra V1 (bersaglio standardizzato), V3 (padding simmetrico) e V13 (entrambi) si sceglie
la variante con la MAE di validazione (fold1 e fold2 con EMA, seme 0, media su 12 soggetti x 2 fold) piu' bassa, solo se migliora la base
V0 = 9,906 gradi (run 63988277, stessa media) di almeno 0,20; altrimenti resta V0.
Regola `h5` (proposta H5, passo A, 10/11/2026): fra W1 (coseno) e W2 (epoca finale a 1e-5), entrambe su V13, si sceglie la variante con la
stessa MAE di validazione piu' bassa, solo se migliora la base V13 = 9,628 gradi (media di 9,590, job 64005316, e 9,666, job 64010753, seme 0)
di almeno 0,20; altrimenti resta V13: la replica si chiude a 7,585 e il passo B non si fa.
In entrambe: a parita' esatta vale l'ordine delle varianti; uno screening incompleto non sceglie; la soglia si confronta a meno di 1e-9 gradi
di arrotondamento float. Questo script legge solo le chiavi di validazione delle unita': i numeri di test restano chiusi.

Uso: python3 scripts/temponet_db8_screen.py <cartella dello screening con una sottocartella per variante> [--rule diagnosis|h5] [--out <cartella>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

V0_VAL_MAE_DEG = 9.906  # run 63988277, seme 0, fold1/fold2, validazione con EMA, media su 12 soggetti x 2 fold
H5_BASE_VAL_MAE_DEG = 9.628  # V13, seme 0: media di 9,590 (job 64005316) e 9,666 (job 64010753), stessa media
MIN_GAIN_DEG = 0.20
FLOAT_SLACK_DEG = 1e-9  # solo arrotondamento: 9,628 - 9,428 in float e' 0,19999999999999929
VARIANTS = ("V1", "V3", "V13")
H5_VARIANTS = ("W1", "W2")
RULES = {
    "diagnosis": {"base": "V0", "base_val_mae_deg": V0_VAL_MAE_DEG, "variants": VARIANTS, "source": "run 63988277", "closing": "",
                  "title": "Screening delle varianti di TEMPONet (passo 2, solo validazione)"},
    "h5": {"base": "V13", "base_val_mae_deg": H5_BASE_VAL_MAE_DEG, "variants": H5_VARIANTS,
           "source": "media di 9,590, job 64005316, e 9,666, job 64010753, seme 0",
           "closing": ", la replica si chiude a 7,585 e il passo B non si fa",
           "title": "Screening degli schedule del learning rate di TEMPONet (H5, passo A, solo validazione)"},
}
FOLDS = ("fold1", "fold2")
N_SUBJECTS = 12


def variant_val(run_dir: Path) -> dict:
    units = [json.loads(p.read_text()) for p in sorted((run_dir / "units").glob("S*_seed0.json"))]
    subjects = sorted(u["subject"] for u in units)
    vals = np.array([[u["models"][f]["val_ema"]["mae_deg"] for f in FOLDS] for u in sorted(units, key=lambda u: u["subject"])],
                    dtype=float).reshape(-1, len(FOLDS))  # anche senza unita' (cartella vuota): nessun errore, variante incompleta
    return {"subjects": subjects, "complete": subjects == list(range(1, N_SUBJECTS + 1)),
            "val_mae_deg": float(vals.mean()) if len(vals) else None,
            "per_fold_val_mae_deg": {f: float(vals[:, i].mean()) for i, f in enumerate(FOLDS)} if len(vals) else {},
            "per_subject_val_mae_deg": {str(s): float(v) for s, v in zip(subjects, vals.mean(axis=1))}}


def choose(results: dict, v0: float = V0_VAL_MAE_DEG, min_gain: float = MIN_GAIN_DEG, base: str = "V0", closing: str = "") -> dict:
    complete = {k: r for k, r in results.items() if r["complete"]}
    if len(complete) != len(results):
        return {"choice": None, "reason": "screening incompleto: nessuna scelta"}
    best = min(complete, key=lambda k: complete[k]["val_mae_deg"])  # a parita' esatta la prima nell'ordine delle varianti
    gain = v0 - complete[best]["val_mae_deg"]
    if gain >= min_gain - FLOAT_SLACK_DEG:
        return {"choice": best, "gain_deg": gain, "reason": f"{best} migliora {base} di {gain:.3f} gradi (soglia {min_gain:.2f})"}
    return {"choice": base, "gain_deg": gain,
            "reason": f"miglior variante {best}, guadagno {gain:.3f} sotto la soglia {min_gain:.2f}: resta {base}{closing}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("screen_dir")
    ap.add_argument("--rule", choices=sorted(RULES), default="diagnosis")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rule = RULES[args.rule]
    root = Path(args.screen_dir)
    variants = rule["variants"]
    results = {v: variant_val(root / v) for v in variants if (root / v / "units").is_dir()}
    if set(results) != set(variants):
        print(f"varianti mancanti: {sorted(set(variants) - set(results))}", file=sys.stderr)
    chosen = (choose(results, rule["base_val_mae_deg"], MIN_GAIN_DEG, rule["base"], rule["closing"]) if set(results) == set(variants)
              else {"choice": None, "reason": "varianti mancanti"})
    if args.rule == "diagnosis":  # chiavi di prima, invariate
        s = {"v0_val_mae_deg": V0_VAL_MAE_DEG, "min_gain_deg": MIN_GAIN_DEG, "variants": results, **chosen}
    else:
        s = {"rule": args.rule, "base": rule["base"], "base_val_mae_deg": rule["base_val_mae_deg"], "min_gain_deg": MIN_GAIN_DEG,
             "variants": results, **chosen}
    lines = [f"# {rule['title']}", "",
             f"Base {rule['base']} ({rule['source']}): {rule['base_val_mae_deg']:.3f} gradi. Soglia di miglioramento: {MIN_GAIN_DEG:.2f}.", "",
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
