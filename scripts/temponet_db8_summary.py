"""Sintesi della replica di TEMPONet su NinaPro DB8 (`scripts/temponet_db8.py`): grandezza primaria, descrittive e verdetto.

Grandezza primaria (fissata prima dei risultati in `docs/decisioni.md`, voce del 09/11/2026): MAE in gradi sull'acquisizione 3 dopo l'EMA,
media su 12 soggetti x 5 DoA x 2 modelli di fold (fold1, fold2) x semi. Verdetto rispetto ai 6,89 gradi di [Z21] Tab. I con tolleranza
simmetrica: dentro -> «replicata»; sopra -> «non replicata»; sotto -> «non coerente (migliore del pubblicato)»; mancano unita' -> «incompleta».
Tutto il resto (modello `full`, uscite senza EMA, accuratezze, alpha, deviazioni standard) e' descrittivo.

Uso: python3 scripts/temponet_db8_summary.py <cartella del run> [--out <cartella>] [--reference 6.89] [--tolerance 0.30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REFERENCE_MAE_DEG = 6.89  # [Z21] Tab. I, TEMPONet int8 con Block III a 64 canali
REFERENCE = {"mae_deg": 6.89, "mae_sd_deg": 2.08, "acc_10deg": 0.814, "acc_15deg": 0.900, "alpha_mean": 0.862, "alpha_sd": 0.044,
             "lstm_mae_deg": 7.04, "mlp_mae_deg": 7.14, "svm_mae_deg": 7.28}
FOLDS = ("fold1", "fold2")
N_SUBJECTS = 12


def load_units(run_dir: Path) -> list[dict]:
    paths = [p for p in sorted((run_dir / "units").glob("S*_seed*.json")) if not p.name.endswith(".partial.json")]
    return [json.loads(p.read_text()) for p in paths]  # le .partial.json sono avanzamenti di unita' non finite


def verdict(m: float, reference: float, tolerance: float, complete: bool) -> str:
    if not complete:
        return "incompleta"
    if abs(m - reference) <= tolerance:
        return "replicata"
    return "non replicata" if m > reference else "non coerente (migliore del pubblicato)"


def summarize(units: list[dict], seeds_expected: list[int] | None = None, reference: float = REFERENCE_MAE_DEG,
              tolerance: float = 0.30) -> dict:
    if not units:
        raise ValueError("nessuna unita'")
    subjects = sorted({u["subject"] for u in units})
    seeds = sorted({u["seed"] for u in units})
    seeds_expected = sorted(seeds_expected) if seeds_expected is not None else seeds
    by_key = {(u["subject"], u["seed"]): u for u in units}
    complete = subjects == list(range(1, N_SUBJECTS + 1)) and all((s, k) in by_key for s in range(1, N_SUBJECTS + 1)
                                                                  for k in seeds_expected)

    def arr(model: str, field: str, metric: str = "mae_per_doa_deg") -> np.ndarray:
        """(soggetti, semi, ...) per un modello; metric per DoA (5) o scalare."""
        return np.array([[np.asarray(by_key[(s, k)]["models"][model][field][metric], dtype=float) for k in seeds] for s in subjects])

    fold_ema = np.stack([arr(f, "test_ema") for f in FOLDS], axis=2)  # (S, K, 2, 5)
    fold_raw = np.stack([arr(f, "test_raw") for f in FOLDS], axis=2)
    full_keys = [(s, k) for s in subjects for k in seeds if "full" in by_key[(s, k)]["models"]]  # full puo' mancare per alcuni semi
    full_ema = np.array([by_key[key]["models"]["full"]["test_ema"]["mae_per_doa_deg"] for key in full_keys], dtype=float).reshape(-1, 5)
    full_raw = np.array([by_key[key]["models"]["full"]["test_raw"]["mae_per_doa_deg"] for key in full_keys], dtype=float).reshape(-1, 5)
    full_seeds = sorted({k for _, k in full_keys})
    primary = float(fold_ema.mean())
    per_seed = fold_ema.mean(axis=(0, 2, 3))
    subj_doa = fold_ema.mean(axis=(1, 2))  # (S, 5): media su semi e fold
    acc = {f"acc_{t}deg": float(np.mean([[by_key[(s, k)]["models"][f]["test_ema"][f"acc_{t}deg"] for f in FOLDS] for s in subjects
                                         for k in seeds])) for t in (10, 15)}
    alphas = np.array([[[by_key[(s, k)]["models"][f]["alpha"] for f in FOLDS] for k in seeds] for s in subjects])  # (S, K, 2, 5)
    out = {
        "criterion": {"reference_mae_deg": reference, "tolerance_deg": tolerance,
                      "quantity": "MAE test A3 dopo EMA, media su soggetti x DoA x fold1/fold2 x semi"},
        "complete": bool(complete),
        "subjects": subjects,
        "seeds": seeds,
        "primary_mae_deg": primary,
        "verdict": verdict(primary, reference, tolerance, complete),
        "delta_vs_reference_deg": primary - reference,
        "below_lstm_7_04": bool(primary < REFERENCE["lstm_mae_deg"]),
        "per_seed_mae_deg": [float(v) for v in per_seed],
        "seed_sd_deg": float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
        "subject_doa_sd_deg": float(subj_doa.std()),  # deviazione standard (ddof=0) sui 60 valori soggetto x DoA
        "per_subject_mae_deg": {str(s): float(v) for s, v in zip(subjects, subj_doa.mean(axis=1))},
        "per_doa_mae_deg": [float(v) for v in fold_ema.mean(axis=(0, 1, 2))],
        "fold_mae_deg": {f: float(fold_ema[:, :, i].mean()) for i, f in enumerate(FOLDS)},
        "raw_fold_mae_deg": float(fold_raw.mean()),
        "full_seeds": full_seeds,
        "full_n_units": len(full_keys),
        "full_mae_deg": float(full_ema.mean()) if full_keys else None,
        "raw_full_mae_deg": float(full_raw.mean()) if full_keys else None,
        "full_per_seed_mae_deg": {str(k): float(np.mean([by_key[(s, kk)]["models"]["full"]["test_ema"]["mae_deg"] for s, kk in full_keys
                                                         if kk == k])) for k in full_seeds},
        "fold_seed0_mae_deg": float(fold_ema[:, seeds.index(0)].mean()) if 0 in seeds else None,
        **acc,
        "alpha_mean": float(alphas.mean()),
        "alpha_sd": float(alphas.std()),
        "reference": REFERENCE,
    }
    return out


def to_markdown(s: dict) -> str:
    lines = [
        "# Replica di TEMPONet su NinaPro DB8: sintesi",
        "",
        f"- Grandezza primaria (MAE test, EMA, fold1/fold2): **{s['primary_mae_deg']:.3f} gradi**, riferimento {s['criterion']['reference_mae_deg']:.2f}"
        f" (differenza {s['delta_vs_reference_deg']:+.3f}), tolleranza {s['criterion']['tolerance_deg']:.2f}: **{s['verdict']}**.",
        f"- Per seme: {', '.join(f'{v:.3f}' for v in s['per_seed_mae_deg'])}"
        + (f" (dev. std. {s['seed_sd_deg']:.3f})" if s["seed_sd_deg"] is not None else "") + ".",
        f"- Dev. std. sui 60 valori soggetto x DoA: {s['subject_doa_sd_deg']:.2f} (pubblicata 2,08, IWASI 2023).",
        f"- Accuratezza a 10 e 15 gradi: {s['acc_10deg']:.3f} e {s['acc_15deg']:.3f} (pubblicate 0,814 e 0,900).",
        f"- Alpha dell'EMA: media {s['alpha_mean']:.3f}, dev. std. {s['alpha_sd']:.3f} (pubblicate 0,862 e 0,044).",
        f"- Senza EMA (fold): {s['raw_fold_mae_deg']:.3f}."
        + (f" Modello `full` (A1+A2; semi {s['full_seeds']}, {s['full_n_units']} unita'): {s['full_mae_deg']:.3f} con EMA,"
           f" {s['raw_full_mae_deg']:.3f} senza; fold con il solo seme 0: {s['fold_seed0_mae_deg']:.3f}." if s["full_mae_deg"] is not None else ""),
        f"- Sotto la LSTM di Koch et al. (7,04): {'si' if s['below_lstm_7_04'] else 'no'}.",
        "",
        "| Soggetto | MAE (gradi) |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in s["per_subject_mae_deg"].items()]
    lines += ["", "| DoA | MAE (gradi) |", "|---|---|"]
    names = ("rotazione pollice", "flessione pollice", "flessione indice", "flessione medio", "flessione anulare+mignolo")
    lines += [f"| {n} | {v:.3f} |" for n, v in zip(names, s["per_doa_mae_deg"])]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="semi attesi per la completezza")
    ap.add_argument("--reference", type=float, default=REFERENCE_MAE_DEG)
    ap.add_argument("--tolerance", type=float, default=0.30)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    s = summarize(load_units(run_dir), args.seeds, args.reference, args.tolerance)
    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    (out / "summary.md").write_text(to_markdown(s))
    print(to_markdown(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
