"""Sintesi della replica di TEMPONet su NinaPro DB8 (`scripts/temponet_db8.py`): grandezza primaria, descrittive e verdetto.

Grandezza primaria (fissata prima dei risultati in `docs/decisioni.md`, voce del 09/11/2026): MAE in gradi sull'acquisizione 3 dopo l'EMA,
media su 12 soggetti x 5 DoA x 2 modelli di fold (fold1, fold2) x semi. Verdetto rispetto ai 6,89 gradi di [Z21] Tab. I con tolleranza
simmetrica: dentro -> «replicata»; sopra -> «non replicata»; sotto -> «non coerente (migliore del pubblicato)»; mancano unita' -> «incompleta».
Tutto il resto (modello `full`, uscite senza EMA, accuratezze, alpha, deviazioni standard) e' descrittivo.
Con `--primary full` (passo 3 della diagnosi, caso H1) la grandezza primaria e' la stessa media sul modello `full`. Nel JSON le chiavi senza
prefisso `full` descrivono sempre i modelli di fold; le descrittive del `full` stanno in `full_details`, e la sintesi in markdown dichiara il
modello di ogni riga (correzione del 10/11/2026: prima le righe per seme, per soggetto e per DoA, sotto la primaria `full`, erano del fold).

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
# NinaPro DB8: «Ten able-bodied (Subjects 1-10) and two right-hand transradial amputee participants (Subjects 11-12)». Simone, 09/11/2026:
# «si tieni separati normali e amputati»: sintesi per gruppo descrittiva; il criterio primario resta sui 12 soggetti, come [Z21].
GROUPS = {"normodotati": tuple(range(1, 11)), "amputati": (11, 12)}


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
              tolerance: float = 0.30, primary_model: str = "fold") -> dict:
    """primary_model: "fold" (replica, run 63988277) o "full" (passo 3 della diagnosi nel caso H1, decisioni.md 09/11/2026)."""
    if primary_model not in ("fold", "full"):
        raise ValueError("primary_model: fold o full")
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
    full_per_seed = {str(k): float(np.mean([by_key[(s, kk)]["models"]["full"]["test_ema"]["mae_deg"] for s, kk in full_keys if kk == k]))
                     for k in full_seeds}
    full_details = None  # descrittive del modello full con le stesse definizioni delle chiavi di fold (correzione del 10/11/2026)
    if full_keys:
        full_subjects = sorted({s for s, _ in full_keys})
        f_subj_doa = np.array([np.mean([by_key[key]["models"]["full"]["test_ema"]["mae_per_doa_deg"] for key in full_keys if key[0] == s],
                                       axis=0) for s in full_subjects])  # (S, 5): media sui semi
        f_alpha = np.array([by_key[key]["models"]["full"]["alpha"] for key in full_keys], dtype=float)
        full_details = {
            "subjects": full_subjects,
            "seed_sd_deg": float(np.std(list(full_per_seed.values()), ddof=1)) if len(full_seeds) > 1 else None,
            "subject_doa_sd_deg": float(f_subj_doa.std()),
            "per_subject_mae_deg": {str(s): float(v) for s, v in zip(full_subjects, f_subj_doa.mean(axis=1))},
            "per_doa_mae_deg": [float(v) for v in f_subj_doa.mean(axis=0)],
            **{f"acc_{t}deg": float(np.mean([by_key[key]["models"]["full"]["test_ema"][f"acc_{t}deg"] for key in full_keys])) for t in (10, 15)},
            "alpha_mean": float(f_alpha.mean()),
            "alpha_sd": float(f_alpha.std()),
        }
    full_complete = all((s, k) in by_key and "full" in by_key[(s, k)]["models"] for s in range(1, N_SUBJECTS + 1) for k in seeds_expected)
    if primary_model == "full":
        primary = float(full_ema.mean()) if full_keys else float("nan")
        complete = complete and full_complete
    else:
        primary = float(fold_ema.mean())
    per_seed = fold_ema.mean(axis=(0, 2, 3))
    subj_doa = fold_ema.mean(axis=(1, 2))  # (S, 5): media su semi e fold
    acc = {f"acc_{t}deg": float(np.mean([[by_key[(s, k)]["models"][f]["test_ema"][f"acc_{t}deg"] for f in FOLDS] for s in subjects
                                         for k in seeds])) for t in (10, 15)}
    alphas = np.array([[[by_key[(s, k)]["models"][f]["alpha"] for f in FOLDS] for k in seeds] for s in subjects])  # (S, K, 2, 5)
    groups = {}
    for gname, gsubjects in GROUPS.items():
        idx = [subjects.index(s) for s in gsubjects if s in subjects]
        if not idx:
            continue
        g_full = [by_key[key]["models"]["full"]["test_ema"]["mae_deg"] for key in full_keys if key[0] in gsubjects]
        g_full_acc = {f"full_acc_{t}deg": float(np.mean([by_key[key]["models"]["full"]["test_ema"][f"acc_{t}deg"] for key in full_keys
                                                          if key[0] in gsubjects])) if g_full else None for t in (10, 15)}
        groups[gname] = {
            "subjects": [subjects[i] for i in idx],
            "fold_mae_deg": float(fold_ema[idx].mean()),
            "raw_fold_mae_deg": float(fold_raw[idx].mean()),
            "per_seed_mae_deg": [float(v) for v in fold_ema[idx].mean(axis=(0, 2, 3))],
            "per_doa_mae_deg": [float(v) for v in fold_ema[idx].mean(axis=(0, 1, 2))],
            "subject_doa_sd_deg": float(subj_doa[idx].std()),
            **{f"acc_{t}deg": float(np.mean([[by_key[(subjects[i], k)]["models"][f]["test_ema"][f"acc_{t}deg"] for f in FOLDS]
                                             for i in idx for k in seeds])) for t in (10, 15)},
            "alpha_mean": float(alphas[idx].mean()),
            "full_mae_deg": float(np.mean(g_full)) if g_full else None,
            "full_n_units": len(g_full),
            **g_full_acc,
        }
    out = {
        "criterion": {"reference_mae_deg": reference, "tolerance_deg": tolerance, "primary_model": primary_model,
                      "quantity": ("MAE test A3 dopo EMA, media su soggetti x DoA x semi del modello full (A1 + A2)" if primary_model == "full"
                                   else "MAE test A3 dopo EMA, media su soggetti x DoA x fold1/fold2 x semi")},
        "fold_mae_all_deg": float(fold_ema.mean()),
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
        "full_per_seed_mae_deg": full_per_seed,
        "fold_seed0_mae_deg": float(fold_ema[:, seeds.index(0)].mean()) if 0 in seeds else None,
        # le chiavi senza prefisso full (per seme, per soggetto, per DoA, deviazioni standard, accuratezze, alpha) descrivono i modelli di fold
        "full_details": full_details,
        **acc,
        "alpha_mean": float(alphas.mean()),
        "alpha_sd": float(alphas.std()),
        "groups": groups,  # descrittivo (richiesta di Simone del 09/11), nessun verdetto per gruppo
        "reference": REFERENCE,
    }
    return out


DOA_NAMES = ("rotazione pollice", "flessione pollice", "flessione indice", "flessione medio", "flessione anulare+mignolo")


def _markdown_full_primary(s: dict) -> str:
    """Grandezza primaria sul modello full: ogni riga descrittiva dichiara il suo modello; il fold e' la grandezza secondaria."""
    fd = s["full_details"]
    seeds = s["full_per_seed_mae_deg"]
    lines = [
        "# Replica di TEMPONet su NinaPro DB8: sintesi",
        "",
        f"- Grandezza primaria (MAE test, EMA, modello full): **{s['primary_mae_deg']:.3f} gradi**, riferimento {s['criterion']['reference_mae_deg']:.2f}"
        f" (differenza {s['delta_vs_reference_deg']:+.3f}), tolleranza {s['criterion']['tolerance_deg']:.2f}: **{s['verdict']}**.",
        f"- `full` per seme: {', '.join(f'{v:.3f}' for v in seeds.values())}"
        + (f" (dev. std. {fd['seed_sd_deg']:.3f})" if fd["seed_sd_deg"] is not None else "") + f"; senza EMA {s['raw_full_mae_deg']:.3f}.",
        f"- `full`, dev. std. sui 60 valori soggetto x DoA: {fd['subject_doa_sd_deg']:.2f} (pubblicata 2,08, IWASI 2023).",
        f"- `full`, accuratezza a 10 e 15 gradi: {fd['acc_10deg']:.3f} e {fd['acc_15deg']:.3f} (pubblicate 0,814 e 0,900).",
        f"- `full`, alpha dell'EMA (dalle predizioni fuori fold): media {fd['alpha_mean']:.3f}, dev. std. {fd['alpha_sd']:.3f}"
        " (pubblicate 0,862 e 0,044).",
        f"- Fold (secondaria): {s['fold_mae_all_deg']:.3f} con EMA (per seme {', '.join(f'{v:.3f}' for v in s['per_seed_mae_deg'])};"
        f" fold1 {s['fold_mae_deg']['fold1']:.3f}, fold2 {s['fold_mae_deg']['fold2']:.3f}), {s['raw_fold_mae_deg']:.3f} senza;"
        f" accuratezza {s['acc_10deg']:.3f} e {s['acc_15deg']:.3f}; alpha medio {s['alpha_mean']:.3f}.",
        f"- Sotto la LSTM di Koch et al. (7,04): {'si' if s['below_lstm_7_04'] else 'no'}.",
        "",
        "| Soggetto | `full` (gradi) | fold (gradi) |",
        "|---|---|---|",
    ]
    lines += [f"| {k}{' (amputato)' if int(k) in GROUPS['amputati'] else ''} | {fd['per_subject_mae_deg'].get(k, float('nan')):.3f} | {v:.3f} |"
              for k, v in s["per_subject_mae_deg"].items()]
    if s.get("groups"):
        lines += ["", "Normodotati e amputati separati (descrittivo, nessun verdetto per gruppo; il criterio resta sui 12 soggetti):", "",
                  "| Gruppo | Soggetti | `full` | `full` acc. 10 / 15 | fold con EMA | fold senza EMA | fold acc. 10 / 15 | fold alpha medio |",
                  "|---|---|---|---|---|---|---|---|"]
        for gname, g in s["groups"].items():
            subj = f"{g['subjects'][0]}-{g['subjects'][-1]}" if len(g["subjects"]) > 1 else str(g["subjects"][0])
            full = (f"{g['full_mae_deg']:.3f} | {g['full_acc_10deg']:.3f} / {g['full_acc_15deg']:.3f}" if g["full_mae_deg"] is not None
                    else "- | -")
            lines.append(f"| {gname} | {subj} | {full} | {g['fold_mae_deg']:.3f} | {g['raw_fold_mae_deg']:.3f} "
                         f"| {g['acc_10deg']:.3f} / {g['acc_15deg']:.3f} | {g['alpha_mean']:.3f} |")
    lines += ["", "| DoA | `full` (gradi) | fold (gradi) |", "|---|---|---|"]
    lines += [f"| {n} | {a:.3f} | {b:.3f} |" for n, a, b in zip(DOA_NAMES, fd["per_doa_mae_deg"], s["per_doa_mae_deg"])]
    return "\n".join(lines) + "\n"


def to_markdown(s: dict) -> str:
    if s["criterion"].get("primary_model") == "full" and s.get("full_details"):
        return _markdown_full_primary(s)
    lines = [
        "# Replica di TEMPONet su NinaPro DB8: sintesi",
        "",
        f"- Grandezza primaria (MAE test, EMA, {'modello full' if s['criterion'].get('primary_model') == 'full' else 'fold1/fold2'}): **{s['primary_mae_deg']:.3f} gradi**, riferimento {s['criterion']['reference_mae_deg']:.2f}"
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
    lines += [f"| {k}{' (amputato)' if int(k) in GROUPS['amputati'] else ''} | {v:.3f} |" for k, v in s["per_subject_mae_deg"].items()]
    if s.get("groups"):
        lines += ["", "Normodotati e amputati separati (descrittivo, nessun verdetto per gruppo; il criterio resta sui 12 soggetti):", "",
                  "| Gruppo | Soggetti | MAE fold con EMA | senza EMA | `full` | acc. 10 gradi | acc. 15 gradi | alpha medio |",
                  "|---|---|---|---|---|---|---|---|"]
        for gname, g in s["groups"].items():
            full = f"{g['full_mae_deg']:.3f}" if g["full_mae_deg"] is not None else "-"
            subj = f"{g['subjects'][0]}-{g['subjects'][-1]}" if len(g["subjects"]) > 1 else str(g["subjects"][0])
            lines.append(f"| {gname} | {subj} | {g['fold_mae_deg']:.3f} | {g['raw_fold_mae_deg']:.3f} | {full} | {g['acc_10deg']:.3f} "
                         f"| {g['acc_15deg']:.3f} | {g['alpha_mean']:.3f} |")
    lines += ["", "| DoA | MAE (gradi) |", "|---|---|"]
    lines += [f"| {n} | {v:.3f} |" for n, v in zip(DOA_NAMES, s["per_doa_mae_deg"])]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="semi attesi per la completezza")
    ap.add_argument("--reference", type=float, default=REFERENCE_MAE_DEG)
    ap.add_argument("--tolerance", type=float, default=0.30)
    ap.add_argument("--primary", choices=("fold", "full"), default="fold", help="modello della grandezza primaria")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    s = summarize(load_units(run_dir), args.seeds, args.reference, args.tolerance, args.primary)
    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    (out / "summary.md").write_text(to_markdown(s))
    print(to_markdown(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
