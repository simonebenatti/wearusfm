"""Inventario del corpus dai riepiloghi di ingest in `results/passo2/` (stampa una tabella markdown).

Uso: python3 scripts/corpus_inventory.py > docs/inventario_corpus.md  (o solo a schermo)
Legge SOLO i riepiloghi committati; per i dataset non ancora ingeriti riporta le stime dichiarate a mano in
`PENDING` (con la fonte): non sono misure.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "passo2"

# (file, etichetta, fs nativa Hz, canali nell'array, topologia, ruolo)
INGESTED = [
    ("capgmyo_ingest_report.json", "CapgMyo-DBa", 1000, 128, "griglia HD", "pretraining"),
    ("grabmyo_ingest_report.json", "GRABMyo", 2048, 28, "4 anelli", "pretraining"),
    ("putemg_ingest_report_summary.json", "putEMG", 5120, 24, "3 anelli", "pretraining"),
    ("csl_hdemg_ingest_report_summary.json", "CSL-hdemg", 2048, 168, "griglia HD", "pretraining"),
    ("camargo_ingest_report_summary.json", "Camargo 2021", 1000, 11, "sparso (arto inferiore)", "pretraining"),
    ("kaifosh_ingest_report_summary.json", "Kaifosh (Discrete Gestures)", 2000, 16, "anello (Meta)", "pretraining"),
    ("ninapro_db2_ingest_report_summary.json", "NinaPro DB2", 2000, 12, "anello 8 + 4 mirati", "pretraining"),
    ("ninapro_db3_ingest_report_summary.json", "NinaPro DB3 (amputati)", 2000, 12, "anello 8 + 4 mirati", "pretraining"),
    ("ninapro_db4_ingest_report_summary.json", "NinaPro DB4", 2000, 12, "anello 8 + 4 mirati", "pretraining"),
    ("ninapro_db6_ingest_report_summary.json", "NinaPro DB6", 2000, 16, "anello 8 + 6 distali (+2 vuote)", "pretraining"),
    ("ninapro_db7_ingest_report_summary.json", "NinaPro DB7 (2 amputati)", 2000, 12, "anello 8 + 4 mirati", "pretraining"),
]
# Non ancora ingeriti: soggetti/ore da documentazione o da conteggi sui file, NON da un ingest.
PENDING = [
    ("NinaPro DB5", 200, 16, "2 anelli (Myo)", 10, "~8 h", "parser pronto; 10 soggetti x ~0,8 h misurati in locale sui file reali", "in coda (job 59054445)"),
    ("Hyser", 2048, 256, "4 griglie 8x8", 20, "n.d.", "76 GB di `*_raw_*`: ore da calcolare a ingest fatto", "in coda (job 59054442)"),
    ("emg2qwerty", 2000, 32, "2 anelli (Meta, sx+dx)", "n.d.", "~346 h", "documentazione (dataset_access.md): ~346 h, 1.136 file", "parser scritto"),
    ("emg2pose", 2000, 16, "anello (Meta)", 193, "n.d.", "25.253 registrazioni di 193 utenti (CSV di metadati)", "parser scritto"),
    ("Zhang 2026", "2000 + 4000", 8, "anatomico vs equidistante", 64, "n.d.", "1.245 file CSV, 35 GB", "serve una decisione di schema"),
]

# Il riepilogo di putEMG non ha la lista dei soggetti: 44 partecipanti da documentazione (docs/dataset_access.md), non da un ingest
SUBJECT_FALLBACK = {"putEMG": 44}


def _subjects(d: dict):
    for k in ("n_subjects", "n_users"):
        if isinstance(d.get(k), int):
            return d[k]
    for k in ("subjects_processed", "participants_processed"):
        if isinstance(d.get(k), list):
            return len(d[k])
    return None


def main() -> None:
    rows, tot_h, tot_s = [], 0.0, 0
    for fname, label, fs, ch, topo, role in INGESTED:
        d = json.loads((ROOT / fname).read_text())
        h, s = d["hours_total"], _subjects(d)
        mark = ""
        if s is None and label in SUBJECT_FALLBACK:
            s, mark = SUBJECT_FALLBACK[label], "*"
        tot_h += h
        tot_s += s or 0
        rows.append(f"| {label} | {s if s is not None else 'n.d.'}{mark} | {h:.1f} | {fs} | {ch} | {topo} | {role} |")
    print("# Inventario del corpus (dai riepiloghi di ingest, 30/09/2026)\n")
    print("Generato da `scripts/corpus_inventory.py`. Ore = durata dei dati ingeriti (`hours_total` dei report), "
          "soggetti = soggetti/utenti distinti del report.\n")
    print("## Ingeriti\n")
    print("| Dataset | Soggetti | Ore | fs nativa (Hz) | Canali | Topologia | Ruolo |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print(f"| **Totale ingerito** | **{tot_s}** | **{tot_h:.1f}** | | | | |\n")
    print("\\* soggetti da documentazione, non da un ingest.\n")
    print("Attenzione: la somma dei soggetti NON e' il numero di persone distinte: la sovrapposizione fra i DB NinaPro "
          "(fatto n. 2) non e' verificata, e Kaifosh conta utenti diversi da quelli di emg2pose/emg2qwerty solo per "
          "costruzione degli identificatori, non per verifica.\n")
    print("## Non ancora ingeriti (stime dichiarate, NON misure)\n")
    print("| Dataset | fs (Hz) | Canali | Topologia | Soggetti | Ore | Fonte della stima | Stato |")
    print("|---|---|---|---|---|---|---|---|")
    for name, fs, ch, topo, subj, hours, src, status in PENDING:
        print(f"| {name} | {fs} | {ch} | {topo} | {subj} | {hours} | {src} | {status} |")
    print("\nSolo harness (fuori dal pretraining): EPN-612 (200 Hz, 8 canali), UCI-EMG (1 kHz, 8 canali). "
          "Esclusi: NinaPro DB1 (100 Hz, inviluppo), DB8 e DB10 (raw non scaricato), DB9 (solo cinematica).")


if __name__ == "__main__":
    main()
