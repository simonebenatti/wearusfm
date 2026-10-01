#!/usr/bin/env python3
"""Tabella del corpus per il manifest D9 (passo 4): ore, soggetti, canali, classe di quota, D_t e D_c per dataset, dai riepiloghi di ingest in
`results/passo2/`. Stampa markdown. Le scelte (classi, ruolo) sono quelle PROPOSTE in `docs/proposta_manifest_d9.md`, non ancora firmate.

Definizioni (v10 §10.3): D_t = time-patch unici = ore x 3600 / patch; D_c = source-channel-patch unici = somma(ore x canali validi) x 3600 / patch,
contati dopo il QC e prima di ogni augmentation. Patch: 25 ms (default di lavoro di D10, non ancora deciso). I canali validi sono una STIMA: canali
dell'array meno quelli scartati dal QC riportati nei riepiloghi (media per sessione dove serve).

  python3 scripts/manifest_table.py                       # tabella e frazioni per classe
  python3 scripts/manifest_table.py --extra emg2pose=400  # con un'ipotesi di ore per un dataset non ancora ingerito
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent / "results" / "passo2"
PATCH_S = 0.025

# chiave: (file di riepilogo, etichetta, classe di quota proposta, canali validi stimati, ruolo proposto)
# Classi (v10 §2.8): A radi/anatomici e misti (NinaPro anello + mirati, Camargo, Zhang anatomico); B anelli e fasce; C griglie HD.
DATASETS = {
    "ninapro_db2": ("ninapro_db2_ingest_report_summary.json", "NinaPro DB2", "A", 12.0, "pretraining"),
    "ninapro_db3": ("ninapro_db3_ingest_report_summary.json", "NinaPro DB3 (amputati)", "A", 12.0 - 4 / 11, "pretraining"),
    "ninapro_db4": ("ninapro_db4_ingest_report_summary.json", "NinaPro DB4", "A", 12.0, "pretraining"),
    "ninapro_db6": ("ninapro_db6_ingest_report_summary.json", "NinaPro DB6", "A", 14.0, "pretraining"),
    "ninapro_db7": ("ninapro_db7_ingest_report_summary.json", "NinaPro DB7", "A", 12.0, "pretraining"),
    "camargo2021": ("camargo_ingest_report_summary.json", "Camargo 2021", "A", 11.0, "pretraining"),
    "zhang2026_anatomical": ("zhang2026_ingest_report_summary.json", "Zhang 2026, anatomical", "A", 8.0, "pretraining"),
    "zhang2026_random": ("zhang2026_ingest_report_summary.json", "Zhang 2026, random", "B", 8.0, "pretraining"),
    "kaifosh": ("kaifosh_ingest_report_summary.json", "Kaifosh (Discrete Gestures)", "B", 16.0, "D3b: da decidere"),
    "emg2qwerty": ("emg2qwerty_ingest_report_summary.json", "emg2qwerty", "B", 32.0, "pretraining"),
    "emg2pose": ("emg2pose_ingest_report_summary.json", "emg2pose", "B", 16.0, "pretraining"),
    "grabmyo": ("grabmyo_ingest_report.json", "GRABMyo", "B", 28.0, "pretraining"),
    "putemg": ("putemg_ingest_report_summary.json", "putEMG", "B", 24.0, "pretraining"),
    "ninapro_db5": ("ninapro_db5_ingest_report_summary.json", "NinaPro DB5 (200 Hz)", "B", 16.0, "pretraining"),
    "ninapro_db8": ("ninapro_db8_ingest_report_summary.json", "NinaPro DB8", "B", 16.0 - 7 / 12, "pretraining"),
    "ninapro_db10": ("ninapro_db10_ingest_report_summary.json", "NinaPro DB10 (MDS1)", "B", 12.0, "pretraining"),
    "capgmyo": ("capgmyo_ingest_report.json", "CapgMyo-DBa", "C", 128.0, "pretraining"),
    "csl_hdemg": ("csl_hdemg_ingest_report_summary.json", "CSL-hdemg", "C", 168.0, "pretraining"),
    "hyser": ("hyser_ingest_report_summary.json", "Hyser", "C", 256.0, "pretraining"),
}


def _subjects(d: dict) -> int | None:
    for k in ("n_subjects", "n_users"):
        if isinstance(d.get(k), int):
            return d[k]
    for k in ("subjects_processed", "participants_processed"):
        if isinstance(d.get(k), list):
            return len(d[k])
    return None


def load_row(key: str, extra: dict[str, float]) -> dict:
    fname, label, cls, ch, role = DATASETS[key]
    p = ROOT / fname
    row = {"key": key, "label": label, "class": cls, "channels": ch, "role": role, "hours": None, "subjects": None, "source": "riepilogo"}
    if p.exists():
        d = json.loads(p.read_text())
        if key.startswith("zhang2026_"):
            mode = key.split("_", 1)[1]
            sess = [s for s in d["per_session"] if s["mode"] == mode]
            row["hours"], row["subjects"] = sum(s["hours"] for s in sess), len({s["subject"] for s in sess})
        else:
            row["hours"], row["subjects"] = d["hours_total"], _subjects(d)
            if key == "putemg" and row["subjects"] is None:
                row["subjects"], row["source"] = 44, "riepilogo (soggetti da documentazione)"
    elif key in extra:
        row["hours"], row["source"] = extra[key], "IPOTESI (ingest non concluso)"
    else:
        row["source"] = "mancante (ingest non concluso)"
    return row


def table(extra: dict[str, float], include_kaifosh: bool) -> tuple[list[dict], dict]:
    rows = [load_row(k, extra) for k in DATASETS]
    per_hour = 3600 / PATCH_S
    for r in rows:
        r["d_t"] = r["hours"] * per_hour if r["hours"] is not None else None
        r["d_c"] = r["hours"] * r["channels"] * per_hour if r["hours"] is not None else None
    used = [r for r in rows if r["hours"] is not None and (include_kaifosh or r["key"] != "kaifosh")]
    tot = sum(r["hours"] for r in used)
    classes = {c: sum(r["hours"] for r in used if r["class"] == c) for c in "ABC"}
    return rows, {"hours_total": tot, "hours_by_class": classes, "fraction_by_class": {c: h / tot for c, h in classes.items()},
                  "d_t": sum(r["d_t"] for r in used), "d_c": sum(r["d_c"] for r in used)}


def passes(quota: float, hours_class: float, hours_total: float, epochs: float = 4.0) -> float:
    """Passaggi sui dati di una classe se la classe riceve `quota` dei campioni e si consumano `epochs` volte le ore totali (v10 §10.3: D~ = 4 D)."""
    return quota * epochs * hours_total / hours_class


def allocate(rows: list[dict], quota: dict[str, float], alpha: float, max_passes: float, hours_total: float, epochs: float = 4.0) -> dict:
    """Ripartizione per dataset con la funzione del manifest (src/wearusfm/data/pretraining_manifest.py).
    Ritorna {"per_dataset": {chiave: (quota, passaggi)}, "unused_quota": {classe: quota}}."""
    from wearusfm.data.pretraining_manifest import allocate as _allocate

    units = {r["key"]: (r["class"], r["hours"]) for r in rows}
    a = _allocate(units, quota, alpha, max_passes, epochs, hours_total=hours_total)
    return {"per_dataset": a["per_unit"], "unused_quota": a["unused_quota"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", action="append", default=[], metavar="CHIAVE=ORE", help="ore ipotizzate per un dataset senza riepilogo")
    ap.add_argument("--without-kaifosh", action="store_true", help="Kaifosh fuori dal pretraining (D3b, opzione benchmark)")
    ap.add_argument("--quota", default=None, help="quote garantite per classe, es. A=0.2,B=0.75,C=0.05: stampa la ripartizione per dataset")
    ap.add_argument("--alpha", type=float, default=0.5, help="pesi dentro la classe proporzionali a ore^alpha")
    ap.add_argument("--max-passes", type=float, default=8.0, help="tetto ai passaggi per dataset (con 4 epoche di consumo totale)")
    args = ap.parse_args(argv)
    extra = {k: float(v) for k, v in (e.split("=", 1) for e in args.extra)}
    rows, tot = table(extra, not args.without_kaifosh)
    print("| Dataset | Classe | Ore | Soggetti | Canali validi (stima) | D_t (milioni) | D_c (miliardi) | Ruolo proposto | Fonte |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        h = f"{r['hours']:.1f}" if r["hours"] is not None else "-"
        dt = f"{r['d_t'] / 1e6:.1f}" if r["d_t"] is not None else "-"
        dc = f"{r['d_c'] / 1e9:.2f}" if r["d_c"] is not None else "-"
        print(f"| {r['label']} | {r['class']} | {h} | {r['subjects'] if r['subjects'] is not None else '-'} | {r['channels']:.1f} | {dt} | {dc} | "
              f"{r['role']} | {r['source']} |")
    print(f"\nTotale nel pretraining{' (senza Kaifosh)' if args.without_kaifosh else ''}: {tot['hours_total']:.1f} h; D_t = {tot['d_t'] / 1e6:.1f} milioni; "
          f"D_c = {tot['d_c'] / 1e9:.2f} miliardi (patch {PATCH_S * 1000:.0f} ms)")
    for c in "ABC":
        print(f"- classe {c}: {tot['hours_by_class'][c]:.1f} h = {100 * tot['fraction_by_class'][c]:.1f}% delle ore")
    print("\nPassaggi sui dati di una classe con 4 epoche di consumo totale, per quota garantita:")
    for c in "AC":
        print(f"- classe {c}: " + ", ".join(f"quota {int(q * 100)}% -> {passes(q, tot['hours_by_class'][c], tot['hours_total']):.1f}"
                                         for q in (0.05, 0.10, 0.20, 0.30, 0.40)))
    if args.quota:
        quota = {k: float(v) for k, v in (x.split("=") for x in args.quota.split(","))}
        used = [r for r in rows if r["hours"] is not None and (not args.without_kaifosh or r["key"] != "kaifosh")]
        a = allocate(used, quota, args.alpha, args.max_passes, tot["hours_total"])
        print(f"\nRipartizione con quote {quota}, pesi ore^{args.alpha}, tetto {args.max_passes:g} passaggi per dataset:")
        print("| Dataset | Classe | Quota dei campioni | Passaggi |")
        print("|---|---|---|---|")
        for r in used:
            sh, ps = a["per_dataset"][r["key"]]
            print(f"| {r['label']} | {r['class']} | {100 * sh:.1f}% | {ps:.1f} |")
        print("Quota non utilizzabile per classe (tutti i dataset al tetto):", {c: f"{100 * v:.1f}%" for c, v in a["unused_quota"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
