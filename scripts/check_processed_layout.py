#!/usr/bin/env python3
"""Controlla che i dati processati rispettino il contratto (`docs/formato_processato.md`).

Cammina sotto ciascuna radice, valida ogni sessione (cartella con `metadata.json`) SENZA leggere i dati (l'array si apre in mmap: solo
forma e tipo), e scrive un riepilogo JSON per dataset: sessioni, soggetti, ore, canali, layout, problemi. Esce con 1 se c'e' almeno
un problema. Sola lettura: e' sicuro sul login node (il costo e' un `stat` + la lettura di un JSON per sessione).

  python scripts/check_processed_layout.py --root kaifosh=$SCRATCH/data/processed/kaifosh --root ... --report out.json
  python scripts/check_processed_layout.py --scan $WORK/data/processed --scan $SCRATCH/data/processed --report out.json

`--scan DIR` tratta ogni sottocartella di DIR come un dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from wearusfm.data.processed import discover_sessions, session_summary, validate_session


def check_dataset(name: str, root: Path, *, max_problems_listed: int = 20) -> dict:
    found = discover_sessions(root, name)
    problems: dict[str, list[str]] = {}
    hours, layouts, scales, fs, n_channels = 0.0, Counter(), Counter(), Counter(), Counter()
    subjects = set()
    for subject, session, path in found:
        subjects.add(subject)
        issues = validate_session(path)
        if issues:
            problems[f"{subject}/{session}"] = issues
            continue
        s = session_summary(path)
        hours += s["hours"]
        layouts[s["layout"]] += 1
        scales[s["scale"]] += 1
        fs[s["fs_hz"]] += 1
        n_channels[s["n_channels"]] += 1
    n_bad = len(problems)
    return {
        "root": str(root), "n_sessions": len(found), "n_subjects": len(subjects), "n_sessions_with_problems": n_bad,
        "n_sessions_conforming": len(found) - n_bad,
        "hours_conforming": round(hours, 3), "layouts": dict(layouts), "scales": dict(scales),
        "fs_hz": {str(k): v for k, v in fs.items()}, "n_channels": {str(k): v for k, v in n_channels.items()},
        "problems_sample": dict(list(problems.items())[:max_problems_listed]),
        "problem_kinds": dict(Counter(p.split(":")[0].split(" ")[0] for v in problems.values() for p in v)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=[], metavar="NOME=DIR", help="un dataset (ripetibile)")
    ap.add_argument("--scan", action="append", default=[], metavar="DIR", help="ogni sottocartella e' un dataset (ripetibile)")
    ap.add_argument("--report", type=Path, default=None, help="JSON di riepilogo (default: solo stampa)")
    args = ap.parse_args(argv)

    targets: list[tuple[str, Path]] = []
    for item in args.root:
        if "=" not in item:
            ap.error(f"--root vuole NOME=DIR, ricevuto {item!r}")
        name, _, path = item.partition("=")
        targets.append((name, Path(path)))
    for d in args.scan:
        base = Path(d)
        if not base.is_dir():
            ap.error(f"--scan: {base} non e' una cartella")
        targets.extend((sub.name, sub) for sub in sorted(base.iterdir()) if sub.is_dir())
    if not targets:
        ap.error("niente da controllare: passa --root o --scan")

    report: dict[str, dict] = {}
    for name, root in targets:
        key = name if name not in report else f"{name}@{root}"  # stesso nome in due dischi ($WORK e $SCRATCH): non si sovrascrive
        report[key] = check_dataset(name, root)
        r = report[key]
        print(
            f"{key}: {r['n_sessions']} sessioni, {r['n_subjects']} soggetti, {r['hours_conforming']} h conformi, "
            f"{r['n_sessions_with_problems']} con problemi",
            flush=True,
        )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    n_bad = sum(r["n_sessions_with_problems"] for r in report.values())
    n_empty = [k for k, r in report.items() if r["n_sessions"] == 0]
    if n_empty:
        print(f"ATTENZIONE: nessuna sessione trovata in {n_empty}", file=sys.stderr)
    return 1 if (n_bad or n_empty) else 0


if __name__ == "__main__":
    sys.exit(main())
