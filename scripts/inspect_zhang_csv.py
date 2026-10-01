#!/usr/bin/env python3
"""Ispezione in SOLA LETTURA dei `sensor_data.csv` di Zhang 2026, per i quattro controlli da fare prima del parser (docs/decisioni.md, «Zhang 2026»):
(1) come sono disposti nelle righe i canali a 2000 e a 4000 Hz (righe vuote, ripetute?); (2) se le colonne hanno la stessa origine dei tempi
(prima e ultima riga non vuota, passo fra righe non vuote); (3) se `anatomical` e `random` hanno gli stessi canali EMG (si confrontano le
intestazioni di piu' file); (4) se i valori usano la virgola decimale. Stampa un JSON; non scrive nulla.

  python scripts/inspect_zhang_csv.py <sequenza>/sensor_data.csv [<altro>/sensor_data.csv ...] --pattern-rows 20000
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
import sys
from pathlib import Path


HEADER_SCAN_ROWS = 50


def _num(s: str) -> float | None:
    """Numero finito (virgola o punto decimale), altrimenti None: «NaN» NON conta come numero (compare nelle intestazioni di Zhang)."""
    try:
        v = float(s.strip().replace(",", "."))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def detect_delimiter(first_line: str) -> str:
    return max((";", ",", "\t"), key=first_line.count)


def inspect_file(path: Path, pattern_rows: int) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        delim = detect_delimiter(first)
        reader = csv.reader(f, delimiter=delim)
        head = [row for _, row in zip(range(HEADER_SCAN_ROWS), reader)]
        # intestazione = fino all'ultima riga (fra le prime HEADER_SCAN_ROWS) con una cella non vuota che non e' un numero finito: in Zhang ci sono
        # righe d'intestazione tutte numeriche (numeri di serie) seguite dalla riga delle unita' di misura
        last_text = max((i for i, row in enumerate(head) if any(c.strip() and _num(c) is None for c in row)), default=-1)
        header, data_head = head[: last_text + 1], head[last_text + 1 :]
        data_first = data_head[0] if data_head else []
        reader = itertools.chain(data_head[1:], reader)
        n_cols = max(len(r) for r in header + [data_first])
        nonempty = [0] * n_cols
        first_idx, last_idx = [None] * n_cols, [None] * n_cols
        prev = [None] * n_cols
        steps = [collections.Counter() for _ in range(n_cols)]
        comma = [0] * n_cols
        samples = [[] for _ in range(n_cols)]
        not_numeric = [0] * n_cols
        n_rows = 0
        for i, row in enumerate(itertools.chain([data_first], reader)):
            for j in range(n_cols):
                c = row[j].strip() if j < len(row) else ""
                if not c:
                    continue
                nonempty[j] += 1
                last_idx[j] = i
                if first_idx[j] is None:
                    first_idx[j] = i
                if i < pattern_rows and prev[j] is not None:
                    steps[j][i - prev[j]] += 1
                prev[j] = i
                if "," in c:
                    comma[j] += 1
                if _num(c) is None:
                    not_numeric[j] += 1
                if len(samples[j]) < 3:
                    samples[j].append(c)
            n_rows += 1
    cols = []
    for j in range(n_cols):
        cols.append({
            "col": j, "header": [r[j] if j < len(r) else "" for r in header], "n_nonempty": nonempty[j],
            "first_row": first_idx[j], "last_row": last_idx[j],
            "steps_first_rows": dict(steps[j].most_common(3)), "frac_comma": comma[j] / nonempty[j] if nonempty[j] else None,
            "n_not_numeric": not_numeric[j], "samples": samples[j],
        })
    return {"file": str(path), "delimiter": delim, "n_header_lines": len(header), "n_data_rows": n_rows, "n_cols": n_cols, "columns": cols}


def compare_headers(results: list[dict], emg_marker: str = "EMG") -> dict:
    """Per (3): le colonne la cui intestazione contiene `emg_marker`, file per file; uguali fra tutti i file?"""
    emg = [[tuple(c["header"]) for c in r["columns"] if any(emg_marker.lower() in h.lower() for h in c["header"])] for r in results]
    return {"emg_columns_per_file": [len(e) for e in emg], "all_equal": all(e == emg[0] for e in emg)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--pattern-rows", type=int, default=20000, help="righe iniziali su cui contare il passo fra righe non vuote")
    ap.add_argument("--columns", action="store_true", help="stampa il dettaglio delle colonne di ogni file (default: solo del primo)")
    args = ap.parse_args(argv)
    res = [inspect_file(p, args.pattern_rows) for p in args.paths]
    def compact(c):
        return (f"{c['col']:2d} | {' / '.join(c['header'])} | n={c['n_nonempty']} righe {c['first_row']}-{c['last_row']} passi {c['steps_first_rows']} "
                f"virgola {c['frac_comma']} non numerici {c['n_not_numeric']} es. {c['samples']}")

    out = {"files": [{**{k: v for k, v in r.items() if k != "columns"}, "header_rows": [list(h) for h in zip(*[c["header"] for c in r["columns"]])],
                      **({"columns": [compact(c) for c in r["columns"]]} if args.columns or i == 0 else {})} for i, r in enumerate(res)],
           "emg_header_comparison": compare_headers(res)}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
