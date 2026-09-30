"""Ingest di NinaPro DB6 (14 Delsys in 16 colonne, 2 kHz, 5 giorni x 2 sessioni): vedi
src/wearusfm/ingest/ninapro_db6.py.

Uso: python3 scripts/ingest_ninapro_db6.py --raw-root <.../ninapro> --out-root <.../ninapro_db6>
       --report <report.json> [--subjects 1 2 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.ninapro_db6 import ingest_subject, scan_db6  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="cartella che contiene DB6/")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=None, help="default: tutti quelli presenti")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    zips = scan_db6(args.raw_root)
    if not zips:
        raise FileNotFoundError(f"nessuno zip DB6_s<N>_<a|b>.zip sotto {args.raw_root}/DB6")
    wanted = sorted(zips) if args.subjects is None else args.subjects
    missing = [s for s in wanted if s not in zips]
    if missing:
        raise FileNotFoundError(f"soggetti senza zip: {missing}")

    t0 = time.time()
    per_subject, failed = [], {}
    for s in wanted:
        print(f"db6 s{s}: ingest in corso...", flush=True)
        try:
            per_subject.append(ingest_subject(zips[s], args.out_root, s))
        except Exception as e:  # un soggetto rotto non ferma gli altri
            failed[s] = f"{type(e).__name__}: {e}"
            print(f"  FALLITO s{s}: {failed[s]}", flush=True)
            continue
        r = per_subject[-1]
        print(f"  fatto: soggetto {r['subject']}, {r['n_sessions']} sessioni, {r['hours']:.2f} h, "
              f"canali scartati {r['discarded_channels_union']}, campi coerenti col nome: {r['all_fields_match_filename']}", flush=True)
    report = {
        "dataset": "ninapro_db6", "subjects_processed": [x["subject"] for x in per_subject], "n_subjects": len(per_subject),
        "n_sessions_total": sum(x["n_sessions"] for x in per_subject), "hours_total": sum(x["hours"] for x in per_subject),
        "discarded_channels_union": sorted({c for x in per_subject for c in x["discarded_channels_union"]}),
        "subjects_with_field_mismatch": [x["subject"] for x in per_subject if not x["all_fields_match_filename"]],
        "subjects_with_label_length_adjustments": {
            x["subject"]: [s for s in x["sessions"] if s["label_length_adjustments"]] for x in per_subject
            if any(s["label_length_adjustments"] for s in x["sessions"])},
        "failed_subjects": failed, "per_subject": per_subject, "elapsed_s": time.time() - t0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_subject"}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
