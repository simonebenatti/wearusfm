"""Ingest di NinaPro DB8 (16 Delsys in due righe da 8, griglia a 2 kHz con banda effettiva fino a 555,5 Hz, opzione A): vedi
src/wearusfm/ingest/ninapro_db8.py.

Uso: python3 scripts/ingest_ninapro_db8.py --raw-root <.../ninapro> --out-root <.../ninapro_db8> --report <report.json> [--subjects 1 11]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.ninapro_db8 import ingest_subject, scan_db8  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="cartella che contiene DB8/")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=None, help="default: tutti quelli presenti")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)

    files = scan_db8(args.raw_root)
    if not files:
        raise FileNotFoundError(f"nessun file S<N>_E1_A<k>.mat sotto {args.raw_root}/DB8")
    wanted = sorted(files) if args.subjects is None else args.subjects
    missing = [s for s in wanted if s not in files]
    if missing:
        raise FileNotFoundError(f"soggetti senza file: {missing}")

    t0 = time.time()
    per_subject, failed = [], {}
    for s in wanted:
        print(f"db8 s{s}: ingest in corso...", flush=True)
        try:
            per_subject.append(ingest_subject(files[s], args.out_root, s))
        except Exception as e:  # un soggetto rotto non ferma gli altri
            failed[s] = f"{type(e).__name__}: {e}"
            print(f"  FALLITO s{s}: {failed[s]}", flush=True)
            continue
        r = per_subject[-1]
        print(f"  fatto: {r['hours']:.2f} h, canali scartati {r['discarded_channels']}, movimenti {r['movements_per_acquisition']}, "
              f"subject nel file {r['subject_field_in_file']}", flush=True)
    report = {
        "dataset": "ninapro_db8", "subjects_processed": [x["subject"] for x in per_subject], "n_subjects": len(per_subject),
        "hours_total": sum(x["hours"] for x in per_subject),
        "discarded_channels_by_subject": {x["subject"]: x["discarded_channels"] for x in per_subject if x["discarded_channels"]},
        "subjects_with_field_mismatch": [x["subject"] for x in per_subject if x["subject_field_in_file"] != [x["subject"]]],
        "subjects_with_label_length_adjustments": [x["subject"] for x in per_subject if x["label_length_adjustments"]],
        "failed_subjects": failed, "per_subject": per_subject, "elapsed_s": time.time() - t0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_subject"}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
