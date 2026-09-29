"""Ingest di NinaPro DB5 (200 Hz, due Myo, 16 canali): vedi src/wearusfm/ingest/ninapro_db5.py.

Uso: python3 scripts/ingest_ninapro_db5.py --raw-root <.../ninapro> --out-root <.../ninapro_db5>
       --report <report.json> [--subjects 1 2 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.ninapro_db5 import ingest_subject, scan_db5  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="cartella che contiene DB5/")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=None, help="default: tutti quelli presenti")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    zips = scan_db5(args.raw_root)
    if not zips:
        raise FileNotFoundError(f"nessuno zip s<N>.zip sotto {args.raw_root}/DB5")
    wanted = sorted(zips) if args.subjects is None else args.subjects
    missing = [s for s in wanted if s not in zips]
    if missing:
        raise FileNotFoundError(f"soggetti senza zip: {missing}")

    t0 = time.time()
    per_subject = []
    for s in wanted:
        print(f"s{s}: ingest in corso...", flush=True)
        per_subject.append(ingest_subject(zips[s], args.out_root, s))
        print(f"  fatto: {per_subject[-1]}", flush=True)
    report = {
        "dataset": "ninapro_db5", "subjects_processed": wanted, "n_subjects": len(wanted),
        "hours_total": sum(x["hours"] for x in per_subject), "n_channels": 16,
        "n_channels_discarded_total": sum(x["n_channels_discarded"] for x in per_subject),
        "per_subject": per_subject, "elapsed_s": time.time() - t0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_subject"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
