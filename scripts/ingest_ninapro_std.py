"""Ingest dei DB NinaPro a 12 elettrodi (DB3, DB4): vedi src/wearusfm/ingest/ninapro_std.py.

Uso: python3 scripts/ingest_ninapro_std.py --db db4 --raw-root <.../ninapro> --out-root <.../ninapro_db4>
       --report <report.json> [--subjects 1 2 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.ninapro_std import DB3, DB4, ingest_subject, scan_zips  # noqa: E402

CONFIGS = {"db3": DB3, "db4": DB4}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", choices=sorted(CONFIGS), required=True)
    p.add_argument("--raw-root", type=Path, required=True, help="cartella che contiene DBn/")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=None, help="default: tutti quelli presenti")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    cfg = CONFIGS[args.db]

    zips = scan_zips(args.raw_root, cfg)
    if not zips:
        raise FileNotFoundError(f"nessuno zip {cfg.zip_re} sotto {args.raw_root}/{cfg.key.upper()}")
    wanted = sorted(zips) if args.subjects is None else args.subjects
    missing = [s for s in wanted if s not in zips]
    if missing:
        raise FileNotFoundError(f"soggetti senza zip: {missing}")

    t0 = time.time()
    per_subject = []
    for s in wanted:
        print(f"{cfg.key} s{s}: ingest in corso...", flush=True)
        per_subject.append(ingest_subject(zips[s], args.out_root, s, cfg))
        print(f"  fatto: {per_subject[-1]}", flush=True)
    report = {
        "dataset": cfg.dataset_name, "subjects_processed": wanted, "n_subjects": len(wanted),
        "hours_total": sum(x["hours"] for x in per_subject), "n_channels": cfg.n_channels,
        "n_channels_discarded_total": sum(x["n_channels_discarded"] for x in per_subject),
        "subjects_with_subject_field_mismatch": [x["subject"] for x in per_subject if not x["subject_field_matches_filename"]],
        "subjects_with_exercise_field_mismatch": [x["subject"] for x in per_subject if not x["exercise_field_matches_filename"]],
        "per_subject": per_subject, "elapsed_s": time.time() - t0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_subject"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
