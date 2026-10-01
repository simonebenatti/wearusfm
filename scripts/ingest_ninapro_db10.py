"""Ingest di NinaPro DB10 = MeganePro MDS1 (12 Delsys, 1926 Hz, segmenti alle pause dell'asse dei tempi): vedi
src/wearusfm/ingest/ninapro_db10.py.

Uso: python3 scripts/ingest_ninapro_db10.py --raw-root <.../ninapro> --out-root <.../ninapro_db10> --report <report.json> [--subjects 10 108] [--skip-existing] [--time-budget-s 9600]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.ninapro_db10 import ingest_subject, scan_db10  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="cartella che contiene DB10/MDS1/")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=None, help="default: tutti quelli presenti")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--time-budget-s", type=float, default=None, help="non comincia un nuovo soggetto oltre il budget")
    args = p.parse_args(argv)

    files = scan_db10(args.raw_root)
    if not files:
        raise FileNotFoundError(f"nessun file S<NNN>_ex1.mat sotto {args.raw_root}/DB10/MDS1")
    wanted = sorted(files) if args.subjects is None else args.subjects
    missing = [s for s in wanted if s not in files]
    if missing:
        raise FileNotFoundError(f"soggetti senza file: {missing}")

    t0 = time.time()
    per_subject, failed, skipped, stopped = [], {}, 0, None
    def write_report(final: bool) -> dict:
        report = {
            "dataset": "ninapro_db10", "subjects_processed": [x["subject"] for x in per_subject], "n_subjects": len(per_subject),
            "hours_total": sum(x["hours"] for x in per_subject), "n_skipped_existing": skipped, "stopped": stopped if final else "in corso",
            "discarded_channels_by_subject": {x["subject"]: x["discarded_channels"] for x in per_subject if x["discarded_channels"]},
            "suspect_low_channels_by_subject": {x["subject"]: x["suspect_low_channels"] for x in per_subject if x["suspect_low_channels"]},
            "subjects_with_label_length_adjustments": [x["subject"] for x in per_subject if x["label_length_adjustments"]],
            "failed_subjects": failed, "per_subject": per_subject, "elapsed_s": time.time() - t0,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        return report

    for s in wanted:
        # il budget si controlla PRIMA di ogni soggetto (anche dopo un fallimento); il report parziale si riscrive a ogni soggetto, cosi' un job
        # ucciso da SLURM lascia comunque il report (review del codice, 02/10/2026)
        if args.time_budget_s is not None and time.time() - t0 > args.time_budget_s:
            stopped = "time-budget"
            break
        if args.skip_existing and (args.out_root / f"s{s:03d}" / "ex1" / "metadata.json").exists():
            skipped += 1
            continue
        print(f"db10 S{s:03d}: ingest in corso...", flush=True)
        try:
            per_subject.append(ingest_subject(files[s], args.out_root, s))
        except Exception as e:  # un soggetto rotto non ferma gli altri
            failed[s] = f"{type(e).__name__}: {e}"
            print(f"  FALLITO s{s}: {failed[s]}", flush=True)
            write_report(False)
            continue
        r = per_subject[-1]
        print(f"  fatto: {r['hours']:.2f} h, {r['n_channels']} canali, {r['n_segments']} segmenti, fs dai ts {r['fs_from_ts_hz']:.2f}, "
              f"scartati {r['discarded_channels']}, sospetti {r['suspect_low_channels']}", flush=True)
        write_report(False)
    report = write_report(True)
    print(json.dumps({k: v for k, v in report.items() if k != "per_subject"}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
