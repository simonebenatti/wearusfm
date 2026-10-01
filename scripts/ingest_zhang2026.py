"""Ingest di Zhang et al. 2026 (8 EMG: 4 a 2000 Hz e 4 a 4000 Hz portati a 2000 Hz; modi anatomical e random): vedi src/wearusfm/ingest/zhang2026.py.

Uso: python3 scripts/ingest_zhang2026.py --raw-root <.../zhang2026> --out-root <.../zhang2026> --report <report.json>
       [--subjects HG_A468E29 ...] [--max-sessions 2] [--skip-existing] [--time-budget-s 9600]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.zhang2026 import ingest_session, load_dominant_hands, lookup_participant, scan_raw  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="cartella con HG_*/ e participants.csv")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--max-sessions", type=int, default=None, help="collaudo: si ferma dopo tante sessioni (soggetto x modo)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--time-budget-s", type=float, default=None)
    args = p.parse_args(argv)

    raw = scan_raw(args.raw_root)
    hands = load_dominant_hands(args.raw_root / "participants.csv")
    wanted = sorted(raw) if args.subjects is None else args.subjects
    missing = [s for s in wanted if s not in raw]
    if missing:
        raise FileNotFoundError(f"soggetti senza sequenze: {missing}")
    t0, done, failed, skipped, stopped = time.time(), [], {}, 0, None
    for subj in wanted:
        pid, hand = lookup_participant(subj, hands)
        for mode, seqs in raw[subj].items():
            if args.max_sessions is not None and len(done) >= args.max_sessions:
                stopped = "max-sessions"
                break
            if args.time_budget_s is not None and time.time() - t0 > args.time_budget_s:
                stopped = "time-budget"
                break
            if args.skip_existing and (args.out_root / subj / mode / "metadata.json").exists():
                skipped += 1
                continue
            print(f"[{len(done) + 1}] {subj} {mode}: {len(seqs)} sequenze...", flush=True)
            try:
                done.append(ingest_session(subj, mode, seqs, args.out_root, hand, pid))
            except Exception as e:  # una sessione rotta non ferma le altre
                failed[f"{subj}/{mode}"] = f"{type(e).__name__}: {e}"
                print(f"  FALLITA: {failed[f'{subj}/{mode}']}", flush=True)
                continue
            r = done[-1]
            print(f"  fatto: {r['hours']:.2f} h, scartati {r['discarded_channels']}, lato {r['laterality'] or '?'}, etichette {r['labels_copied']}",
                  flush=True)
        if stopped:
            break
    report = {
        "dataset": "zhang2026", "n_sessions": len(done), "n_subjects": len({r["subject"] for r in done}), "hours_total": sum(r["hours"] for r in done),
        "n_sequences": sum(r["n_sequences"] for r in done), "n_skipped_existing": skipped, "failed": failed, "stopped": stopped,
        "discarded_channels_by_session": {f"{r['subject']}/{r['mode']}": r["discarded_channels"] for r in done if r["discarded_channels"]},
        "subjects_not_found_in_participants_csv": sorted({r["subject"] for r in done if r["participant_id_in_csv"] is None}),
        "subjects_matched_by_prefix": sorted({r["subject"] for r in done if r["participant_id_in_csv"] not in (None, r["subject"])}),
        "sessions_with_trimmed_samples": {f"{r['subject']}/{r['mode']}": r["trimmed_by_sequence"] for r in done if r["trimmed_by_sequence"]},
        "sessions_with_excluded_sequences": {f"{r['subject']}/{r['mode']}": r["excluded_sequences"] for r in done if r["excluded_sequences"]},
        "sessions_with_empty_label_files": {f"{r['subject']}/{r['mode']}": r["empty_label_files"] for r in done if r["empty_label_files"]},
        "per_session": done, "elapsed_s": time.time() - t0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_session"}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
