"""Ingest di Kaifosh et al. 2025 (Discrete Gestures) dal tar: vedi src/wearusfm/ingest/kaifosh.py.

Uso: python3 scripts/ingest_kaifosh.py --tar <.../full_data.tar> --csv <.../discrete_gestures_corpus.csv>
       --out-root <.../kaifosh> --report <report.json> [--max-recordings N] [--users 0 1 ...]
Un file alla volta, in streaming dal tar (nessun file temporaneo).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.kaifosh import ingest_recording, load_split_table, scan_tar  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tar", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--users", type=int, nargs="+", default=None, help="default: tutti quelli nel tar")
    p.add_argument("--max-recordings", type=int, default=None, help="per il collaudo: al massimo N registrazioni")
    args = p.parse_args()

    split_table = load_split_table(args.csv)
    t0 = time.time()
    per_rec = []
    with tarfile.open(args.tar, "r:") as tf:
        refs = scan_tar(tf)
        if args.users is not None:
            refs = [r for r in refs if r.user in args.users]
        if args.max_recordings is not None:
            refs = refs[: args.max_recordings]
        if not refs:
            raise FileNotFoundError(f"nessuna registrazione in {args.tar} per la selezione richiesta")
        for i, ref in enumerate(refs, 1):
            print(f"[{i}/{len(refs)}] user {ref.user} dataset {ref.dataset}: ingest in corso...", flush=True)
            fobj = tf.extractfile(ref.member)
            per_rec.append(ingest_recording(fobj, ref, args.out_root, split_table))
            print(f"  fatto: {per_rec[-1]}", flush=True)
    report = {
        "dataset": "kaifosh_discrete_gestures", "n_recordings": len(per_rec),
        "n_users": len({r["user"] for r in per_rec}), "hours_total": sum(r["hours"] for r in per_rec),
        "n_channels_discarded_total": sum(r["n_channels_discarded"] for r in per_rec),
        "n_recordings_time_irregular": sum(1 for r in per_rec if not r["time_axis_regular"]),
        "per_recording": per_rec, "elapsed_s": time.time() - t0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_recording"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
