"""Ingest di emg2qwerty (tar.gz, streaming) e di emg2pose (tar non compresso): vedi src/wearusfm/ingest/emg2qwerty.py e
emg2pose.py. Un file alla volta; un file rotto non ferma gli altri; RIPRENDIBILE (`--skip-existing`) e con un budget di
tempo (`--time-budget-s`) per fermarsi in ordine prima del limite di SLURM e scrivere il report.

Uso: python3 scripts/ingest_meta_bracelet.py --dataset emg2qwerty --tar <...tar.gz> --out-root <...> --report <r.json>
       python3 scripts/ingest_meta_bracelet.py --dataset emg2pose --tar <...tar> --csv <...metadata.csv> --out-root ...
Opzioni: --max-recordings N (collaudo), --skip-existing, --time-budget-s S.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest import emg2pose as E2P  # noqa: E402
from wearusfm.ingest import emg2qwerty as E2Q  # noqa: E402


def _write_report(path: Path, dataset: str, done: list, failed: dict, skipped: int, t0: float, stopped: str | None) -> dict:
    report = {
        "dataset": dataset, "n_recordings": len(done), "n_failed": len(failed), "n_skipped_existing": skipped,
        "n_users": len({r["user"] for r in done}), "hours_total": sum(r["hours"] for r in done),
        "n_channels_discarded_total": sum(len(r["discarded_channels"]) for r in done),
        "n_recordings_time_irregular": sum(1 for r in done if not r["time_axis_regular"]),
        "failed": failed, "stopped": stopped, "elapsed_s": time.time() - t0, "per_recording": done,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["emg2qwerty", "emg2pose"], required=True)
    p.add_argument("--tar", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=None, help="emg2pose: CSV di metadati")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-recordings", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--time-budget-s", type=float, default=None)
    args = p.parse_args()

    t0 = time.time()
    done: list[dict] = []
    failed: dict[str, str] = {}
    skipped = 0
    stopped = None

    def out_dir_exists(user_dir: str, sess: str) -> bool:
        return (args.out_root / user_dir / sess / "metadata.json").exists()

    def out_of_budget() -> bool:
        return args.time_budget_s is not None and time.time() - t0 > args.time_budget_s

    if args.dataset == "emg2qwerty":
        for ref, data in E2Q.iter_tar_recordings(args.tar):
            if args.max_recordings is not None and len(done) >= args.max_recordings:
                stopped = "max-recordings"
                break
            if out_of_budget():
                stopped = "time-budget"
                break
            if args.skip_existing and list(args.out_root.glob(f"u*/{ref.session_name}/metadata.json")):
                skipped += 1
                continue
            print(f"[{len(done) + 1}] {ref.session_name}...", flush=True)
            try:
                done.append(E2Q.ingest_recording(ref, data, args.out_root))
            except Exception as e:
                failed[ref.session_name] = f"{type(e).__name__}: {e}"
                print(f"  FALLITO: {failed[ref.session_name]}", flush=True)
                continue
            if len(done) % 50 == 0:
                _write_report(args.report, args.dataset, done, failed, skipped, t0, "in corso")
    else:
        if args.csv is None:
            raise SystemExit("--csv e' obbligatorio per emg2pose")
        meta = E2P.load_metadata_csv(args.csv)
        with tarfile.open(args.tar, "r:") as tf:
            refs = E2P.scan_tar(tf)
            print(f"{len(refs)} registrazioni nel tar, {len(meta)} righe nel CSV", flush=True)
            for ref in refs:
                if args.max_recordings is not None and len(done) >= args.max_recordings:
                    stopped = "max-recordings"
                    break
                if out_of_budget():
                    stopped = "time-budget"
                    break
                row = meta.get(ref.stem)
                user = (row or {}).get("user", "sconosciuto")
                if args.skip_existing and (args.out_root / f"u{E2P._SAFE_RE.sub('_', user)}" / E2P._SAFE_RE.sub("_", ref.stem) / "metadata.json").exists():
                    skipped += 1
                    continue
                try:
                    done.append(E2P.ingest_recording(tf.extractfile(ref.member), ref, args.out_root, row))
                except Exception as e:
                    failed[ref.stem] = f"{type(e).__name__}: {e}"
                    print(f"  FALLITO {ref.stem}: {failed[ref.stem]}", flush=True)
                    continue
                if len(done) % 200 == 0:
                    print(f"  {len(done)} fatti", flush=True)
                    _write_report(args.report, args.dataset, done, failed, skipped, t0, "in corso")
    report = _write_report(args.report, args.dataset, done, failed, skipped, t0, stopped)
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_recording", "failed")}, indent=2))
    if failed:
        print(f"FALLITI: {len(failed)} (primi 5: {dict(list(failed.items())[:5])})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
