"""Ingest di Hyser (HD-sEMG 256 canali a 2048 Hz, solo `*_raw_*`): vedi src/wearusfm/ingest/hyser.py.

Uso: python3 scripts/ingest_hyser.py --raw-root <.../hyser> --out-root <.../hyser_processed> --report <report.json>
       [--subjects 1 2 ...] [--groups 1dof mvc ndof random pr_dynamic pr_maintenance] [--max-groups N]
Un gruppo (soggetto, sessione, sotto-dataset) alla volta; un gruppo rotto non ferma gli altri.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.hyser import ingest_group, scan_hyser  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=None)
    p.add_argument("--groups", nargs="+", default=None, help="1dof mvc ndof random pr_dynamic pr_maintenance")
    p.add_argument("--max-groups", type=int, default=None, help="per il collaudo: al massimo N gruppi")
    args = p.parse_args()

    groups = scan_hyser(args.raw_root)
    if args.subjects is not None:
        groups = [g for g in groups if g.subject in args.subjects]
    if args.groups is not None:
        groups = [g for g in groups if g.key in args.groups]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise FileNotFoundError(f"nessun gruppo in {args.raw_root} per la selezione richiesta")

    t0 = time.time()
    done, failed = [], {}
    for i, g in enumerate(groups, 1):
        tag = f"s{g.subject:02d}/session{g.session}_{g.key}"
        print(f"[{i}/{len(groups)}] {tag}: {len(g.hea_paths)} registrazioni...", flush=True)
        try:
            done.append(ingest_group(g, args.out_root))
        except Exception as e:
            failed[tag] = f"{type(e).__name__}: {e}"
            print(f"  FALLITO {tag}: {failed[tag]}", flush=True)
            continue
        r = done[-1]
        print(f"  fatto: {r['n_trials']} registrazioni, {r['hours']:.3f} h, canali scartati {r['discarded_channels']}", flush=True)
    report = {
        "dataset": "hyser", "n_groups": len(done), "n_groups_failed": len(failed),
        "n_subjects": len({r["subject"] for r in done}), "hours_total": sum(r["hours"] for r in done),
        "n_trials_total": sum(r["n_trials"] for r in done),
        "groups_with_discarded_channels": {f"s{r['subject']:02d}/session{r['session']}_{r['group']}": r["discarded_channels"] for r in done if r["discarded_channels"]},
        "failed_groups": failed, "per_group": done, "elapsed_s": time.time() - t0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_group"}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
