"""Passo 1-bis: verifiche V1-V4 sul tokenizer NeuroRVQ-EMG congelato (D5a, congelata il
29/09/2026: docs/decisioni.md). Da lanciare su Leonardo (serve torch/GPU); l'output JSON va
FUORI dal repo ($WORK/wearusfm_runs/results/step1bis/) e entra in `results/step1bis/` solo
copiato e committato dal Mac.

Il codice del tokenizer NON e' in questo repo: `--repo-dir` e' un clone di
KonstantinosBarmpas/NeuroRVQ al commit 926e770 (licenza CC BY-NC 4.0), tenuto fuori dal repo.
Il report e' un dato, non un verdetto: D5b (adottare, restringere o scartare l'ancora RVQ) resta
una decisione di Simone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.tokenizer_checks.emg2pose import load_emg2pose_session, load_user_map  # noqa: E402
from wearusfm.tokenizer_checks.neurorvq import NeuroRVQRunner  # noqa: E402
from wearusfm.tokenizer_checks.pipeline import Config, run_all  # noqa: E402
from wearusfm.tokenizer_checks.sessions import discover_sessions, load_session  # noqa: E402


def _git(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "sconosciuto"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _loader(name: str, subj: str, sess: str, path: Path):
    return load_session(path, name, subj, sess)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", type=Path, required=True, help="clone di NeuroRVQ (commit 926e770), fuori da questo repo")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["capgmyo", "grabmyo", "putemg", "csl_hdemg", "camargo2021"])
    p.add_argument("--emg2pose-dir", type=Path, required=True, help="cartella con i .hdf5 di emg2pose")
    p.add_argument("--emg2pose-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-groups", type=int, default=78)
    p.add_argument("--v1-windows", type=int, default=80)
    p.add_argument("--max-sessions-per-subject", type=int, default=2)
    p.add_argument("--n-noise-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--skip-v3", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true", help="prova di collaudo: NON e' il run di D5a")
    p.add_argument("--save-arrays-dir", type=Path, default=None, help="cartella (fuori dal repo) per codici e token per dataset")
    args = p.parse_args()

    t0 = time.time()
    cfg = Config(n_groups=args.n_groups, v1_windows=args.v1_windows, seed=args.seed,
                 max_sessions_per_subject=args.max_sessions_per_subject, batch=args.batch,
                 n_noise_seeds=args.n_noise_seeds, skip_v3=args.skip_v3)
    runner = NeuroRVQRunner(args.repo_dir, args.checkpoint, device=args.device)

    umap = load_user_map(args.emg2pose_csv)
    em_files = sorted(args.emg2pose_dir.rglob("*.hdf5"))
    if not em_files:
        raise SystemExit(f"nessun .hdf5 in {args.emg2pose_dir}")
    em_items = [(umap.get(f.stem, "sconosciuto"), f.stem, f) for f in em_files]
    em_loader = lambda subj, sess, path: load_emg2pose_session(path, umap)  # noqa: E731

    datasets = {}
    for name in args.datasets:
        root = args.processed_root / name
        items = discover_sessions(root, name)
        if not items:
            print(f"ATTENZIONE: nessuna sessione in {root}", flush=True)
        datasets[name] = (items, partial(_loader, name))

    partial_path = args.out.with_name(args.out.stem + ".partial.json")  # NON chiamarla `partial`: nasconderebbe functools.partial (bug del 30/09)

    def on_progress(rep: dict) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(rep, indent=2))

    report = run_all(runner, cfg, (em_items, em_loader), datasets,
                     save_arrays_dir=args.save_arrays_dir, on_progress=on_progress)
    report["provenance"] = {
        "smoke": args.smoke,
        "decision": "docs/decisioni.md, D5a definizioni operative di V1-V4 (congelate 29/09/2026)",
        "wearusfm_commit": _git(Path(__file__).resolve().parent.parent),
        "neurorvq_repo_commit": _git(args.repo_dir),
        "checkpoint": str(args.checkpoint), "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "n_emg2pose_files": len(em_files), "elapsed_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    summary = {k: report.get(k) for k in ("scale_calibration", "v1", "v2_all_pass", "excluded_datasets", "stopped")}
    summary["v3_passes"] = (report.get("v3") or {}).get("passes")
    summary["v4_passes"] = (report.get("v4") or {}).get("passes")
    print(json.dumps(summary, indent=2))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
