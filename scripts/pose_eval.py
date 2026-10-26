"""Valutazione della posa fuori campione (emendamento 1 a `pilot_posa_v1`, E1-E3): vedi `src/wearusfm/harness/pose_eval.py`.

Estrazione (GPU, una per modello-endpoint e per insieme):
  python3 scripts/pose_eval.py --stage extract --checkpoint <run>/checkpoint_postdecay.pt --csv <emg2pose_metadata.csv> \\
      --processed-root $SCRATCH/data/processed/emg2pose --pose-root $SCRATCH/data/processed/emg2pose_pose --split val --out <...>/val.npz
  python3 scripts/pose_eval.py --stage extract ... --split train --n-users 40 --n-recordings 200 --out <...>/train_subset.npz
Rapporto (CPU):
  python3 scripts/pose_eval.py --stage report --val <...>/val.npz [--train <...>/train_subset.npz] --out <...>/rapporto.json
Il rapporto applica solo le regole congelate dell'emendamento (E1 appresa / E2 disallineato / E3 accessibile) e le riporta.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.harness import pose_eval as PE  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "report"], required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--which", choices=["teacher", "student"], default="teacher", help="come la sonda: il teacher post-decay")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--pose-root", type=Path)
    ap.add_argument("--split", default="val", help="split del CSV: val (E1-E2) oppure train (sottoinsieme per E3)")
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--n-recordings", type=int, default=None)
    ap.add_argument("--n-windows", type=int, default=10)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=PE.SEED_WINDOWS)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val", type=Path)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.stage == "extract":
        for name in ("checkpoint", "csv", "processed_root", "pose_root"):
            if getattr(args, name) is None:
                ap.error(f"--{name.replace('_', '-')} e' obbligatorio per extract")
        import torch

        from wearusfm.harness.fm_features import load_model
        from wearusfm.training import run as R

        model = load_model(str(args.checkpoint), device=args.device, which=args.which)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # lo stesso loader dei run (filtro, griglia a 200 ms, margini, scala di sessione): il preset dei run della finestra 1
        loader_cfg = R.sanity_config().loader
        ev = PE.EvalConfig(n_windows=args.n_windows, window_s=args.window_s, seed=args.seed, batch=args.batch, device=args.device, loader=loader_cfg)
        recs = PE.recordings_for(args.csv, args.processed_root, args.split, n_users=args.n_users, n_recordings=args.n_recordings, seed=args.seed)
        print(f"{len(recs)} registrazioni dello split {args.split!r}; modello {args.checkpoint} ({args.which}), testa di posa: "
              f"{getattr(model, 'pose_head', None) is not None}", flush=True)
        store = PT.PoseStore(args.pose_root)
        data = PE.extract(model, recs, store, ev)
        data["config"].update({"checkpoint": str(args.checkpoint), "which": args.which, "split": args.split,
                               "code_sha": os.environ.get("WEARUSFM_CODE_SHA"), "pose_root": str(args.pose_root),
                               "checkpoint_pose_stats_sha256": state.get("pose_stats_sha256")})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        PE.save_npz(args.out, data)
        print(json.dumps({"windows": int(data["lat"].shape[0]), "recordings": data["n_recordings"], "has_head": data["has_head"],
                          "elapsed_s": time.time() - t0, "out": str(args.out)}))
        return 0
    if args.val is None:
        ap.error("--val e' obbligatorio per report")
    val = PE.load_npz(args.val)
    train = PE.load_npz(args.train) if args.train else None
    rep = PE.report(val, train)
    rep["inputs"] = {"val": str(args.val), "train": str(args.train) if args.train else None, "elapsed_s": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1, allow_nan=True))
    summary = {k: rep[k] for k in ("n_windows", "n_users", "n_dynamic", "has_head")}
    for key in ("E1", "E2", "E3"):
        if key in rep and "dynamic" in rep[key]:
            d = rep[key]["dynamic"]
            summary[key] = {"ratio": d["ratio"], "ci95": d["ratio_ci95_users"], "median_user": d["median_user_ratio"],
                            "verdict": rep[key].get("learned", rep[key].get("misaligned_ok", rep[key].get("accessible")))}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
