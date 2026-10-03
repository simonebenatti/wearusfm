#!/usr/bin/env python3
"""Training JEPA con ancore (passo 6): oggi il preset `sanity` (valori firmati da Simone il 03/10/2026, `docs/fogli_firma_d9_d10.md` 15-20; le
altre scelte sono in `src/wearusfm/training/run.py`). Scrive in `--out-dir`, FUORI dal repo; riprende da `checkpoint.pt` se c'e' (job concatenati).

  python3 scripts/train_jepa.py --preset sanity --manifest manifest_59253155.json.gz --root $WORK/data/processed --root $SCRATCH/data/processed \\
      --scales session_scales.json --out-dir $WORK/wearusfm_runs/runs/sanity_<id> --time-limit-s 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402

from wearusfm.training import run as R  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["sanity", "small"], required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--scales", type=Path, default=None, help="session_scales.json di scripts/measure_loader.py (senza: calcolate al volo)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--time-limit-s", type=float, default=None, help="si ferma e salva il checkpoint prima del limite del job")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", default=None, help="elenco separato da virgole, o 'all'; default: quello del preset (sanity: emg2qwerty)")
    ap.add_argument("--rvq-codes", type=Path, default=None,
                    help="radice dei codici RVQ precalcolati (scripts/precompute_rvq_codes.py): accende l'ancora RVQ (D5b); senza, l'ancora e' spenta")
    args = ap.parse_args(argv)
    cfg = R.sanity_config() if args.preset == "sanity" else R.small_config()
    if args.rvq_codes is not None:
        cfg = R.with_rvq(cfg, args.rvq_codes)
    if args.max_steps is not None:
        cfg = replace(cfg, max_steps=args.max_steps)
    if args.datasets is not None:
        cfg = replace(cfg, datasets=None if args.datasets == "all" else tuple(args.datasets.split(",")))
    scales = json.loads(args.scales.read_text())["scales"] if args.scales else {}
    print(f"preset {args.preset}: {R.count_parameters(cfg.model) / 1e6:.1f} M parametri, {cfg.max_steps} passi, batch {cfg.batch_size}, "
          f"dispositivo {args.device}, {len(scales)} scale di sessione, ancora RVQ {'accesa: ' + str(args.rvq_codes) if args.rvq_codes else 'spenta'}",
          flush=True)
    summary = R.train(cfg, args.manifest, args.root, args.out_dir, scales=scales, device=args.device, num_workers=args.num_workers,
                      time_limit_s=args.time_limit_s, log=lambda m: print(m, flush=True))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
