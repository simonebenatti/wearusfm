"""Passo 0 di TEMPONet su emg2pose, solo CPU (`docs/temponet_emg2pose_plan.md`; regole ed emendamento in `docs/decisioni.md`, 10/11/2026).

1. Dal tar, per ogni registrazione di validazione e di test (split del CSV ufficiale): maschera dei fallimenti della cinematica inversa
   (`no_ik_failure`, come `utils.get_ik_failures_mask`), inizi delle finestre ufficiali (10.000 + 255 campioni, passo 10.000, come
   `test_analysis`) e angoli esatti a 2 kHz negli istanti valutati. Un .npz per registrazione. Controlli:
   - lunghezza uguale all'EMG processato;
   - istanti tutti senza fallimento;
   - scarto dalle medie a 200 Hz dell'ingest delle pose.
2. Dal processato: deviazione standard dei canali di 1.000 registrazioni di training estratte col seme 0, e scala 10^k con la regola congelata.

Scrive in una cartella NUOVA `<out>/<split>/<u...>/<stem>.npz` (`starts` (finestre,) int64 e `targets` (finestre, 1000, 20) float32 in radianti)
e `manifest.json`, con righe per registrazione, totali per split e condizione, e scala. Nessun numero di modello: solo bersagli e conteggi.

Uso (CPU, `scripts/slurm/emg2pose_eval_targets.sbatch`):
  python3 scripts/emg2pose_eval_targets.py --tar <tar> --csv <csv> --processed-root <emg2pose> --pose-root <emg2pose_pose> --out <cartella>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from wearusfm.harness import emg2pose_regression as R
from wearusfm.ingest import emg2pose as E2P
from wearusfm.ingest import emg2pose_pose as PP

EVAL_SPLITS = ("val", "test")


def measure_scale(rows: list[dict], processed_root: Path, n: int, seed: int) -> dict:
    """Deviazione standard per canale (unita' native = codice / int16_scale) su n registrazioni di training estratte col seme; mediana e k."""
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["filename"])
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(train), size=min(n, len(train)), replace=False))
    sds = []
    for i in pick:
        u, s = R.recording_key(train[i])
        d = processed_root / u / s
        meta = json.loads((d / "metadata.json").read_text())
        x = np.asarray(np.load(d / "data_int16.npy", mmap_mode="r"), dtype=np.float64) / float(meta["int16_scale"])
        sds.extend(float(v) for v in x.std(axis=0))
    med = float(np.median(sds))
    k = R.scale_exponent(med)
    return {"n_recordings": int(len(pick)), "seed": seed, "n_channel_sd": len(sds), "median_channel_sd_native": med,
            "p05_channel_sd_native": float(np.percentile(sds, 5)), "p95_channel_sd_native": float(np.percentile(sds, 95)),
            "exponent_k": k, "factor": 10.0 ** k, "median_channel_sd_scaled": med * 10.0 ** k, "rule_range": list(R.SCALE_SD_RANGE)}


def process(angles: np.ndarray, n_processed: int, blocks: np.ndarray | None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Finestre ufficiali, bersagli e controlli di una registrazione. Ritorna (riga del manifest, starts, targets)."""
    mask = R.no_ik_failure(angles)
    starts = R.official_window_starts(mask)
    inst = R.eval_instants(starts)
    targets = angles[inst].astype(np.float32) if len(starts) else np.zeros((0, R.EVAL_PREDICTED // R.EVAL_STEP, R.N_JOINTS), np.float32)
    row = {"n_samples": int(angles.shape[0]), "n_processed": int(n_processed), "aligned": int(angles.shape[0]) == int(n_processed),
           "n_ik_failure_rows": int((~mask).sum()), "n_windows": int(len(starts)), "n_instants": int(inst.size),
           "instants_all_valid": bool(mask[inst].all()) if inst.size else True}
    if blocks is not None and inst.size:
        b = inst // R.BLOCK
        ok = b < blocks.shape[0]
        diff = np.abs(np.asarray(blocks[b[ok]], dtype=np.float64) - angles[inst[ok]])
        row["block_diff_max_rad"] = float(diff.max()) if diff.size else None
        row["block_diff_median_rad"] = float(np.median(diff)) if diff.size else None
    return row, starts, targets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--pose-root", required=True)
    ap.add_argument("--out", required=True, help="cartella nuova o vuota")
    ap.add_argument("--n-scale", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="solo collaudo: al massimo N registrazioni di validazione e test")
    ap.add_argument("--time-budget-s", type=float, default=0.0, help="ferma l'elaborazione dopo questo tempo (manifest incompleto, uscita 1)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} non e' vuota: serve una cartella nuova", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = R.read_rows(args.csv)
    by_stem = {Path(r["filename"]).stem: r for r in rows}
    wanted = {s for s, r in by_stem.items() if r["split"] in EVAL_SPLITS}
    processed_root, pose_root = Path(args.processed_root), Path(args.pose_root)
    manifest = {"kind": "temponet_emg2pose_eval_targets_v1", "code_sha": os.environ.get("WEARUSFM_CODE_SHA"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "args": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "protocol": {"predicted": R.EVAL_PREDICTED, "context": R.CONTEXT, "stride": R.EVAL_STRIDE, "step": R.EVAL_STEP}}
    manifest["scale"] = measure_scale(rows, processed_root, args.n_scale, args.seed)
    print(f"scala: mediana {manifest['scale']['median_channel_sd_native']:.4g} unita' native -> k = {manifest['scale']['exponent_k']}", flush=True)
    recs, problems = {}, []
    deadline = t0 + args.time_budget_s if args.time_budget_s > 0 else float("inf")
    stopped = False
    with tarfile.open(args.tar, "r:") as tf:
        for ref in E2P.iter_recordings(tf):
            if ref.stem not in wanted or ref.stem in recs:
                continue
            if time.time() > deadline or (args.limit and len(recs) >= args.limit):
                stopped = True
                break
            row = by_stem[ref.stem]
            u, s = R.recording_key(row)
            angles = PP.read_pose(tf.extractfile(ref.info if ref.info is not None else ref.member))["angles"]
            n_proc = json.loads((processed_root / u / s / "metadata.json").read_text())["shape"][0]
            pose_file = pose_root / u / s / "pose_200hz_f16.npy"
            blocks = np.load(pose_file, mmap_mode="r") if pose_file.is_file() else None
            info, starts, targets = process(angles, n_proc, blocks)
            dst = out / row["split"] / u / f"{s}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, starts=starts, targets=targets)
            info.update({"split": row["split"], "generalization": row["generalization"], "user": row["user"], "side": row["side"],
                         "moving_hand": row["moving_hand"], "stage": row["stage"], "file": str(dst.relative_to(out)),
                         "pose_blocks_found": blocks is not None})
            if not info["aligned"]:
                problems.append(f"{ref.stem}: {info['n_samples']} campioni nel raw contro {info['n_processed']} nel processato")
            if not info["instants_all_valid"]:
                problems.append(f"{ref.stem}: istanti valutati con fallimenti della cinematica inversa")
            recs[ref.stem] = info
    missing = sorted(wanted - set(recs))
    totals = defaultdict(lambda: {"recordings": 0, "recordings_without_windows": 0, "windows": 0, "instants": 0, "users": set()})
    for info in recs.values():
        t = totals[f"{info['split']}/{info['generalization']}"]
        t["recordings"] += 1
        t["recordings_without_windows"] += info["n_windows"] == 0
        t["windows"] += info["n_windows"]
        t["instants"] += info["n_instants"]
        t["users"].add(info["user"])
    manifest["totals"] = {k: {**{kk: vv for kk, vv in v.items() if kk != "users"}, "users": len(v["users"])} for k, v in sorted(totals.items())}
    diffs = [i["block_diff_max_rad"] for i in recs.values() if i.get("block_diff_max_rad") is not None]
    manifest["block_check"] = {"recordings": len(diffs), "max_rad": max(diffs) if diffs else None,
                               "median_of_max_rad": float(np.median(diffs)) if diffs else None}
    manifest["recordings"] = recs
    manifest["missing"] = missing if not (stopped or args.limit) else f"{len(missing)} non elaborate (collaudo o tempo esaurito)"
    manifest["problems"] = problems
    manifest["complete"] = not stopped and not args.limit and not missing and not problems
    manifest["seconds"] = round(time.time() - t0, 1)
    text = json.dumps(manifest, indent=1)
    (out / "manifest.json").write_text(text)
    print(f"registrazioni {len(recs)} su {len(wanted)}; totali {json.dumps(manifest['totals'])}; problemi {len(problems)}; "
          f"sha256 manifest {hashlib.sha256(text.encode()).hexdigest()}; {manifest['seconds']:.0f} s", flush=True)
    return 0 if manifest["complete"] or (args.limit and not problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
