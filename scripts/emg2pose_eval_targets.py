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

Lettura del tar (correzione dopo il job 64096497, fallito per I/O: 226 GB letti in 1 h 50 per 2.949 registrazioni, 3 min 24 s di CPU).
Scorrendo i membri con `tarfile` sul file normale, la lettura anticipata di Lustre trascinava i dati dei membri di training saltati. Ora:
- l'indice si costruisce leggendo le sole intestazioni con `pread` su un descrittore con `POSIX_FADV_RANDOM`;
- ogni membro di validazione e di test si legge con un'unica `pread`, in thread paralleli;
- l'HDF5 si apre in memoria.
Il calcolo non cambia.

Uso (CPU, `scripts/slurm/emg2pose_eval_targets.sbatch`):
  python3 scripts/emg2pose_eval_targets.py --tar <tar> --csv <csv> --processed-root <emg2pose> --pose-root <emg2pose_pose> --out <cartella>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from wearusfm.harness import emg2pose_regression as R
from wearusfm.ingest import emg2pose as E2P
from wearusfm.ingest import emg2pose_pose as PP

EVAL_SPLITS = ("val", "test")


class PreadFile(io.RawIOBase):
    """File in sola lettura servito con `os.pread`, senza lettura anticipata (`POSIX_FADV_RANDOM` dove c'e'): per `tarfile`, che ne legge
    solo le intestazioni da 512 byte."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.fd = os.open(path, os.O_RDONLY)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.pos = 0
        self.size = os.fstat(self.fd).st_size
        self.reads = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = offset if whence == io.SEEK_SET else self.pos + offset if whence == io.SEEK_CUR else self.size + offset
        return self.pos

    def readinto(self, b) -> int:
        data = os.pread(self.fd, len(b), self.pos)
        b[: len(data)] = data
        self.pos += len(data)
        self.reads += 1
        return len(data)

    def close(self) -> None:
        if not self.closed:
            os.close(self.fd)
        super().close()


def tar_index(path: str | Path) -> dict[str, tuple[int, int]]:
    """{stem della registrazione: (offset dei dati, dimensione)}, dalle sole intestazioni del tar."""
    out = {}
    with PreadFile(path) as raw, tarfile.open(fileobj=raw, mode="r:") as tf:
        for m in tf:
            if m.isfile() and E2P._is_recording(m.name):
                out[Path(m.name).stem] = (int(m.offset_data), int(m.size))
    return out


def read_exact(fd: int, offset: int, size: int) -> bytes:
    """`size` byte da `offset` con pread (ripetuta se il sistema ne restituisce meno)."""
    parts, done = [], 0
    while done < size:
        chunk = os.pread(fd, size - done, offset + done)
        if not chunk:
            raise OSError(f"fine del file a {offset + done}, attesi {size} byte da {offset}")
        parts.append(chunk)
        done += len(chunk)
    return b"".join(parts)


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
    ap.add_argument("--io-threads", type=int, default=4, help="letture parallele dei membri del tar")
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
    t_idx = time.time()
    index = tar_index(args.tar)
    manifest["tar_index"] = {"members": len(index), "seconds": round(time.time() - t_idx, 1)}
    order = sorted((s for s in wanted if s in index), key=lambda s: index[s][0])  # nell'ordine del tar
    problems += [f"{s}: non trovata nel tar" for s in sorted(wanted - set(index))]
    if args.limit:
        order = order[: args.limit]
    print(f"indice del tar: {len(index)} registrazioni in {manifest['tar_index']['seconds']:.0f} s; da leggere {len(order)}", flush=True)
    fd = os.open(args.tar, os.O_RDONLY)
    bytes_read = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.io_threads)) as ex:
            pending: collections.deque = collections.deque()
            todo = iter(order)

            def submit_next() -> None:
                stem = next(todo, None)
                if stem is not None:
                    pending.append((stem, ex.submit(read_exact, fd, *index[stem])))

            for _ in range(2 * max(1, args.io_threads)):
                submit_next()
            while pending:
                if time.time() > deadline:
                    stopped = True
                    for _, fut in pending:
                        fut.cancel()
                    break
                stem, fut = pending.popleft()
                data = fut.result()
                submit_next()
                bytes_read += len(data)
                row = by_stem[stem]
                u, s = R.recording_key(row)
                angles = PP.read_pose(io.BytesIO(data))["angles"]
                del data
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
                    problems.append(f"{stem}: {info['n_samples']} campioni nel raw contro {info['n_processed']} nel processato")
                if not info["instants_all_valid"]:
                    problems.append(f"{stem}: istanti valutati con fallimenti della cinematica inversa")
                recs[stem] = info
                if len(recs) % 1000 == 0:
                    print(f"  {len(recs)} registrazioni, {bytes_read / 2**30:.1f} GiB, {time.time() - t0:.0f} s", flush=True)
    finally:
        os.close(fd)
    manifest["bytes_read_from_tar"] = bytes_read
    if args.limit and len(order) < len(wanted):
        stopped = True
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
