"""Cache .npy di NinaPro DB8 per la replica di TEMPONet (`docs/temponet_db8_replication.md`): una lettura dei 36 .mat ufficiali.

Per ogni acquisizione `S<s>_E1_A<a>.mat` scrive `S<s>_A<a>_{emg,doa,restimulus,rerepetition}.npy` (emg float32 in volt come alla fonte, DoA
float32 in gradi con la matrice di Krasoulis 2019 eq. S2) e un `manifest.json` con sha256 di sorgenti e uscite, forme, statistiche descrittive
(ampiezze EMG per canale, intervalli di guanto e DoA, movimenti e ripetizioni) e provenienza del codice. Nessun filtro, nessuna normalizzazione.

Uso (job CPU, `scripts/slurm/temponet_db8_cache.sbatch`):
  python3 scripts/temponet_db8_cache.py --raw-root $SCRATCH/data/raw/ninapro --out <cartella NUOVA> --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wearusfm.harness import db8_regression as db8


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def describe(raw: dict[str, np.ndarray], arrays: dict[str, np.ndarray]) -> dict:
    emg = arrays["emg"].astype(np.float64)
    doa = arrays["doa"].astype(np.float64)
    glove = raw["glove"].astype(np.float64)
    restim = arrays["restimulus"]
    rerep = arrays["rerepetition"]
    moves = sorted(int(v) for v in np.unique(restim))
    return {
        "n_samples": int(emg.shape[0]),
        "duration_s": float(emg.shape[0] / db8.FS_HZ),
        "emg_dtype_source": str(raw["emg"].dtype),
        "glove_dtype_source": str(raw["glove"].dtype),
        "emg_std_uV": [float(v) for v in emg.std(axis=0) * 1e6],
        "emg_abs_p50_uV": [float(v) for v in np.percentile(np.abs(emg), 50, axis=0) * 1e6],
        "emg_abs_p99_uV": [float(v) for v in np.percentile(np.abs(emg), 99, axis=0) * 1e6],
        "emg_abs_max_uV": float(np.abs(emg).max() * 1e6),
        "emg_zero_std_channels": [int(c) for c in np.flatnonzero(emg.std(axis=0) == 0)],
        "glove_min": [float(v) for v in glove.min(axis=0)],
        "glove_max": [float(v) for v in glove.max(axis=0)],
        "doa_min_deg": [float(v) for v in doa.min(axis=0)],
        "doa_max_deg": [float(v) for v in doa.max(axis=0)],
        "doa_mean_deg": [float(v) for v in doa.mean(axis=0)],
        "doa_std_deg": [float(v) for v in doa.std(axis=0)],
        "restimulus_values": moves,
        "rest_fraction": float((restim == 0).mean()),
        "repetitions_per_movement": {str(m): int(rerep[restim == m].max()) for m in moves if m != 0},
    }


def one(job: tuple[str, str, int, int]) -> dict:
    raw_root, out_dir, s, a = job
    src = db8.acquisition_path(raw_root, s, a)
    t0 = time.time()
    raw = db8.read_raw(src)
    t_read = time.time() - t0
    arrays = db8.to_cache_arrays(raw)
    entry = {"subject": s, "acquisition": a, "source": str(src), "source_bytes": src.stat().st_size,
             "source_mtime": int(src.stat().st_mtime), "source_sha256": sha256(src), "read_seconds": round(t_read, 1)}
    entry.update(describe(raw, arrays))
    outputs = {}
    for name, arr in arrays.items():
        dst = db8.cache_file(out_dir, s, a, name)
        np.save(dst, arr)
        outputs[name] = {"file": dst.name, "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha256(dst)}
    entry["outputs"] = outputs
    print(f"S{s} A{a}: {entry['n_samples']} campioni ({entry['duration_s']:.1f} s), lettura {t_read:.0f} s", flush=True)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, help="cartella che contiene DB8/S<s>_E1_A<a>.mat")
    ap.add_argument("--out", required=True, help="cartella della cache: deve essere nuova o vuota")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, db8.N_SUBJECTS + 1)))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} non e' vuota: serve una cartella nuova", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.raw_root, str(out), s, a) for s in args.subjects for a in db8.ACQUISITIONS]
    missing = [str(db8.acquisition_path(args.raw_root, s, a)) for _, _, s, a in jobs if not db8.acquisition_path(args.raw_root, s, a).is_file()]
    if missing:
        print("sorgenti mancanti: " + ", ".join(missing), file=sys.stderr)
        return 2
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        entries = list(pool.map(one, jobs))
    manifest = {
        "kind": "temponet_db8_cache_v1",
        "raw_root": str(args.raw_root),
        "dof_to_doa": db8.DOF_TO_DOA.tolist(),
        "doa_names": list(db8.DOA_NAMES),
        "code_dir": os.environ.get("WEARUSFM_CODE_DIR"),
        "code_sha": os.environ.get("WEARUSFM_CODE_SHA"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seconds": round(time.time() - t0, 1),
        "entries": sorted(entries, key=lambda e: (e["subject"], e["acquisition"])),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"cache pronta: {out} ({len(entries)} acquisizioni, {manifest['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
