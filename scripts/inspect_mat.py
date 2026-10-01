#!/usr/bin/env python3
"""Ispezione in SOLA LETTURA di un file .mat (prima di scrivere un parser di ingest): variabili con forma e tipo, valori degli scalari, e per la
variabile EMG la deviazione standard per colonna e la quota di potenza sopra una frequenza (es. 555 Hz per DB8: se l'EMG e' stato acquisito a 1111 Hz
e sovracampionato a 2 kHz, sopra 555 Hz non deve esserci quasi nulla). Legge sia i .mat classici (scipy) sia i v7.3 (HDF5, h5py). Non scrive niente.

  python scripts/inspect_mat.py S1_E1_A1.mat --fs 2000 --above-hz 555 --seconds 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MAX_SCALAR_SIZE = 4  # variabili con al piu' tanti elementi si stampano per intero


def is_v73(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"MATLAB 7.3" in f.read(128)


def list_variables(path: Path) -> list[dict]:
    if is_v73(path):
        import h5py

        out = []
        with h5py.File(path, "r") as f:
            for k, v in f.items():
                if isinstance(v, h5py.Dataset):
                    out.append({"name": k, "shape": list(v.shape[::-1]), "dtype": str(v.dtype)})  # MATLAB salva in ordine di colonna
        return out
    import scipy.io as sio

    return [{"name": n, "shape": list(s), "dtype": t} for n, s, t in sio.whosmat(str(path))]


def load_variables(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    if is_v73(path):
        import h5py

        with h5py.File(path, "r") as f:
            return {n: np.asarray(f[n]).T for n in names if n in f}
    import scipy.io as sio

    mat = sio.loadmat(str(path), variable_names=names)
    return {n: np.asarray(mat[n]) for n in names if n in mat}


def power_fraction_above(x: np.ndarray, fs: float, above_hz: float) -> np.ndarray:
    """Per colonna di x (T, C): potenza sopra `above_hz` / potenza totale (media tolta)."""
    x = x - x.mean(axis=0, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    tot = p.sum(axis=0)
    return np.where(tot > 0, p[f > above_hz].sum(axis=0) / np.where(tot > 0, tot, 1.0), np.nan)


def _guard(fn, *a):
    """Una variabile strana (struct, cell, tabella MATLAB) non deve far fallire l'ispezione delle altre: si riporta l'errore al suo posto."""
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 - e' uno strumento di ispezione
        return f"ERRORE {type(e).__name__}: {e}"


def _label_summary(x) -> dict:
    x = np.asarray(x)
    return {"shape": list(x.shape), "length": int(x.size), "n_distinct": int(np.unique(x).size), "min": float(x.min()), "max": float(x.max())}


def inspect(path: Path, emg_var: str, fs: float, above_hz: float | None, seconds: float) -> dict:
    variables = list_variables(path)
    small = [v["name"] for v in variables if int(np.prod(v["shape"])) <= MAX_SCALAR_SIZE]
    label_like = [v["name"] for v in variables if v["name"] in ("stimulus", "restimulus", "repetition", "rerepetition")]
    data = load_variables(path, small + label_like + [emg_var])
    out: dict = {"file": str(path), "format": "v7.3 (HDF5)" if is_v73(path) else "classico", "variables": variables,
                 "scalars": {n: _guard(lambda v: v.ravel().tolist() if v.dtype.kind in "biuf" else str(v.ravel().tolist())[:200], np.asarray(data[n]))
                             for n in small if n in data}}
    out["labels"] = {n: _guard(_label_summary, data[n]) for n in label_like if n in data}
    if emg_var in data and np.asarray(data[emg_var]).dtype.kind not in "biuf":
        out["emg"] = f"variabile {emg_var!r} di tipo {np.asarray(data[emg_var]).dtype} (non numerica): non analizzata"
    elif emg_var in data:
        emg = np.asarray(data[emg_var], dtype=np.float64)
        out["emg"] = {"shape": list(emg.shape), "std_per_column": emg.std(axis=0).tolist(), "max_abs": float(np.abs(emg).max()),
                      "n_constant_columns": int((emg.std(axis=0) == 0).sum()), "duration_s_at_fs": emg.shape[0] / fs}
        if above_hz is not None:
            chunk = emg[: int(seconds * fs)]
            out["emg"][f"power_fraction_above_{above_hz:g}_hz_first_{seconds:g}_s"] = power_fraction_above(chunk, fs, above_hz).tolist()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--emg-var", default="emg")
    ap.add_argument("--fs", type=float, default=2000.0, help="frequenza della griglia del file (per la quota di potenza)")
    ap.add_argument("--above-hz", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=60.0, help="secondi iniziali su cui calcolare lo spettro")
    ap.add_argument("--compact", action="store_true", help="una riga per variabile, poi scalari, etichette ed EMG in breve")
    args = ap.parse_args(argv)
    r = inspect(args.path, args.emg_var, args.fs, args.above_hz, args.seconds)
    if not args.compact:
        print(json.dumps(r, indent=1))
        return 0
    print(f"FILE {r['file']} ({r['format']})")
    for v in r["variables"]:
        print(f"  {v['name']:24s} {str(v['shape']):22s} {v['dtype']}")
    print("  SCALARI", json.dumps(r["scalars"], ensure_ascii=False)[:600])
    print("  ETICHETTE", json.dumps(r["labels"], ensure_ascii=False)[:600])
    e = r.get("emg")
    if isinstance(e, dict):
        print(f"  EMG shape {e['shape']} max_abs {e['max_abs']:.4g} colonne costanti {e['n_constant_columns']} durata {e['duration_s_at_fs']:.2f} s a fs")
        print("  EMG std", [round(x, 6) for x in e["std_per_column"]])
        for k, v in e.items():
            if k.startswith("power_fraction"):
                print(f"  EMG {k}", [None if x != x else round(x, 4) for x in v])
    else:
        print("  EMG", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
