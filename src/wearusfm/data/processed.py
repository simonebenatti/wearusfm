"""Lettura e validazione dei dati PROCESSATI (uscita degli ingest): il contratto di formato e' in
`docs/formato_processato.md`. Modulo neutro: lo usano le verifiche del tokenizer, lo useranno il costruttore del manifest e il
dataloader. Non dipende da torch ne' dalla vista canonica.

Una SESSIONE e' una cartella con `data_int16.npy` e `metadata.json` (piu', a seconda del dataset, `labels.npz` e altri file). Sotto la
radice del dataset: `<soggetto>/<sessione...>/` (il soggetto e' il primo livello; CapgMyo ha la sessione nella cartella del soggetto).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SessionData:
    dataset: str
    subject: str
    session: str
    fs: float
    segments: list[np.ndarray]  # ciascuno (T, C), float64, dati nativi ricostruiti (codice / scala)
    qc_valid: np.ndarray  # (C,) bool


def qc_flags(meta: dict) -> list:
    """I flag `qc_valid` del montaggio, nell'ordine di scrittura (gruppi, poi canali): e' anche l'ordine delle colonne. Nessun controllo."""
    return [ch["qc_valid"] for g in meta["montage"]["groups"] for ch in g["channels"]]


def qc_valid_from_metadata(meta: dict, n_columns: int) -> np.ndarray:
    """I flag `qc_valid` come array booleano, con il controllo che siano uno per colonna."""
    flags = qc_flags(meta)
    if len(flags) != n_columns:
        raise ValueError(f"{len(flags)} flag qc_valid per {n_columns} colonne")
    return np.asarray(flags, dtype=bool)


def read_scale(meta: dict):
    """`int16_scale` del sidecar: None (int16 nativo, GRABMyo), un numero (uguale per tutti i canali) o un array (C,) (una
    scala per canale: Hyser). Ricostruzione: dato = codice / scala."""
    scale = meta.get("int16_scale")
    if isinstance(scale, list):
        return np.asarray(scale, dtype=np.float64)
    return scale


def split_segments(arr, meta: dict) -> list[np.ndarray]:
    """Le prove di una sessione come segmenti (T_i, C) float64 GREZZI (codici, non ancora divisi per la scala): (n_prove, T, C) ->
    una per prova; (T, C) con `trials` (offset, n_samples) -> una per prova; (T, C) senza `trials` -> un solo segmento."""
    if arr.ndim == 3:
        return [np.asarray(a, dtype=np.float64) for a in arr]
    if arr.ndim == 2:
        trials = meta.get("trials")
        if trials and all("offset" in t and "n_samples" in t for t in trials):
            return [np.asarray(arr[t["offset"] : t["offset"] + t["n_samples"]], dtype=np.float64) for t in trials]
        return [np.asarray(arr, dtype=np.float64)]
    raise ValueError(f"array a {arr.ndim} dimensioni")


def load_session(session_dir: Path, dataset: str, subject: str, session: str) -> SessionData:
    meta = json.loads((session_dir / "metadata.json").read_text())
    arr = np.load(session_dir / "data_int16.npy", mmap_mode="r")
    scale = read_scale(meta)
    try:
        segs = split_segments(arr, meta)
    except ValueError as e:
        raise ValueError(f"{session_dir}: {e}") from None
    if scale is not None and not (np.isscalar(scale) and not scale):
        segs = [s / scale for s in segs]
    qc = qc_valid_from_metadata(meta, segs[0].shape[-1])
    return SessionData(dataset, subject, session, float(meta["native_fs_hz"]), segs, qc)


def discover_sessions(root: Path, dataset: str) -> list[tuple[str, str, Path]]:
    """Le cartelle di sessione sotto `root` (quelle con `metadata.json`), come (soggetto, sessione, percorso). Soggetto = primo livello
    sotto root; sessione = il resto del percorso (CapgMyo ha la sessione nella cartella del soggetto stesso: sessione 's')."""
    out = []
    for meta in sorted(Path(root).rglob("metadata.json")):
        rel = meta.parent.relative_to(root).parts
        if not rel:
            continue
        out.append((rel[0], "/".join(rel[1:]) or "s", meta.parent))
    return out


def choose_sessions(
    found: list[tuple[str, str, Path]], max_sessions_per_subject: int, rng: np.random.Generator
) -> list[tuple[str, str, Path]]:
    """Al piu' `max_sessions_per_subject` sessioni a caso per soggetto (contiene il tempo di calcolo)."""
    by_subject: dict[str, list] = {}
    for item in found:
        by_subject.setdefault(item[0], []).append(item)
    chosen = []
    for subj in sorted(by_subject):
        items = by_subject[subj]
        pick = rng.permutation(len(items))[:max_sessions_per_subject]
        chosen.extend(items[i] for i in sorted(pick))
    return chosen


# --- contratto di formato -----------------------------------------------------------------------------------------------------------


def validate_session(session_dir: Path) -> list[str]:
    """Controlla una sessione contro il contratto (`docs/formato_processato.md`). Ritorna la lista dei problemi (vuota = conforme).
    Non legge i dati: apre l'array in mmap e guarda solo forma e tipo."""
    problems: list[str] = []
    meta_path, data_path = session_dir / "metadata.json", session_dir / "data_int16.npy"
    for p in (meta_path, data_path):
        if not p.exists():
            problems.append(f"manca {p.name}")
    if problems:
        return problems
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        return [f"metadata.json non e' JSON valido: {e}"]
    try:
        arr = np.load(data_path, mmap_mode="r")
    except Exception as e:  # file corrotto o non .npy
        return [f"data_int16.npy illeggibile: {type(e).__name__}: {e}"]

    for key in ("montage", "native_fs_hz", "shape"):
        if key not in meta:
            problems.append(f"manca la chiave `{key}` nel sidecar")
    if arr.dtype != np.int16:
        problems.append(f"dtype {arr.dtype}, atteso int16")
    if arr.ndim not in (2, 3):
        problems.append(f"array a {arr.ndim} dimensioni, attese 2 o 3")
        return problems
    if "shape" in meta and list(arr.shape) != list(meta["shape"]):
        problems.append(f"`shape` del sidecar {meta['shape']} diversa da quella dell'array {list(arr.shape)}")
    fs = meta.get("native_fs_hz")
    if fs is not None and not (isinstance(fs, (int, float)) and fs > 0):
        problems.append(f"native_fs_hz non valida: {fs!r}")

    n_ch = arr.shape[-1]
    if "montage" in meta:
        try:
            flags = qc_flags(meta)
        except (KeyError, TypeError) as e:
            problems.append(f"montaggio malformato ({type(e).__name__}: {e})")
            flags = None
        if flags is not None:
            if len(flags) != n_ch:
                problems.append(f"{len(flags)} canali nel montaggio, {n_ch} colonne nell'array")
            if not all(isinstance(f, bool) for f in flags):
                problems.append("qc_valid non booleani")
            declared = meta.get("n_channels_discarded_by_qc")
            if declared is not None and declared != flags.count(False):
                problems.append(f"n_channels_discarded_by_qc={declared} ma {flags.count(False)} flag falsi")

    scale = meta.get("int16_scale")
    if isinstance(scale, list):
        if len(scale) != n_ch:
            problems.append(f"int16_scale ha {len(scale)} valori per {n_ch} canali")
        elif not all(isinstance(v, (int, float)) and np.isfinite(v) and v > 0 for v in scale):
            problems.append("int16_scale con valori non positivi o non finiti")
    elif scale is not None and not (isinstance(scale, (int, float)) and np.isfinite(scale) and scale > 0):
        problems.append(f"int16_scale non valida: {scale!r}")

    trials = meta.get("trials")
    if trials:
        has_offsets = all(isinstance(t, dict) and "offset" in t and "n_samples" in t for t in trials)
        if arr.ndim == 2 and has_offsets:
            pos = 0
            for i, t in enumerate(trials):
                if t["offset"] != pos:
                    problems.append(f"prova {i}: offset {t['offset']}, atteso {pos} (prove non contigue)")
                    break
                pos += t["n_samples"]
            else:
                if pos != arr.shape[0]:
                    problems.append(f"le prove coprono {pos} campioni, l'array ne ha {arr.shape[0]}")
        elif arr.ndim == 2:  # prove dichiarate ma senza posizione: il lettore le fonderebbe in un solo segmento
            problems.append("array 2D con `trials` senza `offset`/`n_samples` in ogni prova: i confini delle prove non si possono ricostruire")
        elif arr.ndim == 3 and len(trials) != arr.shape[0]:
            problems.append(f"{len(trials)} prove nel sidecar, {arr.shape[0]} nell'array")

    labels_path = session_dir / "labels.npz"
    if labels_path.exists() and arr.ndim == 2:
        try:
            with np.load(labels_path) as lab:
                for k in lab.files:
                    if lab[k].shape[0] != arr.shape[0]:
                        problems.append(f"labels.npz[{k}] ha {lab[k].shape[0]} righe, l'array {arr.shape[0]}")
        except Exception as e:  # file corrotto: e' un problema da riportare, non un crash
            problems.append(f"labels.npz illeggibile: {type(e).__name__}: {e}")
    return problems


def session_summary(session_dir: Path) -> dict:
    """Numeri di una sessione senza leggere i dati: campioni, ore, canali, canali validi, layout."""
    meta = json.loads((session_dir / "metadata.json").read_text())
    arr = np.load(session_dir / "data_int16.npy", mmap_mode="r")
    fs = float(meta["native_fs_hz"])
    n_samples = int(np.prod(arr.shape[:-1]))
    flags = qc_flags(meta)
    return {
        "n_samples": n_samples, "hours": n_samples / fs / 3600, "n_channels": int(arr.shape[-1]),
        "n_channels_valid": int(sum(bool(f) for f in flags)), "fs_hz": fs,
        "layout": "trials3d" if arr.ndim == 3 else "continuous" if not meta.get("trials") else "concatenated_trials",
        "scale": "none" if meta.get("int16_scale") is None else "per_channel" if isinstance(meta["int16_scale"], list) else "scalar",
    }
