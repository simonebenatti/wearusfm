"""Lettura delle sessioni ingerite e campionamento dei token per V1-V4 (D5a, congelata).

Layout letto (uguale per CapgMyo, GRABMyo, putEMG, CSL-hdemg, Camargo): una cartella per
sessione con `data_int16.npy` e `metadata.json`. L'array e' (n_prove, T, C) oppure (T, C)
(putEMG e Camargo: una registrazione continua); `int16_scale`, se presente, e' il fattore per
cui i dati sono stati MOLTIPLICATI (ricostruzione: dati / scala). GRABMyo salva l'int16
nativo senza scala. Poiche' la vista canonica divide per la deviazione standard di sessione,
l'unita' di misura sparisce comunque.

Definizioni (D5a punto 1): "registrazione" = sessione di un soggetto; un TOKEN e' un canale x
una patch da 200 ms; un campione di V2 e' un canale x 16 patch (3,2 s). Le prove di una
sessione si filtrano e ricampionano una per una, si scalano con UN numero di sessione e si
concatenano lungo le patch (ogni prova tronca il resto < 1 patch): la griglia e' ancorata
all'inizio di ogni prova. CapgMyo (prove da 1 s = 5 patch esatte) ha le giunture esattamente
sui bordi delle patch, quindi nessuna patch attraversa due prove, ma un campione da 16 patch
puo' contenere prove diverse: e' un dettaglio d'implementazione, non un cambio di definizione.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wearusfm.preprocessing.canonical_view import (
    PATCH_SAMPLES,
    apply_scale,
    bandpass_resample,
    patchify,
    session_scale,
)

PATCHES_PER_SAMPLE = 16  # 16 patch x 200 ms = 3,2 s per canale


@dataclass
class SessionData:
    dataset: str
    subject: str
    session: str
    fs: float
    segments: list[np.ndarray]  # ciascuno (T, C), float64, dati nativi
    qc_valid: np.ndarray  # (C,) bool


@dataclass
class CanonicalSession:
    dataset: str
    subject: str
    session: str
    stream: np.ndarray  # (C, A_totale, 200): prove concatenate lungo le patch, gia' scalate
    qc_valid: np.ndarray
    scale: float


def qc_valid_from_metadata(meta: dict, n_columns: int) -> np.ndarray:
    """I flag `qc_valid` dei canali, nell'ordine di scrittura (gruppi, poi canali): e' anche
    l'ordine delle colonne."""
    flags = [ch["qc_valid"] for g in meta["montage"]["groups"] for ch in g["channels"]]
    if len(flags) != n_columns:
        raise ValueError(f"{len(flags)} flag qc_valid per {n_columns} colonne")
    return np.asarray(flags, dtype=bool)


def load_session(session_dir: Path, dataset: str, subject: str, session: str) -> SessionData:
    meta = json.loads((session_dir / "metadata.json").read_text())
    arr = np.load(session_dir / "data_int16.npy", mmap_mode="r")
    scale = meta.get("int16_scale")  # numero (uguale per tutti i canali) o lista (una scala per canale: Hyser)
    if isinstance(scale, list):
        scale = np.asarray(scale, dtype=np.float64)
    if arr.ndim == 3:
        segs = [np.asarray(a, dtype=np.float64) for a in arr]
    elif arr.ndim == 2:
        trials = meta.get("trials")
        if trials and all("offset" in t and "n_samples" in t for t in trials):
            # registrazioni concatenate lungo il tempo (Camargo, NinaPro, Hyser): una prova = un segmento, cosi'
            # il filtro non attraversa le giunture (D5a: "le prove si filtrano e ricampionano una per una")
            segs = [np.asarray(arr[t["offset"] : t["offset"] + t["n_samples"]], dtype=np.float64) for t in trials]
        else:
            segs = [np.asarray(arr, dtype=np.float64)]
    else:
        raise ValueError(f"{session_dir}: array a {arr.ndim} dimensioni")
    if scale is not None and not (np.isscalar(scale) and not scale):
        segs = [s / scale for s in segs]
    qc = qc_valid_from_metadata(meta, segs[0].shape[-1])
    return SessionData(dataset, subject, session, float(meta["native_fs_hz"]), segs, qc)


def to_canonical(s: SessionData) -> CanonicalSession:
    """Vista canonica per prova, scala unica di sessione sui canali che passano il QC, poi
    patch. Solleva ValueError se fs < 1 kHz (dataset non eleggibile)."""
    views = [bandpass_resample(seg, s.fs).astype(np.float32) for seg in s.segments]
    scale = session_scale(views, s.qc_valid)
    patched = [
        patchify(apply_scale(v, scale)).astype(np.float32) for v in views if v.shape[0] >= PATCH_SAMPLES
    ]
    if not patched:
        raise ValueError(f"{s.dataset}/{s.subject}/{s.session}: nessuna prova con almeno una patch")
    return CanonicalSession(s.dataset, s.subject, s.session, np.concatenate(patched, axis=1), s.qc_valid, scale)


def discover_sessions(root: Path, dataset: str) -> list[tuple[str, str, Path]]:
    """Le cartelle di sessione sotto `root` (quelle con `metadata.json`), come
    (soggetto, sessione, percorso). Soggetto = primo livello sotto root; sessione = il resto
    del percorso (CapgMyo ha la sessione nella cartella del soggetto stesso: sessione 's')."""
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
    """Al piu' `max_sessions_per_subject` sessioni a caso per soggetto (contiene il tempo di
    calcolo; D5a non impone di usarle tutte)."""
    by_subject: dict[str, list] = {}
    for item in found:
        by_subject.setdefault(item[0], []).append(item)
    chosen = []
    for subj in sorted(by_subject):
        items = by_subject[subj]
        pick = rng.permutation(len(items))[:max_sessions_per_subject]
        chosen.extend(items[i] for i in sorted(pick))
    return chosen


def allocate_quota(
    sessions: list[tuple[str, str, Path]], n_samples: int, rng: np.random.Generator
) -> dict[tuple[str, str], int]:
    """Quota di campioni per (soggetto, sessione): parti uguali fra i soggetti (round-robin),
    dentro il soggetto la sessione e' a caso."""
    by_subject: dict[str, list[str]] = {}
    for subj, sess, _ in sessions:
        by_subject.setdefault(subj, []).append(sess)
    subjects = sorted(by_subject)
    quota: dict[tuple[str, str], int] = {}
    for i in range(n_samples):
        subj = subjects[i % len(subjects)]
        sess = by_subject[subj][rng.integers(len(by_subject[subj]))]
        quota[(subj, sess)] = quota.get((subj, sess), 0) + 1
    return quota


def draw_from_session(
    cs: CanonicalSession, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[str, str, int, int]]]:
    """n campioni canale-per-volta da 16 patch: canale QC-valido e inizio (sulla griglia) a
    caso. Ritorna (array (n, 3200), provenienza [(soggetto, sessione, canale, patch_iniziale)]).
    Array vuoto se la sessione ha meno di 16 patch o nessun canale valido (il chiamante
    registra il mancato quota, non lo nasconde)."""
    if cs.stream.shape[1] < PATCHES_PER_SAMPLE or not cs.qc_valid.any():
        return np.empty((0, PATCHES_PER_SAMPLE * PATCH_SAMPLES), dtype=np.float32), []
    out = np.empty((n, PATCHES_PER_SAMPLE * PATCH_SAMPLES), dtype=np.float32)
    prov = []
    for i in range(n):
        ch = int(rng.choice(np.flatnonzero(cs.qc_valid)))
        a0 = int(rng.integers(0, cs.stream.shape[1] - PATCHES_PER_SAMPLE + 1))
        out[i] = cs.stream[ch, a0 : a0 + PATCHES_PER_SAMPLE].reshape(-1)
        prov.append((cs.subject, cs.session, ch, a0))
    return out, prov
