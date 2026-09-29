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
    scale = meta.get("int16_scale")
    if arr.ndim == 3:
        segs = [np.asarray(a, dtype=np.float64) for a in arr]
    elif arr.ndim == 2:
        segs = [np.asarray(arr, dtype=np.float64)]
    else:
        raise ValueError(f"{session_dir}: array a {arr.ndim} dimensioni")
    if scale:
        segs = [s / scale for s in segs]
    qc = qc_valid_from_metadata(meta, segs[0].shape[-1])
    return SessionData(dataset, subject, session, float(meta["native_fs_hz"]), segs, qc)


def to_canonical(s: SessionData) -> CanonicalSession:
    """Vista canonica per prova, scala unica di sessione sui canali che passano il QC, poi
    patch. Solleva ValueError se fs < 1 kHz (dataset non eleggibile)."""
    views = [bandpass_resample(seg, s.fs) for seg in s.segments]
    scale = session_scale(views, s.qc_valid)
    patched = [patchify(apply_scale(v, scale)) for v in views if v.shape[0] >= PATCH_SAMPLES]
    if not patched:
        raise ValueError(f"{s.dataset}/{s.subject}/{s.session}: nessuna prova con almeno una patch")
    return CanonicalSession(s.dataset, s.subject, s.session, np.concatenate(patched, axis=1), s.qc_valid, scale)


def draw_single_channel_samples(
    sessions: list[CanonicalSession], n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[str, str, int, int]]]:
    """n_samples campioni canale-per-volta da 16 patch, ripartiti in parti uguali fra i
    soggetti, con sessione, canale QC-valido e inizio (allineato alla griglia) a caso.
    Ritorna (array (n, 3200), provenienza [(soggetto, sessione, canale, patch_iniziale)])."""
    by_subject: dict[str, list[CanonicalSession]] = {}
    for s in sessions:
        if s.stream.shape[1] >= PATCHES_PER_SAMPLE and s.qc_valid.any():
            by_subject.setdefault(s.subject, []).append(s)
    if not by_subject:
        raise ValueError("nessuna sessione con almeno 16 patch e un canale valido")
    subjects = sorted(by_subject)
    out = np.empty((n_samples, PATCHES_PER_SAMPLE * PATCH_SAMPLES))
    prov = []
    for i in range(n_samples):
        subj = subjects[i % len(subjects)]  # round-robin: parti uguali fra i soggetti
        cs = by_subject[subj][rng.integers(len(by_subject[subj]))]
        ch = int(rng.choice(np.flatnonzero(cs.qc_valid)))
        a0 = int(rng.integers(0, cs.stream.shape[1] - PATCHES_PER_SAMPLE + 1))
        out[i] = cs.stream[ch, a0 : a0 + PATCHES_PER_SAMPLE].reshape(-1)
        prov.append((subj, cs.session, ch, a0))
    return out, prov
