"""Campionamento dei token per V1-V4 (D5a, congelata). La LETTURA delle sessioni ingerite e' in `wearusfm.data.processed`
(qui ri-esportata); il formato e' in `docs/formato_processato.md`.

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wearusfm.data.processed import (  # noqa: F401  (ri-esportati: erano definiti qui)
    SessionData,
    choose_sessions,
    discover_sessions,
    load_session,
    qc_valid_from_metadata,
)
from wearusfm.preprocessing.canonical_view import (
    PATCH_SAMPLES,
    apply_scale,
    bandpass_resample,
    patchify,
    session_scale,
)

PATCHES_PER_SAMPLE = 16  # 16 patch x 200 ms = 3,2 s per canale




@dataclass
class CanonicalSession:
    dataset: str
    subject: str
    session: str
    stream: np.ndarray  # (C, A_totale, 200): prove concatenate lungo le patch, gia' scalate
    qc_valid: np.ndarray
    scale: float






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
