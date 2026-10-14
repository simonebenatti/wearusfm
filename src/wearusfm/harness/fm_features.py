"""Feature congelate del nostro FM per la sonda lineare (passo 5, regime «encoder congelato + probe»; metriche P1 e P2 di D12, firmata il
04/10/2026). Richiede torch.

**Stesso ingresso del pretraining** (`data.pretraining_loader`): filtro passa-banda 20-450 Hz adattato alla Nyquist con notch a 50 e 60 Hz
(decisione 13), un solo numero di scala per soggetto (mediana dei MAD dei canali, v10 §4.3), codici di canale e insiemi di attenzione del montaggio.
**Normalizzazione senza etichette e senza il test:** la scala di un soggetto si stima dalle sue prime `scale_windows` finestre (v10 §8: «se per
sessione, dai primi N secondi»).

**Feature (scelta di AG, da confermare):** media delle uscite del backbone su istanti validi e latenti (d) concatenata alla media delle uscite
dell'encoder locale su canali e istanti validi (d), cioe' 2d numeri per finestra. Il modello ha visto contesti di 1-4 s: una finestra piu' lunga di
`crop_s` si divide in due ritagli (allineati all'inizio e alla fine) e le feature si mediano.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.harness.feature_sets import FEATURE_SETS, uses_model
from wearusfm.ingest.common import montage_to_dict
from wearusfm.metadata.schema import (AnatomicalIdentity, AnatomicalPrecision, ChannelGroup, ChannelMetadata, Chirality, MontageMetadata,
                                      RawOrEnvelope, SensorCoordinates, Topology)
from wearusfm.model import channel_codes as CC
from wearusfm.model.channel_identity import codes_to_tensors
from wearusfm.model.fm import ModelInputs, WearUsFM
from wearusfm.model.local_encoder import sets_to_tensors
from wearusfm.model.spectral_branches import readout


def myo8_montage(dataset: str, fs: float, side: str = "unknown", mains_hz: int = 50) -> dict:
    """Un bracciale Myo da 8 canali equispaziati (anello D_8, anatomia ignota), come il primo Myo di DB5 (`ingest.ninapro_db5`): EPN-612 e
    UCI-EMG. Orientamento assoluto sul braccio ignoto. Frequenza di rete: da verificare per dataset (default 50 Hz; il notch le toglie entrambe)."""
    chir = {"right": Chirality.RIGHT, "left": Chirality.LEFT}.get(side, Chirality.UNKNOWN)
    channels = [ChannelMetadata(sensor_coords=SensorCoordinates(channel_index=i, ring_angle_deg=45.0 * i), electrode_type="Myo_dry_8bit",
                                native_fs_hz=fs, effective_band_hz=(0.0, fs / 2), mains_frequency_hz=mains_hz, raw_or_envelope=RawOrEnvelope.RAW,
                                anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN), chirality=chir, body_region="forearm")
                for i in range(8)]
    m = MontageMetadata(dataset_name=dataset, subject_id=f"{dataset}_any", session_id="any",
                        groups=[ChannelGroup(group_id="myo", topology=Topology.RING, symmetry="D_8", channels=channels)])
    return montage_to_dict(m, [True] * 8)


NOTCH_HZ = (50.0, 60.0)  # come `LoaderConfig.notch_hz` (v10 §4.3: sempre entrambi)


def _filtered(x: np.ndarray, fs: float, band: tuple[float, float] | None) -> np.ndarray:
    """x: (T, C) -> (C, T) float32, filtrato come nel pretraining."""
    xt = np.ascontiguousarray(np.asarray(x, dtype=np.float64).T)
    return L._filter(xt, fs, band, NOTCH_HZ).astype(np.float32) if band else xt.astype(np.float32)


def subject_scales(windows: np.ndarray, subjects: np.ndarray, fs: float, band, scale_windows: int = 10) -> dict:
    """Per soggetto: mediana dei MAD dei canali sulle sue prime `scale_windows` finestre filtrate (nessuna etichetta, nessun dato di altri)."""
    out = {}
    for s in dict.fromkeys(subjects.tolist()):
        idx = np.flatnonzero(subjects == s)[:scale_windows]
        x = np.concatenate([_filtered(windows[i], fs, band) for i in idx], axis=1)
        mad = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
        v = float(np.median(mad))
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"soggetto {s}: scala nulla o non finita")
        out[s] = v
    return out


def _crops(n: int, fs: float, crop_s: float) -> list[tuple[int, int]]:
    m = int(math.floor(crop_s * fs))
    return [(0, n)] if n <= m else [(0, m), (n - m, n)]


def band_limit_hz(montage: dict, fs: float) -> np.ndarray:
    """(C,) limite superiore di banda per canale, come `pretraining_loader` (banda effettiva dichiarata, al piu' la Nyquist)."""
    hi = [float(ch["effective_band_hz"][1]) for g in montage["groups"] for ch in g["channels"]]
    return np.minimum(np.asarray(hi, dtype=np.float64), fs / 2.0)


# Insiemi di feature (test preliminare dello step 2, «2A»: Simone 13/10/2026): l'embedding del modello ("model"), le 32 statistiche spettrali
# fisse dello Step 1 con la loro maschera ("stats"), la concatenazione, e i controlli con la sola maschera di disponibilita' (una banda
# assente puo' identificare dispositivo o dataset senza dire nulla del segnale). Le statistiche mancanti valgono 0 accanto alla maschera:
# nessun valore stimato, quindi nessuna stima che possa vedere validazione o test. Nomi e colonne in `feature_sets` (anche per la fase probe).


def model_width(model: WearUsFM) -> int:
    """Colonne dell'embedding del modello (`spectral_branches.readout`): 2d piu' i rami spettrali."""
    return 2 * model.cfg.dim + model.cfg.branch_dim * len(model.cfg.spectral_branches)


def feature_width(model: WearUsFM, feature_set: str) -> int:
    """Colonne del modello in un insieme di feature (`feature_sets.cache_record`): l'embedding ("model..."), le sole uscite dei rami
    ("branches"), nessuna (statistiche, maschera)."""
    if feature_set == "branches":
        return model.cfg.branch_dim * len(model.cfg.spectral_branches)
    return model_width(model) if uses_model(feature_set) else 0


def spectral_stats(x: np.ndarray, fs: float, limit: np.ndarray, qc_valid: np.ndarray, guard: int) -> tuple[np.ndarray, np.ndarray]:
    """(32,) e maschera (32,): i target globali dello Step 1 (`global_spectral_targets` sulle ancore multiscala) di un ritaglio gia' filtrato e
    scalato (C, T), con le zone di bordo del pretraining."""
    from wearusfm.model.anchor_targets import anchor_targets
    from wearusfm.model.spectral_step1 import global_spectral_targets

    tg = anchor_targets(x, fs, limit, guard=(guard, guard), multiscale=True)
    y, m = global_spectral_targets([tg], [x.shape[0]], np.asarray(qc_valid, dtype=bool))
    return y[0], m[0]


@torch.no_grad()
def extract_features(model: WearUsFM, windows: np.ndarray, fs: float, montage: dict, subjects: np.ndarray, *, batch: int = 32,
                     device: str = "cpu", band: tuple[float, float] | None = (20.0, 450.0), crop_s: float = 4.0, scale_windows: int = 10,
                     k_neighbors: int = 8, edge_guard_s: float = 0.1, feature_set: str = "model") -> np.ndarray:
    """windows: (N, T, C) alla frequenza `fs`, canali nell'ordine del montaggio. Ritorna (N, D) float32. Con `feature_set` "model" (default):
    D = 2d, piu' i rami spettrali del modello se ne ha (`spectral_branches.readout`, lo stesso readout della loss del training); le altre
    varianti in `FEATURE_SETS`: "stats" = 32 statistiche (0 se mancanti) + 32 indicatori, "mask" = i soli 32 indicatori, "model+..." le
    concatenano all'embedding; "branches" = le sole uscite dei rami (le colonne dopo le prime 2d dell'embedding, calcolate allo stesso modo),
    senza l'encode raw (step 2C: la parte raw e' gia' estratta dal teacher d'origine congelato). Con due ritagli: media delle statistiche dove
    valgono, indicatore se valgono in almeno uno.
    Ogni ritaglio, filtrato da solo, ha i transitori del filtro a tutti e due i bordi: `edge_guard_s` li esclude dal supporto dei rami e delle
    statistiche (come `anchor_edge_guard_s` del pretraining)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set {feature_set!r}: uno di {FEATURE_SETS}")
    use_model, use_stats = uses_model(feature_set), feature_set not in ("model", "branches")
    branches_only = feature_set == "branches"
    if branches_only and getattr(model, "spectral_branches", None) is None:
        raise ValueError("feature_set 'branches': il modello non ha rami spettrali")
    model.eval()
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, k_neighbors)
    c = layout.n_channels
    if windows.shape[2] != c:
        raise ValueError(f"{windows.shape[2]} canali per un montaggio da {c}")
    scales = subject_scales(windows, np.asarray(subjects), fs, band, scale_windows)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))
    feats = np.zeros((len(windows), feature_width(model, feature_set)), dtype=np.float32) if use_model else None
    stat_sum, stat_n = np.zeros((len(windows), 32)), np.zeros((len(windows), 32))
    limit, g = band_limit_hz(montage, fs), int(round(edge_guard_s * fs))
    crops = _crops(windows.shape[1], fs, crop_s)
    for a, b in crops:
        for i in range(0, len(windows), batch):
            idx = range(i, min(i + batch, len(windows)))
            xs = [_filtered(windows[j, a:b], fs, band) / np.float32(scales[subjects[j]]) for j in idx]
            n = len(xs)
            if use_stats:
                for k, x in zip(idx, xs):
                    y, m = spectral_stats(x, fs, limit, layout.qc_valid, g)
                    stat_sum[k] += np.where(m, y, 0.0)
                    stat_n[k] += m
            if not use_model:
                continue
            sig = [torch.from_numpy(x).to(device) for x in xs]
            inp = ModelInputs(sig, [fs] * n, {k: v.to(device) for k, v in codes_to_tensors(CC.pack_codes([code] * n)).items()},
                              {k: v.to(device) for k, v in sets_to_tensors(CC.pack_attention_sets([sets] * n)).items()}, [c] * n,
                              torch.as_tensor(layout.qc_valid).repeat(n).to(device), band_limit=[limit] * n, edge_guard=[(g, g)] * n)
            with amp:
                if branches_only:  # come le colonne dei rami in `readout`, senza l'encode raw
                    z = torch.cat([v.float() for v in model.spectral_branches(inp).values()], -1)
                else:
                    z = readout(model, model.encode(inp, None), inp)
            feats[i:i + n] += z.float().cpu().numpy() / len(crops)
    if not use_stats:
        return feats
    present = stat_n > 0
    stats = np.where(present, stat_sum / np.maximum(stat_n, 1), 0.0).astype(np.float32)
    extra = present.astype(np.float32) if feature_set.endswith("mask") else np.concatenate([stats, present.astype(np.float32)], axis=1)
    return extra if feats is None else np.concatenate([feats, extra], axis=1)


def load_model(checkpoint: str, device: str = "cpu", which: str = "teacher") -> WearUsFM:
    """Il modello (teacher, default, oppure studente) da un `checkpoint.pt` del training, con la sua configurazione."""
    from dataclasses import fields

    from wearusfm.model.fm import FMConfig

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mcfg = state["config"]["model"]
    cfg = FMConfig(**{f.name: mcfg[f.name] for f in fields(FMConfig) if f.name in mcfg})
    model = WearUsFM(cfg)
    model.load_state_dict(state[which])
    return model.to(device).eval()
