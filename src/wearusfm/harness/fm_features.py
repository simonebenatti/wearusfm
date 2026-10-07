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
from wearusfm.ingest.common import montage_to_dict
from wearusfm.metadata.schema import (AnatomicalIdentity, AnatomicalPrecision, ChannelGroup, ChannelMetadata, Chirality, MontageMetadata,
                                      RawOrEnvelope, SensorCoordinates, Topology)
from wearusfm.model import channel_codes as CC
from wearusfm.model.channel_identity import codes_to_tensors
from wearusfm.model.fm import ModelInputs, WearUsFM
from wearusfm.model.local_encoder import sets_to_tensors
from wearusfm.model.spectral_step1 import pool_p1p2


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


@torch.no_grad()
def extract_features(model: WearUsFM, windows: np.ndarray, fs: float, montage: dict, subjects: np.ndarray, *, batch: int = 32,
                     device: str = "cpu", band: tuple[float, float] | None = (20.0, 450.0), crop_s: float = 4.0, scale_windows: int = 10,
                     k_neighbors: int = 8) -> np.ndarray:
    """windows: (N, T, C) alla frequenza `fs`, canali nell'ordine del montaggio. Ritorna (N, 2d) float32."""
    model.eval()
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, k_neighbors)
    c = layout.n_channels
    if windows.shape[2] != c:
        raise ValueError(f"{windows.shape[2]} canali per un montaggio da {c}")
    scales = subject_scales(windows, np.asarray(subjects), fs, band, scale_windows)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))
    feats = np.zeros((len(windows), 2 * model.cfg.dim), dtype=np.float32)
    crops = _crops(windows.shape[1], fs, crop_s)
    for a, b in crops:
        for i in range(0, len(windows), batch):
            idx = range(i, min(i + batch, len(windows)))
            sig = [torch.from_numpy(_filtered(windows[j, a:b], fs, band) / np.float32(scales[subjects[j]])).to(device) for j in idx]
            n = len(sig)
            inp = ModelInputs(sig, [fs] * n, {k: v.to(device) for k, v in codes_to_tensors(CC.pack_codes([code] * n)).items()},
                              {k: v.to(device) for k, v in sets_to_tensors(CC.pack_attention_sets([sets] * n)).items()}, [c] * n,
                              torch.as_tensor(layout.qc_valid).repeat(n).to(device))
            with amp:
                enc = model.encode(inp, None)
            feats[i:i + n] += pool_p1p2(enc, inp.counts, inp.qc_valid).cpu().numpy() / len(crops)
    return feats


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
