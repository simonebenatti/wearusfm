"""Readout per canale dell'encoder congelato (23/10/2026, Simone: «Parti dal readout per canale sui checkpoint esistenti»; diagnosi in
`docs/diagnosi_pretraining_20261023.md`, protocollo in `docs/readout_per_canale_v1.md`).

Il readout delle sonde P1/P2 (`spectral_step1.pool_p1p2`) e' la media dell'encoder locale sui canali piu' la media del backbone su latenti e
istanti: 2d numeri per finestra, in cui *quale* canale e' attivo sopravvive solo se la rete lo ha mescolato col contenuto. Qui l'uscita si
tiene per canale, mediata solo nel tempo, a due profondita' della stessa passata in avanti (nessun parametro nuovo, nessun training):

| strato | cosa | per finestra |
|---|---|---|
| `pool` | il readout di sempre (`pool_p1p2`), per il controllo d'uguaglianza con le cache esistenti | 2d |
| `local` | uscita dell'encoder locale (target dell'opzione a di D11) | C x d |
| `dec` | uscita del decoder a query del modello (teacher) sulla vista senza maschera, interrogato su OGNI (canale, patch) valida: il target JEPA dell'opzione b (D11) esteso a tutte le posizioni; la rappresentazione profonda per canale (le teste delle ancore leggono invece il decoder dello studente sulla vista mascherata, alle sole posizioni nascoste) | C x d |

Non c'e' uno strato «token» (patch + identita', prima dell'encoder locale): il front-end e' lineare nel segnale e la media nel tempo delle
sue uscite su un segnale a media nulla e' ~0 qualunque sia l'ampiezza; resterebbe solo l'identita' di canale, costante. L'ampiezza entra
nella media temporale solo dopo una non linearita' (LayerNorm, attenzione, MLP), cioe' da `local` in poi, come nel readout medio.
La media per canale e' sulle patch valide (esistenti e del QC); un canale scartato dal QC o senza patch vale 0 (come la baseline classica).
L'ordine delle colonne e' (canale del montaggio, dimensione). Preprocessing identico a `fm_features.extract_features`: filtro del
pretraining, scala per soggetto/sessione, due ritagli da `crop_s` mediati, montaggio e insiemi di attenzione del dataset.

**Riduzione a larghezza pari** (sonda): C x d supera di molto i 2d del readout medio; per confrontare a pari larghezza si standardizza sul
train del fold, si proietta con una PCA stimata sul solo train a `k` componenti (default 2d) e si addestra la logistica di sempre senza
ristandardizzare (`harness.probe.linear_probe(scale=False)`, parametro aggiunto il 23/10/2026 con default invariato; Simone: «adesso le
modifiche principali sono quelle che fai tu»): la regolarizzazione resta nello spazio delle colonne standardizzate. Con k >= larghezza
non si riduce nulla e la sonda riceve le colonne come nella calibrazione (controllo d'uguaglianza nei test). Richiede torch per
l'estrazione; la riduzione e' solo numpy/sklearn.
"""

from __future__ import annotations

import numpy as np

LAYERS = ("pool", "local", "dec")
CHANNEL_LAYERS = ("local", "dec")
FEATURE_SET = "channel_readout"  # non in `feature_sets.FEATURE_SETS`: le cache non si confondono con quelle delle sonde di sempre


def forward_layers(model, inp):
    """Una passata in avanti senza maschera, come `WearUsFM.encode(inp, None)`, piu' l'uscita del decoder a query su tutte le (canale, patch)
    valide. Ritorna (enc, dec (C_tot, P, d): 0 dove non valido)."""
    import torch

    from wearusfm.model.fm import Encoded
    from wearusfm.model.perceiver import offsets_from_counts
    from wearusfm.training.jepa import hidden_queries

    tokens, patch_valid, time_valid = model.tokenizer(inp.signals, inp.fs)
    vis = patch_valid & inp.qc_valid[:, None]
    ids = model.identity(inp.codes, None)  # in eval: nessun dropout del muscolo
    local = model.local(tokens + ids[:, None, :], inp.sets, vis)
    offsets = offsets_from_counts(inp.counts)
    lat = model.pool(local, offsets, vis)
    per_sample = torch.stack([vis[a:b].any(dim=0) for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist())])
    key_tv = time_valid & per_sample
    enc = Encoded(local, model.backbone(lat, key_tv), key_tv, ids, patch_valid, vis)
    q_ch, q_t, q_off = hidden_queries(vis, inp.counts)  # tutte le (canale, patch) valide, in ordine di canale
    dec = torch.zeros_like(local)
    if q_ch.numel():
        dec[q_ch, q_t] = model.decode(enc, q_ch, q_t, q_off).to(dec.dtype)
    return enc, dec


def channel_means(x, valid, counts: list[int]):
    """(C_tot, P, d), (C_tot, P) -> (S, C x d) in float32: media sulle patch valide di ogni canale, 0 senza patch valide; tutti i campioni del
    batch hanno lo stesso numero di canali C (stesso montaggio)."""
    import torch

    m = valid.to(torch.float32)[..., None]
    mean = (x.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    if len(set(counts)) != 1:
        raise ValueError(f"canali diversi nello stesso batch: {sorted(set(counts))}")
    return mean.reshape(len(counts), counts[0] * x.shape[-1])


def readouts(model, inp) -> dict:
    """{strato: (S, larghezza)} float32 per un batch gia' preparato (`ModelInputs`)."""
    from wearusfm.model.spectral_step1 import pool_p1p2

    enc, dec = forward_layers(model, inp)
    out = {"pool": pool_p1p2(enc, inp.counts, inp.qc_valid).float()}
    for name, x in (("local", enc.local), ("dec", dec)):
        out[name] = channel_means(x, enc.visible, inp.counts)
    return out


def extract_channel_readouts(model, windows: np.ndarray, fs: float, montage: dict, subjects: np.ndarray, *, batch: int = 32,
                             device: str = "cpu", band: tuple[float, float] | None = (20.0, 450.0), crop_s: float = 4.0,
                             scale_windows: int = 10, k_neighbors: int = 8, edge_guard_s: float = 0.1) -> dict:
    """windows: (N, T, C) alla frequenza `fs`, canali nell'ordine del montaggio. Ritorna {strato: (N, larghezza)}: `pool` float32 (uguale a
    `fm_features.extract_features(..., feature_set="model")`), gli strati per canale float16 (C x d), piu' "channels" (C). Stesso
    preprocessing e stessi ritagli di `extract_features`."""
    import torch

    from wearusfm.harness import fm_features as FF
    from wearusfm.model import channel_codes as CC
    from wearusfm.model.channel_identity import codes_to_tensors
    from wearusfm.model.fm import ModelInputs
    from wearusfm.model.local_encoder import sets_to_tensors

    if getattr(model, "spectral_branches", None) is not None:
        raise ValueError("modello con rami spettrali: il suo readout non e' `pool_p1p2` e questo protocollo non lo copre")
    model.eval()
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, k_neighbors)
    c = layout.n_channels
    if windows.shape[2] != c:
        raise ValueError(f"{windows.shape[2]} canali per un montaggio da {c}")
    subjects = np.asarray(subjects)
    scales = FF.subject_scales(windows, subjects, fs, band, scale_windows)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))
    d = model.cfg.dim
    widths = {"pool": 2 * d, **{k: c * d for k in CHANNEL_LAYERS}}
    acc = {k: np.zeros((len(windows), w), dtype=np.float32) for k, w in widths.items()}
    limit, g = FF.band_limit_hz(montage, fs), int(round(edge_guard_s * fs))
    crops = FF._crops(windows.shape[1], fs, crop_s)
    with torch.no_grad():
        for a, b in crops:
            for i in range(0, len(windows), batch):
                idx = range(i, min(i + batch, len(windows)))
                xs = [FF._filtered(windows[j, a:b], fs, band) / np.float32(scales[subjects[j]]) for j in idx]
                n = len(xs)
                sig = [torch.from_numpy(x).to(device) for x in xs]
                inp = ModelInputs(sig, [fs] * n, {k: v.to(device) for k, v in codes_to_tensors(CC.pack_codes([code] * n)).items()},
                                  {k: v.to(device) for k, v in sets_to_tensors(CC.pack_attention_sets([sets] * n)).items()}, [c] * n,
                                  torch.as_tensor(layout.qc_valid).repeat(n).to(device), band_limit=[limit] * n, edge_guard=[(g, g)] * n)
                with amp:
                    out = readouts(model, inp)
                for k, v in out.items():
                    acc[k][i:i + n] += v.float().cpu().numpy() / len(crops)
    result = {"pool": acc["pool"], "channels": c}
    result.update({k: acc[k].astype(np.float16) for k in CHANNEL_LAYERS})
    return result


def effective_k(k: int | None, n_train: int, width: int) -> int | None:
    """Componenti della PCA davvero stimabili: None (nessuna riduzione) se `k` manca, e' 0 o raggiunge la larghezza; altrimenti min(k, n_train - 1):
    le feature standardizzate e centrate di n_train righe hanno rango al piu' n_train - 1, e con k >= rango la PCA e' una base ortonormale
    dello spazio dei dati di train, in cui la soluzione L2 della logistica (che vive in quello spazio) e' la stessa della larghezza intera: la
    «pari larghezza» su un dataset piccolo (UCI-EMG: ~220-240 finestre di train per fold) non vincola nulla, e si dichiara (`k_capped`)."""
    if k is None or k <= 0 or k >= width:
        return None
    if n_train < 2:
        raise ValueError(f"{n_train} finestre di train: nessuna riduzione possibile")
    return min(k, n_train - 1)


def fit_reduction(x_train: np.ndarray, k: int | None, seed: int = 0):
    """Standardizzazione (solo train) e, se `effective_k` lo chiede, PCA a quel numero di componenti stimata sul solo train. Ritorna (scaler, pca|None)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    k_eff = effective_k(k, len(x_train), x_train.shape[1])
    pca = None if k_eff is None else PCA(n_components=k_eff, svd_solver="randomized", random_state=seed).fit(scaler.transform(x_train))
    return scaler, pca


def apply_reduction(scaler, pca, x: np.ndarray) -> np.ndarray:
    z = scaler.transform(x)
    return z if pca is None else pca.transform(z)
