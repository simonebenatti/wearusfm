"""Insiemi di feature delle sonde P1/P2 (`fm_features.FEATURE_SETS`, test 2A dello step 2) e la loro identita' nelle cache: ogni cache estratta
registra l'insieme e la larghezza dell'embedding, e la fase probe verifica che coincidano con quanto dichiarato (review GPT del 13/10/2026: una
cache di statistiche da 64 colonne veniva accettata come "mask"). "branches" (step 2C, encoder raw congelato) sono le sole uscite dei rami
spettrali, senza l'encode raw: la parte raw e' quella gia' estratta dal teacher d'origine. Solo numpy: la fase probe gira sulla CPU senza torch."""

from __future__ import annotations

import numpy as np

FEATURE_SETS = ("model", "stats", "model+stats", "mask", "model+mask", "branches", "td")
EXTRA_COLUMNS = {"model": 0, "stats": 64, "model+stats": 64, "mask": 32, "model+mask": 32, "branches": 0, "td": 0}
# "td" (baseline classica, 23/10/2026): feature nel tempo per canale (`fm_features.TD_FEATURES`), nessun modello; le colonne dipendono dai
# canali del dataset e si registrano in `model_dim` come «colonne della cache»


def uses_model(feature_set: str) -> bool:
    """L'insieme contiene colonne del modello (l'embedding, o le sole uscite dei rami)."""
    return feature_set.startswith("model") or feature_set == "branches"


def cache_record(feature_set: str, model_dim: int, td_channels: int | None = None) -> dict:
    """Le chiavi da salvare nella cache: insieme di feature e colonne del modello (0 senza modello; con "branches", quelle dei rami)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set {feature_set!r}: uno di {FEATURE_SETS}")
    out = {"feature_set": np.asarray(feature_set),
           "model_dim": np.asarray(int(model_dim) if uses_model(feature_set) or feature_set == "td" else 0)}
    if feature_set == "td":  # baseline classica: le colonne sono canali x feature (4, `fm_features.TD_FEATURES`)
        if td_channels is None or int(model_dim) != 4 * int(td_channels):
            raise ValueError(f"td: {model_dim} colonne per {td_channels} canali (attese 4 per canale)")
        out["td_channels"] = np.asarray(int(td_channels))
    return out


def check_cache(z, requested: str, width: int) -> None:
    """`z`: la cache aperta (np.load). Una cache senza registrazione (estratta prima degli insiemi di feature) e' un embedding ("model").
    L'insieme registrato deve essere quello dichiarato alla sonda, e le colonne = embedding registrato + colonne dell'insieme."""
    recorded = str(z["feature_set"]) if "feature_set" in z.files else "model"
    if recorded != requested:
        raise ValueError(f"cache estratta come {recorded!r}, sonda dichiarata {requested!r}")
    if recorded == "td" and ("td_channels" not in z.files or width != 4 * int(z["td_channels"])):
        raise ValueError(f"cache td con {width} colonne: attese 4 per canale registrato")
    if "model_dim" in z.files:
        expected = int(z["model_dim"]) + EXTRA_COLUMNS[recorded]
        if width != expected:
            raise ValueError(f"cache {recorded!r} con {width} colonne, attese {expected}")
    elif recorded != "model":
        raise ValueError("cache senza larghezza registrata")
