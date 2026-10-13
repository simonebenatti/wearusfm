"""Insiemi di feature delle sonde P1/P2 (`fm_features.FEATURE_SETS`, test 2A dello step 2) e la loro identita' nelle cache: ogni cache estratta
registra l'insieme e la larghezza dell'embedding, e la fase probe verifica che coincidano con quanto dichiarato (review GPT del 13/10/2026: una
cache di statistiche da 64 colonne veniva accettata come "mask"). Solo numpy: la fase probe gira sulla CPU senza torch."""

from __future__ import annotations

import numpy as np

FEATURE_SETS = ("model", "stats", "model+stats", "mask", "model+mask")
EXTRA_COLUMNS = {"model": 0, "stats": 64, "model+stats": 64, "mask": 32, "model+mask": 32}


def cache_record(feature_set: str, model_dim: int) -> dict:
    """Le chiavi da salvare nella cache: insieme di feature e colonne dell'embedding (0 senza embedding)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set {feature_set!r}: uno di {FEATURE_SETS}")
    return {"feature_set": np.asarray(feature_set), "model_dim": np.asarray(int(model_dim) if feature_set.startswith("model") else 0)}


def check_cache(z, requested: str, width: int) -> None:
    """`z`: la cache aperta (np.load). Una cache senza registrazione (estratta prima degli insiemi di feature) e' un embedding ("model").
    L'insieme registrato deve essere quello dichiarato alla sonda, e le colonne = embedding registrato + colonne dell'insieme."""
    recorded = str(z["feature_set"]) if "feature_set" in z.files else "model"
    if recorded != requested:
        raise ValueError(f"cache estratta come {recorded!r}, sonda dichiarata {requested!r}")
    if "model_dim" in z.files:
        expected = int(z["model_dim"]) + EXTRA_COLUMNS[recorded]
        if width != expected:
            raise ValueError(f"cache {recorded!r} con {width} colonne, attese {expected}")
    elif recorded != "model":
        raise ValueError("cache senza larghezza registrata")
