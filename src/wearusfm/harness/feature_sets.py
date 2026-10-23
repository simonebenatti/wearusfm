"""Insiemi di feature delle sonde P1/P2 (`fm_features.FEATURE_SETS`, test 2A dello step 2) e la loro identita' nelle cache: ogni cache estratta
registra l'insieme e la larghezza dell'embedding, e la fase probe verifica che coincidano con quanto dichiarato (review GPT del 13/10/2026: una
cache di statistiche da 64 colonne veniva accettata come "mask"). "branches" (step 2C, encoder raw congelato) sono le sole uscite dei rami
spettrali, senza l'encode raw: la parte raw e' quella gia' estratta dal teacher d'origine. Solo numpy: la fase probe gira sulla CPU senza torch."""

from __future__ import annotations

import json

import numpy as np

FEATURE_SETS = ("model", "stats", "model+stats", "mask", "model+mask", "branches", "td")
EXTRA_COLUMNS = {"model": 0, "stats": 64, "model+stats": 64, "mask": 32, "model+mask": 32, "branches": 0, "td": 0}
# "td" (baseline classica, 23/10/2026): feature di Hudgins per canale (`fm_features.td_features`), nessun modello; le colonne dipendono dai
# canali del dataset e si registrano in `model_dim` come «colonne della cache», coi canali dei dati e la ricetta (verificate in `check_cache`)
TD_FEATURES = ("mav", "zc", "ssc", "wl")
TD_DEADZONE = 0.0  # soglia di ZC e SSC: quella di default di `features_hudgins` e di `HudginsLDABaseline` (nessuna soglia scelta qui)
TD_EDGE_GUARD_S = 0.1
TD_BAND = (20.0, 450.0)
TD_RECIPE = {"features": list(TD_FEATURES), "deadzone": TD_DEADZONE, "edge_guard_s": TD_EDGE_GUARD_S, "band_hz": list(TD_BAND),
             "qc": "canali scartati dal QC a 0", "window": "intera", "module": "harness.features_hudgins"}


def uses_model(feature_set: str) -> bool:
    """L'insieme contiene colonne del modello (l'embedding, o le sole uscite dei rami)."""
    return feature_set.startswith("model") or feature_set == "branches"


def cache_record(feature_set: str, model_dim: int, td_channels: int | None = None) -> dict:
    """Le chiavi da salvare nella cache: insieme di feature e colonne del modello (0 senza modello; con "branches", quelle dei rami)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set {feature_set!r}: uno di {FEATURE_SETS}")
    out = {"feature_set": np.asarray(feature_set),
           "model_dim": np.asarray(int(model_dim) if uses_model(feature_set) or feature_set == "td" else 0)}
    if feature_set == "td":  # baseline classica: le colonne sono canali dei dati x feature (`TD_FEATURES`), con la ricetta
        if td_channels is None or int(model_dim) != len(TD_FEATURES) * int(td_channels):
            raise ValueError(f"td: {model_dim} colonne per {td_channels} canali (attese 4 per canale)")
        out["td_channels"] = np.asarray(int(td_channels))
        out["td_recipe"] = np.asarray(json.dumps(TD_RECIPE, sort_keys=True))
    return out


def check_cache(z, requested: str, width: int) -> None:
    """`z`: la cache aperta (np.load). Una cache senza registrazione (estratta prima degli insiemi di feature) e' un embedding ("model").
    L'insieme registrato deve essere quello dichiarato alla sonda, e le colonne = embedding registrato + colonne dell'insieme."""
    recorded = str(z["feature_set"]) if "feature_set" in z.files else "model"
    if recorded != requested:
        raise ValueError(f"cache estratta come {recorded!r}, sonda dichiarata {requested!r}")
    if recorded == "td":
        if "td_channels" not in z.files or width != len(TD_FEATURES) * int(z["td_channels"]):
            raise ValueError(f"cache td con {width} colonne: attese 4 per canale registrato")
        if "td_recipe" not in z.files or json.loads(str(z["td_recipe"])) != TD_RECIPE:
            raise ValueError("cache td con una ricetta diversa da quella corrente (feature, soglia, bordi, banda, QC)")
    if "model_dim" in z.files:
        expected = int(z["model_dim"]) + EXTRA_COLUMNS[recorded]
        if width != expected:
            raise ValueError(f"cache {recorded!r} con {width} colonne, attese {expected}")
    elif recorded != "model":
        raise ValueError("cache senza larghezza registrata")
