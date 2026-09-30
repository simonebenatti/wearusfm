"""Regola dei rami dell'ancora RVQ (D5b, FIRMATA e CORRETTA il 30/09/2026: `docs/proposta_ancora_rvq.md`).

Per ogni ramo, solo livello 0, sulle classi accese: accuratezza della sonda dataset-ID di V3 da due rappresentazioni dell'unita' da 256 token,
(1) l'istogramma dei codici e (2) la media dei vettori del codebook scelti; si prende la piu' alta. Il ramo e' idoneo se questa accuratezza supera
quella delle 5 potenze di banda (stesse classi, stesso split) di non piu' di Y = 10 punti (la stessa soglia di V3, `metrics.v3_passes`).

Lo split per soggetto e' quello di V3: si calcola sui 6 dataset del run (`continuous_features.v3_setup`) e POI si restringe alle classi accese, cosi'
nessun soggetto cambia parte rispetto al run. Solo numpy e scikit-learn: il codebook entra come array (vedi `scripts/rvq_branch_probe.py`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse

from wearusfm.tokenizer_checks import metrics as M
from wearusfm.tokenizer_checks.continuous_features import probe_summary as _probe
from wearusfm.tokenizer_checks.continuous_features import standardize_with_train, v3_setup

ENABLED_DEFAULT = ("camargo2021", "capgmyo", "emg2pose", "grabmyo")  # D5b ristretta: V2 <= 2 misurato, piu' il riferimento emg2pose


def level0_histogram(codes_list: Sequence[np.ndarray], branch: int) -> sparse.csr_matrix:
    """(n_unita', 8192): frazione dei token dell'unita' per ciascun codice del livello 0 del ramo `branch`. codes_list: array (4, 16, n_token)."""
    rows, cols, vals = [], [], []
    for i, codes in enumerate(codes_list):
        c = np.asarray(codes)[branch, 0]
        if c.min() < 0 or c.max() >= M.N_CODE:
            raise ValueError("codice fuori da [0, 8192)")
        rows.append(np.full(c.shape, i))
        cols.append(c)
        vals.append(np.full(c.shape, 1.0 / c.shape[0]))
    m = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(codes_list), M.N_CODE))
    return m.tocsr()


def level0_codebook_mean(codes_list: Sequence[np.ndarray], branch: int, codebook: np.ndarray) -> np.ndarray:
    """(n_unita', d): media sui token dell'unita' del vettore del codebook scelto al livello 0 del ramo. codebook: (8192, d) del ramo."""
    if codebook.shape[0] != M.N_CODE:
        raise ValueError(f"codebook con {codebook.shape[0]} righe, attese {M.N_CODE}")
    out = []
    for codes in codes_list:
        c = np.asarray(codes)[branch, 0]
        if c.min() < 0 or c.max() >= M.N_CODE:  # un codice negativo indicizzerebbe dalla fine senza errore
            raise ValueError("codice fuori da [0, 8192)")
        out.append(codebook[c].mean(axis=0))
    return np.stack(out).astype(np.float64)


def branch_probe(
    per_dataset: Mapping[str, Mapping],
    codebooks: np.ndarray,
    enabled: Sequence[str] = ENABLED_DEFAULT,
    seed: int = 0,
    setup: tuple | None = None,
) -> dict:
    """Applica la regola dei rami. per_dataset[nome]: `codes` (4,16,n), `tokens` (n,200), `group_subject` (G,), per TUTTI i dataset del run (servono
    a ricostruire lo split di V3). codebooks: (4, 8192, d), livello 0 dei 4 rami."""
    codebooks = np.asarray(codebooks)
    if codebooks.ndim != 3 or codebooks.shape[:2] != (M.N_BRANCHES, M.N_CODE):
        raise ValueError(f"codebooks forma {codebooks.shape}, attesa (4, 8192, d)")
    missing = sorted(set(enabled) - set(per_dataset))
    if missing:
        raise ValueError(f"classi accese assenti dagli array: {missing}")
    # `setup`: il risultato di v3_setup(per_dataset, seed) gia' calcolato dal chiamante (lo script lo usa anche per riprodurre V3)
    names, codes_l, tokens_l, y, units, part = setup if setup is not None else v3_setup(per_dataset, seed)
    keep = np.flatnonzero(np.isin(np.asarray(names)[y], list(enabled)))
    codes_l = [codes_l[i] for i in keep]
    tokens_l = [tokens_l[i] for i in keep]
    y, units, part = y[keep], units[keep], part[keep]
    for p in ("train", "val", "test"):
        if len(np.unique(y[part == p])) != len(enabled):
            raise ValueError(f"la parte {p} non contiene tutte le classi accese")

    bands = _probe(M.band_power_features(tokens_l), y, units, part, seed)
    out: dict = {
        "rule": "ramo idoneo se max(acc istogramma, acc vettore medio) - acc 5 bande <= Y (livello 0, stesse classi, split di V3)",
        "y_points": M.Y_POINTS, "enabled": sorted(enabled), "n_units": int(len(y)), "n_test_units": int((part == "test").sum()),
        "bands": bands, "branches": {},
    }
    for b in range(M.N_BRANCHES):
        hist = _probe(level0_histogram(codes_l, b), y, units, part, seed)
        dense = _probe(standardize_with_train(level0_codebook_mean(codes_l, b, codebooks[b]), part), y, units, part, seed)
        best = "histogram" if hist["balanced_accuracy"] >= dense["balanced_accuracy"] else "codebook_mean"
        acc = max(hist["balanced_accuracy"], dense["balanced_accuracy"])
        out["branches"][str(b)] = {
            "histogram": hist, "codebook_mean": dense, "branch_accuracy": acc, "from": best,
            "diff_minus_bands": acc - bands["balanced_accuracy"],
            "eligible": bool(M.v3_passes(acc, bands["balanced_accuracy"])),
        }
    out["eligible_branches"] = [int(b) for b, v in out["branches"].items() if v["eligible"]]
    out["anchor_starts"] = bool(out["eligible_branches"])
    return out


BASE_BRANCH = 0  # esito della regola del 30/09/2026 (decisioni.md): si predice solo il livello 0 del ramo 0


def confirm_new_dataset(
    base_per_dataset: Mapping[str, Mapping],
    new_name: str,
    new_arrays: Mapping,
    codebooks: np.ndarray,
    base_enabled: Sequence[str] = ENABLED_DEFAULT,
    seed: int = 0,
) -> dict:
    """Opzione (b) della conferma (decisioni.md, 30/09/2026): il dataset nuovo entra se, con classi = base accesa + lui, il livello 0 del ramo 0 resta
    idoneo con la stessa regola. Si prova da solo. Lo split per soggetto dei dataset della base deve restare quello di oggi: se cambia, ValueError."""
    if new_name in base_per_dataset:
        raise ValueError(f"{new_name} e' gia' nella base")
    b_names, _, _, b_y, b_units, b_part = v3_setup(base_per_dataset, seed)
    per = {**base_per_dataset, new_name: new_arrays}
    setup = v3_setup(per, seed)
    n_names, _, _, n_y, n_units, n_part = setup

    def split_of_base(names, y, units, part):
        return {u: p for u, p, yy in zip(units, part, y) if names[yy] in base_enabled}

    if split_of_base(b_names, b_y, b_units, b_part) != split_of_base(n_names, n_y, n_units, n_part):
        raise ValueError(f"aggiungendo {new_name} lo split dei dataset della base cambia: mi fermo")
    r = branch_probe(per, codebooks, enabled=[*base_enabled, new_name], seed=seed, setup=setup)
    b0 = r["branches"][str(BASE_BRANCH)]
    return {
        "dataset": new_name, "enters": bool(b0["eligible"]), "classes": r["enabled"], "n_units": r["n_units"], "n_test_units": r["n_test_units"],
        "bands": r["bands"], "branch0": b0, "rule": "entra se il livello 0 del ramo 0 resta idoneo con classi = base + dataset (opzione b)",
    }
