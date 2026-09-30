"""ANALISI DESCRITTIVA, NON CONGELATA (D5a resta com'e'): le FEATURE CONTINUE del tokenizer (prima della quantizzazione) come alternativa ai
codici discreti come bersaglio del pretraining. Due domande, sugli stessi materiali del run vero di V1-V4 (gli array salvati):

(a) `dataset_id_probe_continuous`: quanta identita' del dataset portano le feature continue, rispetto alle 5 bande (baseline congelata di V3)
    e ai codici? Stesso campione, stesso split di soggetti, stessa sonda di V3: cambia solo la rappresentazione.
(b) `stability_continuous`: quanto si spostano le feature sotto il rumore di V4, rispetto a quanto differiscono gia' fra token diversi?

La logica e' in numpy; il modello entra come `feature_fn(windows (n, 3200) float32) -> (4, n*16, 128)` (in produzione `NeuroRVQRunner.features`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from wearusfm.tokenizer_checks import metrics as M

N_TOKENS_PER_WINDOW = 16  # patch da 200 ms in una finestra da 3,2 s
WINDOW_SAMPLES = N_TOKENS_PER_WINDOW * 200
FeatureFn = Callable[[np.ndarray], np.ndarray]


def token_features(feature_fn: FeatureFn, tokens: np.ndarray, batch: int = 32) -> np.ndarray:
    """tokens: (n, 200), n multiplo di 16 (finestre da 16 patch consecutive, come nel run). Ritorna (4, n, 128) float32."""
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 200 or tokens.shape[0] % N_TOKENS_PER_WINDOW:
        raise ValueError(f"tokens forma {tokens.shape}: attesa (n multiplo di {N_TOKENS_PER_WINDOW}, 200)")
    windows = tokens.reshape(-1, WINDOW_SAMPLES)
    parts = []
    for i in range(0, len(windows), batch):
        f = np.asarray(feature_fn(windows[i : i + batch]))
        if f.shape != (4, len(windows[i : i + batch]) * N_TOKENS_PER_WINDOW, f.shape[-1]):
            raise ValueError(f"feature_fn ha restituito {f.shape}")
        parts.append(f)
    return np.concatenate(parts, axis=1)


def pool_units(feats: np.ndarray, n_per_unit: int = M.N_TOKENS_PER_SAMPLE, *, with_std: bool = False) -> np.ndarray:
    """(4, n, d) -> (n // n_per_unit, 4*d): media (e, se `with_std`, deviazione standard) delle feature dei 256 token di un'unita' V3,
    per ramo, concatenando i rami. Le unita' sono blocchi consecutivi di `n_per_unit` token, come nel run."""
    n_branch, n, d = feats.shape
    if n % n_per_unit:
        raise ValueError(f"{n} token non divisibili in unita' da {n_per_unit}")
    blocks = feats.reshape(n_branch, n // n_per_unit, n_per_unit, d)
    pooled = [blocks.mean(axis=2)]
    if with_std:
        pooled.append(blocks.std(axis=2))
    return np.concatenate([p.transpose(1, 0, 2).reshape(n // n_per_unit, n_branch * d) for p in pooled], axis=1)


def standardize_with_train(x: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Standardizza le colonne con media e std del SOLO train (senza guardare val/test); colonne costanti restano a zero."""
    mu = x[part == "train"].mean(axis=0)
    sd = x[part == "train"].std(axis=0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def v3_setup(per_dataset: Mapping[str, Mapping], seed: int = 0):
    """Ricostruisce il campione di V3 come in `pipeline.run_v3` (stessi rng, stesso split), a partire dagli array salvati.
    per_dataset[nome] ha `codes` (4,16,n), `tokens` (n,200), `group_subject` (G,). Ritorna (nomi, codes_list, tokens_list, y, units, part)."""
    names = sorted(per_dataset)
    codes_l, tokens_l, y, units, ds_subj = [], [], [], [], {}
    for di, name in enumerate(names):
        d = per_dataset[name]
        for k, subj in enumerate(d["group_subject"]):
            sl = slice(k * M.N_TOKENS_PER_SAMPLE, (k + 1) * M.N_TOKENS_PER_SAMPLE)
            codes_l.append(d["codes"][:, :, sl])
            tokens_l.append(d["tokens"][sl])
            y.append(di)
            units.append(f"{name}/{subj}")
        ds_subj[name] = sorted(set(str(s) for s in d["group_subject"]))
    part_map = M.split_subjects(ds_subj, np.random.default_rng([seed, 3, 0]))
    part = np.array([part_map[(names[yy], u.split("/", 1)[1])] for yy, u in zip(y, units)])
    return names, codes_l, tokens_l, np.array(y), np.array(units), part


def probe_summary(x, y, units, part, seed) -> dict:
    """La sonda dataset-ID di V3 (stesso rng) ridotta a un dizionario serializzabile."""
    r = M.dataset_id_probe(x, y, units, part, np.random.default_rng([seed, 3, 1]))
    return {"balanced_accuracy": r.balanced_accuracy, "ci95": list(r.ci95), "chance": r.chance, "model": r.model, "n_test": r.n_test}


def dataset_id_probe_continuous(
    per_dataset: Mapping[str, Mapping], features: Mapping[str, np.ndarray], seed: int = 0
) -> dict:
    """Sonda dataset-ID di V3 su rappresentazioni diverse dello STESSO campione e dello STESSO split.
    `features[nome]` = (4, n, 128) come da `token_features`. Congelate (identiche a `run_v3`): 5 bande, codici. Non congelate: feature
    continue con media per ramo (512-d) e con media + deviazione standard (1024-d), standardizzate sul train; e, come controllo di
    scala, le 5 bande standardizzate allo stesso modo."""
    names, codes_l, tokens_l, y, units, part = v3_setup(per_dataset, seed)
    pooled = np.concatenate([pool_units(features[n], with_std=False) for n in names])
    pooled_sd = np.concatenate([pool_units(features[n], with_std=True) for n in names])
    if len(pooled) != len(y):
        raise ValueError(f"{len(pooled)} unita' di feature per {len(y)} campioni V3")
    bands = M.band_power_features(tokens_l)
    out = {
        "datasets": names, "n_samples": int(len(y)), "n_test_units": int((part == "test").sum()),
        "frozen_bands": probe_summary(bands, y, units, part, seed),
        "frozen_codes": probe_summary(M.code_histogram_features(codes_l), y, units, part, seed),
        "extra_bands_standardized": probe_summary(standardize_with_train(bands, part), y, units, part, seed),
        "extra_continuous_mean": probe_summary(standardize_with_train(pooled, part), y, units, part, seed),
        "extra_continuous_mean_std": probe_summary(standardize_with_train(pooled_sd, part), y, units, part, seed),
    }
    for k in range(4):  # un ramo alla volta: quale porta l'identita'
        sl = slice(k * 128, (k + 1) * 128)
        out[f"extra_continuous_mean_branch{k}"] = probe_summary(standardize_with_train(pooled[:, sl], part), y, units, part, seed)
    return out


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def stability_continuous(
    feature_fn: FeatureFn,
    tokens: np.ndarray,
    noise_floor_rms: float,
    *,
    n_windows: int = 200,
    seed: int = 0,
    batch: int = 32,
    noise_levels: Sequence[float] = (1.0, 0.1),
) -> dict:
    """Coseno fra le feature di un token pulito e dello stesso token con rumore bianco gaussiano (V4: alla soglia, e 0,1 x soglia), per
    ramo; a confronto, il coseno fra token DIVERSI (quanto sono gia' simili fra loro senza nessun rumore). Finestre: ogni k-esima, al
    piu' `n_windows`, uguali per pulito e rumoroso. Ritorna {livello: {mediana, q05 per ramo}} e il riferimento."""
    windows = np.asarray(tokens, dtype=np.float32).reshape(-1, WINDOW_SAMPLES)
    step = max(1, len(windows) // n_windows)
    x = windows[::step][:n_windows]
    clean = token_features(feature_fn, x.reshape(-1, 200), batch)
    rng = np.random.default_rng([seed, 6])
    out: dict = {"n_windows": int(len(x)), "noise_floor_rms": float(noise_floor_rms), "noise": {}}
    for lvl in noise_levels:
        noisy = token_features(feature_fn, (x + rng.normal(scale=lvl * noise_floor_rms, size=x.shape).astype(np.float32)).reshape(-1, 200), batch)
        cs = [cosine_rows(clean[k], noisy[k]) for k in range(4)]
        out["noise"][f"{lvl:g}x_floor"] = {
            "median_per_branch": [float(np.median(c)) for c in cs], "q05_per_branch": [float(np.quantile(c, 0.05)) for c in cs],
        }
    perm = np.random.default_rng([seed, 7]).permutation(clean.shape[1])
    ref = [cosine_rows(clean[k], clean[k][perm]) for k in range(4)]
    out["between_different_tokens"] = {
        "median_per_branch": [float(np.median(c)) for c in ref], "q95_per_branch": [float(np.quantile(c, 0.95)) for c in ref],
    }
    return out
