"""Esecuzione di V1-V4 (D5a, congelata il 29/09/2026) con un `runner` intercambiabile
(`NeuroRVQRunner` su Leonardo; un finto runner nei test locali).

Interfaccia del runner: `run(x (B, n_ch, n_time*200), spatial_idx, want_recon=True)` -> dict con
`codes` (4, 16, B*n_ch*n_time) e, se want_recon, `std_x`, `std_xrec` (B*n_ch*n_time, 200);
`spatial_index(nome_canale)` -> indice spaziale.

Disegno del campionamento (D5a punto 1): `n_groups` gruppi da 16 finestre; un gruppo e' un
campione di V3 (256 token = 16 finestre x 16 patch della STESSA sessione) e, insieme agli altri,
serve anche a V2 e V4. I gruppi sono ripartiti in parti uguali fra i soggetti. Le finestre sono
canale-per-volta da 3,2 s. Nessun mancato quota e' nascosto: finisce nel report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wearusfm.tokenizer_checks import metrics as M
from wearusfm.tokenizer_checks.neurorvq import SPATIAL_INDEX_FIXED
from wearusfm.tokenizer_checks.sessions import (
    PATCHES_PER_SAMPLE,
    CanonicalSession,
    SessionData,
    allocate_quota,
    choose_sessions,
    draw_from_session,
    to_canonical,
)

WINDOWS_PER_GROUP = 16
WINDOW_SAMPLES = 3200
SCALE_UNIT = "unit"
SCALE_NATIVE = "emg2pose_native"

Loader = Callable[[str, str, Path], SessionData]


@dataclass
class Config:
    n_groups: int = 78  # 78 x 256 = 19.968 token ~ 20.000 (D5a punto 1)
    v1_windows: int = 80  # 80 x 256 = 20.480 token
    seed: int = 0
    max_sessions_per_subject: int = 2
    batch: int = 64
    n_noise_seeds: int = 3
    skip_v3: bool = False


@dataclass
class Draw:
    windows: np.ndarray  # (G*16, 3200) float32, vista canonica, scala unitaria
    group_subject: list[str]  # (G,)
    group_session: list[str]
    shortfall_groups: int = 0
    sessions_used: list[str] = field(default_factory=list)


def run_single(runner, windows: np.ndarray, batch: int, want_recon: bool = True) -> dict:
    """Canale-per-volta con l'indice spaziale fisso. windows: (n, 3200)."""
    parts = [
        runner.run(windows[i : i + batch, None, :], [SPATIAL_INDEX_FIXED], want_recon=want_recon)
        for i in range(0, len(windows), batch)
    ]
    out = {"codes": np.concatenate([p["codes"] for p in parts], axis=2)}
    if want_recon:
        out["std_x"] = np.concatenate([p["std_x"] for p in parts])
        out["std_xrec"] = np.concatenate([p["std_xrec"] for p in parts])
    return out


def draw_groups(
    items: list[tuple[str, str, Path]], loader: Loader, dataset: str, cfg: Config, rng: np.random.Generator
) -> Draw:
    chosen = choose_sessions(items, cfg.max_sessions_per_subject, rng)
    quota = allocate_quota(chosen, cfg.n_groups, rng)
    paths = {(s, ss): p for s, ss, p in chosen}
    wins, g_subj, g_sess, used, short = [], [], [], [], 0
    for (subj, sess), g in sorted(quota.items()):
        cs = to_canonical(loader(subj, sess, paths[(subj, sess)]))  # ValueError se fs < 1 kHz
        w, prov = draw_from_session(cs, g * WINDOWS_PER_GROUP, rng)
        if len(w) == 0:
            short += g
            continue
        wins.append(w)
        g_subj += [subj] * g
        g_sess += [sess] * g
        used.append(f"{subj}/{sess}")
    if not wins:
        raise ValueError(f"{dataset}: nessuna finestra disponibile")
    return Draw(np.concatenate(wins), g_subj, g_sess, short, used)


def _tokens(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(-1, 200)


def calibrate_scale(runner, ref: Draw, native_scale: float, batch: int) -> dict:
    """D5a punto 1: la scala di arrivo si sceglie SOLO su emg2pose: vince il candidato con
    l'errore mediano di ricostruzione piu' basso, sugli stessi token."""
    res = {}
    for name, factor in ((SCALE_UNIT, 1.0), (SCALE_NATIVE, native_scale)):
        out = run_single(runner, ref.windows * factor, batch)
        res[name] = float(np.median(M.token_nmse(out["std_x"], out["std_xrec"])))
    chosen = min(res, key=res.get)
    return {"median_nmse": res, "chosen": chosen, "factor": 1.0 if chosen == SCALE_UNIT else native_scale,
            "native_scale": native_scale}


def run_v1(runner, emg2pose: list[CanonicalSession], factor: float, cfg: Config, rng: np.random.Generator) -> dict:
    """V1: multi-canale nativo (16 canali, nomi c1..c16 come nel repo) contro canale-per-volta
    con indice fisso, sugli STESSI token (finestre di 16 canali x 16 patch)."""
    usable = [s for s in emg2pose if s.stream.shape[1] >= PATCHES_PER_SAMPLE and s.qc_valid.all()]
    if not usable:
        raise ValueError("emg2pose: nessuna sessione utilizzabile per V1")
    by_subject: dict[str, list[CanonicalSession]] = {}
    for s in usable:
        by_subject.setdefault(s.subject, []).append(s)
    subjects = sorted(by_subject)
    wins = np.empty((cfg.v1_windows, 16, WINDOW_SAMPLES), dtype=np.float32)
    for i in range(cfg.v1_windows):
        pool = by_subject[subjects[i % len(subjects)]]
        cs = pool[int(rng.integers(len(pool)))]
        a0 = int(rng.integers(0, cs.stream.shape[1] - PATCHES_PER_SAMPLE + 1))
        wins[i] = cs.stream[:, a0 : a0 + PATCHES_PER_SAMPLE].reshape(16, -1)
    wins *= np.float32(factor)
    idx = [runner.spatial_index(f"c{j + 1}") for j in range(16)]
    multi_nmse, single_nmse = [], []
    for i in range(0, cfg.v1_windows, max(1, cfg.batch // 16)):
        chunk = wins[i : i + max(1, cfg.batch // 16)]
        out = runner.run(chunk, idx)
        multi_nmse.append(M.token_nmse(out["std_x"], out["std_xrec"]))
    multi_nmse = np.concatenate(multi_nmse)
    single = run_single(runner, wins.reshape(-1, WINDOW_SAMPLES), cfg.batch)
    single_nmse = M.token_nmse(single["std_x"], single["std_xrec"])
    return {
        "median_nmse_multi": float(np.median(multi_nmse)),
        "median_nmse_single": float(np.median(single_nmse)),
        "ratio_single_over_multi": float(np.median(single_nmse) / np.median(multi_nmse)),
        "threshold": M.V1_FACTOR,
        "passes": M.v1_passes(single_nmse, multi_nmse),
        "n_tokens": int(len(multi_nmse)),
    }


def analyze_dataset(runner, name: str, draw: Draw, factor: float, cfg: Config, ref_median_nmse: float | None) -> dict:
    """V2 (e V4) su un dataset; restituisce anche i materiali per V3 (`_v3`)."""
    windows = draw.windows * np.float32(factor)
    out = run_single(runner, windows, cfg.batch)
    nmse = M.token_nmse(out["std_x"], out["std_xrec"])
    res = {
        "n_windows": int(len(windows)), "n_groups": len(draw.group_subject), "shortfall_groups": draw.shortfall_groups,
        "n_subjects": len(set(draw.group_subject)), "sessions_used": draw.sessions_used,
        "v2_median_nmse": float(np.median(nmse)),
    }
    if ref_median_nmse is not None:
        ratio = float(np.median(nmse) / ref_median_nmse)
        res.update({"v2_ratio_vs_emg2pose": ratio, "v2_threshold": M.X_RATIO, "v2_passes": M.v2_passes(ratio)})
    # V4: rumore bianco alla soglia di rumore, scala di sessione fissa (gia' applicata alle finestre)
    tokens = _tokens(windows)
    floor = M.noise_floor_rms(tokens)
    fracs = []
    for k in range(cfg.n_noise_seeds):
        nr = np.random.default_rng([cfg.seed, 4, k])
        noisy = windows + nr.normal(scale=floor, size=windows.shape).astype(np.float32)
        codes_noisy = run_single(runner, noisy, cfg.batch, want_recon=False)["codes"]
        fracs.append(M.code_stability(out["codes"], codes_noisy))
    frac = np.mean(fracs, axis=0)
    res.update({"v4_noise_floor_rms": floor, "v4_fraction_unchanged": frac.tolist()})
    res["_v3"] = {"codes": out["codes"], "tokens": tokens}
    res["_frac"] = frac
    return res


def run_v3(per_dataset: dict[str, dict], draws: dict[str, Draw], cfg: Config) -> dict:
    """Sonda dataset-ID sui codici contro le potenze di banda (D5a punto 4)."""
    names = sorted(per_dataset)
    codes_list, tokens_list, y, units, ds_subjects = [], [], [], [], {}
    for di, name in enumerate(names):
        d = draws[name]
        mat = per_dataset[name]["_v3"]
        for k, subj in enumerate(d.group_subject):
            sl = slice(k * 256, (k + 1) * 256)
            codes_list.append(mat["codes"][:, :, sl])
            tokens_list.append(mat["tokens"][sl])
            y.append(di)
            units.append(f"{name}/{subj}")
        ds_subjects[name] = sorted(set(d.group_subject))
    short = {n: len(s) for n, s in ds_subjects.items() if len(s) < 3}
    if short:
        return {"computable": False, "reason": f"meno di 3 soggetti: {short}"}
    part_map = M.split_subjects(ds_subjects, np.random.default_rng([cfg.seed, 3, 0]))
    part = np.array([part_map[(names[yy], u.split("/", 1)[1])] for yy, u in zip(y, units)])
    y = np.array(y)
    units = np.array(units)
    x_codes = M.code_histogram_features(codes_list)
    x_bands = M.band_power_features(tokens_list)
    r_codes = M.dataset_id_probe(x_codes, y, units, part, np.random.default_rng([cfg.seed, 3, 1]))
    r_bands = M.dataset_id_probe(x_bands, y, units, part, np.random.default_rng([cfg.seed, 3, 1]))
    return {
        "computable": True, "datasets": names, "n_samples": int(len(y)),
        "codes": _probe_dict(r_codes), "bands": _probe_dict(r_bands),
        "diff_codes_minus_bands": r_codes.balanced_accuracy - r_bands.balanced_accuracy,
        "threshold": M.Y_POINTS, "passes": M.v3_passes(r_codes.balanced_accuracy, r_bands.balanced_accuracy),
    }


def _probe_dict(r: M.ProbeResult) -> dict:
    return {"balanced_accuracy": r.balanced_accuracy, "ci95": list(r.ci95), "chance": r.chance,
            "model": r.model, "n_test": r.n_test}


def run_v4(per_dataset: dict[str, dict]) -> dict:
    fr = {n: v["_frac"] for n, v in per_dataset.items()}
    per, overall = M.stable_levels(fr)
    return {"datasets": sorted(fr), "stable_by_dataset": {n: b.tolist() for n, b in per.items()},
            "stable_overall": overall.tolist(), "threshold": M.STABILITY_MIN, "passes": M.v4_passes(overall)}


def _public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def run_all(
    runner,
    cfg: Config,
    emg2pose: tuple[list[tuple[str, str, Path]], Loader],
    datasets: dict[str, tuple[list[tuple[str, str, Path]], Loader]],
) -> dict:
    """Esegue V1-V4. Si ferma dopo V1 se V1 non passa (il ripiego richiede una nuova decisione,
    D5a punto 2). Un dataset con fs < 1 kHz o senza dati e' escluso E registrato."""
    rng = np.random.default_rng(cfg.seed)
    em_items, em_loader = emg2pose
    em_chosen = choose_sessions(em_items, cfg.max_sessions_per_subject, rng)
    em_cs = [to_canonical(em_loader(*it)) for it in em_chosen]
    native_scale = float(np.median([c.scale for c in em_cs]))
    ref = draw_groups(em_items, em_loader, "emg2pose", cfg, rng)
    calib = calibrate_scale(runner, ref, native_scale, cfg.batch)
    factor = calib["factor"]
    report: dict = {"config": cfg.__dict__.copy(), "scale_calibration": calib}
    v1 = run_v1(runner, em_cs, factor, cfg, rng)
    report["v1"] = v1
    if not v1["passes"]:
        report["stopped"] = "V1 non passa: ripiego (mappare i canali sui 16 elettrodi) richiede una nuova decisione"
        return report
    per: dict[str, dict] = {}
    draws: dict[str, Draw] = {"emg2pose": ref}
    per["emg2pose"] = analyze_dataset(runner, "emg2pose", ref, factor, cfg, None)
    ref_median = per["emg2pose"]["v2_median_nmse"]
    excluded = {}
    for name, (items, loader) in datasets.items():
        try:
            draws[name] = draw_groups(items, loader, name, cfg, rng)
        except ValueError as e:
            excluded[name] = str(e)
            continue
        per[name] = analyze_dataset(runner, name, draws[name], factor, cfg, ref_median)
    report["excluded_datasets"] = excluded
    report["v2"] = {n: {k: v for k, v in _public(r).items() if k.startswith(("v2", "n_", "shortfall", "sessions"))}
                    for n, r in per.items()}
    report["v2_all_pass"] = all(r.get("v2_passes", True) for r in per.values())
    report["v3"] = {"skipped": True} if cfg.skip_v3 else run_v3(per, draws, cfg)
    report["v4"] = run_v4(per)
    report["v4_per_dataset"] = {n: {"noise_floor_rms": r["v4_noise_floor_rms"],
                                    "fraction_unchanged": r["v4_fraction_unchanged"]} for n, r in per.items()}
    return report
