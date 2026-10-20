#!/usr/bin/env python3
"""Confronto dei bracci dell'obiettivo con un riferimento, con le regole firmate da Simone il 16/10/2026 (`docs/bracci_obiettivo_finestra1.md`,
«Decisioni del 16/10/2026 sulla critica del punto 5»; `docs/decisioni.md`). Nessuna soglia nuova: solo quelle firmate.

Regola (uguale a quella del 2A/2C), per un compito: confronto accoppiato per seme fra riferimento e braccio, coi soggetti ricampionati insieme
per i quattro modelli (riferimento e braccio, semi 0 e 1: `harness.downstream_ab.paired_component`); il braccio «batte» il riferimento se il
delta medio e' >= +0,01 (+1 punto), il limite inferiore dell'intervallo al 95% e' > 0 e il delta e' > 0 in tutti e due i semi; «non dannoso»
se il delta medio e' >= -0,01.
- **Scelta del default (regola 2):** sulla validation interna S (`scripts/calibration_probe.py`: P1 dalle partizioni train, P2 dai soggetti
  fuori dai test dei tre split), le predizioni fuori fold convertite in conteggi per soggetto e classe. Un braccio sostituisce il riferimento
  solo se lo batte su S della P1 **e** su S della P2. Se piu' bracci lo battono, la scelta e' di Simone (nessuna regola firmata per il caso).
- **Lettura scientifica (regola 1):** su P1/P2 ufficiali (`probe_p1.py`, `probe_p2.py`) con `harness.downstream_ab.compare_reports`; non sceglie
  il default. Se S e test ufficiali non concordano, il default segue S e il disaccordo si registra.
- Un braccio fermato dall'allarme di collasso (STOP con «allarme») e' «collassato» e non batte il riferimento (firma, punto 6). Un run fermato
  per un altro motivo non e' valutabile: errore, si decide a mano. Il riferimento deve essere completo.

Spec (JSON): {"reference": "A", "arms": ["JEPA-only", ...], "runs": {nome: {"0": {"run_dir", "s_p1", "s_p2", "p1", "p2"}, "1": {...}}}}
(per un braccio collassato bastano i run_dir). Uso: python3 scripts/arms_compare.py spec.json --out esito.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import downstream_ab as DA  # noqa: E402

MIN_DELTA, HARM_DELTA = 0.01, -0.01
SEEDS = ("0", "1")
N_BOOT, BOOT_SEED = 1000, 20261009  # come compare_reports


def _select():
    spec = importlib.util.spec_from_file_location("calibration_select", ROOT / "scripts" / "calibration_select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verdict(row: dict) -> dict:
    """La regola firmata su una riga di confronto (delta, ci95, delta_by_model_seed)."""
    beats = bool(row["delta"] >= MIN_DELTA and row["ci95"][0] > 0 and all(d > 0 for d in row["delta_by_model_seed"]))
    return {"beats": beats, "not_harmful": bool(row["delta"] >= HARM_DELTA)}


def s_comparison(ref: list[dict], arm: list[dict], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Confronto accoppiato su S (rapporti di `calibration_probe.py` dello stesso compito, semi 0 e 1 di riferimento e braccio)."""
    CS = _select()
    reps = [ref[0], arm[0], ref[1], arm[1]]  # ordine di paired_component: A0, B0, A1, B1
    if len({r.get("task", "p1") for r in reps}) != 1 or len({json.dumps(r["weights"], sort_keys=True) for r in reps}) != 1:
        raise ValueError("rapporti di S di compiti o pesi diversi")
    CS.check_pairing(reps)  # split, fold, ricetta e righe uguali prima di confrontare (review GPT 41ce569)
    oofs = [CS.load_oof(r) for r in reps]
    weights = reps[0]["weights"]
    datasets = sorted(weights)
    rng = np.random.default_rng(seed)
    w = np.array([weights[d] for d in datasets])
    rows, draws = {}, []
    for d in datasets:
        y0, _, s0 = oofs[0][d]
        if any(not (np.array_equal(o[d][0], y0) and np.array_equal(o[d][2], s0)) for o in oofs):
            raise ValueError(f"{d}: finestre o soggetti diversi fra i modelli: il confronto non e' accoppiato")
        stats = [DA.subject_counts(y0, o[d][1], s0) for o in oofs]
        result, sample = DA.paired_component(stats, rng, n_boot)
        rows[d] = result
        draws.append(sample)
    samples = w @ np.asarray(draws)
    out = {k: float(w @ np.array([rows[d][k] for d in datasets])) for k in ("A", "B", "delta")}
    out["delta_by_model_seed"] = (w @ np.array([rows[d]["delta_by_model_seed"] for d in datasets])).tolist()
    out["ci95"] = np.quantile(samples, [.025, .975]).tolist()
    out["paired_subject_se"] = float(np.std(samples, ddof=1))
    out["datasets"] = rows
    return {**out, **verdict(out)}


def official_comparison(ref: dict, arm: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Regola 1 su P1/P2 ufficiali: `compare_reports` col riferimento al posto di A e il braccio al posto di B."""
    reports = {t: {"A_seed0": ref["0"][t], "B_seed0": arm["0"][t], "A_seed1": ref["1"][t], "B_seed1": arm["1"][t]} for t in ("p1", "p2")}
    cmp = DA.compare_reports(reports, n_boot=n_boot, seed=seed)
    tasks = {t: {**{k: cmp["tasks"][t][k] for k in ("A", "B", "delta", "delta_by_model_seed", "ci95", "paired_subject_se")},
                 **verdict(cmp["tasks"][t])} for t in ("p1", "p2")}
    return {"tasks": tasks, "beats": all(v["beats"] for v in tasks.values()), "not_harmful": all(v["not_harmful"] for v in tasks.values()),
            "uncertainty": cmp["uncertainty"]}


def run_status(run_dir: Path) -> str:
    """completo / collassato (STOP dell'allarme di collasso) / fallito (altro STOP o run incompleto)."""
    run_dir = Path(run_dir)
    stop = run_dir / "STOP"
    if stop.exists():
        return "collassato" if "allarme" in stop.read_text() else "fallito"
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    return "completo" if summary.get("stopped") == "max_steps" else "fallito"


def compare(spec: dict) -> dict:
    ref_name = spec["reference"]
    runs = spec["runs"]
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731

    def status_of(name):
        st = {sd: run_status(runs[name][sd]["run_dir"]) for sd in SEEDS}
        if "fallito" in st.values():
            raise ValueError(f"{name}: run fermato per un motivo diverso dal collasso ({st}): non valutabile, si decide a mano")
        return "collassato" if "collassato" in st.values() else "completo"

    if status_of(ref_name) != "completo":
        raise ValueError(f"il riferimento {ref_name} non e' completo: nessun confronto")
    ref = {sd: runs[ref_name][sd] for sd in SEEDS}
    ref_official = {sd: {t: load(ref[sd][t]) for t in ("p1", "p2")} for sd in SEEDS}
    out = {"reference": ref_name, "rule": {"min_delta": MIN_DELTA, "harm_delta": HARM_DELTA, "ci": "95%, limite inferiore > 0",
                                           "seeds": "delta > 0 in tutti e due i semi"}, "arms": {}}
    for arm in spec["arms"]:
        st = status_of(arm)
        if st == "collassato":  # firma, punto 6: collassato = non batte il riferimento
            out["arms"][arm] = {"status": "collassato", "S": {"beats": False}, "official": {"beats": False, "not_harmful": False}}
            continue
        S = {t: s_comparison([load(ref[sd][f"s_{t}"]) for sd in SEEDS], [load(runs[arm][sd][f"s_{t}"]) for sd in SEEDS]) for t in ("p1", "p2")}
        S["beats"] = S["p1"]["beats"] and S["p2"]["beats"]
        arm_official = {sd: {t: load(runs[arm][sd][t]) for t in ("p1", "p2")} for sd in SEEDS}
        official = official_comparison(ref_official, arm_official)
        out["arms"][arm] = {"status": "completo", "S": S, "official": official, "S_and_official_agree": S["beats"] == official["beats"]}
    winners = [a for a, r in out["arms"].items() if r["S"]["beats"]]
    if not winners:
        out["default"] = {"choice": ref_name, "why": "nessun braccio batte il riferimento su S della P1 e della P2"}
    elif len(winners) == 1:
        out["default"] = {"choice": winners[0], "why": "unico braccio che batte il riferimento su S della P1 e della P2"}
    else:
        out["default"] = {"choice": None, "candidates": winners,
                          "why": "piu' bracci battono il riferimento su S: la scelta e' di Simone (nessuna regola firmata per il caso)"}
    out["disagreements"] = [a for a, r in out["arms"].items() if r["status"] == "completo" and not r["S_and_official_agree"]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = compare(json.loads(args.spec.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["default"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
