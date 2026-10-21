#!/usr/bin/env python3
"""Criteri del collaudo breve su dbg dei due bracci nuovi (JEPA-only, ancore-only), congelati prima del lancio
(`docs/bracci_obiettivo_finestra1.md`, «Collaudo dei bracci nuovi»). Collaudo dell'infrastruttura, non un risultato scientifico: dice se i
due bracci girano in DDP a 4 GPU col checkout dei bracci, coi pesi della loss voluti, fino al checkpoint post-decay. Non decide il lancio
dei bracci: lo decide Simone con l'esito.

Legge la cartella del run copiata sul Mac (config.json, summary.json, metrics.jsonl; i checkpoint restano sul cluster: se ne passa
l'elenco con `--listing`, output di `ls -l` della cartella del run) e scrive l'esito.

Uso: python scripts/arms_collaudo.py RUN_DIR --arm JEPA-only|ancore-only --steps 1300 --expected-sha SHA --listing ls.txt [--out esito.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "w1arm-collaudo-v1"
ARM_WEIGHTS = {"JEPA-only": (1.0, 0.0), "ancore-only": (0.0, 0.2)}  # (jepa_weight, anchor_weight), firma dei bracci del 16/10
ANCHORS = ("log_rms", "band_shape", "log_env", "ms_fast", "ms_slow")  # le ancore attuali (`training/jepa.py`, anchor_losses)
WORLD, BATCH_PER_PROCESS, WINDOWS = 4, 8, 32
REL_TOL = 1e-6  # il totale del contratto e' la somma pesata delle componenti in float64 (`run._accumulate_terms`)
EDGE = 50  # record mediati all'inizio e alla fine per «il termine allenato scende»; e passi esclusi dal ritmo (avvio)
CHECKPOINTS = ("checkpoint_predecay.pt", "checkpoint_postdecay.pt")
EVAL_EVERY = 500  # diagnostiche di non-collasso del preset sanity: ai multipli di 500 e all'ultimo passo
# A a 30M (elapsed / passi), solo per confronto: C2 W1CAL-C2-30M-lr0p0001-s1 e il piu' lento di C1 (W1CAL-C1-30M-lr0p0001-s0)
A_S_PER_STEP = {"C2": 18153.877413511276 / 24523, "C1_max": 0.7776}


def _select():
    spec = importlib.util.spec_from_file_location("calibration_select", ROOT / "scripts" / "calibration_select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clean(obj):
    """Non finiti -> None, ricorsivamente: l'esito si scrive anche (soprattutto) quando il run ha valori non finiti (re-review, rilievo 5)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def listing_sizes(text: str) -> dict[str, int]:
    """{nome: byte} dalle righe di `ls -l` (permessi, link, utente, gruppo, dimensione, data, nome)."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0][:1] in "-l" and parts[4].isdigit():
            out[parts[-1]] = int(parts[4])
    return out


def expected_evaluations(steps: int) -> list[int]:
    return sorted({s for s in range(EVAL_EVERY, steps + 1, EVAL_EVERY)} | {steps})


def check(run_dir: Path, arm: str, steps: int, expected_sha: str, listing: str) -> dict:
    if arm not in ARM_WEIGHTS:
        raise ValueError(f"braccio {arm!r}: uno di {sorted(ARM_WEIGHTS)}")
    CS = _select()
    run_dir = Path(run_dir)
    wj, wa = ARM_WEIGHTS[arm]
    crit = {}

    def put(name, ok, detail):
        crit[name] = {"ok": bool(ok), "detail": detail}

    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    stop = (run_dir / "STOP").read_text().strip()[:300] if (run_dir / "STOP").exists() else None
    put("completo", stop is None and summary.get("stopped") == "max_steps" and summary.get("steps") == steps,
        {"STOP": stop, "stopped": summary.get("stopped"), "steps": summary.get("steps"), "attesi": steps})
    code = (summary.get("code") or {}).get("sha")
    got = {"sha": code, "world_size": summary.get("world_size"), "batch_per_process": summary.get("batch_per_process"),
           "accum_steps": summary.get("accum_steps")}
    put("codice_e_ddp", got == {"sha": expected_sha, "world_size": WORLD, "batch_per_process": BATCH_PER_PROCESS, "accum_steps": 1}, got)

    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    jepa = cfg.get("jepa") or {}
    weights = {k: jepa.get(k) for k in ("jepa_weight", "anchor_weight", "keep_weight", "branch_weight")}
    put("pesi", weights == {"jepa_weight": wj, "anchor_weight": wa, "keep_weight": 0.0, "branch_weight": 0.0},
        {"config": weights, "attesi": {"jepa_weight": wj, "anchor_weight": wa, "keep_weight": 0.0, "branch_weight": 0.0}})

    recs = CS._records(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
    seq = [r.get("step") for r in recs]
    bad_windows = [r.get("step") for r in recs if r.get("windows") != WINDOWS]
    non_finite = CS._non_finite(recs) if recs else []
    put("record", seq == list(range(1, steps + 1)) and not bad_windows and not non_finite,
        {"records": len(recs), "in_sequenza": seq == list(range(1, steps + 1)), "passi_con_finestre_diverse": bad_windows[:5],
         "passi_non_finiti": non_finite[:5]})

    missing, off = [], []
    for r in recs:
        keys = ("total", "jepa") + ANCHORS
        if any(not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k]) for k in keys):  # inf - inf non supera nessuna tolleranza
            missing.append(r.get("step"))
            continue
        expected = wj * r["jepa"] + wa * sum(r[a] for a in ANCHORS)
        if abs(r["total"] - expected) > REL_TOL * max(1.0, abs(r["total"])):
            off.append(r.get("step"))
    put("composizione_della_loss", bool(recs) and not missing and not off,
        {"formula": f"total = {wj:g} x jepa + {wa:g} x somma delle ancore {list(ANCHORS)}", "tolleranza_relativa": REL_TOL,
         "passi_senza_componenti_finite": missing[:5], "passi_fuori_tolleranza": off[:5]})

    totals = [r["total"] for r in recs if isinstance(r.get("total"), (int, float))]
    first = statistics.fmean(totals[:EDGE]) if len(totals) >= 2 * EDGE else None
    last = statistics.fmean(totals[-EDGE:]) if len(totals) >= 2 * EDGE else None
    put("apprendimento", first is not None and last < first,
        {"termine_allenato": "total", "media_primi": first, "media_ultimi": last, "record_mediati": EDGE})

    diags = [r for r in recs if "student_collapse_ratio" in r]
    alarms = [r.get("alarm") for r in recs if r.get("alarm")]
    want = expected_evaluations(steps)
    put("non_collasso", not alarms and [r["step"] for r in diags] == want,
        {"allarmi": alarms, "valutazioni": [r["step"] for r in diags], "attese": want})

    sizes = listing_sizes(listing)
    put("checkpoint", all(sizes.get(n, 0) > 0 for n in CHECKPOINTS), {n: sizes.get(n) for n in CHECKPOINTS})

    later = [r for r in recs if r.get("step", 0) > EDGE]
    t_step = [r["t_step_s"] for r in later if isinstance(r.get("t_step_s"), (int, float))]
    t_data = [r["t_data_s"] for r in later if isinstance(r.get("t_data_s"), (int, float))]

    def edge_means(key):
        vals = [r[key] for r in recs if isinstance(r.get(key), (int, float))]
        return [statistics.fmean(vals[:EDGE]), statistics.fmean(vals[-EDGE:])] if len(vals) >= 2 * EDGE else None

    descriptive = {
        "s_per_passo_mediana": statistics.median(t_step) if t_step else None,
        "s_per_passo_A": A_S_PER_STEP,
        "attesa_dati_frazione": sum(t_data) / sum(t_step) if t_step and sum(t_step) > 0 else None,
        "elapsed_s": summary.get("elapsed_s"), "gpu_max_memory_gib": summary.get("gpu_max_memory_gib"),
        "jepa_inizio_fine": edge_means("jepa"),
        "ancore_inizio_fine": {a: edge_means(a) for a in ANCHORS},
        "non_collasso": [{k: r.get(k) for k in ("step", "student_collapse_ratio", "teacher_collapse_ratio", "student_erank", "teacher_erank",
                                                  "student_var_samples", "student_var_queries", "teacher_var_samples")} for r in diags],
        "lettura": ("JEPA-only: le teste delle ancore non sono allenate, le loss delle ancore registrate non dicono nulla; "
                    "ancore-only: la JEPA registrata misura un predittore non allenato (docs/bracci_obiettivo_finestra1.md)"),
    }
    return _clean({"protocol": PROTOCOL, "arm": arm, "run_dir": str(run_dir), "steps": steps, "passed": all(c["ok"] for c in crit.values()),
                   "criteria": crit, "descriptive": descriptive})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_WEIGHTS))
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--listing", type=Path, required=True, help="output di `ls -l` della cartella del run sul cluster")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = check(args.run_dir, args.arm, args.steps, args.expected_sha, args.listing.read_text())
    text = json.dumps(res, indent=1, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    for name, c in res["criteria"].items():
        print(f"{'OK ' if c['ok'] else 'NO '} {name}: {json.dumps(c['detail'])[:300]}")
    print(f"{res['arm']}: {'SUPERATO' if res['passed'] else 'NON SUPERATO'}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
