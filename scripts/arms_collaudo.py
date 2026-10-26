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
ARM_WEIGHTS = {"JEPA-only": (1.0, 0.0), "ancore-only": (0.0, 0.2),  # (jepa_weight, anchor_weight), firma dei bracci del 16/10
               "A": (1.0, 0.2),  # obiettivo A: i bracci del pilot degli obiettivi (R0, R1, R2, R5) ne tengono i pesi (24/10/2026)
               "R3": (0.0, 0.2)}  # seconda ondata (firma del 26/10/2026): ancore + testa di posa, senza JEPA
ARM_POSE_WEIGHT = {"R3": 1.0}  # peso del termine di posa (docs/pilot_posa_v1.md §2 (iii)); 0 per gli altri bracci
# cancello congelato del collaudo dbg di R3 (docs/pilot_posa_v1.md §2 (iii) e decisioni.md del 26/10/2026), letto da metrics.jsonl sui soli
# passi con conteggio di posa > 0; posa pesata = pose_weight x pose, ancore pesate = anchor_weight x somma delle ancore
POSE_GATE = {"start_steps": (1, 100), "start_min_ratio": 0.1,  # scala all'avvio: media 1-100, posa pesata >= 0,1 x ancore pesate
             "mid_steps": (500, 1300), "max_ratio": 2.0,  # tetto: media 500-1.300, posa pesata <= 2 x ancore pesate
             "descent_from": (500, 599), "descent_to": (1201, 1300), "descent_factor": 0.9,  # discesa: media 1.201-1.300 <= 0,9 x media 500-599
             "min_count_fraction": 0.9}  # conteggio di posa > 0 in almeno il 90% dei passi 1-1.300
ARM_MAX_S_PER_STEP = {"R3": 0.81}  # soglia di ritmo congelata (mediana di t_step_s oltre l'avvio): sopra, piano rigenerato con un job di scorta
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


def pose_panel_errors(run_dir: Path) -> list[dict] | None:
    """Dal pannello fisso (`fixed_eval.jsonl`): per ogni pannello il passo, l'errore assoluto medio di posa standardizzato (`pose_mae_std`), in
    radianti (`pose_mae`, deviazioni per finestra) e per giunto (`pose_mae_jNN`), con il conteggio delle finestre di posa. Descrittivo."""
    path = Path(run_dir) / "fixed_eval.jsonl"
    if not path.exists():
        return None
    out = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "pose_mae" not in r and "pose_mae_std" not in r:
            continue
        out.append({"step": r.get("step"), "pose_mae_std": r.get("pose_mae_std"), "pose_mae": r.get("pose_mae"),
                    "pose_windows": r.get("pose_windows"), "pose_count": r.get("pose_count"),
                    "per_joint": {k: v for k, v in r.items() if k.startswith("pose_mae_j")}})
    return out or None


POSE_GRAD_STEPS = (500, 1300)  # emendamento 1, E4: passi su cui si legge il rapporto fra le norme dei gradienti condivisi
POSE_GRAD_MIN_RATIO = 0.01  # sotto l'1% delle ancore: «la posa non muove l'encoder», si registra prima del lancio (descrittivo, non cancello)


def pose_gradients(recs: list[dict]) -> dict | None:
    """E4 (descrittivo): media sui passi `POSE_GRAD_STEPS` dei record che portano `shared_grad_pose`/`shared_grad_anchors` (ogni
    `--pose-grad-every` passi) del rapporto fra le norme e del coseno; `flag_sotto_1pct` vero se il rapporto medio sta sotto `POSE_GRAD_MIN_RATIO`."""
    rows = [r for r in recs if isinstance(r.get("shared_grad_pose"), (int, float)) and isinstance(r.get("shared_grad_anchors"), (int, float))
            and POSE_GRAD_STEPS[0] <= r.get("step", 0) <= POSE_GRAD_STEPS[1]]
    if not rows:
        return None
    ratios = [r["shared_grad_pose"] / r["shared_grad_anchors"] for r in rows if r["shared_grad_anchors"] > 0]
    cos = [r["shared_grad_cos"] for r in rows if isinstance(r.get("shared_grad_cos"), (int, float))]
    mean_ratio = statistics.fmean(ratios) if ratios else None
    return {"passi": list(POSE_GRAD_STEPS), "n_record": len(rows), "rapporto_norme_medio": mean_ratio,
            "rapporto_norme_min_max": [min(ratios), max(ratios)] if ratios else None, "coseno_medio": statistics.fmean(cos) if cos else None,
            "flag_sotto_1pct": bool(mean_ratio is not None and mean_ratio < POSE_GRAD_MIN_RATIO), "soglia_flag": POSE_GRAD_MIN_RATIO}


def pose_gate(recs: list[dict], wp: float, wa: float, steps: int) -> list[tuple[str, bool, dict]]:
    """I quattro criteri congelati del cancello di R3 (`POSE_GATE`), sui soli record con conteggio di posa > 0 (`pose_windows`). Un record
    senza `pose_windows` o senza componenti finite non entra nelle medie. Ritorna [(nome, ok, dettaglio)]."""
    g = POSE_GATE

    def finite(r, k):
        return isinstance(r.get(k), (int, float)) and math.isfinite(r[k])

    with_pose = [r for r in recs if finite(r, "pose") and finite(r, "pose_windows") and r["pose_windows"] > 0 and all(finite(r, a) for a in ANCHORS)]

    def means(lo, hi):
        sel = [r for r in with_pose if lo <= r.get("step", 0) <= hi]
        if not sel:
            return None, None, 0
        p = statistics.fmean(wp * r["pose"] for r in sel)
        a = statistics.fmean(wa * sum(r[x] for x in ANCHORS) for r in sel)
        return p, a, len(sel)

    out = []
    p0, a0, n0 = means(*g["start_steps"])
    ratio0 = (p0 / a0) if p0 is not None and a0 else None
    out.append(("posa_scala_avvio", ratio0 is not None and ratio0 >= g["start_min_ratio"],
                {"passi": list(g["start_steps"]), "n_record": n0, "posa_pesata": p0, "ancore_pesate": a0, "rapporto": ratio0,
                 "minimo": g["start_min_ratio"]}))
    p1, a1, n1 = means(*g["mid_steps"])
    ratio1 = (p1 / a1) if p1 is not None and a1 else None
    out.append(("posa_tetto", ratio1 is not None and ratio1 <= g["max_ratio"],
                {"passi": list(g["mid_steps"]), "n_record": n1, "posa_pesata": p1, "ancore_pesate": a1, "rapporto": ratio1, "massimo": g["max_ratio"]}))
    pf, _, nf = means(*g["descent_from"])
    pt, _, nt = means(*g["descent_to"])
    out.append(("posa_discesa", pf is not None and pt is not None and pt <= g["descent_factor"] * pf,
                {"media_posa_pesata_da": pf, "passi_da": list(g["descent_from"]), "media_posa_pesata_a": pt, "passi_a": list(g["descent_to"]),
                 "fattore": g["descent_factor"], "n_record": [nf, nt]}))
    counted = [r for r in recs if finite(r, "pose_windows")]
    frac = (sum(1 for r in counted if r["pose_windows"] > 0) / steps) if steps else 0.0
    out.append(("posa_conteggio", bool(counted) and frac >= g["min_count_fraction"],
                {"frazione_passi_con_posa": frac, "minimo": g["min_count_fraction"], "record_con_conteggio": len(counted), "passi": steps}))
    return out


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
    wp = ARM_POSE_WEIGHT.get(arm, 0.0)
    weights = {k: jepa.get(k) for k in ("jepa_weight", "anchor_weight", "keep_weight", "branch_weight")}
    weights["pose_weight"] = jepa.get("pose_weight", 0.0)  # i config precedenti alla posa non hanno la chiave: vale 0
    expected_w = {"jepa_weight": wj, "anchor_weight": wa, "keep_weight": 0.0, "branch_weight": 0.0, "pose_weight": wp}
    put("pesi", weights == expected_w, {"config": weights, "attesi": expected_w})
    if wp > 0:  # R3: testa nel modello e albero della posa nel loader, con lo SHA delle statistiche registrato
        model_cfg, loader_cfg = cfg.get("model") or {}, cfg.get("loader") or {}
        put("posa_configurata", bool(model_cfg.get("pose_head")) and bool(loader_cfg.get("pose_root")) and bool(cfg.get("pose_stats_sha256")),
            {"pose_head": model_cfg.get("pose_head"), "pose_root": loader_cfg.get("pose_root"), "pose_stats_sha256": cfg.get("pose_stats_sha256")})

    recs = CS._records(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
    seq = [r.get("step") for r in recs]
    bad_windows = [r.get("step") for r in recs if r.get("windows") != WINDOWS]
    non_finite = CS._non_finite(recs) if recs else []
    put("record", seq == list(range(1, steps + 1)) and not bad_windows and not non_finite,
        {"records": len(recs), "in_sequenza": seq == list(range(1, steps + 1)), "passi_con_finestre_diverse": bad_windows[:5],
         "passi_non_finiti": non_finite[:5]})

    missing, off = [], []
    for r in recs:
        keys = ("total", "jepa") + ANCHORS + (("pose",) if wp > 0 else ())
        if any(not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k]) for k in keys):  # total infinito: tolleranza infinita
            missing.append(r.get("step"))
            continue
        expected = wj * r["jepa"] + wa * sum(r[a] for a in ANCHORS) + (wp * r["pose"] if wp > 0 else 0.0)
        if abs(r["total"] - expected) > REL_TOL * max(1.0, abs(r["total"])):
            off.append(r.get("step"))
    put("composizione_della_loss", bool(recs) and not missing and not off,
        {"formula": f"total = {wj:g} x jepa + {wa:g} x somma delle ancore {list(ANCHORS)}" + (f" + {wp:g} x pose" if wp > 0 else ""),
         "tolleranza_relativa": REL_TOL, "passi_senza_componenti_finite": missing[:5], "passi_fuori_tolleranza": off[:5]})
    if wp > 0:
        for name, ok, detail in pose_gate(recs, wp, wa, steps):
            put(name, ok, detail)
    max_rate = ARM_MAX_S_PER_STEP.get(arm)
    if max_rate is not None:
        rates = [r["t_step_s"] for r in recs if r.get("step", 0) > EDGE and isinstance(r.get("t_step_s"), (int, float))]
        med = statistics.median(rates) if rates else None
        put("ritmo", med is not None and med <= max_rate, {"s_per_passo_mediana": med, "soglia": max_rate, "passi_esclusi_avvio": EDGE})

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
        "posa_inizio_fine": edge_means("pose") if wp > 0 else None,
        "posa_errore_pannello_fisso": pose_panel_errors(run_dir) if wp > 0 else None,
        "gradienti_condivisi": pose_gradients(recs) if wp > 0 else None,
        "posa_finestre_medie": (statistics.fmean(r["pose_windows"] for r in recs if isinstance(r.get("pose_windows"), (int, float)))
                                if wp > 0 and any(isinstance(r.get("pose_windows"), (int, float)) for r in recs) else None),
        "non_collasso": [{k: r.get(k) for k in ("step", "student_collapse_ratio", "teacher_collapse_ratio", "student_erank", "teacher_erank",
                                                  "student_var_samples", "student_var_queries", "teacher_var_samples")} for r in diags],
        "lettura": ("A: JEPA e ancore allenate insieme; il collaudo misura ritmo, memoria e contratto, non confronta gli obiettivi "
                    "(docs/pilot_obiettivi_v1.md)" if arm == "A" else
                    "R3: ancore e testa di posa, JEPA registrata ma non allenata; il cancello della posa controlla scala, tetto, discesa e "
                    "conteggio, non quanto la testa impari (docs/pilot_posa_v1.md §2 (iii))" if arm == "R3" else
                    "JEPA-only: le teste delle ancore non sono allenate, le loss delle ancore registrate non dicono nulla; "
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
