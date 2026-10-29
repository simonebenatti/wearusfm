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
from collections import Counter, defaultdict
import hashlib
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
               "R3": (0.0, 0.2),  # seconda ondata (firma del 26/10/2026): ancore + testa di posa, senza JEPA
               "R3-lungo": (0.0, 0.2)}  # collaudo lungo (B) di R3: la ricetta ESATTA di R3 fermata a 5.000 passi (28/10/2026)
ARM_POSE_WEIGHT = {"R3": 1.0, "R3-lungo": 1.0}  # peso del termine di posa (docs/pilot_posa_v1.md §2 (iii)); 0 per gli altri bracci
# cancello congelato del collaudo dbg di R3 (docs/pilot_posa_v1.md §2 (iii) e decisioni.md del 26/10/2026), letto da metrics.jsonl sui soli
# passi con conteggio di posa > 0; posa pesata = pose_weight x pose, ancore pesate = anchor_weight x somma delle ancore
POSE_GATE = {"start_steps": (1, 100), "start_min_ratio": 0.1,  # scala all'avvio: media 1-100, posa pesata >= 0,1 x ancore pesate
             "mid_steps": (500, 1300), "max_ratio": 2.0,  # tetto: media 500-1.300, posa pesata <= 2 x ancore pesate
             "descent_from": (500, 599), "descent_to": (1201, 1300), "descent_factor": 0.9,  # discesa: media 1.201-1.300 <= 0,9 x media 500-599
             "min_count_fraction": 0.9}  # conteggio di posa > 0 in almeno il 90% dei passi 1-1.300
ARM_MAX_S_PER_STEP = {"R3": 0.81, "R3-lungo": 0.81}  # soglia di ritmo congelata (mediana di t_step_s oltre l'avvio): sopra, piano rigenerato
# B: scala, tetto e conteggio restano controlli tecnici. La pendenza del training e la varianza delle predizioni sono DESCRITTIVE.
POSE_GATE_LONG = {**POSE_GATE, "mid_steps": (500, 5000), "descent_from": (2001, 2500), "descent_to": (4501, 5000)}
# L'apprendimento si decide solo sul pannello integro, appaiato per utente: MSE(5000) <= .9 MSE(0), IC95 superiore del rapporto < 1.
PANEL_GATE = {"step_from": 0, "step_to": 5000, "every": 500, "factor": 0.9, "ci_upper_max": 1.0, "n_boot": 1000, "seed": 20261027,
              "users": 32, "recordings_per_user": 2, "windows_per_recording": 4}
B_PROTOCOL = "r3-long-collaudo-v1"
B_POSE_STATS_SHA = "757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e"
ARM_CHECKPOINTS = {"R3-lungo": ("checkpoint.pt",)}  # fermato a 5.000 di 49.046: ne' predecay ne' postdecay, resta il checkpoint di ripresa
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


def pose_gradients(recs: list[dict], interval: tuple[int, int] = POSE_GRAD_STEPS) -> dict | None:
    """E4 (descrittivo): media sui passi `interval` dei record che portano `shared_grad_pose`/`shared_grad_anchors` (ogni
    `--pose-grad-every` passi) del rapporto fra le norme e del coseno; `flag_sotto_1pct` vero se il rapporto medio sta sotto `POSE_GRAD_MIN_RATIO`."""
    rows = [r for r in recs if isinstance(r.get("shared_grad_pose"), (int, float)) and isinstance(r.get("shared_grad_anchors"), (int, float))
            and interval[0] <= r.get("step", 0) <= interval[1]]
    if not rows:
        return None
    ratios = [r["shared_grad_pose"] / r["shared_grad_anchors"] for r in rows if r["shared_grad_anchors"] > 0]
    cos = [r["shared_grad_cos"] for r in rows if isinstance(r.get("shared_grad_cos"), (int, float))]
    mean_ratio = statistics.fmean(ratios) if ratios else None
    return {"passi": list(interval), "n_record": len(rows), "rapporto_norme_medio": mean_ratio,
            "rapporto_norme_min_max": [min(ratios), max(ratios)] if ratios else None, "coseno_medio": statistics.fmean(cos) if cos else None,
            "flag_sotto_1pct": bool(mean_ratio is not None and mean_ratio < POSE_GRAD_MIN_RATIO), "soglia_flag": POSE_GRAD_MIN_RATIO}


def pose_gate(recs: list[dict], wp: float, wa: float, steps: int, gate: dict = POSE_GATE) -> list[tuple[str, bool, dict]]:
    """I quattro criteri congelati del cancello di R3 (`POSE_GATE`; `POSE_GATE_LONG` per il collaudo lungo), sui soli record con conteggio di
    posa > 0 (`pose_windows`). Un record senza `pose_windows` o senza componenti finite non entra nelle medie. Ritorna [(nome, ok, dettaglio)]."""
    g = gate

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


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _has_nonfinite(value) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_nonfinite(v) for v in value)
    return False


def _strict_json(path: Path, *, lines: bool = False):
    """B fails closed on malformed JSON, duplicate keys and non-object records; legacy readers are unchanged."""
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"chiave duplicata: {key}")
            out[key] = value
        return out

    errors, records = [], []
    try:
        raw = path.read_text()
    except OSError as exc:
        return ([] if lines else {}), [f"{path.name}: {exc.strerror}"]
    for index, text in enumerate(raw.splitlines() if lines else [raw], 1):
        if lines and not text.strip():
            continue
        try:
            value = json.loads(text, object_pairs_hook=pairs)
            if not isinstance(value, dict):
                raise ValueError("atteso oggetto JSON")
            records.append(value)
        except (ValueError, TypeError) as exc:
            errors.append(f"{path.name}, record {index}: {exc}")
    return (records if lines else (records[0] if records else {})), errors


def _panel_window_key(row: dict):
    window = row.get("window")
    if not isinstance(window, list) or len(window) != 3:
        raise ValueError("coordinate della finestra mancanti")
    trial, start, stop = window
    if (trial is not None and type(trial) is not int) or type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError("coordinate della finestra non valide")
    if any(not isinstance(row.get(k), str) or not row[k] for k in ("user", "session")):
        raise ValueError("utente o registrazione mancanti")
    return row["user"], row["session"], trial, start, stop


def _panel_integrity(panel: list[dict], manifest: dict, g: dict) -> tuple[bool, dict]:
    """The manifest fixes identities AND valid support; aggregates never substitute for the individual windows."""
    errors = []
    expected_n = g["users"] * g["recordings_per_user"] * g["windows_per_recording"]
    wanted_steps = list(range(g["step_from"], g["step_to"] + 1, g["every"]))
    if [r.get("step") for r in panel] != wanted_steps or any(type(r.get("step")) is not int for r in panel):
        errors.append("valutazioni mancanti, duplicate, fuori ordine o inattese")
    if _has_nonfinite(panel):
        errors.append("numero non finito nel pannello")
    keys, support = [], []
    try:
        if manifest.get("schema_version") != 1 or manifest.get("window_s") != 4.0 or manifest.get("sampling") != "nonoverlapping_pose_valid":
            raise ValueError("schema o campionamento del manifest diverso dal protocollo")
        if manifest.get("mask_strategy") != "training_mask_fixed":
            raise ValueError("maschere del pannello diverse dal protocollo")
        if manifest.get("seed") != [0, 7272]:
            raise ValueError("seme del pannello diverso dal protocollo")
        windows, users, support = manifest["windows"], manifest["window_users"], manifest["n_valid"]
        if not all(isinstance(x, list) and len(x) == expected_n for x in (windows, users, support)):
            raise ValueError("dimensione windows/window_users/n_valid diversa dal protocollo")
        for w, u in zip(windows, users):
            if not isinstance(w, list) or len(w) != 4:
                raise ValueError("coordinate del manifest non valide")
            keys.append(_panel_window_key({"user": u, "session": w[0], "window": w[1:]}))
        if len(set(keys)) != expected_n or len({k[1:] for k in keys}) != expected_n:
            raise ValueError("finestre duplicate nel manifest")
        if any(type(n) is not int or n <= 0 or n > 160 * 20 for n in support):
            raise ValueError("supporto valido del manifest assente o impossibile")
        by_user, by_recording, owners = defaultdict(Counter), defaultdict(list), defaultdict(set)
        for u, rec, trial, start, stop in keys:
            by_user[u][rec] += 1
            by_recording[rec, trial].append((start, stop))
            owners[rec].add(u)
        if any(len(us) != 1 for us in owners.values()):
            raise ValueError("registrazione assegnata a utenti diversi")
        if len(by_user) != g["users"] or manifest.get("users") != sorted(by_user):
            raise ValueError("utenti del manifest diversi dal protocollo")
        if any(len(rs) != g["recordings_per_user"] or set(rs.values()) != {g["windows_per_recording"]} for rs in by_user.values()):
            raise ValueError("quota registrazioni/finestre per utente diversa dal protocollo")
        if any(a[1] > b[0] for spans in by_recording.values() for a, b in zip(sorted(spans), sorted(spans)[1:])):
            raise ValueError("finestre sovrapposte nella stessa registrazione")
        digests = manifest.get("batch_sha256")
        if not isinstance(digests, list) or len(digests) != g["users"] * g["recordings_per_user"] or any(not _is_sha(x, 64) for x in digests):
            raise ValueError("digest dei batch mancanti o non validi")
        digest = _canonical_sha256(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"manifest: {exc}")
        return False, {"errori": errors, "finestre_attese": expected_n}

    baseline = None
    for record in panel:
        at = record.get("step")
        per = record.get("per_window")
        if record.get("panel_sha256") != digest:
            errors.append(f"passo {at}: digest del pannello diverso dal manifest")
        if not isinstance(per, list) or len(per) != expected_n:
            errors.append(f"passo {at}: numero di finestre diverso dal manifest")
            continue
        try:
            got = [_panel_window_key(w) for w in per]
            if got != keys:
                raise ValueError("identita'/ordine delle finestre diverso dal manifest")
            if [w.get("n_valid") for w in per] != support or any(type(w.get("n_valid")) is not int for w in per):
                raise ValueError("supporto valido cambiato o mancante")
            for w in per:
                if any(not _finite_number(w.get(k)) or w[k] < 0 for k in ("mse", "mae", "mse_const", "pred_var")):
                    raise ValueError("metrica per finestra mancante, negativa o non finita")
                if at == g["step_from"]:
                    counts, means = w.get("n_valid_per_joint"), w.get("pred_mean_per_joint")
                    if not all(isinstance(v, list) and len(v) == 20 for v in (counts, means)):
                        raise ValueError("baseline: supporti/medie per giunto mancanti o di lunghezza diversa da 20")
                    if any(type(c) is not int or c < 0 for c in counts) or sum(counts) != w["n_valid"]:
                        raise ValueError("baseline: supporti per giunto non interi, negativi o incoerenti con n_valid")
                    if any((not _finite_number(m) or abs(m) > 1e-6) if c > 0 else
                           (m is not None and not _finite_number(m)) for c, m in zip(counts, means)):
                        raise ValueError("baseline: media predetta per giunto non nulla, non finita o mancante sul supporto valido")
            for k in ("mse", "mae", "mse_const", "pred_var"):
                if not _finite_number(record.get(k)) or not math.isclose(record[k], statistics.fmean(w[k] for w in per), rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(f"aggregato {k} incoerente con le finestre")
            for k, v in (("n_windows", expected_n), ("n_windows_valid", expected_n), ("n_users", g["users"])):
                if type(record.get(k)) is not int or record[k] != v:
                    raise ValueError(f"conteggio {k} incoerente")
            const = [w["mse_const"] for w in per]
            if baseline is None:
                baseline = const
            if any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(const, baseline)):
                raise ValueError("MSE del costante cambiato: target/supporto non fisso")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"passo {at}: {exc}")
    return not errors, {"errori": errors, "finestre_attese": expected_n, "utenti_attesi": g["users"], "panel_sha256": digest}


def _is_sha(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _panel_user_means(rec: dict) -> dict[str, float]:
    acc = defaultdict(list)
    for w in rec["per_window"]:
        acc[w["user"]].append(w["mse"])
    return {u: statistics.fmean(v) for u, v in acc.items()}


def panel_gate(panel: list[dict], manifest: dict | None = None, g: dict = PANEL_GATE) -> list[tuple[str, bool, dict]]:
    """B: paired user bootstrap only after checking the complete immutable panel and zero-head baseline."""
    import numpy as np

    have = [r.get("step") for r in panel]
    want = list(range(g["step_from"], g["step_to"] + 1, g["every"]))
    integrity, detail = _panel_integrity(panel, manifest or {}, g)
    out = [("pannello_valutazioni", have == want, {"valutazioni": have, "attese": want}),
           ("pannello_integrita", integrity, detail)]
    gates = ("pannello_baseline", "pannello_bootstrap_valido", "pannello_discesa", "pannello_ic95_utenti")
    if not integrity:
        return out + [(name, False, {"motivo": "pannello non valutabile: integrita' non verificata"}) for name in gates]
    by_step = {r["step"]: r for r in panel}
    a, b = by_step[g["step_from"]], by_step[g["step_to"]]
    ua, ub = _panel_user_means(a), _panel_user_means(b)
    baseline_ok = all(math.isclose(w["mse"], w["mse_const"], rel_tol=1e-6, abs_tol=1e-8) and w["pred_var"] <= 1e-12
                      for w in a["per_window"]) and sum(ua.values()) > 0
    out.append(("pannello_baseline", baseline_ok, {"mse_zero": a["mse"], "mse_costante": a["mse_const"],
                                                   "motivo": "testa a zero e MSE iniziale globale positiva"}))
    if not baseline_ok:
        return out + [(name, False, {"motivo": "baseline iniziale incoerente o nulla"}) for name in gates[1:]]
    # Equal quotas make window and user means identical; compute from the individual windows, never trust cached aggregates.
    users = sorted(ua)
    x, y = np.array([ua[u] for u in users]), np.array([ub[u] for u in users])
    ratio = float(y.mean() / x.mean())
    out.append(("pannello_discesa", ratio <= g["factor"], {"passi": [g["step_from"], g["step_to"]], "mse_iniziale": float(x.mean()),
                "mse_finale": float(y.mean()), "rapporto": ratio, "fattore": g["factor"], "mae_iniziale_finale": [a["mae"], b["mae"]]}))
    idx = np.random.default_rng(g["seed"]).integers(0, len(users), size=(g["n_boot"], len(users)))
    denominators = x[idx].mean(axis=1)
    valid_den = np.isfinite(denominators) & (denominators > 0)
    out.append(("pannello_bootstrap_valido", bool(valid_den.all()),
                {"n_boot": g["n_boot"], "repliche_denominatore_non_positivo_o_non_finito": int((~valid_den).sum()), "seme": g["seed"]}))
    if not valid_den.all():
        return out + [("pannello_ic95_utenti", False, {"motivo": "IC non valutabile: denominatore bootstrap nullo o non finito"})]
    ratios = y[idx].mean(axis=1) / denominators
    lo, hi = float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))
    out.append(("pannello_ic95_utenti", math.isfinite(hi) and hi < g["ci_upper_max"],
                {"n_utenti": len(users), "rapporto_medie_utente": ratio, "ic95": [lo, hi], "massimo_limite_superiore": g["ci_upper_max"],
                 "n_boot": g["n_boot"], "seme": g["seed"], "utenti_in_discesa": int((y < x).sum()),
                 "per_utente": {u: [ua[u], ub[u]] for u in users}}))
    return out


def pose_panel_summary(panel: list[dict]) -> list[dict]:
    """Per ogni valutazione del pannello della posa: passo, MSE, MAE, MSE del costante, varianza delle predizioni, finestre e utenti (descrittivo)."""
    return [{k: r.get(k) for k in ("step", "mse", "mae", "mse_const", "pred_var", "n_windows_valid", "n_users")} for r in panel]


def long_configuration(cfg: dict, summary: dict, prov: dict, manifest: dict, recs: list[dict], steps: int) -> tuple[bool, dict]:
    """Check the effective recipe, including the real schedule truncated at 5000 rather than decayed there."""
    expected = {"max_steps": 5000, "warmup_steps": 1962, "lr": 1e-4, "weight_decay": 0.05, "grad_clip": 1.0,
                "batch_size": 32, "accum_steps": 1, "seed": 0, "freeze_encoder": False, "pose_grad_every": 500,
                "pose_panel_every": 500, "pose_panel_users": 32, "pose_panel_recordings": 2, "pose_panel_windows": 4,
                "nested": "1", "eval_every": 500, "amp_bf16": True, "prefetch": True,
                "schedule.kind": "wsd", "schedule.total_steps": 49046, "schedule.decay_fraction": 0.2,
                "schedule.keep_checkpoints": [19618], "model.pose_head": True, "model.dim": 384, "model.patch_ms": 25.0,
                "model.n_heads": 6, "model.k_latents": 64, "model.local_layers": 2, "model.backbone_layers": 8,
                "model.decoder_layers": 2, "model.muscle_dropout": 0.4, "model.grad_checkpoint": True,
                "jepa.loss_per_window": True, "loader.patch_ms": 25.0, "loader.pose_unprompted": True}
    actual = {}
    for path in expected:
        value = cfg
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        actual[path] = value
    mismatch = {k: {"atteso": v, "letto": actual[k]} for k, v in expected.items()
                if actual[k] != v or (isinstance(v, bool) != isinstance(actual[k], bool))}
    if steps != 5000:
        mismatch["lettore.steps"] = {"atteso": 5000, "letto": steps}
    sha = summary.get("pose_stats_sha256")
    if sha != B_POSE_STATS_SHA or prov.get("pose_stats_sha256") != sha or manifest.get("pose_stats_sha256") != sha:
        mismatch["pose_stats_sha256"] = {"atteso": B_POSE_STATS_SHA, "summary": sha, "provenance": prov.get("pose_stats_sha256"),
                                        "manifest": manifest.get("pose_stats_sha256")}
    root = cfg.get("loader", {}).get("pose_root")
    if not isinstance(root, str) or not root or root != prov.get("pose_root") or prov.get("pose_unprompted") is not True:
        mismatch["provenienza_posa"] = {"pose_root_config": root, "provenance": prov}
    bad_lr = [r.get("step") for r in recs if not _finite_number(r.get("lr")) or
              not math.isclose(r["lr"], 1e-4 * min(1.0, r["step"] / 1962), rel_tol=1e-9, abs_tol=1e-12)]
    if bad_lr:
        mismatch["lr_metrics"] = {"passi_errati": bad_lr[:10]}
    return not mismatch, {"differenze": mismatch, "warmup_da_steps_at_d": 98094,
                          "nota": "steps_at_d congelato nel comando; config verifica il warmup effettivo 1962"}


def check(run_dir: Path, arm: str, steps: int, expected_sha: str, listing: str) -> dict:
    if arm not in ARM_WEIGHTS:
        raise ValueError(f"braccio {arm!r}: uno di {sorted(ARM_WEIGHTS)}")
    CS = _select()
    run_dir = Path(run_dir)
    wj, wa = ARM_WEIGHTS[arm]
    crit = {}

    def put(name, ok, detail):
        crit[name] = {"ok": bool(ok), "detail": detail}

    is_long = arm == "R3-lungo"
    input_errors = []
    manifest, panel = {}, []
    if is_long:
        summary, errors = _strict_json(run_dir / "summary.json")
        input_errors.extend(errors)
        cfg, errors = _strict_json(run_dir / "config.json")
        input_errors.extend(errors)
        prov, errors = _strict_json(run_dir / "pose_provenance.json")
        input_errors.extend(errors)
        manifest, errors = _strict_json(run_dir / "pose_panel_set.json")
        input_errors.extend(errors)
        panel, errors = _strict_json(run_dir / "pose_panel.jsonl", lines=True)
        input_errors.extend(errors)
        recs, errors = _strict_json(run_dir / "metrics.jsonl", lines=True)
        input_errors.extend(errors)
        if any(type(r.get("step")) is not int for r in recs):
            input_errors.append("metrics.jsonl: step non intero")
            recs = [r for r in recs if type(r.get("step")) is int]
        for key in ("jepa", "model", "loader", "schedule"):
            if not isinstance(cfg.get(key), dict):
                input_errors.append(f"config.json: {key} non e' un oggetto")
                cfg[key] = {}
        if not isinstance(summary.get("code"), dict):
            input_errors.append("summary.json: code non e' un oggetto")
            summary["code"] = {}
        put("file_integri", not input_errors, {"errori": input_errors})
    else:
        summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    stop = (run_dir / "STOP").read_text().strip()[:300] if (run_dir / "STOP").exists() else None
    put("completo", stop is None and summary.get("stopped") == "max_steps" and summary.get("steps") == steps,
        {"STOP": stop, "stopped": summary.get("stopped"), "steps": summary.get("steps"), "attesi": steps})
    code = (summary.get("code") or {}).get("sha")
    got = {"sha": code, "world_size": summary.get("world_size"), "batch_per_process": summary.get("batch_per_process"),
           "accum_steps": summary.get("accum_steps")}
    put("codice_e_ddp", got == {"sha": expected_sha, "world_size": WORLD, "batch_per_process": BATCH_PER_PROCESS, "accum_steps": 1}, got)

    if not is_long:
        cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    jepa = cfg.get("jepa") or {}
    wp = ARM_POSE_WEIGHT.get(arm, 0.0)
    weights = {k: jepa.get(k) for k in ("jepa_weight", "anchor_weight", "keep_weight", "branch_weight")}
    weights["pose_weight"] = jepa.get("pose_weight", 0.0)  # i config precedenti alla posa non hanno la chiave: vale 0
    expected_w = {"jepa_weight": wj, "anchor_weight": wa, "keep_weight": 0.0, "branch_weight": 0.0, "pose_weight": wp}
    put("pesi", weights == expected_w, {"config": weights, "attesi": expected_w})
    if wp > 0:  # R3: testa nel modello e albero della posa nel loader, con lo SHA delle statistiche registrato
        model_cfg, loader_cfg = cfg.get("model") or {}, cfg.get("loader") or {}
        # lo SHA di pose_stats.json sta in summary.json e in pose_provenance.json (il trainer lo tiene FUORI dal config, che e' l'identita' del
        # run: `training/run.py`); fino al 27/10/2026 il lettore lo cercava solo nel config e dava NO a un run configurato bene (collaudo
        # 62429050): corretto, dichiarato in decisioni.md
        if not is_long:
            prov = json.loads((run_dir / "pose_provenance.json").read_text()) if (run_dir / "pose_provenance.json").exists() else {}
        pose_sha = summary.get("pose_stats_sha256") or prov.get("pose_stats_sha256") or cfg.get("pose_stats_sha256")
        put("posa_configurata", bool(model_cfg.get("pose_head")) and bool(loader_cfg.get("pose_root")) and bool(pose_sha),
            {"pose_head": model_cfg.get("pose_head"), "pose_root": loader_cfg.get("pose_root"), "pose_stats_sha256": pose_sha,
             "sha_in": [k for k, v in (("summary.json", summary.get("pose_stats_sha256")), ("pose_provenance.json", prov.get("pose_stats_sha256")),
                                       ("config.json", cfg.get("pose_stats_sha256"))) if v]})

    if not is_long:
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
    training_descent = None
    if wp > 0:
        for name, ok, detail in pose_gate(recs, wp, wa, steps, POSE_GATE_LONG if is_long else POSE_GATE):
            if is_long and name == "posa_discesa":
                training_descent = detail  # descriptive: no second 10% requirement after warmup
            else:
                put(name, ok, detail)
    if is_long:
        ok, detail = long_configuration(cfg, summary, prov, manifest, recs, steps)
        put("configurazione_b", ok, detail)
        for name, ok, detail in panel_gate(panel, manifest):
            put(name, ok, detail)
    max_rate = ARM_MAX_S_PER_STEP.get(arm)
    if max_rate is not None:
        rates = [r["t_step_s"] for r in recs if r.get("step", 0) > EDGE and isinstance(r.get("t_step_s"), (int, float))]
        med = statistics.median(rates) if rates else None
        put("ritmo", med is not None and med <= max_rate, {"s_per_passo_mediana": med, "soglia": max_rate, "passi_esclusi_avvio": EDGE})

    totals = [r["total"] for r in recs if isinstance(r.get("total"), (int, float))]
    first = statistics.fmean(totals[:EDGE]) if len(totals) >= 2 * EDGE else None
    last = statistics.fmean(totals[-EDGE:]) if len(totals) >= 2 * EDGE else None
    total_trend = {"termine_allenato": "total", "media_primi": first, "media_ultimi": last, "record_mediati": EDGE}
    if not is_long:
        put("apprendimento", first is not None and last < first, total_trend)

    diags = [r for r in recs if "student_collapse_ratio" in r]
    alarms = [r.get("alarm") for r in recs if r.get("alarm")]
    want = expected_evaluations(steps)
    put("non_collasso", not alarms and [r["step"] for r in diags] == want,
        {"allarmi": alarms, "valutazioni": [r["step"] for r in diags], "attese": want})

    sizes = listing_sizes(listing)
    wanted = ARM_CHECKPOINTS.get(arm, CHECKPOINTS)
    put("checkpoint", all(sizes.get(n, 0) > 0 for n in wanted), {n: sizes.get(n) for n in wanted})

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
        "pannello_posa": pose_panel_summary(panel) if is_long else None,
        "gradienti_condivisi": pose_gradients(recs, (500, 5000) if is_long else POSE_GRAD_STEPS) if wp > 0 else None,
        "posa_finestre_medie": (statistics.fmean(r["pose_windows"] for r in recs if isinstance(r.get("pose_windows"), (int, float)))
                                if wp > 0 and any(isinstance(r.get("pose_windows"), (int, float)) for r in recs) else None),
        "non_collasso": [{k: r.get(k) for k in ("step", "student_collapse_ratio", "teacher_collapse_ratio", "student_erank", "teacher_erank",
                                                  "student_var_samples", "student_var_queries", "teacher_var_samples")} for r in diags],
        "lettura": ("A: JEPA e ancore allenate insieme; il collaudo misura ritmo, memoria e contratto, non confronta gli obiettivi "
                    "(docs/pilot_obiettivi_v1.md)" if arm == "A" else
                    "R3: ancore e testa di posa, JEPA registrata ma non allenata; il cancello della posa controlla scala, tetto, discesa e "
                    "conteggio, non quanto la testa impari (docs/pilot_posa_v1.md §2 (iii))" if arm == "R3" else
                    "R3-lungo: pannello fisso integro, MSE(5000) <= 0,9 MSE(0) e IC95 appaiato per utente con limite superiore < 1. "
                    "La baseline verifica la testa a zero; training a blocchi e varianza sono descrittivi. Non dimostra generalizzazione "
                    "o dipendenza dinamica dall'EMG." if arm == "R3-lungo" else
                    "JEPA-only: le teste delle ancore non sono allenate, le loss delle ancore registrate non dicono nulla; "
                    "ancore-only: la JEPA registrata misura un predittore non allenato (docs/bracci_obiettivo_finestra1.md)"),
    }
    status = {}
    if is_long:
        descriptive.update(posa_training_blocchi=training_descent, total_training_inizio_fine=total_trend)
        technical = ("file_integri", "completo", "codice_e_ddp", "configurazione_b", "record", "pannello_valutazioni", "pannello_integrita",
                     "pannello_baseline", "pannello_bootstrap_valido")
        evaluable = all(crit[k]["ok"] for k in technical)
        status = {"evaluable": evaluable, "status": "non_valutabile" if not evaluable else
                  ("superato" if all(c["ok"] for c in crit.values()) else "non_superato")}
    return _clean({**status, "protocol": B_PROTOCOL if is_long else PROTOCOL, "arm": arm, "run_dir": str(run_dir), "steps": steps, "passed": all(c["ok"] for c in crit.values()),
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
