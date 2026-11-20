"""Confronto dei bracci dello studio delle loss di TEMPONet su emg2pose con il braccio A (L1, 11/11), con la regola congelata in
`docs/decisioni.md` il 18/11/2026 (`docs/temponet_emg2pose_loss_plan_20261118.md`, sezione «Lettura»).

Per condizione di test (`user`, `stage`, `user_stage`): Δ = MAE con EMA (media fra utenti) del braccio - quella di A. |Δ| ≤ 0,30 gradi: non
distinguibile da L1 con un seme; Δ < -0,30: migliore, da confermare con altri semi; Δ > +0,30: peggiore, da confermare. Descrittivi: utenti
migliori, peggiori e pari (confronto diretto dei valori per utente, conteggi interi), validazione con EMA e cancello.

Uso: python3 scripts/temponet_emg2pose_loss_compare.py --reference <eval.json di A> --arm smoothl1=<eval.json> --arm mse=<eval.json> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THRESHOLD_DEG = 0.30
CONDITIONS = ("user", "stage", "user_stage")


def reading(delta: float) -> str:
    if abs(delta) <= THRESHOLD_DEG:
        return "non distinguibile da L1 con un seme"
    return "migliore di L1, da confermare con altri semi" if delta < 0 else "peggiore di L1, da confermare con altri semi"


def users_compare(arm: dict, ref: dict) -> dict:
    common = sorted(set(arm) & set(ref))
    better = sum(1 for u in common if arm[u] < ref[u])
    worse = sum(1 for u in common if arm[u] > ref[u])
    return {"users": len(common), "better": better, "worse": worse, "equal": len(common) - better - worse,
            "only_in_one": len(set(arm) ^ set(ref))}


def compare(ref: dict, arm: dict) -> dict:
    out = {"train_loss": arm.get("train_loss"), "val_ema_deg": arm["val"]["ema"]["mean_over_users_deg"],
           "val_ema_deg_reference": ref["val"]["ema"]["mean_over_users_deg"], "gate_ratio": arm["gate"]["ratio"],
           "gate_passed": arm["gate"]["passed"]}
    if isinstance(arm["test"], str):
        out["test"] = arm["test"]
        return out
    out["conditions"] = {}
    for cond in CONDITIONS:
        a = arm["test"]["conditions"][cond]["ema"]
        r = ref["test"]["conditions"][cond]["ema"]
        if a["mean_over_users_deg"] is None or r["mean_over_users_deg"] is None:
            out["conditions"][cond] = None
            continue
        delta = a["mean_over_users_deg"] - r["mean_over_users_deg"]
        out["conditions"][cond] = {"arm_deg": a["mean_over_users_deg"], "arm_sd_deg": a["sd_over_users_deg"],
                                   "reference_deg": r["mean_over_users_deg"], "delta_deg": delta, "reading": reading(delta),
                                   "per_user": users_compare(a["per_user_deg"], r["per_user_deg"])}
    return out


def to_markdown(results: dict) -> str:
    lines = ["| Braccio | Condizione | MAE con EMA | A (L1) | Δ | Lettura | Utenti migliori / peggiori / pari |", "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        if "conditions" not in res:
            lines.append(f"| {name} | - | - | - | - | {res['test']} | - |")
            continue
        for cond, c in res["conditions"].items():
            if c is None:
                lines.append(f"| {name} | {cond} | - | - | - | nessun utente | - |")
                continue
            p = c["per_user"]
            lines.append(f"| {name} | {cond} | {c['arm_deg']:.2f} ± {c['arm_sd_deg'] or 0:.2f} | {c['reference_deg']:.2f} | {c['delta_deg']:+.2f} | "
                         f"{c['reading']} | {p['better']} / {p['worse']} / {p['equal']} su {p['users']} |")
    lines += ["", "| Braccio | Validazione con EMA | A | Cancello |", "|---|---|---|---|"]
    for name, res in results.items():
        lines.append(f"| {name} | {res['val_ema_deg']:.3f} | {res['val_ema_deg_reference']:.3f} | {res['gate_ratio']:.3f} "
                     f"({'superato' if res['gate_passed'] else 'NON superato'}) |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="eval.json del braccio A (L1)")
    ap.add_argument("--arm", action="append", required=True, help="nome=percorso di eval.json")
    ap.add_argument("--out", default=None, help="cartella per confronto.json e confronto.md")
    args = ap.parse_args(argv)
    ref = json.loads(Path(args.reference).read_text())
    if ref.get("train_loss", "l1") != "l1":
        ap.error(f"il riferimento non e' L1: {ref.get('train_loss')}")
    results = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--arm {spec}: serve nome=percorso")
        arm = json.loads(Path(path).read_text())
        if name in ("smoothl1", "mse", "mse_vel") and arm.get("train_loss") != name:  # la loss vera e' train_loss, non args.loss
            ap.error(f"--arm {name}: l'eval.json e' di un training con loss {arm.get('train_loss')}")
        results[name] = compare(ref, arm)
    md = to_markdown(results)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "confronto.json").write_text(json.dumps({"threshold_deg": THRESHOLD_DEG, "arms": results}, indent=1))
        (out / "confronto.md").write_text(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
