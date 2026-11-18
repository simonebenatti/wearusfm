"""Conferma con altri semi del confronto L1 / MSE di TEMPONet su emg2pose: regola congelata in `docs/decisioni.md` il 18/11/2026
(proposta `docs/temponet_emg2pose_loss_seeds_proposal_20261118.md`, emendata dopo la review del critico prima del primo job).

Per condizione (`user`, `stage`, `user_stage`), Δ_s = MSE_s - L1_s dai `mean_over_users_deg` non arrotondati. La lettura usa SOLO i semi
nuovi 1 e 2:
- «MSE peggiore di L1, confermato»: Δ_1 > 0, Δ_2 > 0 e media dei due > +0,30 gradi;
- «MSE migliore di L1, confermato»: Δ_1 < 0, Δ_2 < 0 e media < -0,30;
- «non leggibile» se a uno dei quattro bracci nuovi manca il test;
- altrimenti «non confermato».
Il seme 0 e la media dei tre semi sono descrittivi.

Controlli prima di leggere (il seme del training sta nel `run.json`; `args.seed` dell'`eval.json` e' quello della chiamata di valutazione):
`run.json` con stato ok, seme e loss attesi (senza `args.loss` vale l1); `train_loss` dell'`eval.json` uguale alla loss (senza campo vale l1);
il checkpoint valutato sta nella cartella di uscita del training.

Uso: python3 scripts/temponet_emg2pose_seeds_compare.py --arm l1:0:<cartella training>:<cartella valutazione> ... [--out <cartella>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import PurePosixPath, Path

import numpy as np

THRESHOLD_DEG = 0.30
CONDITIONS = ("user", "stage", "user_stage")
NEW_SEEDS = (1, 2)


def load_arm(loss: str, seed: int, train_dir: str, eval_dir: str) -> dict:
    run = json.loads((Path(train_dir) / "run.json").read_text())
    ev = json.loads((Path(eval_dir) / "eval.json").read_text())
    problems = []
    if run.get("status") != "ok":
        problems.append(f"training non concluso: {run.get('status')}")
    if int(run["args"]["seed"]) != seed:
        problems.append(f"seme del training {run['args']['seed']}, atteso {seed}")
    if run["args"].get("loss", "l1") != loss:
        problems.append(f"loss del training {run['args'].get('loss', 'l1')}, attesa {loss}")
    if ev.get("train_loss", "l1") != loss:
        problems.append(f"train_loss dell'eval.json {ev.get('train_loss', 'l1')}, attesa {loss}")
    if PurePosixPath(ev["checkpoint"]).parent != PurePosixPath(run["args"]["out"]):
        problems.append(f"checkpoint {ev['checkpoint']} fuori dalla cartella del training {run['args']['out']}")
    test = None if isinstance(ev["test"], str) else {c: ev["test"]["conditions"][c]["ema"] for c in CONDITIONS}
    return {"loss": loss, "seed": seed, "problems": problems, "test": test, "gate": ev["gate"], "train_job": run.get("slurm_job_id"),
            "eval_job": ev.get("slurm_job_id"), "code_sha": run.get("code_sha")}


def reading(d1: float | None, d2: float | None) -> str:
    if d1 is None or d2 is None:
        return "non leggibile"
    mean = (d1 + d2) / 2
    if d1 > 0 and d2 > 0 and mean > THRESHOLD_DEG:
        return "MSE peggiore di L1, confermato"
    if d1 < 0 and d2 < 0 and mean < -THRESHOLD_DEG:
        return "MSE migliore di L1, confermato"
    return "non confermato"


def compare(arms: dict) -> dict:
    out = {"threshold_deg": THRESHOLD_DEG, "new_seeds": list(NEW_SEEDS), "conditions": {}}
    seeds = sorted({s for (_, s) in arms})
    for cond in CONDITIONS:
        val = {(l, s): (a["test"][cond]["mean_over_users_deg"] if a["test"] and a["test"][cond]["mean_over_users_deg"] is not None else None)
               for (l, s), a in arms.items()}
        delta = {s: (val[("mse", s)] - val[("l1", s)]) if val.get(("mse", s)) is not None and val.get(("l1", s)) is not None else None
                 for s in seeds}
        per_user = {}
        for s in seeds:
            if delta[s] is None:
                continue
            a, r = arms[("mse", s)]["test"][cond]["per_user_deg"], arms[("l1", s)]["test"][cond]["per_user_deg"]
            common = sorted(set(a) & set(r))
            per_user[s] = {"worse": sum(a[u] > r[u] for u in common), "better": sum(a[u] < r[u] for u in common), "users": len(common)}
        known = [delta[s] for s in seeds if delta[s] is not None]
        stats = {}
        for loss in ("l1", "mse"):
            v = [val[(loss, s)] for s in seeds if val.get((loss, s)) is not None]
            stats[loss] = {"mean_deg": float(np.mean(v)) if v else None, "sd_over_seeds_deg": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                           "n_seeds": len(v)}
        out["conditions"][cond] = {
            "values_deg": {f"{l}_s{s}": v for (l, s), v in sorted(val.items())},
            "delta_deg": {f"s{s}": d for s, d in delta.items()},
            "reading": reading(delta.get(1), delta.get(2)),
            "mean_delta_new_seeds_deg": (delta[1] + delta[2]) / 2 if delta.get(1) is not None and delta.get(2) is not None else None,
            "descriptive": {"mean_delta_all_seeds_deg": float(np.mean(known)) if known else None,
                            "sd_delta_all_seeds_deg": float(np.std(known, ddof=1)) if len(known) > 1 else None,
                            "per_loss": stats, "per_user_by_seed": per_user}}
    return out


def to_markdown(res: dict) -> str:
    lines = ["| Condizione | Δ seme 0 | Δ seme 1 | Δ seme 2 | media semi 1-2 | Lettura | L1 media ± dev. fra semi | MSE media ± dev. fra semi |",
             "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:+.2f}"  # noqa: E731
    g = lambda s: "-" if s["mean_deg"] is None else f"{s['mean_deg']:.2f} ± {s['sd_over_seeds_deg'] or 0:.2f} ({s['n_seeds']} semi)"  # noqa: E731
    for cond, c in res["conditions"].items():
        d = c["delta_deg"]
        lines.append(f"| {cond} | {f(d.get('s0'))} | {f(d.get('s1'))} | {f(d.get('s2'))} | {f(c['mean_delta_new_seeds_deg'])} | {c['reading']} | "
                     f"{g(c['descriptive']['per_loss']['l1'])} | {g(c['descriptive']['per_loss']['mse'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="loss:seme:cartella del training:cartella della valutazione")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    arms = {}
    for spec in args.arm:
        loss, seed, train_dir, eval_dir = spec.split(":", 3)
        if loss not in ("l1", "mse"):
            ap.error(f"loss {loss}: solo l1 o mse")
        arms[(loss, int(seed))] = load_arm(loss, int(seed), train_dir, eval_dir)
    bad = {f"{l}_s{s}": a["problems"] for (l, s), a in arms.items() if a["problems"]}
    if bad:
        sys.stderr.write(f"controlli falliti, nessuna lettura: {json.dumps(bad, ensure_ascii=False)}\n")
        return 2
    res = compare(arms)
    res["arms"] = {f"{l}_s{s}": {k: a[k] for k in ("train_job", "eval_job", "code_sha", "gate")} for (l, s), a in sorted(arms.items())}
    md = to_markdown(res)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "confronto_semi.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        (out / "confronto_semi.md").write_text(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
