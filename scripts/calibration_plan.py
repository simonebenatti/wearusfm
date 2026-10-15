"""Pianificatore della calibrazione C1-C4 della finestra 1 (accordo GPT-Claude del 15/10/2026, `docs/accordo_operativo_finestra1_20261015.md`,
sezione «Calibrazione»): configurazioni, passi, checkpoint, memoria, tempo, preventivo e tetto, e i piani per `scripts/wsd_chain.py`. Non
sottomette nulla e non sceglie nulla: i finalisti sono decisioni esplicite di GPT (file `--decisions`), mai dedotte dalla training loss.

Fasi (obiettivo di riferimento A: JEPA target b + ancore 0,2; keep e rami spenti; batch globale 32; un nodo, 4 GPU; prefetch):
  C1  30M, LR 1e-4 / 3e-4 / 1e-3, 1 epoca, seme 0 (3 run)
  C2  30M, i due finalisti di C1, 1 epoca, seme 1 (2 run)
  C3  100M, i due finalisti confermati da C2, 0,5 epoche, seme 0 (2 run)
  C4  300M, il LR proposto dopo C3, 0,25 epoche, seme 0 (1 verifica)
Una fase senza la decisione che la precede e' bloccata, col motivo; una conferma fallita in C2 ferma C3 e C4 (nessuna sostituzione di LR,
nessuna espansione della griglia). WSD D14: warmup riferito a D = 4 epoche (`wsd_warmup_steps` dei passi a D), decay sul 20% finale di
ciascun endpoint; passi = floor(epoche x finestre per epoca / batch), un solo floor, dal manifest derivato al 100% (rapporto del costruttore).
Stesso seme entro taglia: inizializzazione e stream accoppiati fra i candidati.

File delle decisioni (JSON, scritto da GPT con le sue regole congelate prima di C1):
  {"C1": {"finalists": [lr, lr], "record": "..."}, "C2": {"confirmed": [lr, ...], "record": "..."}, "C3": {"proposed_300m_lr": lr, "record": "..."}}

Uso: python scripts/calibration_plan.py rapporto_dei_manifest_derivati.json [--decisions decisioni.json] [--campaign campagna.json]
     [--out cartella]   -> tabella e preventivo; con --out i piani per wsd_chain.py delle fasi pronte e il riepilogo JSON
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GRID_C1 = (1e-4, 3e-4, 1e-3)
PHASES = {"C1": {"preset": "sanity", "label": "30M", "epochs": 1.0, "seed": 0},
          "C2": {"preset": "sanity", "label": "30M", "epochs": 1.0, "seed": 1},
          "C3": {"preset": "100m", "label": "100M", "epochs": 0.5, "seed": 0},
          "C4": {"preset": "300m", "label": "300M", "epochs": 0.25, "seed": 0}}
BATCH = 32
# misure: s/passo DDP 4 GPU batch 32 col prefetch (w1pf, w1gd) e memoria massima per GPU (w1gd, 8 finestre per GPU)
S_PER_STEP = {"sanity": 0.70, "100m": 1.39, "300m": 2.77}
MEMORY_GIB = {"sanity": 10.0, "100m": 17.4, "300m": 28.7}
CKPT_GB = {"sanity": 0.45, "100m": 1.7, "300m": 4.7}
DRY_CAMPAIGN = {"campaign": "w1-calibrazione-DRYRUN", "agent": "claude", "code_dir": "<CODE_DIR>", "expected_sha": "0" * 40,
                "manifest": "<derivato_100>", "scales": "<session_scales.json>", "window_seconds": "<window_seconds.json>",
                "run_root": "<RUN_ROOT>", "slurm": {"qos": "normal", "time": "24:00:00"}, "margin_min": 40, "startup_min": 15, "slack": 1.15,
                # un job per run, dimensionato con la stima pessimista: nessuna scorta (un run che non finisce si ferma pulito al limite di
                # tempo e la continuazione ha un suo preventivo, accordo del 15/10)
                "spare_jobs": 0, "gpus": 4}


def _wsd_chain():
    spec = importlib.util.spec_from_file_location("wsd_chain", ROOT / "scripts" / "wsd_chain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lr_tag(lr: float) -> str:
    """Etichetta del LR negli ID e nelle cartelle, senza perdita (`repr` del float): 0.00025 -> '0p00025', 1e-4 -> '0p0001'."""
    return repr(float(lr)).replace(".", "p").replace("-", "m").replace("+", "")


def resolve_phases(decisions: dict | None) -> dict:
    """LR per fase dalle decisioni, oppure il motivo per cui la fase e' bloccata. Errori su decisioni malformate."""
    d = decisions or {}
    out = {"C1": {"lrs": list(GRID_C1)}}
    c1 = d.get("C1", {}).get("finalists")
    if c1 is None:
        out.update({p: {"blocked": "in attesa dei finalisti di C1 (decisione di GPT con criteri congelati prima di C1)"} for p in ("C2", "C3", "C4")})
        return out
    if len(c1) != 2 or len(set(c1)) != 2 or any(x not in GRID_C1 for x in c1):
        raise ValueError(f"finalisti di C1 {c1}: servono due LR distinti della griglia {GRID_C1}")
    out["C2"] = {"lrs": sorted(c1)}
    confirmed = d.get("C2", {}).get("confirmed")
    if confirmed is None:
        out.update({p: {"blocked": "in attesa della conferma di C2"} for p in ("C3", "C4")})
        return out
    if any(x not in c1 for x in confirmed):
        raise ValueError(f"confermati di C2 {confirmed}: devono essere finalisti di C1 {sorted(c1)}")
    if sorted(set(confirmed)) != sorted(c1):
        why = "un finalista ha fallito la conferma in C2: C3 e C4 si fermano (nessuna sostituzione di LR, nessuna espansione della griglia)"
        out.update({p: {"blocked": why} for p in ("C3", "C4")})
        return out
    out["C3"] = {"lrs": sorted(c1)}
    lr300 = d.get("C3", {}).get("proposed_300m_lr")
    if lr300 is None:
        out["C4"] = {"blocked": "in attesa del LR proposto per il 300M dopo C3"}
        return out
    if not (isinstance(lr300, (int, float)) and math.isfinite(lr300) and lr300 > 0):
        raise ValueError(f"LR proposto per il 300M {lr300!r}: positivo e finito")
    out["C4"] = {"lrs": [float(lr300)]}
    return out


def plan(build_report: dict, decisions: dict | None = None, campaign: dict | None = None) -> dict:
    from wearusfm.training import run as R

    frac = build_report["fractions"]["1"]
    wpe = frac["windows_per_epoch"]
    steps_at_d = math.floor(4.0 * wpe / BATCH)
    warmup = R.wsd_warmup_steps(steps_at_d)
    phases = resolve_phases(decisions)
    W = _wsd_chain()
    base = {**DRY_CAMPAIGN, **(campaign or {})}
    dry = campaign is None
    rows, plans, blocked = [], {}, {}
    for name, spec in PHASES.items():
        info = phases[name]
        if "blocked" in info:
            blocked[name] = info["blocked"]
            continue
        total = math.floor(spec["epochs"] * wpe / BATCH)
        runs = [{"id": f"W1CAL-{name}-{spec['label']}-lr{_lr_tag(lr)}-s{spec['seed']}", "preset": spec["preset"], "seed": spec["seed"],
                 "batch_size": BATCH, "accum_steps": 1, "steps_at_d": steps_at_d, "total_steps": total, "keep_checkpoints": [],
                 "args": f"--lr {lr:g} --nested 1"} for lr in info["lrs"]]
        p = {**copy.deepcopy(base), "campaign": f"{base['campaign']}-{name.lower()}", "runs": runs,
             "s_per_step": {f"{spec['preset']}/{BATCH}": S_PER_STEP[spec["preset"]]}, "ckpt_gb": {spec["preset"]: CKPT_GB[spec["preset"]]},
             "common_args": "--window1-rules --datasets all --prefetch"}
        if len({r["id"] for r in runs}) != len(runs):
            raise ValueError(f"{name}: ID di run ripetuti {[r['id'] for r in runs]}")
        jobs = W.plan_jobs(p)
        plans[name] = p
        for run, lr in zip(jobs["runs"], info["lrs"]):
            hours = run["total_steps"] * run["s_per_step"] / 3600
            rows.append({"phase": name, "id": run["id"], "size": spec["label"], "preset": spec["preset"], "seed": spec["seed"], "lr": lr,
                         "epochs": spec["epochs"], "steps": run["total_steps"], "warmup_steps": run["warmup_steps"],
                         "decay_start": run["decay_start"], "checkpoints": ["checkpoint_predecay.pt", "checkpoint_postdecay.pt", "checkpoint.pt"],
                         "memory_gib_per_gpu": MEMORY_GIB[spec["preset"]], "hours": round(hours, 2), "job_time": run["time"],
                         "jobs": run["jobs"], "gpu_hours_expected": run["gpu_hours_expected"], "gpu_hours_max": run["gpu_hours_max"]})
        if any(r["warmup_steps"] != warmup for r in jobs["runs"]):
            raise AssertionError("warmup diverso da quello di D")
    totals = {"runs": len(rows), "gpu_hours_expected": round(sum(r["gpu_hours_expected"] for r in rows), 1),
              "gpu_hours_max": round(sum(r["gpu_hours_max"] for r in rows), 1),
              "gpu_hours_training_only": round(sum(4 * r["steps"] * S_PER_STEP[r["preset"]] / 3600 for r in rows), 1)}
    return {"dry_run": dry, "batch": BATCH, "windows_per_epoch": wpe, "steps_at_d": steps_at_d, "warmup_steps": warmup,
            "manifest": {"hash": frac.get("hash"), "file_sha256": frac.get("file_sha256"), "file": frac.get("file")},
            "runs": rows, "blocked": blocked, "totals": totals, "plans": plans,
            "excluded_from_estimate": "estrazioni, sonde e selezione (GPT), collaudi; batch 64 e decay 10/20% fuori da questa griglia"}


def table(result: dict) -> str:
    lines = ["| Fase | Run | LR | Seme | Passi | Warmup | Decay da | Memoria/GPU | Ore | Job | GPU-ora attese / tetto |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        lines.append(f"| {r['phase']} | {r['id']} | {r['lr']:g} | {r['seed']} | {r['steps']} | {r['warmup_steps']} | {r['decay_start']} | "
                     f"{r['memory_gib_per_gpu']} GiB | {r['hours']} | {r['jobs']} x {r['job_time']} | {r['gpu_hours_expected']} / {r['gpu_hours_max']} |")
    for p, why in result["blocked"].items():
        lines.append(f"| {p} | bloccata | | | | | | | | | {why} |")
    t = result["totals"]
    lines.append(f"| **Totale** | {t['runs']} run | | | | | | | | | {t['gpu_hours_expected']} / {t['gpu_hours_max']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build_report", type=Path)
    ap.add_argument("--decisions", type=Path)
    ap.add_argument("--campaign", type=Path, help="campi di campagna per wsd_chain.py (code_dir, expected_sha, manifest, ...); senza: dry-run")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result = plan(json.loads(args.build_report.read_text()), json.loads(args.decisions.read_text()) if args.decisions else None,
                  json.loads(args.campaign.read_text()) if args.campaign else None)
    print(table(result))
    print(f"\npassi a D (4 epoche) {result['steps_at_d']}, warmup {result['warmup_steps']}; solo training "
          f"{result['totals']['gpu_hours_training_only']} GPU-ora; {'DRY-RUN (percorsi segnaposto)' if result['dry_run'] else 'campagna'}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, p in result["plans"].items():
            (args.out / f"calibrazione_{name}.json").write_text(json.dumps(p, indent=1))
        (args.out / "calibrazione_riepilogo.json").write_text(json.dumps({k: v for k, v in result.items() if k != "plans"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
