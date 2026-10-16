#!/usr/bin/env python3
"""Pianificatore dei bracci dell'obiettivo della finestra 1 (protocollo firmato il 16/10/2026, `docs/bracci_obiettivo_finestra1.md`): piani di
training per `scripts/wsd_chain.py` e piano di valutazione per `scripts/evaluation_chain.py` (eval-plan-v0), con il preventivo. Non sottomette
nulla e non decide nulla: il confronto lo fa `scripts/arms_compare.py` con le regole firmate.

Bracci (firma, punto 1): A (JEPA target b + ancore 0,2), JEPA-only (`--anchor-weight 0`), ancore-only (`--jepa-weight 0`); keep separata
(firma, punto 4, opzione 1): A' (testa della keep a peso 0) e B (keep 0,05), solo con la calibrazione della keep ricalcolata sul corpus
intero (`--keep-calibration`, percorso sul cluster). 30M (preset `sanity`), D = 4 epoche del derivato al 100% (passi = floor(4 x finestre
per epoca / 32)), WSD D14 (warmup da D, decay sul 20% finale), semi 0 e 1, stesso LR per tutti: quello di 30M uscito da C1-C2 (`--lr`,
scelto con la regola firmata). Stesso seme = stessa inizializzazione e stesso flusso dei dati fra i bracci (stessa architettura).

Valutazione per run (dopo la guardia del checkpoint post-decay): P1 e P2 ufficiali (`probe_p1*.sbatch`, `probe_p2*.sbatch`) e la validation
interna S della P1 e della P2 (`calibration_extract.sbatch`/`calibration_probe.sbatch`, `--task p2`); poi `arms_compare.py` (un nodo per
confronto: bracci contro A; B contro A'). Le sonde sulla seriale girano una alla volta (limite di core per utente): `--probes-on-gpu` le
mette su nodi GPU (1 GPU ciascuna) in parallelo. `--failed-runs`: run fermati prima della valutazione (niente nodi; nel confronto contano
col loro stato: collassato = non batte).

Uso: python scripts/arms_plan.py rapporto_cluster.json --lr LR --campaign campagna.json [--keep-calibration NPZ] [--probes-on-gpu]
     [--failed-runs ID ...] [--out cartella]
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

BATCH, PRESET, EPOCHS, SEEDS = 32, "sanity", 4.0, (0, 1)
OBJECTIVE_ARMS = {"A": "", "JEPA-only": "--anchor-weight 0", "ancore-only": "--jepa-weight 0"}
KEEP_ARMS = {"Aprime": "--keep-weight 0", "B": "--keep-weight 0.05"}  # con --keep-calibration
P1_DS, P2_DS = ("ninapro_db2", "ninapro_db3", "ninapro_db6"), ("epn612", "uci_emg")
# nodi della valutazione: (launcher, tetto, attese in ore); estrazioni 1 GPU; sonde 8 core (seriale) o 1 GPU con --probes-on-gpu.
# Misure: P1 ufficiale 13:42 e sonde 22 min, P2 ufficiale 41:19 e sonde 33 min (docs/step1_downstream_ab.md); S della P1 17 min e 1 h 48
# (w1_selezione_v1); S della P2 stimata come meta' dell'estrazione della P2 e due volte la sua sonda (4 fold).
EVAL = {"p1": {"extract": ("probe_p1.sbatch", "00:45:00", 0.25), "probe": ("probe_p1_probe.sbatch", "02:00:00", 0.4)},
        "p2": {"extract": ("probe_p2.sbatch", "01:30:00", 0.7), "probe": ("probe_p2_probe.sbatch", "02:00:00", 0.6)},
        "s_p1": {"extract": ("calibration_extract.sbatch", "00:45:00", 0.3), "probe": ("calibration_probe.sbatch", "03:00:00", 1.8)},
        "s_p2": {"extract": ("calibration_extract.sbatch", "01:30:00", 0.5), "probe": ("calibration_probe.sbatch", "03:00:00", 1.2)}}
PROBE_CORES = 8
GPU_PROBE_SBATCH = ["--partition=boost_usr_prod", "--qos=normal", "--gres=gpu:1"]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hours(hms: str) -> float:
    h, m, s = (int(x) for x in hms.split(":"))
    return h + m / 60 + s / 3600


def arm_runs(steps: int, lr: float, keep_calibration: str | None) -> dict[str, list[dict]]:
    """{gruppo: [run]} per wsd_chain: "objective" (A, JEPA-only, ancore-only) e, con la calibrazione della keep, "keep" (A', B)."""
    groups = {"objective": OBJECTIVE_ARMS}
    if keep_calibration:
        groups["keep"] = {k: f"--keep-calibration {keep_calibration} {v}" for k, v in KEEP_ARMS.items()}
    out = {}
    for g, arms in groups.items():
        out[g] = [{"id": f"W1ARM-{arm}-30M-s{seed}", "preset": PRESET, "seed": seed, "batch_size": BATCH, "accum_steps": 1, "steps_at_d": steps,
                   "total_steps": steps, "keep_checkpoints": [], "args": f"--lr {lr:g} --nested 1 {flags}".strip(), "arm": arm}
                  for arm, flags in arms.items() for seed in SEEDS]
    return out


def eval_nodes(run: dict, code: dict, ev: dict, sbatch_probe: list[str]) -> tuple[list[dict], dict]:
    """Guardia (dal checkpoint), estrazioni e sonde di un run; ritorna i nodi e i rapporti per arms_compare."""
    rid, rd = run["id"], run["run_dir"]
    ckpt = f"{rd}/checkpoint_postdecay.pt"
    feat, res = f"{ev['features_root']}/{rid}", f"{ev['results_root']}/{rid}"
    raw = ev["raw_root"]
    p2_roots = ["--epn-root", f"{raw}/epn612/EMG-EPN612 Dataset/trainingJSON", "--epn-root", f"{raw}/epn612/EMG-EPN612 Dataset/testingJSON",
                "--uci-root", f"{raw}/uci_emg"]
    nodes, reports = [], {}
    for kind in ("p1", "p2", "s_p1", "s_p2"):
        (ext_l, ext_t, _), (prb_l, prb_t, _) = EVAL[kind]["extract"], EVAL[kind]["probe"]
        f, rep = f"{feat}/{kind}", f"{res}/{kind}.json"
        ds = P1_DS if kind in ("p1", "s_p1") else P2_DS
        npzs = [f"{f}/{d}.npz" for d in ds]
        task = ["--task", "p2"] if kind == "s_p2" else []
        ext_args = [ckpt, f] + (task + p2_roots if kind == "s_p2" else [])
        exports = {"RAW_ROOT": raw} if kind == "p2" else {}
        ext = {"id": f"{rid}-{kind}-ext", "kind": "extract", "after": [], "queue": "boost_usr_prod", "gpu_hours": round(_hours(ext_t), 3), **code,
               "launcher": f"scripts/slurm/{ext_l}", "args": ext_args, "exports": exports, "sbatch": [f"--time={ext_t}"], "inputs": [ckpt],
               "outputs": npzs, "checkpoint": {"run_dir": rd, "name": "checkpoint_postdecay.pt", "training_job": run["job_ids"][-1],
                                                "min_steps": run["total_steps"], "sha256": None}}
        if kind in ("p1", "p2"):
            prb_args, outs = [ckpt, f, rep], [rep]
        else:
            prb_args, outs = [f, rep] + task, [rep, rep[:-5] + ".oof.npz"]
        prb = {"id": f"{rid}-{kind}-probe", "kind": "probe", "after": [ext["id"]], "queue": "boost_usr_prod" if sbatch_probe else "lrd_all_serial",
               "gpu_hours": round(_hours(prb_t), 3) if sbatch_probe else 0, **code, "launcher": f"scripts/slurm/{prb_l}", "args": prb_args,
               "exports": {}, "sbatch": [f"--time={prb_t}", *sbatch_probe], "inputs": npzs, "outputs": outs}
        nodes += [ext, prb]
        reports[kind] = rep
    return nodes, reports


def plan(build_report: dict, lr: float, campaign: dict, keep_calibration: str | None = None, probes_on_gpu: bool = False,
         failed_runs: frozenset = frozenset(), plans_dir: str | None = None) -> dict:
    from wearusfm.training import run as R

    CP, W = _mod("calibration_plan"), _mod("wsd_chain")
    if not str((build_report.get("code") or {}).get("source", "")).startswith("launcher"):
        raise ValueError("serve il rapporto della costruzione dei derivati sul cluster (code.source: launcher)")
    frac = build_report["fractions"]["1"]
    if Path(campaign["manifest"]).name != frac.get("file"):
        raise ValueError(f"manifest della campagna {campaign['manifest']}: il rapporto descrive {frac.get('file')}")
    if not (isinstance(lr, float) and math.isfinite(lr) and lr > 0):
        raise ValueError(f"LR {lr!r}: positivo e finito (quello di 30M da C1-C2)")
    steps = math.floor(EPOCHS * frac["windows_per_epoch"] / BATCH)
    base = {k: v for k, v in campaign.items() if k != "eval"}
    ev = campaign["eval"]
    code = {"code_dir": ev.get("code_dir", base["code_dir"]), "expected_sha": ev.get("expected_sha", base["expected_sha"])}
    panel = _mod("calibration_select").PANEL_EVERY
    groups = arm_runs(steps, lr, keep_calibration)
    plans, evals, rows, specs = {}, {}, [], {}
    all_ids = {r["id"] for g in groups.values() for r in g}
    unknown = set(failed_runs) - all_ids
    if unknown:
        raise ValueError(f"run falliti {sorted(unknown)} non sono run dei bracci")
    sbatch_probe = list(GPU_PROBE_SBATCH) if probes_on_gpu else []
    for g, runs in groups.items():
        p = {**copy.deepcopy(base), "campaign": f"{base['campaign']}-{g}", "runs": [{k: v for k, v in r.items() if k != "arm"} for r in runs],
             "s_per_step": {f"{PRESET}/{BATCH}": CP.S_PER_STEP[PRESET]}, "ckpt_gb": {PRESET: CP.CKPT_GB[PRESET]},
             "common_args": f"--window1-rules --datasets all --prefetch --fixed-eval-every {panel} --manifest-sha256 {frac['file_sha256']}"}
        jobs = W.plan_jobs(p)
        plans[g] = p
        nodes, spec_runs, probes = [], {}, []
        for run, meta in zip(jobs["runs"], runs):
            rows.append({"group": g, "id": run["id"], "arm": meta["arm"], "seed": run["seed"], "steps": run["total_steps"],
                         "warmup_steps": run["warmup_steps"], "decay_start": run["decay_start"], "jobs": run["jobs"], "job_time": run["time"],
                         "gpu_hours_expected": run["gpu_hours_expected"], "gpu_hours_max": run["gpu_hours_max"]})
            entry = {"run_dir": run["run_dir"]}
            if run["id"] not in failed_runs:
                n, reps = eval_nodes(run, code, ev, sbatch_probe)
                nodes += n
                probes += [x["id"] for x in n if x["kind"] == "probe"]
                entry.update(reps)
            spec_runs.setdefault(meta["arm"], {})[str(run["seed"])] = entry
        ref = "A" if g == "objective" else "Aprime"
        spec = {"reference": ref, "arms": [a for a in spec_runs if a != ref], "runs": spec_runs}
        if any(set(failed_runs) & {f"W1ARM-{ref}-30M-s{s}" for s in SEEDS}):
            raise ValueError(f"{g}: il riferimento {ref} non e' arrivato alla valutazione: nessun confronto")
        spec_path = f"{plans_dir}/confronto_{g}.json" if plans_dir else "-"
        specs[g] = spec
        out = f"{ev['results_root']}/confronto_{g}.json"
        reports = [v for e in spec_runs.values() for x in e.values() for k, v in x.items() if k != "run_dir"]
        oofs = [r[:-5] + ".oof.npz" for r in reports if r.split("/")[-1].startswith("s_")]
        nodes.append({"id": f"W1ARM-{g}-compare", "kind": "compare", "after": probes, "queue": "lrd_all_serial", "gpu_hours": 0, **code,
                      "launcher": "scripts/slurm/arms_compare.sbatch", "args": [spec_path, out], "exports": {}, "sbatch": ["--time=00:30:00"],
                      "inputs": reports + oofs, "outputs": [out]})
        n_eval = sum(1 for r in runs if r["id"] not in failed_runs)
        ext_exp = sum(EVAL[k]["extract"][2] for k in EVAL)
        ext_max = sum(_hours(EVAL[k]["extract"][1]) for k in EVAL)
        prb_exp = sum(EVAL[k]["probe"][2] for k in EVAL)
        prb_max = sum(_hours(EVAL[k]["probe"][1]) for k in EVAL)
        evals[g] = {"schema": "eval-plan-v0", "campaign": f"{base['campaign']}-{g}-eval", "agent": base["agent"],
                    "guard": {**code, "report_root": f"{ev['results_root']}/guards"}, "limits": dict(CP.EVAL_LIMITS), "nodes": nodes,
                    "estimate": {"gpu_hours_expected": round(n_eval * (ext_exp + (prb_exp if probes_on_gpu else 0)), 1),
                                 "gpu_hours_max": round(n_eval * (ext_max + (prb_max if probes_on_gpu else 0)), 1),
                                 "cpu_core_hours_expected": 0 if probes_on_gpu else round(n_eval * PROBE_CORES * prb_exp, 1),
                                 "serial_probe_hours": 0 if probes_on_gpu else round(n_eval * prb_exp, 1)}}
    totals = {"runs": len(rows), "gpu_hours_expected": round(sum(r["gpu_hours_expected"] for r in rows), 1),
              "gpu_hours_max": round(sum(r["gpu_hours_max"] for r in rows), 1),
              "eval_gpu_hours_expected": round(sum(e["estimate"]["gpu_hours_expected"] for e in evals.values()), 1),
              "eval_gpu_hours_max": round(sum(e["estimate"]["gpu_hours_max"] for e in evals.values()), 1)}
    return {"lr": lr, "steps": steps, "warmup_steps": R.wsd_warmup_steps(steps), "keep": bool(keep_calibration), "probes_on_gpu": probes_on_gpu,
            "runs": rows, "plans": plans, "eval_plans": evals, "specs": specs, "totals": totals,
            "blocked": {} if keep_calibration else {"keep": "manca la calibrazione della keep ricalcolata sul corpus intero (firma, punto 4)"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_report", type=Path)
    ap.add_argument("--lr", type=float, required=True, help="LR di 30M da C1-C2 (regola firmata)")
    ap.add_argument("--campaign", type=Path, required=True, help="come quella della calibrazione, con eval.raw_root per la P2")
    ap.add_argument("--keep-calibration", default=None, help="calibrazione della keep sul corpus intero (percorso sul cluster)")
    ap.add_argument("--probes-on-gpu", action="store_true")
    ap.add_argument("--failed-runs", nargs="*", default=[])
    ap.add_argument("--out", type=Path, help="cartella dei piani (relativa: nel repo, cosi' il checkout della valutazione li contiene)")
    args = ap.parse_args(argv)
    res = plan(json.loads(args.build_report.read_text()), args.lr, json.loads(args.campaign.read_text()), args.keep_calibration,
               args.probes_on_gpu, frozenset(args.failed_runs), str(args.out) if args.out else None)
    for r in res["runs"]:
        print(f"{r['group']:9s} {r['id']:28s} seme {r['seed']} passi {r['steps']} decay da {r['decay_start']} {r['jobs']} x {r['job_time']} "
              f"GPU-ora {r['gpu_hours_expected']} / {r['gpu_hours_max']}")
    t = res["totals"]
    print(f"training {t['gpu_hours_expected']} / {t['gpu_hours_max']} GPU-ora; valutazione {t['eval_gpu_hours_expected']} / "
          f"{t['eval_gpu_hours_max']}; bloccati: {res['blocked'] or 'nessuno'}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for g, p in res["plans"].items():
            (args.out / f"bracci_{g}.json").write_text(json.dumps(p, indent=1))
            (args.out / f"valutazione_{g}.json").write_text(json.dumps(res["eval_plans"][g], indent=1))
            (args.out / f"confronto_{g}.json").write_text(json.dumps(res["specs"][g], indent=1))
        (args.out / "bracci_riepilogo.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("plans", "eval_plans", "specs")},
                                                                   indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
