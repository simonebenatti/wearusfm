"""Pianificatore della calibrazione C1-C4 della finestra 1 (accordo GPT-Claude del 15/10/2026, `docs/accordo_operativo_finestra1_20261015.md`,
sezione «Calibrazione»): configurazioni, passi, checkpoint, memoria, tempo, preventivo e tetto, e i piani per `scripts/wsd_chain.py`. Non
sottomette nulla e non sceglie nulla: i finalisti sono decisioni esplicite (file `--decisions`, da `scripts/calibration_select.py` con le
regole congelate prima di C1), mai dedotte dalla training loss.

Fasi (obiettivo di riferimento A: JEPA target b + ancore 0,2; keep e rami spenti; batch globale 32; un nodo, 4 GPU; prefetch):
  C1  30M, LR 1e-4 / 3e-4 / 1e-3, 1 epoca, seme 0 (3 run)
  C2  30M, i due finalisti di C1, 1 epoca, seme 1 (2 run)
  C3  100M, i due finalisti confermati da C2, 0,5 epoche, seme 0 (2 run)
  C4  300M, il LR proposto dopo C3, 0,25 epoche, seme 0 (1 verifica)
Una fase senza la decisione che la precede e' bloccata, col motivo; uno stop di C1 o C3 o una conferma fallita in C2 ferma le fasi successive
(nessuna sostituzione di LR, nessuna espansione della griglia). Il pannello fisso del trainer e' acceso (`--fixed-eval-every`, la cadenza
del criterio di stabilita' A4 di `scripts/calibration_select.py`). WSD D14: warmup riferito a D = 4 epoche (`wsd_warmup_steps` dei passi a D), decay sul 20% finale di
ciascun endpoint; passi = floor(epoche x finestre per epoca / batch), un solo floor, dal manifest derivato al 100% (rapporto del costruttore).
Stesso seme entro taglia: inizializzazione e stream accoppiati fra i candidati.

File delle decisioni (JSON, da `scripts/calibration_select.py merge`, con le regole congelate prima di C1 in
`docs/calibrazione_c1_c4_selezione.md`):
  {"C1": {"finalists": [lr, lr]} | {"stop": "..."}, "C2": {"confirmed": [lr, ...]}, "C3": {"proposed_300m_lr": lr} | {"stop": "..."},
   "C4": {"lr": lr, "passed": bool}}
Piano di valutazione (`--out` con una campagna che ha "eval"): per fase pronta `valutazione_<fase>.json` (schema eval-plan-v0 di
`scripts/evaluation_chain.py`): per ogni run guardia del checkpoint post-decay -> estrazione (1 GPU) -> sonda a fold (CPU), poi la selezione
della fase (CPU) dopo tutte le sonde.

Uso (il rapporto e' quello della costruzione sul cluster, results/infra/derived_manifests_v1_1_build_cluster_20261015.json):
     python scripts/calibration_plan.py rapporto_dei_manifest_derivati.json [--decisions decisioni.json] [--campaign campagna.json]
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
DRY_EVAL = {"features_root": "<FEATURES_ROOT>", "results_root": "<RESULTS_ROOT>"}
P1_DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6")
# valutazione: misure del collaudo w1_selezione_v1 (16/10/2026) al 30M: estrazione 17:51 e 16:34 (~10 min di CPU, il resto GPU e I/O), sonda
# 1 h 44 e 1 h 51 a 8 core. 100M e 300M: STIME, estrazione scalata coi parametri sulla parte GPU, sonda con la larghezza delle feature (2d:
# 768 / 1280 / 1792). Sulla seriale (tetto 4 h) la sonda del 300M (~4,2 h) non ci sta: prima di C4 va spostata su un nodo GPU o divisa per
# dataset (registro, 16/10). Le sonde girano una alla volta per il limite di core per utente della seriale.
EXTRACT_TIME = {"sanity": "00:30:00", "100m": "01:30:00", "300m": "03:00:00"}
EXTRACT_EXPECTED_H = {"sanity": 0.3, "100m": 0.65, "300m": 1.5}
PROBE_CORES = 8
PROBE_TIME = {"sanity": "03:00:00", "100m": "04:00:00", "300m": "04:00:00"}
PROBE_EXPECTED_H = {"sanity": 1.8, "100m": 3.0, "300m": 4.2}
SELECT_TIME, SELECT_CORES = "00:30:00", 2
EVAL_LIMITS = {"boost_usr_prod": 8, "lrd_all_serial": 8}


def _wsd_chain():
    spec = importlib.util.spec_from_file_location("wsd_chain", ROOT / "scripts" / "wsd_chain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _select():
    spec = importlib.util.spec_from_file_location("calibration_select", ROOT / "scripts" / "calibration_select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hours(hms: str) -> float:
    h, m, s = (int(x) for x in hms.split(":"))
    return h + m / 60 + s / 3600


def _lr_tag(lr: float) -> str:
    """Etichetta del LR negli ID e nelle cartelle, senza perdita (`repr` del float): 0.00025 -> '0p00025', 1e-4 -> '0p0001'."""
    return repr(float(lr)).replace(".", "p").replace("-", "m").replace("+", "")


def resolve_phases(decisions: dict | None) -> dict:
    """LR per fase dalle decisioni, oppure il motivo per cui la fase e' bloccata. Errori su decisioni malformate."""
    d = decisions or {}
    out = {"C1": {"lrs": list(GRID_C1)}}
    if "stop" in d.get("C1", {}):
        out.update({p: {"blocked": f"C1 si e' fermata: {d['C1']['stop']}"} for p in ("C2", "C3", "C4")})
        return out
    c1 = d.get("C1", {}).get("finalists")
    if c1 is None:
        out.update({p: {"blocked": "in attesa dei finalisti di C1 (selezione con le regole congelate prima di C1)"} for p in ("C2", "C3", "C4")})
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
    if "stop" in d.get("C3", {}):
        out["C4"] = {"blocked": f"C3 si e' fermata: {d['C3']['stop']}"}
        return out
    lr300 = d.get("C3", {}).get("proposed_300m_lr")
    if lr300 is None:
        out["C4"] = {"blocked": "in attesa del LR proposto per il 300M dopo C3"}
        return out
    if not (isinstance(lr300, (int, float)) and math.isfinite(lr300) and lr300 > 0):
        raise ValueError(f"LR proposto per il 300M {lr300!r}: positivo e finito")
    if float(lr300) not in [float(x) for x in c1]:
        raise ValueError(f"LR proposto per il 300M {lr300}: deve essere uno dei due LR di C3 {sorted(c1)}")
    out["C4"] = {"lrs": [float(lr300)]}
    if "C4" in d and (float(d["C4"].get("lr", -1)) != float(lr300) or not isinstance(d["C4"].get("passed"), bool)):
        raise ValueError(f"esito di C4 {d['C4']}: serve il LR proposto {lr300} e passed vero o falso")
    return out


def eval_plan(name: str, spec: dict, jobs: dict, lrs: list[float], base: dict, ev: dict, plan_path: str = "-",
              failed: frozenset = frozenset()) -> dict:
    """Il piano di valutazione della fase per `scripts/evaluation_chain.py` (schema eval-plan-v0). `plan_path`: il piano di training della
    fase come lo vede la selezione (relativo al checkout della valutazione, che deve contenerlo); `failed`: run il cui training si e' fermato
    prima della sonda (niente guardia, estrazione e sonda: entrano nella selezione come inammissibili, review di GPT del 16/10/2026).
    Il codice della valutazione puo' essere un checkout diverso da quello del training (`ev["code_dir"]`, `ev["expected_sha"]`)."""
    code = {"code_dir": ev.get("code_dir", base["code_dir"]), "expected_sha": ev.get("expected_sha", base["expected_sha"])}
    unknown = set(failed) - {r["id"] for r in jobs["runs"]}
    if unknown:
        raise ValueError(f"{name}: run falliti {sorted(unknown)} non sono run della fase")
    if failed and len(failed) == len(jobs["runs"]):
        raise ValueError(f"{name}: nessun run arrivato alla sonda: la fase si ferma, decisione da registrare a mano")
    nodes, cands, probe_outs = [], [], []
    for run, lr in zip(jobs["runs"], lrs):
        if run["id"] in failed:
            cands += [repr(float(lr)), run["run_dir"], "-", str(run["total_steps"])]
            continue
        rid, ckpt = run["id"], f"{run['run_dir']}/checkpoint_postdecay.pt"
        feat, rep = f"{ev['features_root']}/{name}/{rid}", f"{ev['results_root']}/{name}/{rid}.json"
        npzs, oof = [f"{feat}/{d}.npz" for d in P1_DATASETS], str(Path(rep).with_suffix(".oof.npz"))
        t = EXTRACT_TIME[spec["preset"]]
        nodes.append({"id": f"{rid}-ext", "kind": "extract", "after": [], "queue": "boost_usr_prod", "gpu_hours": round(_hours(t), 3), **code,
                      "launcher": "scripts/slurm/calibration_extract.sbatch", "args": [ckpt, feat], "exports": {}, "sbatch": [f"--time={t}"],
                      "inputs": [ckpt], "outputs": npzs,
                      "checkpoint": {"run_dir": run["run_dir"], "name": "checkpoint_postdecay.pt", "training_job": run["job_ids"][-1],
                                     "min_steps": run["total_steps"], "sha256": None}})
        nodes.append({"id": f"{rid}-probe", "kind": "probe", "after": [f"{rid}-ext"], "queue": "lrd_all_serial", "gpu_hours": 0, **code,
                      "launcher": "scripts/slurm/calibration_probe.sbatch", "args": [feat, rep], "exports": {}, "sbatch": [f"--time={PROBE_TIME[spec['preset']]}"],
                      "inputs": npzs, "outputs": [rep, oof]})
        cands += [repr(float(lr)), run["run_dir"], rep, str(run["total_steps"])]
        probe_outs += [rep, oof]
    out, record = f"{ev['results_root']}/{name}/decisione_{name}.json", f"{ev['results_root']}/{name}/registro_{name}.json"
    nodes.append({"id": f"W1CAL-{name}-select", "kind": "compare", "after": [n["id"] for n in nodes if n["kind"] == "probe"], "queue": "lrd_all_serial",
                  "gpu_hours": 0, **code, "launcher": "scripts/slurm/calibration_select.sbatch", "args": [name, out, record, plan_path, *cands],
                  "exports": {},
                  "sbatch": [f"--time={SELECT_TIME}"], "inputs": probe_outs, "outputs": [out, record]})
    n_runs = len(lrs) - len(failed)
    return {"schema": "eval-plan-v0", "campaign": f"{base['campaign']}-{name.lower()}-eval", "agent": base["agent"],
            "guard": {**code, "report_root": f"{ev['results_root']}/guards"}, "limits": dict(EVAL_LIMITS), "nodes": nodes,
            "estimate": {"gpu_hours_expected": round(n_runs * EXTRACT_EXPECTED_H[spec["preset"]], 2),
                         "gpu_hours_max": round(n_runs * _hours(EXTRACT_TIME[spec["preset"]]), 2),
                         "cpu_core_hours_expected": round(n_runs * PROBE_CORES * PROBE_EXPECTED_H[spec["preset"]], 1),
                         "probe_fits_the_cap": PROBE_EXPECTED_H[spec["preset"]] <= _hours(PROBE_TIME[spec["preset"]]),
                         "cpu_core_hours_max": round(n_runs * PROBE_CORES * _hours(PROBE_TIME[spec["preset"]]) + SELECT_CORES * _hours(SELECT_TIME)
                                                     + n_runs * _hours("00:20:00"), 1)}}


def plan(build_report: dict, decisions: dict | None = None, campaign: dict | None = None, plans_dir: str | None = None,
         failed_runs: frozenset = frozenset()) -> dict:
    from wearusfm.training import run as R

    frac = build_report["fractions"]["1"]
    wpe = frac["windows_per_epoch"]
    # il rapporto della costruzione SUL CLUSTER: lo SHA del file e' quello che il training verifica (--manifest-sha256) e che entra nella
    # firma dei dati; quello della costruzione locale e' di un file che sul cluster non esiste (review del critico, 16/10/2026)
    if not str((build_report.get("code") or {}).get("source", "")).startswith("launcher"):
        raise ValueError("rapporto dei manifest derivati non costruito da un launcher sul cluster (code.source): serve quello della costruzione "
                         "sul cluster, results/infra/derived_manifests_v1_1_build_cluster_20261015.json")
    if not (isinstance(frac.get("file_sha256"), str) and len(frac["file_sha256"]) == 64):
        raise ValueError("rapporto dei manifest derivati senza file_sha256 del derivato al 100%: serve il rapporto della costruzione sul cluster")
    steps_at_d = math.floor(4.0 * wpe / BATCH)
    warmup = R.wsd_warmup_steps(steps_at_d)
    phases = resolve_phases(decisions)
    W = _wsd_chain()
    base = {**DRY_CAMPAIGN, **(campaign or {})}
    ev = {**DRY_EVAL, **base.pop("eval", {})}
    dry = campaign is None
    if not dry and Path(base["manifest"]).name != frac.get("file"):
        raise ValueError(f"manifest della campagna {base['manifest']}: il rapporto descrive {frac.get('file')}")
    panel = _select().PANEL_EVERY
    rows, plans, blocked, evals = [], {}, {}, {}
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
             "common_args": f"--window1-rules --datasets all --prefetch --fixed-eval-every {panel} --manifest-sha256 {frac['file_sha256']}"}
        if len({r["id"] for r in runs}) != len(runs):
            raise ValueError(f"{name}: ID di run ripetuti {[r['id'] for r in runs]}")
        jobs = W.plan_jobs(p)
        plans[name] = p
        evals[name] = eval_plan(name, spec, jobs, info["lrs"], base, ev, f"{plans_dir}/calibrazione_{name}.json" if plans_dir else "-",
                                frozenset(r["id"] for r in jobs["runs"] if r["id"] in failed_runs))
        for run, lr in zip(jobs["runs"], info["lrs"]):
            hours = run["total_steps"] * run["s_per_step"] / 3600
            rows.append({"phase": name, "id": run["id"], "size": spec["label"], "preset": spec["preset"], "seed": spec["seed"], "lr": lr,
                         "epochs": spec["epochs"], "steps": run["total_steps"], "warmup_steps": run["warmup_steps"],
                         "decay_start": run["decay_start"], "checkpoints": ["checkpoint_predecay.pt", "checkpoint_postdecay.pt", "checkpoint.pt"],
                         "memory_gib_per_gpu": MEMORY_GIB[spec["preset"]], "hours": round(hours, 2), "job_time": run["time"],
                         "jobs": run["jobs"], "gpu_hours_expected": run["gpu_hours_expected"], "gpu_hours_max": run["gpu_hours_max"]})
        if any(r["warmup_steps"] != warmup for r in jobs["runs"]):
            raise AssertionError("warmup diverso da quello di D")
    unknown = set(failed_runs) - {r["id"] for r in rows}
    if unknown:
        raise ValueError(f"run falliti {sorted(unknown)} non sono run della fase (run pianificati: {sorted(r['id'] for r in rows)})")
    totals = {"runs": len(rows), "gpu_hours_expected": round(sum(r["gpu_hours_expected"] for r in rows), 1),
              "gpu_hours_max": round(sum(r["gpu_hours_max"] for r in rows), 1),
              "gpu_hours_training_only": round(sum(4 * r["steps"] * S_PER_STEP[r["preset"]] / 3600 for r in rows), 1),
              "eval_gpu_hours_expected": round(sum(e["estimate"]["gpu_hours_expected"] for e in evals.values()), 2),
              "eval_gpu_hours_max": round(sum(e["estimate"]["gpu_hours_max"] for e in evals.values()), 2),
              "eval_cpu_core_hours_max": round(sum(e["estimate"]["cpu_core_hours_max"] for e in evals.values()), 1)}
    return {"dry_run": dry, "batch": BATCH, "windows_per_epoch": wpe, "steps_at_d": steps_at_d, "warmup_steps": warmup,
            "manifest": {"hash": frac.get("hash"), "file_sha256": frac.get("file_sha256"), "file": frac.get("file")},
            "runs": rows, "blocked": blocked, "totals": totals, "plans": plans, "eval_plans": evals,
            "excluded_from_estimate": "collaudi; batch 64 e decay 10/20% fuori da questa griglia"}


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
    for name, e in result["eval_plans"].items():
        x = e["estimate"]
        lines.append(f"| {name} valutazione | {sum(n['kind'] == 'extract' for n in e['nodes'])} estrazioni + sonde + selezione | | | | | | | | | "
                     f"{x['gpu_hours_expected']} / {x['gpu_hours_max']} (CPU tetto {x['cpu_core_hours_max']} core-ora) |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build_report", type=Path)
    ap.add_argument("--decisions", type=Path)
    ap.add_argument("--campaign", type=Path, help="campi di campagna per wsd_chain.py (code_dir, expected_sha, manifest, ...); senza: dry-run")
    ap.add_argument("--out", type=Path, help="cartella dei piani (relativa: nel repo, cosi' il checkout della valutazione li contiene)")
    ap.add_argument("--failed-runs", nargs="*", default=[], help="run fermati prima della sonda: entrano nella selezione come inammissibili")
    args = ap.parse_args(argv)
    result = plan(json.loads(args.build_report.read_text()), json.loads(args.decisions.read_text()) if args.decisions else None,
                  json.loads(args.campaign.read_text()) if args.campaign else None, str(args.out) if args.out else None,
                  frozenset(args.failed_runs))
    print(table(result))
    print(f"\npassi a D (4 epoche) {result['steps_at_d']}, warmup {result['warmup_steps']}; solo training "
          f"{result['totals']['gpu_hours_training_only']} GPU-ora; {'DRY-RUN (percorsi segnaposto)' if result['dry_run'] else 'campagna'}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, p in result["plans"].items():
            (args.out / f"calibrazione_{name}.json").write_text(json.dumps(p, indent=1))
        for name, e in result["eval_plans"].items():
            (args.out / f"valutazione_{name}.json").write_text(json.dumps(e, indent=1))
        (args.out / "calibrazione_riepilogo.json").write_text(json.dumps({k: v for k, v in result.items() if k not in ("plans", "eval_plans")},
                                                                         indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
