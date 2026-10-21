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

LR (rilievo 2 del critico, 21/10): si deriva con la regola firmata da `w1cal_decisioni.json` e dal registro di C2 (`signed_lr`); un `--lr`
diverso e' un errore. La spec del confronto porta piano di training (percorso relativo al checkout e SHA-256), LR e ID dei run, perche'
`arms_compare.py` verifichi l'identita' di ogni run e la provenienza di ogni rapporto (rilievo 1). La cartella dei piani e' relativa al
repo (rilievo 4): il nodo del confronto la legge dal checkout della valutazione.

Uso: python scripts/arms_plan.py rapporto_cluster.json --decisions w1cal_decisioni.json --registro-c2 registro_C2.json --campaign campagna.json
     --out cartella_relativa [--lr LR] [--keep-calibration NPZ --keep-calibration-sha256 SHA] [--probes-on-gpu] [--failed-runs ID ...]
     [--frozen-plans cartella_dei_piani_lanciati]   (alla valutazione: i piani rigenerati devono esserle identici, la spec punta a quelli)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BATCH, PRESET, EPOCHS, SEEDS = 32, "sanity", 4.0, (0, 1)
OBJECTIVE_ARMS = {"A": "", "JEPA-only": "--anchor-weight 0", "ancore-only": "--jepa-weight 0"}
KEEP_ARMS = {"Aprime": "--keep-weight 0", "B": "--keep-weight 0.05"}  # con --keep-calibration
REFERENCE = {"objective": "A", "keep": "Aprime"}
P1_DS, P2_DS = ("ninapro_db2", "ninapro_db3", "ninapro_db6"), ("epn612", "uci_emg")
# nodi della valutazione: (launcher, tetto, attese in ore); estrazioni 1 GPU; sonde 8 core (seriale) o 1 GPU con --probes-on-gpu.
# Misure: P1 ufficiale 13:42 e sonde 22 min, P2 ufficiale 41:19 e sonde 33 min (docs/step1_downstream_ab.md); S della P1 17 min e 1 h 48
# (w1_selezione_v1); S della P2 stimata come meta' dell'estrazione della P2 e due volte la sua sonda (4 fold).
EVAL = {"p1": {"extract": ("probe_p1.sbatch", "00:45:00", 0.25), "probe": ("probe_p1_probe.sbatch", "02:00:00", 0.4)},
        "p2": {"extract": ("probe_p2.sbatch", "01:30:00", 0.7), "probe": ("probe_p2_probe.sbatch", "02:00:00", 0.6)},
        "s_p1": {"extract": ("calibration_extract.sbatch", "00:45:00", 0.3), "probe": ("calibration_probe.sbatch", "03:00:00", 1.8)},
        "s_p2": {"extract": ("calibration_extract.sbatch", "01:30:00", 0.5), "probe": ("calibration_probe.sbatch", "03:00:00", 1.2)}}
PROBE_CORES = 8
QOS_MAX_SECONDS = {"boost_qos_dbg": 1800}  # limiti di durata delle QoS che un nodo puo' ereditare dall'intestazione del launcher (R3, GPT 3272398)
GPU_PROBE_SBATCH = ["--partition=boost_usr_prod", "--qos=normal", "--gres=gpu:1"]
# calibrazione della keep dello Step 1 (solo emg2qwerty, decisioni.md): la firma (punto 4) chiede di ricalcolarla sul corpus intero
STEP1_KEEP_CALIBRATION_SHA256 = "4640dfcf2c6819fb8ba2870e1506f08e4b024cdf38dda267c4cd37652cbc1f89"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hours(hms: str) -> float:
    h, m, s = (int(x) for x in hms.split(":"))
    return h + m / 60 + s / 3600


def plan_text(p: dict) -> str:
    """Il testo del piano di training come lo scrive `main` (bracci_<gruppo>.json): il suo SHA-256 va nella spec del confronto."""
    return json.dumps(p, indent=1)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed_lr(decisions: dict, registro: dict) -> tuple[float, dict]:
    """LR di 30M per i bracci con la regola firmata il 16/10 (firma, punto 2): l'unico LR confermato da C2, o, se C2 ne conferma due, quello
    con S migliore al seme 1, a parita' il piu' basso. Parita' = quella della calibrazione (`rules.tie` del registro: differenza accoppiata
    entro 1 SE), la sola definita nei protocolli firmati. Il registro di C2 deve dare la stessa decisione del file delle decisioni."""
    confirmed = sorted(float(x) for x in decisions["C2"]["confirmed"])
    if registro.get("phase") != "C2" or sorted(float(x) for x in registro["decision"]["confirmed"]) != confirmed:
        raise ValueError(f"registro di C2 incoerente col file delle decisioni (C2 confermati {confirmed})")
    if len(confirmed) == 1:
        return confirmed[0], {"confirmed": confirmed, "why": "unico LR confermato da C2"}
    if len(confirmed) != 2:
        raise ValueError(f"C2 conferma {confirmed}: la regola firmata dei bracci copre uno o due LR")
    lo, hi = confirmed
    rows = [r for r in registro.get("descriptive_comparisons", []) if {float(r["a"]), float(r["b"])} == {lo, hi}]
    if len(rows) != 1:
        raise ValueError(f"registro di C2: serve un solo confronto fra {lo:g} e {hi:g}, trovati {len(rows)}")
    row = rows[0]
    s = {float(k): v["S"] for k, v in registro["candidates"].items()}
    delta_lo = row["delta"] if float(row["a"]) == lo else -row["delta"]  # S(lo) - S(hi)
    if not math.isclose(delta_lo, s[lo] - s[hi], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"registro di C2: delta {row['delta']} incoerente con le S {s}")
    tie = bool(row["tie"])
    choice = lo if tie or delta_lo > 0 else hi
    return choice, {"confirmed": confirmed, "S_seed1": {f"{k:g}": v for k, v in s.items()}, "delta_low_minus_high": delta_lo,
                    "se": row["se"], "tie": tie, "rule": registro["rules"]["tie"],
                    "why": "parita' (entro 1 SE): il piu' basso" if tie else "S migliore al seme 1"}


def run_id(arm: str, seed: int | str) -> str:
    return f"W1ARM-{arm}-30M-s{seed}"


def arm_runs(steps: int, lr: float, keep_calibration: str | None) -> dict[str, list[dict]]:
    """{gruppo: [run]} per wsd_chain: "objective" (A, JEPA-only, ancore-only) e, con la calibrazione della keep, "keep" (A', B).
    `arms_compare.py` la richiama per verificare che il piano di training sia esattamente quello dei bracci del gruppo."""
    groups = {"objective": OBJECTIVE_ARMS}
    if keep_calibration:
        groups["keep"] = {k: f"--keep-calibration {keep_calibration} {v}" for k, v in KEEP_ARMS.items()}
    out = {}
    for g, arms in groups.items():
        out[g] = [{"id": run_id(arm, seed), "preset": PRESET, "seed": seed, "batch_size": BATCH, "accum_steps": 1, "steps_at_d": steps,
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
               # QoS esplicita (R3 della review GPT 3272398): probe_p1.sbatch ha boost_qos_dbg nell'intestazione (30 min), l'estrazione ne chiede 45
               "launcher": f"scripts/slurm/{ext_l}", "args": ext_args, "exports": exports, "sbatch": [f"--time={ext_t}", "--qos=normal"],
               "inputs": [ckpt],
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


def check_qos_walltime(nodes: list[dict]) -> None:
    """Il tempo chiesto da ogni nodo entra nel limite della sua QoS EFFETTIVA (opzioni del nodo piu' intestazione del launcher allo SHA di
    deploy, `evaluation_chain.node_labels`): errore altrimenti (R3 della review GPT 3272398: estrazioni P1 da 45 minuti con boost_qos_dbg)."""
    EC = _mod("evaluation_chain")
    for n in nodes:
        t = EC.sbatch_options(n.get("sbatch", [])).get("--time")
        if t is None:
            continue
        secs = int(_hours(t) * 3600 + 0.5)
        for qos, limit in QOS_MAX_SECONDS.items():
            if qos in EC.node_labels(n) and secs > limit:
                raise ValueError(f"{n['id']}: --time={t} oltre il limite di {qos} ({limit} s), QoS effettiva da opzioni e intestazione")


def plan(build_report: dict, lr: float, campaign: dict, keep_calibration: str | None = None, probes_on_gpu: bool = False,
         failed_runs: frozenset = frozenset(), *, plans_dir: str, keep_calibration_sha256: str | None = None,
         frozen_plans: str | None = None) -> dict:
    from wearusfm.training import run as R

    for d in (plans_dir, frozen_plans) if frozen_plans is not None else (plans_dir,):
        if not d or Path(d).is_absolute() or ".." in Path(d).parts:
            raise ValueError(f"cartella dei piani {d!r}: relativa al repo, senza '..' (il confronto la legge dal checkout della valutazione)")
    if keep_calibration is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", keep_calibration_sha256 or ""):
            raise ValueError("--keep-calibration richiede --keep-calibration-sha256 (64 caratteri esadecimali) della ricalibrazione sul corpus intero")
        if keep_calibration_sha256 == STEP1_KEEP_CALIBRATION_SHA256:
            raise ValueError("calibrazione della keep dello Step 1 (solo emg2qwerty): la firma chiede quella ricalcolata sul corpus intero")
    elif keep_calibration_sha256 is not None:
        raise ValueError("--keep-calibration-sha256 senza --keep-calibration")

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
            entry = {"id": run["id"], "run_dir": run["run_dir"]}
            if run["id"] not in failed_runs:
                n, reps = eval_nodes(run, code, ev, sbatch_probe)
                nodes += n
                probes += [x["id"] for x in n if x["kind"] == "probe"]
                entry.update(reps)
            spec_runs.setdefault(meta["arm"], {})[str(run["seed"])] = entry
        ref = REFERENCE[g]
        plan_path = f"{frozen_plans or plans_dir}/bracci_{g}.json"
        if frozen_plans is not None:  # rilievo 4 della re-review: la valutazione si lega al piano LANCIATO, non a una sua rigenerazione
            frozen = ROOT / plan_path
            if not frozen.is_file() or frozen.read_text() != plan_text(p):
                raise ValueError(f"{g}: il piano rigenerato non e' identico al piano congelato {plan_path}")
        spec = {"group": g, "reference": ref, "arms": [a for a in spec_runs if a != ref], "plan": plan_path,
                "plan_sha256": hashlib.sha256(plan_text(p).encode()).hexdigest(), "lr": lr, "runs": spec_runs}
        if g == "keep":
            spec.update(keep_calibration=keep_calibration, keep_calibration_sha256=keep_calibration_sha256)
        if any(set(failed_runs) & {run_id(ref, s) for s in SEEDS}):
            raise ValueError(f"{g}: il riferimento {ref} non e' arrivato alla valutazione: nessun confronto")
        spec_path = f"{plans_dir}/confronto_{g}.json"
        specs[g] = spec
        out = f"{ev['results_root']}/confronto_{g}.json"
        reports = [v for e in spec_runs.values() for x in e.values() for k, v in x.items() if k not in ("id", "run_dir")]
        oofs = [r[:-5] + ".oof.npz" for r in reports if r.split("/")[-1].startswith("s_")]
        nodes.append({"id": f"W1ARM-{g}-compare", "kind": "compare", "after": probes, "queue": "lrd_all_serial", "gpu_hours": 0, **code,
                      "launcher": "scripts/slurm/arms_compare.sbatch", "args": [spec_path, out], "exports": {}, "sbatch": ["--time=00:30:00"],
                      "inputs": reports + oofs, "outputs": [out]})
        n_eval = sum(1 for r in runs if r["id"] not in failed_runs)
        ext_exp = sum(EVAL[k]["extract"][2] for k in EVAL)
        ext_max = sum(_hours(EVAL[k]["extract"][1]) for k in EVAL)
        prb_exp = sum(EVAL[k]["probe"][2] for k in EVAL)
        prb_max = sum(_hours(EVAL[k]["probe"][1]) for k in EVAL)
        check_qos_walltime(nodes)
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
    ap.add_argument("--decisions", type=Path, required=True, help="results/infra/w1cal_decisioni.json")
    ap.add_argument("--registro-c2", type=Path, required=True, help="registro della selezione di C2 (registro_C2.json)")
    ap.add_argument("--lr", type=float, default=None, help="facoltativo: se dato, deve essere quello della regola firmata")
    ap.add_argument("--campaign", type=Path, required=True, help="come quella della calibrazione, con eval.raw_root per la P2")
    ap.add_argument("--keep-calibration", default=None, help="calibrazione della keep sul corpus intero (percorso sul cluster)")
    ap.add_argument("--keep-calibration-sha256", default=None, help="SHA-256 atteso della calibrazione della keep sul corpus intero")
    ap.add_argument("--probes-on-gpu", action="store_true")
    ap.add_argument("--failed-runs", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True, help="cartella dei piani, relativa al repo: il checkout della valutazione li contiene")
    ap.add_argument("--frozen-plans", default=None, help="alla valutazione: cartella (relativa al repo) coi piani di training LANCIATI; "
                    "i piani rigenerati devono esserle identici e la spec punta a quelli")
    args = ap.parse_args(argv)
    lr, why = signed_lr(json.loads(args.decisions.read_text()), json.loads(args.registro_c2.read_text()))
    if args.lr is not None and args.lr != lr:
        raise SystemExit(f"--lr {args.lr:g}: la regola firmata da C2 da' {lr:g} ({why['why']})")
    res = plan(json.loads(args.build_report.read_text()), lr, json.loads(args.campaign.read_text()), args.keep_calibration,
               args.probes_on_gpu, frozenset(args.failed_runs), plans_dir=str(args.out), keep_calibration_sha256=args.keep_calibration_sha256,
               frozen_plans=args.frozen_plans)
    res["lr_rule"] = {**why, "decisions": str(args.decisions), "decisions_sha256": _sha256_file(args.decisions),
                      "registro_c2": str(args.registro_c2), "registro_c2_sha256": _sha256_file(args.registro_c2)}
    for r in res["runs"]:
        print(f"{r['group']:9s} {r['id']:28s} seme {r['seed']} passi {r['steps']} decay da {r['decay_start']} {r['jobs']} x {r['job_time']} "
              f"GPU-ora {r['gpu_hours_expected']} / {r['gpu_hours_max']}")
    t = res["totals"]
    print(f"LR {lr:g} ({why['why']}); training {t['gpu_hours_expected']} / {t['gpu_hours_max']} GPU-ora; valutazione "
          f"{t['eval_gpu_hours_expected']} / {t['eval_gpu_hours_max']}; bloccati: {res['blocked'] or 'nessuno'}")
    args.out.mkdir(parents=True, exist_ok=True)
    for g, p in res["plans"].items():
        if args.frozen_plans is None:  # con i piani congelati la spec punta a quelli: nessuna copia
            (args.out / f"bracci_{g}.json").write_text(plan_text(p))
        (args.out / f"valutazione_{g}.json").write_text(json.dumps(res["eval_plans"][g], indent=1))
        (args.out / f"confronto_{g}.json").write_text(json.dumps(res["specs"][g], indent=1))
    (args.out / "bracci_riepilogo.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("plans", "eval_plans", "specs")},
                                                               indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
