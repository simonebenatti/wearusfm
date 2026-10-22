"""Catene WSD della finestra 1 (15/10/2026, Simone: «sì, vai coi launcher delle catene WSD»): da un piano JSON ai comandi di
`scripts/submit_job.py`, con il preventivo. Non sottomette nulla: i comandi li esegue leonardo-ops, nell'ordine, dopo l'autorizzazione.

Ogni run del piano e' una catena di job su un nodo (DDP a 4 GPU, `scripts/slurm/train_ddp.sbatch`) collegati con `--dopo` (afterok): ogni
job riprende dal checkpoint della cartella del run e si ferma da solo `margin_min` minuti prima della fine del job (`TIME_LIMIT_S`); un
allarme o un errore numerico scrivono STOP e i job successivi escono subito; un run completo fa uscire subito i job di scorta. Un ramo
(`branch_of`, D14: rami di decay del pilot sulle epoche, continuazione a 2D) parte dal checkpoint con nome della fase stabile del tronco
(`checkpoint_step<N>.pt`), o dal pre-decay (`checkpoint_predecay.pt`) per la continuazione a 2D quando parte dall'inizio del decay del
tronco, e dipende dal job del tronco che lo ha sicuramente scritto, con la stima pessimista dei passi per job
(s/passo x `slack`); se il checkpoint non c'e' ancora il primo job del ramo esce con 3 e la catena del ramo si ferma.

Le regole di D14 si controllano col codice del trainer prima di generare i comandi (`with_wsd`, `wsd_warmup_steps`, `_check_wsd`, fase
stabile del tronco e del ramo): un piano incoerente non produce comandi.

Piano (JSON):
  campaign, agent, code_dir, expected_sha (40 caratteri), manifest, scales, window_seconds, run_root: percorsi sul cluster;
  slurm: {"qos": "normal", "time": "24:00:00"}; margin_min (margine per il salvataggio finale), startup_min (avvio prima del primo passo),
  slack (>= 1, sui secondi per passo), spare_jobs (job di scorta per run), gpus (per job, 4);
  s_per_step: {"<preset>/<batch>": secondi per aggiornamento} misurati (DDP a 4 GPU, prefetch);
  ckpt_gb: {"<preset>": GB di un checkpoint con optimizer};
  common_args: argomenti di train_jepa.py comuni a tutti i run (senza virgole: sbatch --export le usa come separatore);
  runs: lista di
    {"id", "preset", "seed", "batch_size", "accum_steps", "steps_at_d", "total_steps", "keep_checkpoints": [...], "args": ""}  tronco
    {"id", "branch_of": "<id>", "from_step": N, "total_steps"}                                                               ramo
  Un ramo eredita dal tronco preset, seme, batch, accumulo, passi a D e argomenti (il trainer accetta un ramo solo se cambiano schedule e
  max_steps).

Uso: python scripts/wsd_chain.py piano.json [--commands comandi.txt] [--json preventivo.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUNK_KEYS = {"id", "preset", "seed", "batch_size", "accum_steps", "steps_at_d", "total_steps", "keep_checkpoints", "args"}
BRANCH_KEYS = {"id", "branch_of", "from_step", "total_steps"}
# opzioni del trainer che impostano il pianificatore (D14, batch, semi) o il launcher (preset, dati, uscite, processi): negli argomenti liberi
# falserebbero i controlli o il run pianificato. Si confrontano le destinazioni di argparse, dopo aver risolto le abbreviazioni come lui.
PLANNER_DESTS = ("schedule", "total_steps", "steps_at_d", "warmup_steps", "decay_fraction", "keep_checkpoints", "max_steps", "batch_size",
                 "accum_steps", "seed", "window_seconds", "time_limit_s", "branch_from")
LAUNCHER_DESTS = ("preset", "manifest", "root", "scales", "out_dir", "num_workers", "distributed")
PLAN_KEYS = {"campaign", "agent", "code_dir", "expected_sha", "manifest", "scales", "window_seconds", "run_root", "slurm", "margin_min",
             "startup_min", "slack", "spare_jobs", "gpus", "s_per_step", "ckpt_gb", "common_args", "runs"}


def _seconds(hms: str) -> int:
    parts = [int(x) for x in hms.split(":")]
    if len(parts) != 3:
        raise ValueError(f"slurm.time {hms!r}: serve HH:MM:SS")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _schedule(preset: str, total_steps: int, steps_at_d: int, keep: tuple[int, ...]):
    """Lo schedule col codice del trainer: stessi controlli di `train_jepa.py --schedule wsd`."""
    from wearusfm.training import run as R

    base = R.sanity_config() if preset == "sanity" else R.ladder_config(preset)
    return R.with_wsd(base, total_steps, warmup_steps=R.wsd_warmup_steps(steps_at_d), keep_checkpoints=keep).schedule


def _trainer_parser():
    spec = importlib.util.spec_from_file_location("train_jepa_options", ROOT / "scripts" / "train_jepa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_parser()


def free_arg_dests(text: str, parser) -> set[str]:
    """Le destinazioni del trainer toccate dagli argomenti liberi `text`, spezzati come li spezza la shell del launcher (`${EXTRA_ARGS}`),
    con le abbreviazioni delle opzioni lunghe risolte come argparse (`--batch` -> `--batch-size`). Opzioni sconosciute, ambigue o brevi e
    `--` sono errori: argparse le rifiuterebbe nel job o le interpreterebbe in modo diverso."""
    options = {o: a.dest for a in parser._actions for o in a.option_strings}
    dests = set()
    for tok in text.split():
        if not tok.startswith("-") or re.fullmatch(r"-\.?\d.*", tok):
            continue  # valore (anche un numero negativo)
        name = tok.split("=", 1)[0]
        if name in options:
            dests.add(options[name])
            continue
        hits = sorted(o for o in options if name.startswith("--") and len(name) > 2 and o.startswith(name))
        if len(hits) != 1:
            raise ValueError(f"opzione {tok!r}: " + (f"ambigua fra {hits}" if hits else "sconosciuta al trainer"))
        dests.add(options[hits[0]])
    return dests


def _warmup(steps_at_d: int) -> int:
    from wearusfm.training import run as R

    return R.wsd_warmup_steps(steps_at_d)


def resolve(plan: dict) -> list[dict]:
    """I run del piano coi campi del tronco ereditati dai rami e lo schedule controllato; i tronchi prima dei loro rami."""
    unknown = set(plan) - PLAN_KEYS
    if unknown:
        raise ValueError(f"piano: campi sconosciuti {sorted(unknown)}")
    if len(plan["expected_sha"]) != 40:
        raise ValueError("expected_sha: serve lo SHA completo (40 caratteri)")
    if plan["slack"] < 1 or plan["spare_jobs"] < 0 or plan["margin_min"] < 0 or plan["startup_min"] < 0:
        raise ValueError("slack >= 1, spare_jobs, margin_min e startup_min >= 0")
    parser = _trainer_parser()
    for where, text in [("common_args", plan["common_args"])] + [(r["id"], r.get("args", "")) for r in plan["runs"]]:
        try:
            bad = sorted(free_arg_dests(text, parser) & set(PLANNER_DESTS + LAUNCHER_DESTS))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        if bad:
            raise ValueError(f"{where}: {bad} li impostano il pianificatore (schedule, batch, semi) o il launcher (preset, dati, uscite): "
                             "non negli argomenti liberi")
    by_id, out = {}, []
    for r in plan["runs"]:
        if r["id"] in by_id:
            raise ValueError(f"run {r['id']} ripetuto")
        if "branch_of" in r:
            if set(r) != BRANCH_KEYS:
                raise ValueError(f"ramo {r['id']}: campi {sorted(set(r) ^ BRANCH_KEYS)} (un ramo cambia solo from_step e total_steps)")
            if r["branch_of"] not in by_id or "branch_of" in by_id[r["branch_of"]]:
                raise ValueError(f"ramo {r['id']}: il tronco {r['branch_of']} deve essere un tronco elencato prima")
            trunk = by_id[r["branch_of"]]
            run = {**{k: trunk[k] for k in TRUNK_KEYS - {"id", "total_steps", "keep_checkpoints"}}, **r, "keep_checkpoints": []}
            if r["from_step"] not in trunk["keep_checkpoints"] and r["from_step"] != trunk["schedule"].decay_start:
                raise ValueError(f"ramo {r['id']}: il passo {r['from_step']} non e' fra i keep_checkpoints del tronco {trunk['id']} ne' il suo "
                                 f"pre-decay ({trunk['schedule'].decay_start})")
            if r["from_step"] > trunk["schedule"].decay_start:
                raise ValueError(f"ramo {r['id']}: il passo {r['from_step']} e' nel decay del tronco (da {trunk['schedule'].decay_start})")
        else:
            if set(r) != TRUNK_KEYS:
                raise ValueError(f"run {r['id']}: campi {sorted(set(r) ^ TRUNK_KEYS)}")
            run = dict(r)
        run["schedule"] = _schedule(run["preset"], run["total_steps"], run["steps_at_d"], tuple(run["keep_checkpoints"]))
        run["warmup_steps"] = _warmup(run["steps_at_d"])
        if "branch_of" in run and run["from_step"] > run["schedule"].decay_start:
            raise ValueError(f"ramo {run['id']}: il decay del ramo comincia (passo {run['schedule'].decay_start}) prima del passo "
                             f"{run['from_step']} da cui parte")
        if f"{run['preset']}/{run['batch_size']}" not in plan["s_per_step"]:
            raise ValueError(f"run {run['id']}: manca s_per_step[{run['preset']}/{run['batch_size']}] (misurato, non stimato)")
        by_id[run["id"]] = run
        out.append(run)
    return out


def extra_args(plan: dict, run: dict) -> str:
    """EXTRA_ARGS di `train_ddp.sbatch` per un run risolto (`resolve`): la stessa stringa che arriva a `train_jepa.py` nel job, usata anche
    dalla selezione della calibrazione per ricostruire la configurazione attesa (`scripts/calibration_select.py`)."""
    extra = (f"{plan['common_args']} --window-seconds {plan['window_seconds']} --batch-size {run['batch_size']} "
             f"--accum-steps {run['accum_steps']} --seed {run['seed']} --schedule wsd --total-steps {run['total_steps']} "
             f"--steps-at-d {run['steps_at_d']}")
    if run["keep_checkpoints"]:
        extra += " --keep-checkpoints " + ":".join(str(x) for x in sorted(run["keep_checkpoints"]))
    if run["args"]:
        extra += " " + run["args"]
    return extra.strip()


def plan_jobs(plan: dict) -> dict:
    """Job, dipendenze, comandi e preventivo."""
    runs = resolve(plan)
    wall = _seconds(plan["slurm"]["time"])
    limit = wall - 60 * plan["margin_min"]
    usable = limit - 60 * plan["startup_min"]
    if usable <= 0:
        raise ValueError("margin_min + startup_min oltre la durata del job")
    gpus, out_runs, commands = plan["gpus"], [], []
    jobs_of: dict[str, list[str]] = {}
    for run in runs:
        sps = plan["s_per_step"][f"{run['preset']}/{run['batch_size']}"]
        per_job = usable / (sps * plan["slack"])  # passi per job, stima pessimista
        if per_job < 1:
            raise ValueError(f"run {run['id']}: meno di un passo per job")
        start = run.get("from_step", 0)
        todo = run["total_steps"] - start
        needed = math.ceil(todo / per_job)
        n = needed + plan["spare_jobs"]
        # un run che sta in un job (i rami del pilot) chiede solo il tempo che gli serve, con la stima pessimista, arrotondato al quarto d'ora
        run_wall = wall if needed > 1 else min(wall, 900 * math.ceil((todo * sps * plan["slack"] + 60 * (plan["startup_min"] +
                                                                                                         plan["margin_min"])) / 900))
        run_limit = run_wall - 60 * plan["margin_min"]
        ids = [f"{run['id']}-j{k:02d}" for k in range(1, n + 1)]
        jobs_of[run["id"]] = ids
        run_dir = f"{plan['run_root']}/{run['id']}"
        after = None
        branch_from = None
        if "branch_of" in run:
            trunk = next(r for r in out_runs if r["id"] == run["branch_of"])
            k = min(max(1, math.ceil(run["from_step"] / trunk["steps_per_job_pessimistic"])), trunk["jobs_needed"])
            after = jobs_of[trunk["id"]][k - 1]
            name = f"checkpoint_step{run['from_step']:07d}.pt" if run["from_step"] in trunk["keep_checkpoints"] else "checkpoint_predecay.pt"
            branch_from = f"{plan['run_root']}/{trunk['id']}/{name}"  # continuazione a 2D (D14): dal pre-decay
        extra = extra_args(plan, run)
        exports = [f"CODE_DIR={plan['code_dir']}", f"EXPECTED_SHA={plan['expected_sha']}", f"PRESET={run['preset']}",
                   f"TIME_LIMIT_S={run_limit}"] + ([f"BRANCH_FROM={branch_from}"] if branch_from else []) + [f"EXTRA_ARGS={extra.strip()}"]
        for item in exports:
            if "," in item:
                raise ValueError(f"run {run['id']}: virgola in {item.split('=')[0]} (sbatch --export la prende come separatore)")
        export = "--export=ALL," + ",".join(exports)
        for i, jid in enumerate(ids):
            dep = after if i == 0 else ids[i - 1]
            cmd = ["python3", "scripts/submit_job.py", "--id", jid, "--agente", plan["agent"], "--campagna", plan["campaign"],
                   "--gpu-ore", f"{gpus * run_wall / 3600:g}"] + (["--dopo", dep] if dep else []) + \
                  ["--", export, f"--qos={plan['slurm']['qos']}", f"--time={_hms(run_wall)}",  # nome e commento li impone submit_job.py
                   f"{plan['code_dir']}/scripts/slurm/train_ddp.sbatch", plan["manifest"], plan["scales"], run_dir, str(run["total_steps"])]
            commands.append(" ".join(shlex.quote(c) for c in cmd))
        s = run["schedule"]
        copies = 1 + len(run["keep_checkpoints"]) + 2  # checkpoint.pt, fase stabile, pre e post decay
        out_runs.append({
            "id": run["id"], "branch_of": run.get("branch_of"), "from_step": run.get("from_step"), "after": after, "preset": run["preset"],
            "seed": run["seed"], "batch_size": run["batch_size"], "accum_steps": run["accum_steps"], "steps_at_d": run["steps_at_d"],
            "warmup_steps": run["warmup_steps"], "total_steps": run["total_steps"], "decay_start": s.decay_start, "keep_checkpoints": run["keep_checkpoints"],
            "s_per_step": sps, "steps_per_job_pessimistic": per_job, "jobs_needed": needed, "jobs": n, "job_ids": ids,
            "gpu_hours_expected": round(gpus * (todo * sps + needed * 60 * plan["startup_min"]) / 3600, 1),
            "gpu_hours_max": round(gpus * n * run_wall / 3600, 1), "time": _hms(run_wall), "time_limit_s": run_limit,
            "checkpoint_gb": round(copies * plan["ckpt_gb"][run["preset"]], 1), "run_dir": run_dir, "branch_from": branch_from})
    return {"campaign": plan["campaign"], "runs": out_runs, "commands": commands,
            "totals": {"jobs": sum(r["jobs"] for r in out_runs),
                       "gpu_hours_expected": round(sum(r["gpu_hours_expected"] for r in out_runs), 1),
                       "gpu_hours_max": round(sum(r["gpu_hours_max"] for r in out_runs), 1),
                       "checkpoint_gb": round(sum(r["checkpoint_gb"] for r in out_runs), 1)}}


def table(result: dict) -> str:
    rows = ["| Run | Da | Passi | Decay da | Job (+scorta) x durata | Dopo | GPU-ora attese / tetto | Checkpoint GB |",
            "|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        src = f"{r['branch_of']} @ {r['from_step']}" if r["branch_of"] else "—"
        rows.append(f"| {r['id']} | {src} | {r['total_steps']} | {r['decay_start']} | {r['jobs_needed']} (+{r['jobs'] - r['jobs_needed']}) x {r['time']} | "
                    f"{r['after'] or '—'} | {r['gpu_hours_expected']} / {r['gpu_hours_max']} | {r['checkpoint_gb']} |")
    t = result["totals"]
    rows.append(f"| **Totale** | | | | {t['jobs']} | | {t['gpu_hours_expected']} / {t['gpu_hours_max']} | {t['checkpoint_gb']} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan", type=Path)
    ap.add_argument("--commands", type=Path, help="file in cui scrivere i comandi di submit_job.py, uno per riga, nell'ordine")
    ap.add_argument("--json", type=Path, help="file in cui scrivere job, dipendenze e preventivo")
    args = ap.parse_args(argv)
    result = plan_jobs(json.loads(args.plan.read_text()))
    print(table(result))
    if args.commands:
        args.commands.write_text("\n".join(result["commands"]) + "\n")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
