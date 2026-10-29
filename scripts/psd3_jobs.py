#!/usr/bin/env python3
"""Piano dei job e file dei comandi per PSD-3 (docs/psd3_readers_20261029.md; esecuzione affidata a Claude, prompt GPT
docs/prompt_claude_psd3_dopo_sonde_20261029.md): un job per cella della matrice, dal checkout di campagna congelato, tramite il submitter
comune (`scripts/submit_job.py`) e il launcher `scripts/slurm/psd3_readout.sbatch`. Nessun job viene sottomesso da qui: il file dei comandi lo
esegue semplicione, blocco per blocco.

Blocchi (al piu' 10 job per blocco, limite della coda seriale):
  1  inventario con verifica SHA delle dieci cache (1 job CPU)
  2  i quattro benchmark train-only previsti dal protocollo: B x {DB6, EPN612} x {pool, dec} (tempi e memoria per fissare le risorse degli shard)
  3  i venti shard modello x dataset x vista (due blocchi da dieci), con le risorse fissate dopo il benchmark (--shard-*)
  4  il confronto sui venti shard (dopo che sono tutti COMPLETED 0:0)

  python3 scripts/psd3_jobs.py --code-sha <sha a 40 caratteri> [--shard-cpus 4 --shard-mem 32G --shard-time 01:00:00] [--out-dir ...]
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = "psd3-frozen-readers-20261029-v1"
MODELS = ("B", "R3c")
DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6", "epn612", "uci_emg")
VIEWS = ("pool", "dec")
BENCHMARKS = tuple(("B", d, v) for d in ("ninapro_db6", "epn612") for v in VIEWS)
MAX_QUEUED = 10
SERIAL_WALLTIME = "04:00:00"
PLAN_REL = "results/infra/psd3_preparation_20261029/plan.json"


def _hours(hms: str) -> float:
    h, m, s = (int(v) for v in hms.split(":"))
    return h + m / 60 + s / 3600


def build_jobs(code_sha: str, campaign: str, results_root: str, agent: str, ledger_campaign: str, *, shard_cpus: int, shard_mem: str,
               shard_time: str, benchmark_cpus: int, benchmark_mem: str, benchmark_time: str, inventory_time: str, compare_time: str,
               date: str = "20261029") -> dict:
    if len(code_sha) != 40 or any(c not in "0123456789abcdef" for c in code_sha):
        raise ValueError("code_sha deve essere uno SHA a 40 caratteri")
    if _hours(shard_time) > _hours(SERIAL_WALLTIME) or _hours(benchmark_time) > _hours(SERIAL_WALLTIME):
        raise ValueError(f"walltime oltre il limite della seriale {SERIAL_WALLTIME}")
    if shard_cpus != benchmark_cpus:
        raise ValueError("benchmark e shard devono usare gli stessi core: la firma numerica registra i thread")
    checkout = f"/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/{campaign}"
    launcher = f"{checkout}/scripts/slurm/psd3_readout.sbatch"
    export = f"--export=ALL,CODE_DIR={checkout},EXPECTED_SHA={code_sha}"
    jobs: list[dict] = []

    def job(block: int, ident: str, cpus: int, mem: str, time_: str, out: str, stage: str, args: list[str], expected_min: float | None = None) -> dict:
        argv = ["python3", "scripts/submit_job.py", "--id", ident, "--agente", agent, "--campagna", ledger_campaign, "--gpu-ore", "0",
                "--", export, f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={time_}", launcher, out, stage, *args]
        j = {"block": block, "id": ident, "stage": stage, "cpus": cpus, "mem": mem, "time": time_, "core_hours_cap": cpus * _hours(time_),
             "expected_minutes": expected_min, "out": out, "depends_on": [], "command": shlex.join(argv)}
        jobs.append(j)
        return j

    job(1, f"PILOT-PSD3-inventory-{date}", 2, "8G", inventory_time, f"{results_root}/inventory.json", "inventory", ["--verify-sha"], expected_min=5)
    for model, dataset, view in BENCHMARKS:
        job(2, f"PILOT-PSD3-benchmark-{model}-{dataset}-{view}-{date}", benchmark_cpus, benchmark_mem, benchmark_time,
            f"{results_root}/benchmark/{model}_{dataset}_{view}.json", "benchmark", ["--model", model, "--dataset", dataset, "--view", view],
            expected_min=10)
    cells = [(m, d, v) for m in MODELS for d in DATASETS for v in VIEWS]
    for i, (model, dataset, view) in enumerate(cells):
        job(3 + i // MAX_QUEUED, f"PILOT-PSD3-run-{model}-{dataset}-{view}-{date}", shard_cpus, shard_mem, shard_time,
            f"{results_root}/shards/{model}_{dataset}_{view}.json", "run", ["--model", model, "--dataset", dataset, "--view", view])
    shard_outs = [j["out"] for j in jobs if j["stage"] == "run"]
    job(5, f"PILOT-PSD3-compare-{date}", 4, "16G", compare_time, f"{results_root}/compare.json", "compare", ["--reports", *shard_outs], expected_min=5)
    titles = {1: "inventario delle dieci cache con verifica SHA completa", 2: "quattro benchmark train-only (B x DB6/EPN612 x pool/dec)",
              3: "shard 1-10 della matrice", 4: "shard 11-20 della matrice", 5: "confronto sui venti shard"}
    blocks = []
    for b in sorted(titles):
        in_block = [j for j in jobs if j["block"] == b]
        blocks.append({"block": b, "title": titles[b], "n_jobs": len(in_block), "core_hours_cap": sum(j["core_hours_cap"] for j in in_block),
                       "submit_after": "subito" if b == 1 else "il blocco precedente e' interamente COMPLETED 0:0 (leonardo_watch.py --mode all)"
                       + ("; prima del blocco 3 pubblicare le risorse degli shard fissate dal benchmark" if b == 3 else "")})
        if len(in_block) > MAX_QUEUED:
            raise ValueError(f"blocco {b} con {len(in_block)} job: oltre il limite della coda seriale")
    return {"schema_version": 1, "protocol": PROTOCOL, "status": "PRONTO_PER_SEMPLICIONE_DOPO_IL_CONGELAMENTO", "analysis_code_sha": code_sha,
            "campaign_checkout": checkout, "plan": f"{checkout}/{PLAN_REL}", "results_root": results_root, "ledger_campaign": ledger_campaign,
            "ledger_agent": agent, "gpu_hours_cap": 0,
            "shard_resources": {"cpus": shard_cpus, "mem": shard_mem, "time": shard_time,
                                "note": "da fissare dopo il benchmark (blocco 2); stessi core per tutti gli shard e per i benchmark"},
            "core_hours_cap": {f"block_{b['block']}": b["core_hours_cap"] for b in blocks} | {"total": sum(b["core_hours_cap"] for b in blocks)},
            "serial_partition": {"walltime_limit": SERIAL_WALLTIME, "max_queued_per_user": MAX_QUEUED},
            "automatic_retry": False, "blocks": blocks, "jobs": jobs}


def write_outputs(plan: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path, cmd_path = out_dir / "piano_job.json", out_dir / "comandi.txt"
    plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    lines = [f"# PSD-3 ({PROTOCOL}): comandi per semplicione, dal Mac, UNA riga per comando, nell'ordine. Codice {plan['analysis_code_sha']} nel checkout",
             f"# {plan['campaign_checkout']} (da creare prima con campaign_checkout.sbatch). Ogni blocco quando il precedente e' interamente COMPLETED 0:0;",
             "# le risorse degli shard (blocchi 3-4) si fissano dopo il benchmark e il file si rigenera. Nessun retry automatico: un job fallito si porta a Claude.", ""]
    for b in plan["blocks"]:
        lines.append(f"# ---- blocco {b['block']}: {b['title']} ({b['n_jobs']} job, tetto {b['core_hours_cap']:.2f} core-ore; sottomettere: {b['submit_after']})")
        lines += [j["command"] for j in plan["jobs"] if j["block"] == b["block"]]
        lines.append("")
    cmd_path.write_text("\n".join(lines))
    return plan_path, cmd_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-sha", required=True)
    ap.add_argument("--campaign", default="pilot-psd3-20261029")
    ap.add_argument("--results-root", default="/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/psd3_20261029")
    ap.add_argument("--agente", default="gpt", choices=["claude", "gpt", "simone"], help="categoria del ledger (l'operatore e' semplicione)")
    ap.add_argument("--ledger-campaign", default="pilot-psd3-20261029")
    ap.add_argument("--shard-cpus", type=int, default=4)
    ap.add_argument("--shard-mem", default="32G")
    ap.add_argument("--shard-time", default="01:00:00")
    ap.add_argument("--benchmark-cpus", type=int, default=4)
    ap.add_argument("--benchmark-mem", default="32G")
    ap.add_argument("--benchmark-time", default="00:30:00")
    ap.add_argument("--inventory-time", default="00:30:00")
    ap.add_argument("--compare-time", default="00:30:00")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "infra" / "psd3_20261029")
    a = ap.parse_args(argv)
    plan = build_jobs(a.code_sha, a.campaign, a.results_root, a.agente, a.ledger_campaign, shard_cpus=a.shard_cpus, shard_mem=a.shard_mem,
                      shard_time=a.shard_time, benchmark_cpus=a.benchmark_cpus, benchmark_mem=a.benchmark_mem, benchmark_time=a.benchmark_time,
                      inventory_time=a.inventory_time, compare_time=a.compare_time)
    plan_path, cmd_path = write_outputs(plan, a.out_dir)
    print(json.dumps({"complete": True, "jobs": len(plan["jobs"]), "core_hours_cap": plan["core_hours_cap"], "plan": str(plan_path),
                      "commands": str(cmd_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
