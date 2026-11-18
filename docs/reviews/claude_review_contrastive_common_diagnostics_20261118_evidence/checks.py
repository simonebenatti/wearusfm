"""Controlli riproducibili della review Claude sulla consegna GPT 365cb1e (diagnostici da stato comune).

Uso, da un checkout git che contenga i due commit di GPT:
    python3 checks.py
Legge solo oggetti git (`git show`), non importa codice dei commit e non contatta Leonardo.
"""
import hashlib
import json
import subprocess

FROZEN = "365cb1e29bc410013d4ee0494b9c2e92835bc7f1"
PLAN_COMMIT = "35ea94ba"
PACKAGE = "results/infra/contrastive_comparison_20261116/common_state_diagnostics_20261118"
COHORT = "results/infra/contrastive_comparison_20261116/calibrated_cohort_20261118_bf16"
RELEASE = "results/infra/contrastive_comparison_20261116/releases/bf16_20261118"
START, FORK, END = 576, 580, 588


def show(rev, path):
    return subprocess.check_output(["git", "show", f"{rev}:{path}"])


def load(rev, path):
    return json.loads(show(rev, path))


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, prefix + k + ".") if isinstance(v, dict) else {prefix + k: v})
    return out


plan = load(PLAN_COMMIT, f"{PACKAGE}/submission_plan.json")
resources = load(FROZEN, f"{PACKAGE}/resources.json")
print("1. piano di invio: frozen_commit == 365cb1e:", plan["frozen_commit"] == FROZEN)
for name, digest in plan["local_transport_and_driver_sha256"].items():
    print("   pin", "OK " if hashlib.sha256(show(FROZEN, name)).hexdigest() == digest else "BAD", name)
submitter = hashlib.sha256(show(FROZEN, "scripts/submit_job.py")).hexdigest()
print("   submitter 365cb1e == piano == resources:",
      submitter == plan["submitter_sha256"] == resources["shared_submitter_sha256"], submitter[:16])

cohort = load(FROZEN, f"{COHORT}/cohort.json")
configs = {arm: flat(load(FROZEN, f"{COHORT}/{spec['path']}")) for arm, spec in cohort["configs"].items()}
keys = set().union(*configs.values())
differ = sorted(k for k in keys if len({json.dumps(c.get(k), sort_keys=True) for c in configs.values()}) > 1)
print("2. ricette C0/CN/CR della coorte: campi diversi =", differ)

for arm in ("CN", "CR"):
    job = load(FROZEN, f"{RELEASE}/full_{arm}.job.json")
    cfg = load(FROZEN, job["config"]["path"])
    s, c = cfg["schedule"], cfg["contrastive"]
    decay_start = s["total_steps"] - int(round(s["decay_fraction"] * s["total_steps"]))
    saves = [n for n in range(START + 1, END + 1) if n % cfg["ckpt_every"] == 0 or n == FORK]
    evals = [n for n in range(START + 1, END + 1) if n % cfg["eval_every"] == 0 or n == END]
    factor = lambda step: min(1., max(0., (step + 1 - c["fill_steps"]) / c["ramp_steps"]))
    print(f"3. {arm}: max_steps {cfg['max_steps']}, schedule {s['kind']}, decay_start {decay_start}, "
          f"{FORK} valido come keep: {s['kind'] == 'wsd' and FORK not in s['keep_checkpoints'] and START < FORK <= decay_start}, "
          f"{END} <= total_steps: {END <= s['total_steps']}")
    print(f"   salvataggi in {START + 1}..{END}: {saves} + finale; valutazioni: {evals}; "
          f"rapporti di copertura (blocco {c['coverage_block']}): "
          f"{[n for n in range(FORK + 1, END + 1) if n % c['coverage_block'] == 0]}")
    print(f"   fattore del peso: update 255 (passo 254) = {factor(254)}, update 581..588 = "
          f"{sorted({factor(n - 1) for n in range(FORK + 1, END + 1)})}")

cell, ck = resources["each_gpu_cell"], resources["checkout"]
gpu = len(cell["cells"]) * cell["gpus"] * cell["wall_seconds"] / 3600
core = len(cell["cells"]) * cell["cpus"] * cell["wall_seconds"] / 3600 + ck["cpus"] * ck["wall_seconds"] / 3600
used = resources["before_diagnostics_used"]
print(f"4. risorse: incrementale {gpu:g} GPU-ore e {core:.4f} core-ore "
      f"(dichiarati {resources['incremental_cap']['gpu_hours']:g} e {resources['incremental_cap']['core_hours']:.4f}); "
      f"cumulativo {used['gpu_hours'] + gpu:.4f} / {resources['campaign_cap']['gpu_hours']:g} GPU-ore, "
      f"{used['core_hours'] + core:.4f} / {resources['campaign_cap']['core_hours']:.2f} core-ore")
