"""Pure CPU envelope check for the already approved post-P1 reader chain.

No filesystem, scheduler, subprocess or submitter access. The Mac controller
supplies the complete job tables, ledger receipts and accounting already checked
against raw sacct. Missing accounting reserves the full allocation; it never
means zero cost or successful completion. Scientific stage ceilings stay frozen.
"""
from fractions import Fraction
import math
import re

PROTOCOL = "scaling-p1-reader-cpu-envelope-20261117-v1"
CAMPAIGN = "scaling-300m-20261104"
STAGES = ("checkout-plans", "checkout-budgets", "checkout-inputs", "pose-validate",
          "pose-benchmark", "probe-benchmark", "pose-ridge", "probe-fit", "probe-aggregate", "compare")
OVERTIME_SECONDS = 180
TOTAL_CAP = Fraction(351)
FIXED_PRIOR_RESERVE = Fraction(2396389, 500000)  # exactly 4.792778 core-hours
FUTURE_MAX = Fraction(2071, 6)  # all remaining stages, including THREE metadata checkouts
BASELINE = FIXED_PRIOR_RESERVE + FUTURE_MAX  # 349.917778 + actual two P1 checkouts (150 CPU-seconds)
FIT_REQUESTED_CAP = Fraction(240)
FIT_RESERVED_CAP = Fraction(246)
PREFIX = "SCALING-19618-"
BENCHMARK_SECONDS = {
    "30m-natural": {"ninapro-db2":2700, "ninapro-db3":1200, "ninapro-db6":1200, "epn612":1800, "uci-emg":600},
    "300m-natural": {"ninapro-db2":3600, "ninapro-db3":1800, "ninapro-db6":3600, "epn612":7200, "uci-emg":600},
    "300m-matched": {"ninapro-db2":2700, "ninapro-db3":1200, "ninapro-db6":1200, "epn612":1800, "uci-emg":600},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hours(cpus, seconds):
    return Fraction(cpus * seconds, 3600)


def _amount(value):
    return {"exact": str(value), "core_hours": float(value)}


def require_total(total):
    """Use the approved 351 cap exactly, never a caller-selected larger limit."""
    require(isinstance(total, Fraction) and total >= 0, "exact nonnegative cumulative amount required")
    require(total <= TOTAL_CAP, "cumulative CPU-only envelope exceeds 351 core-hours")
    return True


def _fixed_specs():
    """Exact logical cells and allocations from the frozen stage-B table."""
    def entry(suffix, cpus, seconds, memory="30G"):
        return (PREFIX + suffix + "-20261106", cpus, seconds, memory)
    return {
        **{"checkout-"+name: [entry(name+"-checkout", 2, 300, "4G")]
           for name in ("plans", "budgets", "inputs")},
        "pose-validate": [entry("pose-validate",2,1800)],
        "pose-benchmark": [entry("pose-bench-"+cell,8,600) for cell in BENCHMARK_SECONDS],
        "probe-benchmark": [entry("probe-bench-"+cell+"-"+dataset,8,seconds)
                            for cell, datasets in BENCHMARK_SECONDS.items() for dataset, seconds in datasets.items()],
        "pose-ridge": [entry("pose-ridge",8,3600)],
        "probe-aggregate": [entry("probe-aggregate-%02d" % n,2,600) for n in range(1,7)],
        "compare": [entry("compare",8,1800)],
    }


def _validate_tables(tables, fit_plan_ready):
    require(type(fit_plan_ready) is bool, "fit-plan readiness must be explicit")
    require(isinstance(tables, dict) and set(tables) == set(STAGES), "complete exact reader-stage table required")
    all_jobs = {}
    for stage, jobs in tables.items():
        require(isinstance(jobs,list), "stage table must be a list: "+stage)
        for job in jobs:
            require(isinstance(job,dict) and isinstance(job.get("id"),str), "invalid job table row")
            require(job["id"] not in all_jobs, "duplicate logical job in cumulative budget")
            require(all(type(job.get(k)) is int for k in ("cpus","seconds","gpus")), "integer allocations required")
            require(job["gpus"] == 0 and job["seconds"] > 0 and job["cpus"] in (2,8), "invalid CPU-only job allocation")
            require(job.get("stage") == ("checkout" if stage.startswith("checkout-") else stage), "job stage differs")
            all_jobs[job["id"]] = job
    for stage, specs in _fixed_specs().items():
        actual = {(j["id"],j["cpus"],j["seconds"],j.get("memory")) for j in tables[stage]}
        require(len(actual)==len(tables[stage])==len(specs) and actual==set(specs), "frozen stage scope/allocation differs: "+stage)
    fits = tables["probe-fit"]
    if not fit_plan_ready:
        require(not fits, "unknown fit plan must reserve the entire fit ceiling")
    else:
        require(1 <= len(fits) <= 15, "fit plan must contain 1..15 shards")
        require({j["id"] for j in fits} == {PREFIX+"probe-fit-%02d-20261106" % n for n in range(1,len(fits)+1)},
                "fit logical IDs differ from the registered consecutive shard list")
        require(all(j["cpus"]==8 and j["memory"]=="30G" and j["seconds"]<=10800 for j in fits),
                "fit exceeds eight CPUs, 30GiB or three hours")
        requested = sum((_hours(j["cpus"],j["seconds"]) for j in fits), Fraction())
        require(requested<=FIT_REQUESTED_CAP, "fits exceed the frozen 240 requested core-hours")
    return all_jobs


def _time(seconds):
    return "%02d:%02d:%02d" % (seconds//3600, seconds%3600//60, seconds%60)


def _ledger(logical, job, state):
    require(isinstance(state,dict) and state.get("id")==logical and state.get("stato")=="sottomesso" and
            state.get("agente") in ("gpt","claude") and state.get("campagna")==CAMPAIGN,
            "unknown, refused or ambiguous ledger state: "+logical)
    slurm = state.get("job_slurm")
    require(type(slurm) in (str,int) and str(slurm).isdigit() and int(slurm)>0, "missing valid SLURM identity")
    require(_number(state.get("gpu_ore_stimate")) and state["gpu_ore_stimate"]==0, "CPU ledger reserves GPU time")
    argv = state.get("sbatch")
    require(isinstance(argv,list) and argv and all(isinstance(s,str) for s in argv), "full sbatch argv required")
    for key, expected in (("--cpus-per-task=",str(job["cpus"])),("--mem=",job["memory"]),("--time=",_time(job["seconds"]))):
        require([x[len(key):] for x in argv if x.startswith(key)]==[expected], "ledger allocation differs: "+logical+" "+key)
    return str(slurm)


def _actual(job, state, record):
    require(isinstance(record,dict) and record.get("JobIDRaw")==str(state["job_slurm"]) and
            record.get("State")=="COMPLETED" and record.get("ExitCode")=="0:0", "accounting not the accepted completed job")
    elapsed,cpus = record.get("ElapsedRaw"),record.get("AllocCPUS")
    require(type(elapsed) is int and 0<elapsed<=job["seconds"] and type(cpus) is int and cpus==job["cpus"],
            "accounting elapsed/CPU allocation differs")
    require(type(record.get("gpus")) is int and record["gpus"]==0, "CPU accounting has unexpected GPUs")
    tres = record.get("AllocTRES")
    require(isinstance(tres,str), "raw allocation TRES required")
    try:
        parts=dict(x.split("=",1) for x in tres.split(","))
    except ValueError as e:
        raise ValueError("malformed accounting TRES") from e
    require(parts.get("cpu")==str(cpus) and parts.get("node")=="1" and
            parts.get("gres/gpu","0")=="0", "accounting TRES differs")
    require(parts.get("mem") in (job["memory"],str(int(job["memory"][:-1])*1024)+"M"), "accounting memory differs")
    require(record.get("Timelimit")==_time(job["seconds"]), "accounting time limit differs")
    raw = record.get("raw_sacct")
    require(isinstance(raw,dict) and isinstance(raw.get("path"),str) and raw["path"] and
            isinstance(raw.get("sha256"),str) and re.fullmatch("[0-9a-f]{64}",raw["sha256"]) is not None,
            "accounting must retain its verified raw sacct pin")
    actual = _hours(cpus,elapsed)
    require(_number(record.get("core_hours")) and math.isclose(record["core_hours"],float(actual),rel_tol=0,abs_tol=1e-9),
            "accounting core-hours do not match parent allocation")
    return actual


def evaluate_budget(tables, submissions, accounting, *, fit_plan_ready):
    """Substitute verified actual costs for reservations, never add both.

    `tables` contains S-style job rows for all STAGES. Before fit resources exist,
    tables['probe-fit'] is empty and fit_plan_ready=False: its full 246 core-hour
    reserve remains. `submissions` maps logical IDs to unmodified state.json.
    `accounting` maps IDs to independently checked S.accounting records. The
    caller must reread and verify raw sacct; this pure function verifies identity
    and arithmetic again. A submitted job without accounting keeps its full cap.
    """
    all_jobs = _validate_tables(tables, fit_plan_ready)
    require(isinstance(submissions,dict) and isinstance(accounting,dict), "ledger and accounting maps required")
    require(set(submissions)<=set(all_jobs), "ledger contains a job outside the complete published stage tables")
    require(set(accounting)<=set(submissions), "accounting without a submitted logical ID")
    slurm_ids = [_ledger(logical,all_jobs[logical],s) for logical,s in submissions.items()]
    require(len(set(slurm_ids))==len(slurm_ids), "duplicate SLURM allocation counted under different logical IDs")
    stages, details = {}, {}
    for stage in STAGES:
        amount = Fraction()
        for job in tables[stage]:
            logical = job["id"]
            reserved = _hours(job["cpus"],job["seconds"]+OVERTIME_SECONDS)
            actual = _actual(job,submissions[logical],accounting[logical]) if logical in accounting else None
            charged = reserved if actual is None else actual
            amount += charged
            details[logical] = {"stage":stage,"status":"actual" if actual is not None else "reserved",
                                "submitted":logical in submissions,"charged":_amount(charged),"reservation":_amount(reserved)}
        if stage=="probe-fit" and not fit_plan_ready:
            amount=FIT_RESERVED_CAP
        stages[stage] = _amount(amount)
    remaining = sum((Fraction(r["exact"]) for r in stages.values()),Fraction())
    total=FIXED_PRIOR_RESERVE+remaining
    require_total(total)
    return {"protocol":PROTOCOL,"complete":True,"within_total_cap":True,"within_stage_caps":True,
            "fixed_prior_reserve":_amount(FIXED_PRIOR_RESERVE),"baseline_max":_amount(BASELINE),
            "remaining_chain":_amount(remaining),"total":_amount(total),"cap_core_hours":351,
            "margin":_amount(TOTAL_CAP-total),"fit_plan_ready":fit_plan_ready,
            "unknown_fit_reserve":_amount(FIT_RESERVED_CAP) if not fit_plan_ready else None,
            "stages":stages,"jobs":details,"actual_jobs":len(accounting),
            "reserved_jobs":len(all_jobs)-len(accounting),"no_new_scope_authorized":True}
