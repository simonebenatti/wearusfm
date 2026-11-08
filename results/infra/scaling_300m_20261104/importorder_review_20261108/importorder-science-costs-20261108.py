"""Read-only cost calculation for scaling recovery and optional full WSD schedule.

No deployment, submission, protocol amendment or authorization. Amounts for the
49k proposal exclude its evaluation and require a new execution plan.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
base = a.repo / "results/infra/scaling_300m_20261104"
freeze_path = base / "continuation19618_budget_freeze_20261105_v2.json"
freeze = json.loads(freeze_path.read_text())
summary_path = base / "scientific5000_completion_20261105/summary.json"
summary = json.loads(summary_path.read_text())
times = freeze["training"]["inputs"]
assert summary["elapsed_s"] == times["trainer300M5000_seconds"]
speed = max(summary["elapsed_s"] / summary["steps"],
            times["preflight300M_112_update_mean_seconds"])
gpu, cpu = 4, 32

def row(start, end):
    steps = end-start
    core = steps*speed
    maximum = math.ceil((core*1.20+1200)/900)*900
    return {"start":start, "end":end, "steps":steps,
            "updates_projected_seconds":core,
            "updates_projected_gpu_hours":gpu*core/3600,
            "expected_with_20min_overhead_seconds":core+1200,
            "expected_with_20min_overhead_gpu_hours":gpu*(core+1200)/3600,
            "requested_cap_seconds":maximum,
            "requested_cap_gpu_hours":gpu*maximum/3600,
            "requested_cap_allocated_cpu_hours":cpu*maximum/3600,
            "fits_24h_partition": maximum <= 86400,
            "scheduler_grace_gpu_hours_at_180s":gpu*180/3600}

segments = [row(5000,19618), row(19618,39237), row(39237,49046)]
assert segments[0]["requested_cap_seconds"] == freeze["training"]["allocation"]["walltime_seconds"]
def total(rows):
    keys = [k for k in rows[0] if k.endswith("seconds") or k.endswith("hours") or k=="steps"]
    return {k:sum(r[k] for r in rows) for k in keys}

# Previously published actual GPU allocations; count allocation rows only.
with (base/"scientific5000_completion_20261105/sacct.txt").open() as f:
    sacct = list(csv.DictReader(f, delimiter="|"))
train = next(x for x in sacct if x["JobID"]=="63532932")
s3 = json.loads((base/"production_memory_63527446/accounting.json").read_text())
s3row = next(x for x in s3["sacct_rows"] if x["JobIDRaw"]=="63527446")
extraction_paths = [base/"pose_completed_20261105/accounting.json"]
extraction_paths += sorted((base/"classification_completed_20261105").glob("*/*/accounting.json"))
extraction_sum = sum(json.loads(x.read_text())["gpu_hours"] for x in extraction_paths)
assert len(extraction_paths)==5
past = {
    "preflight_63526191": {"seconds":546,"gpus":4,"basis":"docs/scaling_300m_execution_20261104.md:112 published duration"},
    "production_memory_63527446": {"seconds":int(s3row["ElapsedRaw"]),"gpus":4,"basis":"production_memory_63527446/accounting.json allocation row"},
    "scientific5000_63532932": {"seconds":int(train["ElapsedRaw"]),"gpus":4,"basis":"scientific5000_completion_20261105/sacct.txt allocation row"},
    "extractions_5000_five_jobs": {"gpu_hours":extraction_sum,"basis":"five published accounting.json files, four classification plus pose"},
    "failed_continuation_63696864": {"seconds":205,"gpus":4,"basis":"diagnosis README.md published duration 3m25s; no resumed update accepted"},
    "diagnosis_gpu_63861825": {"seconds":138,"gpus":4,"basis":"diagnosis README.md published duration 2m18s; other three diagnostic jobs CPU only"}
}
for r in past.values():
    if "gpu_hours" not in r: r["gpu_hours"] = r["seconds"]*r["gpus"]/3600
result = {
    "source_pins": {str(q.relative_to(a.repo)):hashlib.sha256(q.read_bytes()).hexdigest() for q in (freeze_path, summary_path)},
    "selected_seconds_per_update":speed,
    "recipe": "same 4 GPUs, global batch32, WSD total49046/warmup1962/decay39237; no restart or shortening of schedule",
    "formula": "ceil900(steps*max(14665.05355668068/5000,2.7666786608873997)*1.20+1200) per segment",
    "segments": segments,
    "sum_5000_to_49046_three_segments":total(segments),
    "extra_19618_to_49046_two_segments":total(segments[1:]),
    "invalid_single_job_19618_to_49046":row(19618,49046),
    "past_gpu_costs":past,
    "past_gpu_hours_sum":sum(x["gpu_hours"] for x in past.values()),
    "failed_and_diagnosis_gpu_hours":past["failed_continuation_63696864"]["gpu_hours"]+past["diagnosis_gpu_63861825"]["gpu_hours"],
    "past_scope": "line1 preflight+S3+scientific5k+five extractions+failed continuation+diagnosis; excludes historical30M training and line2/line3",
    "limitations": "Timing extrapolation, not measurement at later dose. Queue wait excluded. 49k extractions/CPU fits and metadata checkouts not budgeted here. Original19k evaluation caps7GPUh/350CPUh remain separate; cannot claim reusable49k control fit without source/support/recipe audit.",
    "authorized_new_49046_budget":False,
    "submission_performed":False
}
a.out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
