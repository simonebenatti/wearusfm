"""Amendment of the line-1 CPU envelope: the four checkouts of stage B at 2 CPUs (11/11/2026). Proposed, Simone decides.

`lrd_all_serial` allocates whole cores, two threads each: the code checkout 64194074, published and submitted at 1 CPU, ran at
2 (`sacct_64194074.txt`, `scheduler_reading_64194074.txt`). The freeze v2 already writes 2 CPUs per checkout; the stage-B table
had put its four checkouts at 1 CPU to fit five checkouts in the frozen 0.8 core-hours. This script recomputes, in exact
arithmetic, the amended v3 envelope with those four checkouts at 2 CPUs; GPU-hours do not change. `--check` rebuilds and compares.
"""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INFRA = "results/infra/scaling_300m_20261104"
V3 = INFRA + "/continuation19618_20261109_v3/budget_envelope.json"
FREEZE = INFRA + "/continuation19618_budget_freeze_20261105_v2.json"
SACCT, READING = HERE / "sacct_64194074.txt", HERE / "scheduler_reading_64194074.txt"
OUT = HERE / "budget_amendment.json"
CHECKOUTS = ("code", "plans", "budgets", "inputs")


def exact(x: Fraction) -> dict:
    return {"value": round(x.numerator / x.denominator, 6), "exact": f"{x.numerator}/{x.denominator}"}


def frac(entry: dict) -> Fraction:
    numerator, denominator = entry["exact"].split("/")
    return Fraction(int(numerator), int(denominator))


def table(text: str) -> list[dict]:
    lines = text.strip().splitlines()
    header = lines[0].split("|")
    return [dict(zip(header, line.split("|"))) for line in lines[1:]]


def build() -> bytes:
    v3_bytes, freeze_bytes = (ROOT / V3).read_bytes(), (ROOT / FREEZE).read_bytes()
    v3, readers = json.loads(v3_bytes), json.loads(freeze_bytes)["readers"]
    frozen, overtime = readers["stages"]["checkout"], readers["scheduler_overtime_seconds"]
    seconds = frozen["seconds_each"]
    assert (frozen["cpus"], seconds, overtime) == (2, 300, 180)
    # The stage-B table as reviewed: the training checkout at 2 CPUs and four checkouts at 1 CPU = the frozen 0.8.
    before_row = Fraction((2 + 4 * 1) * (seconds + overtime), 3600)
    assert before_row == Fraction(str(frozen["core_hours_with_scheduler_margin"])) == Fraction(4, 5)
    after_row = Fraction((2 + 4 * 2) * (seconds + overtime), 3600)
    delta = after_row - before_row
    amended = v3["amended"]
    cpu_before, gpu = frac(amended["cpu_only_core_hours_with_margin"]), frac(amended["gpu_hours_with_margin"])
    stage_before = frac(amended["checkout_stage_core_hours_with_margin"])
    cpu_after, cap = cpu_before + delta, Fraction(amended["cpu_only_core_hours_cap"])
    parent = [r for r in table(SACCT.read_text()) if r["JobIDRaw"] == "64194074"]
    assert len(parent) == 1 and parent[0]["State"] == "COMPLETED" and parent[0]["ExitCode"] == "0:0"
    reading = READING.read_text()
    requested = table(reading.split("## sacct -X -P", 1)[1].split("\n", 1)[1].split("##", 1)[0])[0]
    assert requested["JobIDRaw"] == "64194074" and requested["ReqCPUS"] == "1" and requested["AllocCPUS"] == parent[0]["AllocCPUS"] == "2"
    assert "ThreadsPerCore=2" in reading and "NodeName=login13" in reading
    elapsed = int(parent[0]["ElapsedRaw"])
    document = {
        "protocol": "scaling-19618-checkout-cpus-amendment-20261111-v1",
        "status": "proposed: requires Simone's decision (checkout stage above the amended envelope, total within the 350 cap)",
        "base_envelope": {"path": V3, "sha256": hashlib.sha256(v3_bytes).hexdigest()},
        "freeze": {"path": FREEZE, "sha256": hashlib.sha256(freeze_bytes).hexdigest(), "checkout": frozen},
        "cause": {"logical_id": "SCALING-19618-code-checkout-20261106", "job_slurm": "64194074", "partition": parent[0]["Partition"],
                  "node": parent[0]["NodeList"], "threads_per_core": 2, "requested_cpus": 1, "allocated_cpus": 2,
                  "elapsed_seconds": elapsed, "allocated_core_hours": exact(Fraction(2 * elapsed, 3600)),
                  "sacct": {"path": f"{INFRA}/{HERE.name}/{SACCT.name}", "sha256": hashlib.sha256(SACCT.read_bytes()).hexdigest()},
                  "scheduler_reading": {"path": f"{INFRA}/{HERE.name}/{READING.name}",
                                        "sha256": hashlib.sha256(READING.read_bytes()).hexdigest()}},
        "change": {"checkouts": list(CHECKOUTS), "cpus_before": 1, "cpus_after": 2, "seconds_each": seconds,
                   "scheduler_overtime_seconds": overtime},
        "delta_core_hours_with_margin": exact(delta),
        "stage_b_checkout_row_with_margin": {"jobs": 5, "before": exact(before_row), "after": exact(after_row)},
        "envelope_checkout_stage_with_margin": {"before": exact(stage_before), "after": exact(stage_before + delta)},
        "cpu_only_core_hours_with_margin": {"before": exact(cpu_before), "after": exact(cpu_after)},
        "cpu_only_core_hours_cap": int(cap), "within_cpu_only_cap": cpu_after <= cap, "margin_to_cap": exact(cap - cpu_after),
        "gpu_hours_with_margin": {"unchanged": exact(gpu)},
        "not_included": "no new job, no retry: the code checkout 64194074 is kept; the other three checkouts are asked at 2 CPUs",
    }
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def main(argv: list[str]) -> int:
    data = build()
    if argv[1:] == ["--check"]:
        same = OUT.read_bytes() == data
        print(json.dumps({"check": "budget_amendment.json", "identical": same}))
        return 0 if same else 1
    if OUT.exists():
        raise SystemExit("budget_amendment.json exists: use --check")
    OUT.write_bytes(data)
    print(json.dumps({"written": str(OUT.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
