"""Consuntivo of the failed v1 attempt and amended envelope of the recovery (v2) of the 300M continuation 5000 -> 19618.

Usage (from the repository root, with the project Python):
    python -B results/infra/scaling_300m_20261104/continuation19618_20261109_v2/make_envelope.py [--check]
Writes budget_envelope.json next to the plan and never over an existing file; --check rebuilds it and compares. Exact
arithmetic (Fraction). The frozen budget v2 and its pins are untouched: this amendment is a separate document and needs a
registered decision of Simone before the recovery is submitted (CLAUDE.md: a step budget is exceeded only by a decision).
Costs of the failed attempt and of the diagnosis stay in the account; the evaluation (stage B) is unchanged.
"""
from fractions import Fraction
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_plans19618 as S  # noqa: E402

PROTOCOL = "scaling-continuation19618-recovery-envelope-20261109-v1"
OUT = HERE / "budget_envelope.json"
V1_CHECKOUT = S.PACKAGE + "/evidence/checkout-training/training/accounting.json"  # collected on 06/11 with its raw sacct
# The other five jobs already run, as read with sacct on 08/11/2026 (docs/decisioni.md, entry of 08/11): elapsed seconds,
# CPUs and GPUs allocated. Their raw `sacct -P` is not in the repository yet: remote_read.sh reads it again before the
# deploy, it is kept among the evidence of the chain, and any difference from these values stops the recovery.
SPENT = (("v1 training", "SCALING-300M-train19618-20261106", "63696864", "FAILED 1:0", 205, 32, 4),
         ("diagnosis", "SCALING-300M-diag-fixedeval-20261108", "63860499", None, 48, 4, 0),
         ("diagnosis", "SCALING-300M-diag-fixedeval2-20261108", "63861166", None, 100, 4, 0),
         ("diagnosis", "SCALING-300M-diag-fixedeval-gpu-20261108", "63861825", None, 138, 32, 4),
         ("diagnosis", "SCALING-300M-diag-importorder-20261108", "63864244", None, 204, 8, 0))


def exact(value):
    value = Fraction(value)
    return {"value": round(float(value), 6), "exact": f"{value.numerator}/{value.denominator}"}


def decimal(value):
    """A decimal written in the freeze (66.35, 2.8, 0.8) as the exact number it denotes."""
    return Fraction(str(value))


def hours(units, seconds):
    return Fraction(units * seconds, 3600)


def build():
    freeze_bytes = (ROOT / S.FREEZE).read_bytes()
    plan_bytes = (ROOT / S.RECOVERY["plan"]).read_bytes()
    freeze, plan = json.loads(freeze_bytes), json.loads(plan_bytes)
    ceilings, readers = freeze["ceilings"], freeze["readers"]
    margin = readers["scheduler_overtime_seconds"]
    v1 = json.loads((ROOT / V1_CHECKOUT).read_text())
    spent = [{"part": "v1 checkout", "logical_id": "SCALING-300M-cont19618-checkout-20261106", "slurm_job_id": v1["JobIDRaw"],
              "state": "%s %s" % (v1["State"], v1["ExitCode"]), "elapsed_seconds": v1["ElapsedRaw"], "cpus": v1["AllocCPUS"],
              "gpus": 0, "attested_by": V1_CHECKOUT}]
    spent += [{"part": part, "logical_id": logical, "slurm_job_id": job, "state": state, "elapsed_seconds": seconds, "cpus": cpus,
               "gpus": gpus, "attested_by": None} for part, logical, job, state, seconds, cpus, gpus in SPENT]
    for row in spent:
        row.update(core_hours=exact(hours(row["cpus"], row["elapsed_seconds"])), gpu_hours=exact(hours(row["gpus"], row["elapsed_seconds"])))
    spent_gpu = sum(hours(r["gpus"], r["elapsed_seconds"]) for r in spent)
    spent_gpu_job_cpu = sum(hours(r["cpus"], r["elapsed_seconds"]) for r in spent if r["gpus"])
    diagnosis_cpu_only = sum(hours(r["cpus"], r["elapsed_seconds"]) for r in spent if r["part"] == "diagnosis" and not r["gpus"])
    recovery = []
    for part, job in (("checkout", S.checkout_job("training")), ("identity pre-check", S.precheck_job(plan)),
                      ("training", S.training_job(plan))):
        recovery.append({"part": part, "logical_id": job["id"], "cpus": job["cpus"], "gpus": job["gpus"], "memory": job["memory"],
                         "seconds": job["seconds"], "core_hours_cap": exact(hours(job["cpus"], job["seconds"])),
                         "core_hours_with_margin": exact(hours(job["cpus"], job["seconds"] + margin)),
                         "gpu_hours_cap": exact(hours(job["gpus"], job["seconds"])),
                         "gpu_hours_with_margin": exact(hours(job["gpus"], job["seconds"] + margin))})
    checkout, precheck = (hours(r["cpus"], r["seconds"] + margin) for r in recovery[:2])
    planned = Fraction(S.budget_table()["core_hours_with_scheduler_margin"]).limit_denominator(3600)
    frozen = {"gpu_hours_with_margin": decimal(ceilings["additional_total_gpu_hours_with_scheduler_margin"]),
              "checkout_stage_core_hours_with_margin": decimal(readers["stages"]["checkout"]["core_hours_with_scheduler_margin"]),
              "gpu_job_cpu_core_hours_with_margin": decimal(ceilings["allocated_cpu_core_hours_in_gpu_jobs"]) +
              decimal(ceilings["scheduler_margin_gpu_job_cpu_core_hours"])}
    cpu_only = planned + checkout + precheck + diagnosis_cpu_only
    cap = readers["cpu_only_core_hours_cap"]
    return {
        "protocol": PROTOCOL,
        "status": "proposal: needs Simone's registered decision before the recovery is submitted; the frozen budget v2 is unchanged",
        "freeze": {"path": S.FREEZE, "sha256": S.digest(freeze_bytes), "bytes": len(freeze_bytes)},
        "plan": {"path": S.RECOVERY["plan"], "sha256": S.digest(plan_bytes), "bytes": len(plan_bytes)},
        "scheduler_margin_seconds": margin,
        "spent": spent,
        "spent_totals": {"gpu_hours": exact(spent_gpu), "core_hours_in_gpu_jobs": exact(spent_gpu_job_cpu),
                         "diagnosis_cpu_only_core_hours": exact(diagnosis_cpu_only),
                         "v1_checkout_core_hours": spent[0]["core_hours"]},
        "recovery": recovery,
        "evaluation": {"unchanged": True, "extraction_gpu_hours_cap": ceilings["additional_extraction_gpu_hours"],
                       "cpu_only_core_hours_with_margin": exact(planned),
                       "note": "stage B as frozen and reviewed; its CPU-only table counts five checkouts, the training one being "
                               "the slot already used by the v1 checkout, so the recovery checkout is an extra job"},
        "frozen": {key: exact(value) for key, value in frozen.items()} | {"cpu_only_core_hours_cap": cap},
        "amended": {"gpu_hours_with_margin": exact(frozen["gpu_hours_with_margin"] + spent_gpu),
                    "cpu_only_core_hours_with_margin": exact(cpu_only), "cpu_only_core_hours_cap": cap,
                    "within_cpu_only_cap": cpu_only <= cap,
                    "checkout_stage_core_hours_with_margin": exact(frozen["checkout_stage_core_hours_with_margin"] + checkout),
                    "identity_precheck_core_hours_with_margin": exact(precheck),
                    "diagnosis_cpu_only_core_hours": exact(diagnosis_cpu_only),
                    "gpu_job_cpu_core_hours_with_margin": exact(frozen["gpu_job_cpu_core_hours_with_margin"] + spent_gpu_job_cpu)},
        "changes_needing_decision": [
            "a second attempt of the same single training segment 5000 -> 19618 (the freeze: one segment, no blind retry)",
            "GPU hours with margin: the failed attempt and the GPU diagnosis stay in the account",
            "checkout stage: a sixth checkout, the recovery one",
            "new line: the CPU identity pre-check (0 GPU)",
            "CPU core-hours in GPU jobs: the failed attempt and the GPU diagnosis stay in the account"],
        "not_included": ["any dose beyond 19618 (the 49046 proposal of GPT, 08/11, is a separate decision of Simone)", "queue times"],
        "attestation_pending": [r["slurm_job_id"] for r in spent if r["attested_by"] is None],
    }


def main(argv):
    built = json.dumps(build(), indent=1) + "\n"
    if "--check" in argv:
        same = OUT.read_text() == built
        print(json.dumps({"envelope_matches_published": same}))
        return 0 if same else 1
    with OUT.open("x") as stream:
        stream.write(built)
    print(OUT.relative_to(ROOT), S.digest(built.encode()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
