"""Consuntivo of the attempts already made and amended envelope of the recovery (v3) of the 300M continuation 5000 -> 19618.

Usage (from the repository root, with the project Python):
    python -B results/infra/scaling_300m_20261104/continuation19618_20261109_v3/make_envelope.py [--check]
Writes budget_envelope.json next to the plan and never over an existing file; --check rebuilds it and compares. Exact
arithmetic (Fraction). Every job already run is read from attested evidence of the chain (collected accounting, or the raw
`sacct -P` of 09/11 compared with the signed v2 envelope); the frozen budget v2 and the signed v2 envelope are untouched.
Amendment of the signed v2 envelope: one more checkout and one more identity pre-check, CPU only (Simone, 09/11 12:06).
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

PROTOCOL = "scaling-continuation19618-recovery-envelope-20261109-v2"
OUT = HERE / "budget_envelope.json"
EVIDENCE = S.PACKAGE + "/evidence/"
SIGNED = S.INFRA + "/continuation19618_20261109_v2/budget_envelope.json"  # signed by Simone on 09/11 at 11:40
ACCOUNTED = (("v1 checkout", EVIDENCE + "checkout-training/training/accounting.json", "SCALING-300M-cont19618-checkout-20261106"),
             ("v2 checkout", EVIDENCE + "checkout-training-v2/training/accounting.json", "SCALING-300M-cont19618-checkout-20261109-v2"),
             ("v2 identity pre-check", EVIDENCE + "identity-precheck/accounting.json", "SCALING-300M-cont19618-idcheck-20261109-v2"))
SACCT = EVIDENCE + "recovery-reread-20261109/attestation.json"  # v1 training and the four diagnoses, raw sacct -P of 09/11 11:43
APPROVAL = ("Simone, 09/11 alle 12:06, alla domanda «Preparo il v3 col cancello corretto e un nuovo pre-controllo (solo CPU, "
            "+1,8 core-ore)?»: «Sì, prepara il v3»")


def exact(value):
    value = Fraction(value)
    return {"value": round(float(value), 6), "exact": f"{value.numerator}/{value.denominator}"}


def decimal(value):
    """A decimal written in the freeze (66.35, 2.8, 0.8) as the exact number it denotes."""
    return Fraction(str(value))


def hours(units, seconds):
    return Fraction(units * seconds, 3600)


def spent_rows():
    rows = []
    for part, path, logical in ACCOUNTED:
        record = json.loads((ROOT / path).read_text())
        rows.append({"part": part, "logical_id": logical, "slurm_job_id": record["JobIDRaw"],
                     "state": "%s %s" % (record["State"], record["ExitCode"]), "elapsed_seconds": record["ElapsedRaw"],
                     "cpus": record["AllocCPUS"], "gpus": record["gpus"], "allocated_seconds": S.slurm_seconds(record["Timelimit"]),
                     "attested_by": path})
    attestation = json.loads((ROOT / SACCT).read_text())
    assert attestation["all_match"] is True
    for job in attestation["jobs"]:
        if job["slurm_job_id"] == "63695634":  # the v1 checkout comes from its collected accounting above
            continue
        measured = job["sacct"]
        part = "v1 training" if job["slurm_job_id"] == "63696864" else "diagnosis"
        rows.append({"part": part, "logical_id": measured["logical_id"], "slurm_job_id": job["slurm_job_id"],
                     "state": measured["state"], "elapsed_seconds": measured["elapsed_seconds"], "cpus": measured["cpus"],
                     "gpus": measured["gpus"], "allocated_seconds": None, "attested_by": SACCT})
    order = {"v1 checkout": 0, "v1 training": 1, "diagnosis": 2, "v2 checkout": 3, "v2 identity pre-check": 4}
    return sorted(rows, key=lambda r: (order[r["part"]], r["slurm_job_id"]))


def build():
    freeze_bytes, plan_bytes = (ROOT / S.FREEZE).read_bytes(), (ROOT / S.RECOVERY["plan"]).read_bytes()
    signed_bytes = (ROOT / SIGNED).read_bytes()
    freeze, plan = json.loads(freeze_bytes), json.loads(plan_bytes)
    ceilings, readers = freeze["ceilings"], freeze["readers"]
    margin = readers["scheduler_overtime_seconds"]
    spent = spent_rows()
    for row in spent:
        row.update(core_hours=exact(hours(row["cpus"], row["elapsed_seconds"])), gpu_hours=exact(hours(row["gpus"], row["elapsed_seconds"])))
    spent_gpu = sum(hours(r["gpus"], r["elapsed_seconds"]) for r in spent)
    spent_gpu_job_cpu = sum(hours(r["cpus"], r["elapsed_seconds"]) for r in spent if r["gpus"])
    diagnosis_cpu_only = sum(hours(r["cpus"], r["elapsed_seconds"]) for r in spent if r["part"] == "diagnosis" and not r["gpus"])
    # The v2 checkout and pre-check count at their allocation with margin, like the planned jobs of the chain.
    v2_checkout, v2_precheck = (hours(r["cpus"], r["allocated_seconds"] + margin) for r in spent if r["part"].startswith("v2 "))
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
    cpu_only = planned + v2_checkout + v2_precheck + checkout + precheck + diagnosis_cpu_only
    cap = readers["cpu_only_core_hours_cap"]
    return {
        "protocol": PROTOCOL,
        "status": "amendment of the signed v2 envelope, approved: " + APPROVAL + "; the frozen budget v2 is unchanged",
        "freeze": {"path": S.FREEZE, "sha256": S.digest(freeze_bytes), "bytes": len(freeze_bytes)},
        "signed_v2_envelope": {"path": SIGNED, "sha256": S.digest(signed_bytes), "bytes": len(signed_bytes)},
        "plan": {"path": S.RECOVERY["plan"], "sha256": S.digest(plan_bytes), "bytes": len(plan_bytes)},
        "scheduler_margin_seconds": margin,
        "spent": spent,
        "spent_totals": {"gpu_hours": exact(spent_gpu), "core_hours_in_gpu_jobs": exact(spent_gpu_job_cpu),
                         "diagnosis_cpu_only_core_hours": exact(diagnosis_cpu_only),
                         "v2_jobs_core_hours": exact(sum(hours(r["cpus"], r["elapsed_seconds"]) for r in spent if r["part"].startswith("v2 ")))},
        "recovery": recovery,
        "evaluation": {"unchanged": True, "extraction_gpu_hours_cap": ceilings["additional_extraction_gpu_hours"],
                       "cpu_only_core_hours_with_margin": exact(planned),
                       "note": "stage B as frozen and reviewed; its CPU-only table counts five checkouts, the training one being "
                               "the slot already used by the v1 checkout: the v2 and v3 checkouts are extra jobs"},
        "frozen": {key: exact(value) for key, value in frozen.items()} | {"cpu_only_core_hours_cap": cap},
        "amended": {"gpu_hours_with_margin": exact(frozen["gpu_hours_with_margin"] + spent_gpu),
                    "cpu_only_core_hours_with_margin": exact(cpu_only), "cpu_only_core_hours_cap": cap,
                    "within_cpu_only_cap": cpu_only <= cap,
                    "checkout_stage_core_hours_with_margin": exact(frozen["checkout_stage_core_hours_with_margin"] + v2_checkout + checkout),
                    "identity_precheck_core_hours_with_margin": exact(v2_precheck + precheck),
                    "diagnosis_cpu_only_core_hours": exact(diagnosis_cpu_only),
                    "gpu_job_cpu_core_hours_with_margin": exact(frozen["gpu_job_cpu_core_hours_with_margin"] + spent_gpu_job_cpu)},
        "not_included": ["any dose beyond 19618 (the 49046 proposal of GPT, 08/11, is a separate decision of Simone)", "queue times"],
        "attestation_pending": [r["slurm_job_id"] for r in spent if not r["attested_by"]],
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
