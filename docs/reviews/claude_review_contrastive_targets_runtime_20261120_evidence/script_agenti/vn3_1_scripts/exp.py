"""R3-N3 reproduction. argv: clone receipts_dir step. Writes only in the temp clone / receipts dir."""
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
RECEIPTS = Path(sys.argv[2]).resolve()
STEP = sys.argv[3]
sys.path.insert(0, str(CLONE / "scripts"))
import contrastive_release as C  # noqa: E402

assert C.ROOT == CLONE
binding = json.loads((CLONE / "results/infra/contrastive_targets_20261118/vn3_binding.json").read_text())
preflight = json.loads(C.PREFLIGHT.read_text())
audit_id = [c["id"] for c in preflight["cells"] if c["mode"] == "audit"][0]
CALIB = "CONTRAST-CALIB-targets-v3-20261118"
OUT = preflight["fresh_root"] + "/calibration_vn3"
HDR = "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n"


def outcome(label, fn):
    try:
        value = fn()
        print(f"[{label}] OK", (json.dumps(value)[:200] if not isinstance(value, dict) else "keys=" + ",".join(sorted(value))[:200]))
        return value
    except Exception as error:  # report the type and the innermost frame
        tb = traceback.extract_tb(error.__traceback__)[-1]
        print(f"[{label}] {type(error).__name__}: {str(error)[:180]} @ {Path(tb.filename).name}:{tb.lineno}")
        return None


def receipts(base, *, relative, sacct_text=None, job_id="12345678"):
    base.mkdir(parents=True, exist_ok=True)
    audit = base / "audit.json"
    audit.write_text(json.dumps({"source_sha256": C.source_digest(), "passed": True}) + "\n")
    tsv = base / "sacct.tsv"
    tsv.write_text(sacct_text if sacct_text is not None else
                   HDR + f"{job_id}|COMPLETED|0:0|5000|8|120\n{job_id}.batch|COMPLETED|0:0|5000|8|\n")
    acc = base / "accounting.json"
    acc.write_text(json.dumps({"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(),
                               "audit_sha256": C.file_sha(audit), "logical_id": audit_id, "job_id": job_id,
                               "sacct": C.pin(tsv, relative=relative)}, indent=2) + "\n")
    return {"audit": C.pin(audit, relative=relative), "audit_sacct": C.pin(acc, relative=relative)}


if STEP == "A":
    hist = json.loads((CLONE / "results/infra/contrastive_comparison_20261116/releases/calibration.job.json").read_text())
    print("historical audit pin:", hist["audit"]["path"])
    ev = {"audit": hist["audit"], "audit_sacct": {"path": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/audit_accounting.json", "sha256": "0" * 64}}
    outcome("A make_job calib, pin assoluto Leonardo", lambda: C.make_job(binding, CALIB, ev, OUT))
    outcome("A0 make_job audit (controllo: nessuna lettura locale)", lambda: C.make_job(binding, audit_id, {}, OUT))

elif STEP == "B":
    ev = receipts(RECEIPTS, relative=False)
    print("pin audit (default di C.pin):", ev["audit"]["path"])
    job = outcome("B1 make_job calib, pin assoluti del Mac", lambda: C.make_job(binding, CALIB, ev, OUT))
    jobfile = CLONE / "results/infra/contrastive_targets_20261118/vn3_calibration.job.json"
    C.write_new(jobfile, job)
    os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
    import contrastive_job as J
    outcome("B2 build_command, ricevute presenti", lambda: J.build_command(json.loads(jobfile.read_text()), "gpu", 1800))
    moved = RECEIPTS.with_name(RECEIPTS.name + "_moved")
    RECEIPTS.rename(moved)  # the Mac path does not exist on Leonardo
    outcome("B3 build_command, ricevute assenti (cluster)", lambda: J.build_command(json.loads(jobfile.read_text()), "gpu", 1800))
    moved.rename(RECEIPTS)

elif STEP == "C":
    inside = CLONE / "results/infra/contrastive_targets_20261118/vn3_receipts"
    ev = receipts(inside, relative=True)
    print("pin audit relativo:", ev["audit"]["path"], "sacct nel JSON:", json.loads((inside / "accounting.json").read_text())["sacct"]["path"])
    job = outcome("C1 make_job calib, pin relativi", lambda: C.make_job(binding, CALIB, ev, OUT))
    C.write_new(CLONE / "results/infra/contrastive_targets_20261118/vn3_calibration_rel.job.json", job)

elif STEP == "C2":  # run inside the SECOND clone (different absolute ROOT, like the cluster checkout)
    os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
    import contrastive_job as J
    job = json.loads((CLONE / "results/infra/contrastive_targets_20261118/vn3_calibration_rel.job.json").read_text())
    outcome("C2 build_command in un altro ROOT, pin relativi", lambda: J.build_command(job, "gpu", 1800))

elif STEP == "D":
    hist = (CLONE / "results/infra/contrastive_comparison_20261116/audit_completed_20261117/accounting_and_integrity.txt").read_text().splitlines()
    rows = "\n".join(hist[:3]) + "\n"
    cols = "JobIDRaw|State|ExitCode|DerivedExitCode|Start|End|ElapsedRaw|TimelimitRaw|AllocCPUS|AllocTRES|TotalCPU|MaxRSS|MaxVMSize|Reason\n"
    print("campi riga 1 storica:", len(hist[0].split("|")), "intestazione ipotizzata:", len(cols.strip().split("|")))
    for label, text in (("D1 storico, senza intestazione", rows),
                        ("D2 storico + intestazione (stesse colonne)", cols + rows),
                        ("D3 stesse colonne storiche, riga COMPLETED 120 min", cols + rows.replace("TIMEOUT", "COMPLETED").replace("|3787|60|8|", "|3787|120|8|", 1)),
                        ("D4 -n -P con i sei campi (senza intestazione)", "64652559|COMPLETED|0:0|3787|8|120\n")):
        base = RECEIPTS / label.split()[0]
        ev = receipts(base, relative=False, sacct_text=text, job_id="64652559")
        outcome(label, lambda: C.validate_audit_accounting(C.checked_pin(ev["audit"]), ev["audit_sacct"]))

elif STEP == "E":
    import contrastive_prepare as P
    f = CLONE / "results/infra/contrastive_targets_20261118/plan.json"
    print("C.pin default:", C.pin(f)["path"])
    print("C.pin relative=True:", C.pin(f, relative=True)["path"])
    print("P._pin default:", P._pin(f)["path"])

elif STEP == "F":  # audit/accounting pins relative, but the sacct pin INSIDE the accounting JSON made like the test helper: C.pin(tsv)
    inside = CLONE / "results/infra/contrastive_targets_20261118/vn3_receipts_abs"
    inside.mkdir(parents=True, exist_ok=True)
    audit = inside / "audit.json"; audit.write_text(json.dumps({"source_sha256": C.source_digest()}) + "\n")
    tsv = inside / "sacct.tsv"; tsv.write_text(HDR + "12345678|COMPLETED|0:0|5000|8|120\n")
    acc = inside / "accounting.json"
    acc.write_text(json.dumps({"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(), "audit_sha256": C.file_sha(audit),
                               "logical_id": audit_id, "job_id": "12345678", "sacct": C.pin(tsv)}) + "\n")
    print("sacct pin interno (C.pin default, come nel test):", json.loads(acc.read_text())["sacct"]["path"][-60:])
    ev = {"audit": C.pin(audit, relative=True), "audit_sacct": C.pin(acc, relative=True)}
    job = outcome("F1 make_job calib", lambda: C.make_job(binding, CALIB, ev, OUT))
    C.write_new(CLONE / "results/infra/contrastive_targets_20261118/vn3_calibration_abs.job.json", job)

elif STEP == "F2":
    os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
    import contrastive_job as J
    job = json.loads((CLONE / "results/infra/contrastive_targets_20261118/vn3_calibration_abs.job.json").read_text())
    outcome("F2 build_command in altro ROOT, sacct interno assoluto assente", lambda: J.build_command(job, "gpu", 1800))
