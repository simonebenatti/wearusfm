"""Scettico R3-M3: l'ID SLURM dell'accounting e' ancorato all'esecuzione dell'audit?"""
import json, os, subprocess, sys, hashlib, time
from pathlib import Path
CL = Path(sys.argv[1]); WORKD = Path(sys.argv[2])
sys.path.insert(0, str(CL / "scripts")); sys.path.insert(0, str(CL / "src"))
import contrastive_release as C
from contrastive_prepare import _read
print("ROOT", C.ROOT, "source", C.source_digest()[:12])
pre = _read(C.PREFLIGHT)
AUDIT_ID = [c["id"] for c in pre["cells"] if c["mode"] == "audit"][0]
print("audit logical id", AUDIT_ID)

def dump(p, v):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v, indent=2) + "\n"); return p

HDR = "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n"
# Report dell'audit "reale" (gia' scritto, poi allocazione andata in TIMEOUT oltre il margine)
audit = dump(WORKD / "audit/report.json", {"source_sha256": C.source_digest(), "passed": True, "steps": 1024})
def acct(name, job_id, rows, logical=AUDIT_ID):
    tsv = WORKD / (name + ".tsv"); tsv.write_text(HDR + "".join(r + "\n" for r in rows))
    return dump(WORKD / (name + ".json"), {"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(),
                "audit_sha256": C.file_sha(audit), "logical_id": logical, "job_id": job_id, "sacct": C.pin(tsv)})

def run(label, fn):
    try:
        r = fn(); print(label, "ACCETTATO", {k: r[k] for k in ("job_id", "state", "elapsed_seconds")} if isinstance(r, dict) and "job_id" in r else "")
    except Exception as e:
        print(label, "RESPINTO", type(e).__name__, str(e)[:110])

# S1: accounting vero del job dell'audit (TIMEOUT, 7387 s): deve essere respinto
real = acct("real_64900000", "64900000", ["64900000|TIMEOUT|0:0|7387|8|120", "64900000.batch|COMPLETED|0:0|7380|8|"])
run("S1 vero job audit TIMEOUT", lambda: C.validate_audit_accounting(audit, real))
# S2: stesso report con la coppia accounting/TSV di un ALTRO job (8 CPU, 120 min, COMPLETED)
other = acct("other_64900001", "64900001", ["64900001|COMPLETED|0:0|7000|8|120", "64900001.batch|COMPLETED|0:0|6999|8|"])
run("S2 report + accounting di altro job", lambda: C.validate_audit_accounting(audit, other))
run("S2p stesso, pin dict", lambda: C.validate_audit_accounting(audit, C.pin(other)))
# S3: con expected_job_id del job reale (non passato da nessun chiamante di produzione)
run("S3 expected_job_id=64900000", lambda: C.validate_audit_accounting(audit, other, expected_job_id="64900000"))
# S4: un job estraneo con risorse diverse e' invece respinto (limiti delle collisioni accidentali)
for lab, row in (("checkout 4CPU/4min", "64900002|COMPLETED|0:0|35|4|4"), ("readout 8CPU/60min", "64900003|COMPLETED|0:0|900|8|60"),
                 ("GPU 32CPU/30min", "64900004|COMPLETED|0:0|700|32|30"), ("altro 8CPU/240min", "64900005|COMPLETED|0:0|5000|8|240")):
    jid = row.split("|")[0]
    a = acct("x_" + jid, jid, [row])
    run("S4 " + lab, lambda a=a: C.validate_audit_accounting(audit, a))

# S5: integrazione reale in make_job (calibrazione) con binding vero (review committata nel clone temporaneo)
pin = json.loads((CL / "results/infra/contrastive_targets_20261118/amendment_20261119/delivery_pin.json").read_text())
block = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": pin["content_commit"],
         "source_sha256": pin["source_sha256"], "science_plan_sha256": pin["science_plan_sha256"],
         "preflight_plan_sha256": pin["preflight_plan_sha256"]}
rv = CL / "docs/reviews/v3m3_review.md"
rv.write_text("# review sintetica\n\n```contrastive-release-review\n" + json.dumps(block, indent=2) + "\n```\n")
g = lambda *a: subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=CL).decode().strip()
g("add", str(rv.relative_to(CL))); g("commit", "-q", "-m", "synthetic review"); rc = g("rev-parse", "HEAD")
t = time.time(); binding = C.bind_review(rc, str(rv.relative_to(CL)), rv); print("bind_review ok in %.1fs" % (time.time() - t))
calib = [c["id"] for c in pre["cells"] if c["mode"] == "calibration"][0]
ev = lambda a: {"audit": C.pin(audit), "audit_sacct": C.pin(a)}
run("S5a make_job calibrazione + accounting altro job", lambda: (C.make_job(binding, calib, ev(other), "/x/out"), {"job_id": "n/a", "state": "", "elapsed_seconds": ""})[1])
run("S5b make_job calibrazione + accounting vero (TIMEOUT)", lambda: (C.make_job(binding, calib, ev(real), "/x/out"), {"job_id": "n/a", "state": "", "elapsed_seconds": ""})[1])
# S6: dispatcher build_command (calibrazione) con l'accounting dell'altro job: supera il gate? (si ferma dopo)
os.environ["WORK"] = str(WORKD / "WORK")
import contrastive_job as J
job = C.make_job(binding, calib, ev(other), str(WORKD / "WORK/wearusfm_runs/contrastive_targets_x/out"))
try:
    J.build_command(job, "gpu", 1800); print("S6 build_command: comando costruito")
except Exception as e:
    print("S6 build_command si ferma dopo il gate:", type(e).__name__, str(e)[:120])
job2 = C.make_job(binding, calib, ev(other), "/x/out"); job2["audit_sacct"] = C.pin(real)
try:
    J.build_command(job2, "gpu", 1800); print("S6b ?")
except Exception as e:
    print("S6b build_command con accounting vero TIMEOUT:", type(e).__name__, str(e)[:120])
