"""Scettico R4-N7: make_job/build_command accettano CUDA senza calibrazione accettata e full con ricevuta CUDA fallita.

Pacchetto sintetico in una cartella temporanea (come la fixture di tests/cpu/test_contrastive_release.py);
nessuna scrittura nella copia di sola lettura.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

work = Path(sys.argv[1]).resolve()
work.mkdir(parents=True, exist_ok=False)

import contrastive_release as C
import contrastive_job as J
import contrastive_run as runner


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


root = work / "repo"
root.mkdir()
source = root / "scripts/contrastive_one.py"
source.parent.mkdir()
source.write_text("# reviewed execution\n")
digest = hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest()
science = dump(root / "science.json", {"input_sha256": {"manifest": "1" * 64}})
cells = [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200},
         {"id": "new-cuda", "mode": "cuda", "arm": None, "seconds": 1800},
         {"id": "new-full-CN", "mode": "full", "arm": "CN", "seconds": 1800}]
preflight = dump(root / "preflight.json", {"source_sha256": digest,
                 "science_plan": {"path": "science.json", "sha256": C.file_sha(science)}, "cells": cells})
C.ROOT, C.PLAN, C.PREFLIGHT = root, science, preflight
C.source_paths = lambda: [source]
C.source_digest = lambda: digest


def git(*args):
    return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                    "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                                   cwd=root, stderr=subprocess.PIPE).decode().strip()


git("init", "-q")
git("add", ".")
git("commit", "-qm", "Reviewed package")
payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": git("rev-parse", "HEAD"),
           "source_sha256": digest, "science_plan_sha256": C.file_sha(science), "preflight_plan_sha256": C.file_sha(preflight)}
review = root / "review.md"
review.write_text("# review\n\n```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
git("add", "review.md")
git("commit", "-qm", "Actual package review")
binding = C.bind_review(git("rev-parse", "HEAD"), "review.md", review)

# Ricevute di audit valide (come accounting() della fixture)
audit = dump(root / "audit.json", {"source_sha256": digest, "passed": True})
tsv = root / "sacct.tsv"
tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n123|COMPLETED|0:0|7200|8|120\n"
               "123.batch|COMPLETED|0:0|7199|8|120\n")
receipt = dump(root / "accounting.json", {"schema": C.AUDIT_SCHEMA, "source_sha256": digest, "audit_sha256": C.file_sha(audit),
                                          "logical_id": "new-audit", "job_id": "123", "sacct": C.pin(tsv)})
upstream = {"audit": C.pin(audit), "audit_sacct": C.pin(receipt)}

# 1) Cella CUDA con calibrazione FALLITA (status/failure_reasons) e checkpoint qualsiasi
failed_calibration = dump(root / "calib/calibration_report.json",
                          {"status": "FAILED_REQUIRES_REVIEW", "failure_reasons": ["coverage_gate_failed"]})
checkpoint = root / "calib/checkpoint.pt"
checkpoint.write_bytes(b"not a calibrated checkpoint")
job = C.make_job(binding, "new-cuda", {**upstream, "calibration_checkpoint": C.pin(checkpoint),
                                        "calibration_report": C.pin(failed_calibration)}, "/remote/out-cuda")
print("1 make_job CUDA con calibrazione fallita:", job["mode"], job["status"])

# 2) Celle full con ricevuta CUDA fallita oppure assente
failed_cuda = dump(root / "cuda/report.json", {"passed": False})
job = C.make_job(binding, "new-full-CN", {**upstream, "cuda_report": C.pin(failed_cuda)}, "/remote/out-full")
print("2a make_job full con CUDA passed=False:", job["mode"], job["status"], "cuda_report" in job)
job = C.make_job(binding, "new-full-CN", dict(upstream), "/remote/out-full")
print("2b make_job full senza cuda_report:", job["mode"], job["status"], "cuda_report" in job)

# 3) La stessa ricevuta e' respinta solo da validate_cuda_prerequisite (chiamata da contrastive_run.validate_release)
try:
    C.validate_cuda_prerequisite(C.pin(failed_cuda), {"source_sha256": digest, "initial_state_sha256": "0" * 64},
                                 {"manifest": "1" * 64, "scales": "2" * 64, "window_seconds": "3" * 64, "pose_stats": "4" * 64},
                                 root / "calib")
    print("3 validate_cuda_prerequisite: ACCETTATA (inatteso)")
except ValueError as error:
    print("3 validate_cuda_prerequisite rifiuta:", error)

# 4) make_job non rifiuta l'audit mancante solo per la cella audit; per le altre lo richiede
try:
    C.make_job(binding, "new-cuda", {"calibration_checkpoint": C.pin(checkpoint)}, "/remote/out-cuda2")
    print("4 make_job CUDA senza audit: ACCETTATA (inatteso)")
except (KeyError, ValueError) as error:
    print("4 make_job CUDA senza audit rifiuta:", type(error).__name__, error)

# 5) Dispatcher (contrastive_job.build_command) per la cella CUDA con calibrazione fallita:
#    stesse sostituzioni della fixture job_env di tests/cpu/test_contrastive_job.py
os.environ["WORK"] = str(work / "work")
(work / "work/wearusfm_runs").mkdir(parents=True)
roots = [work / "data1", work / "data2"]
for path in roots:
    path.mkdir()
J.DATA_ROOTS = set(roots)
J.source_digest = lambda: "a" * 64
J.validate_input_bindings = lambda _: None
runner.validate_recipe = lambda *args: None
C.validate_review_binding = lambda value: None
C.validate_audit_accounting = lambda *args: None
dispatcher_preflight = dump(work / "dispatcher_preflight.json", {"fresh_root": str(work / "work/wearusfm_runs"),
                            "cells": [{"id": "test-cuda", "mode": "cuda", "arm": None, "repeat": None, "seconds": 1800}]})
C.PREFLIGHT = dispatcher_preflight


def jpin(name, value=None, raw=None):
    path = work / "disp" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_bytes(raw)
    else:
        path.write_text(json.dumps({} if value is None else value))
    return {"path": str(path), "sha256": runner.sha(path)}


cuda_job = {"id": "test-cuda", "arm": None, "repeat": None, "package_review": {"fixture": True}, "mode": "cuda",
            "status": "APPROVED_FOR_PREFLIGHT", "source_sha256": "a" * 64, "review": jpin("review.json"),
            "out": str(work / "work/wearusfm_runs/cuda"), "config": jpin("config.json"), "roots": list(map(str, roots)),
            "inputs": {k: jpin(k + ".json") for k in ("manifest", "scales", "window_seconds", "pose_stats")},
            "audit_sacct": jpin("audit_accounting.json"), "audit": jpin("audit.json"),
            "calibration_report": jpin("calibration.json", {"status": "FAILED_REQUIRES_REVIEW",
                                                             "failure_reasons": ["coverage_gate_failed"]}),
            "calibration_checkpoint": jpin("checkpoint.pt", raw=b"x")}
cuda_job["package_review"]["review"] = cuda_job["review"]
command = J.build_command(cuda_job, "gpu", 1800)
print("5 build_command cuda con calibrazione fallita: comando costruito;",
      "contrastive_cuda_check.py" in " ".join(command), "--calibration-report" in command, "--cohort" in command)
