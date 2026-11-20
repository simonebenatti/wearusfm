"""Probe R5-7 (verifier 2): real bind_review -> make_job -> contrastive_job.build_command on the
actual package (plans/sources at 0efa4d45) in a throwaway clone; the review lives in a local,
unpublished commit of that clone. Physical inputs exist only on Leonardo, so the expected stop
is the first input pin (FileNotFoundError) AFTER binding, review, cell, output, config and recipe."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TMP = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import contrastive_release as C
import contrastive_job as J
from contrastive_prepare import source_digest, _read

REVIEWED = "0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba"
REL = "docs/reviews/probe_v57b_review.md"


def git(*args):
    return subprocess.check_output(["git", "-c", "user.name=probe", "-c", "user.email=probe@example.invalid", *args],
                                   cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()


def attempt(label, fn):
    try:
        value = fn()
        print(label, "OK", value if not isinstance(value, (dict, list)) else "")
        return value
    except Exception as error:  # report, never hide
        print(label, type(error).__name__, str(error)[:160])
        return None


payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": REVIEWED,
           "source_sha256": source_digest(), "science_plan_sha256": C.file_sha(C.PLAN),
           "preflight_plan_sha256": C.file_sha(C.PREFLIGHT)}
review = ROOT / REL
review.write_text("# Probe review (throwaway clone)\n\n```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
git("add", REL)
git("commit", "-qm", "probe review (throwaway, unpublished)")
review_commit = git("rev-parse", "HEAD")
binding = attempt("bind_review", lambda: C.bind_review(review_commit, REL, review))
attempt("validate_review_binding", lambda: C.validate_review_binding(binding)["reviewed_commit"])

science = _read(C.PLAN)
preflight = _read(C.PREFLIGHT)
fresh = preflight["fresh_root"]
inputs = {k: {"path": f"/leonardo_work/IscrB_WearUsFM/probe_absent/{k}.json", "sha256": v}
          for k, v in science["input_sha256"].items()}
config_pin = C.pin(ROOT / "results/infra/contrastive_targets_20261118/review_configs/calibration.config.json", relative=True)
evidence = {"inputs": inputs, "config": config_pin, "roots": sorted(str(p) for p in J.DATA_ROOTS)}
audit_cell = next(c["id"] for c in preflight["cells"] if c["mode"] == "audit")
job = attempt("make_job audit", lambda: C.make_job(binding, audit_cell, evidence, fresh + "/probe_audit"))
print("job keys", sorted(job) if job else None)
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
attempt("build_command audit cpu 7200", lambda: J.build_command(job, "cpu", 7200))
attempt("build_command audit gpu 1800 (must refuse)", lambda: J.build_command(job, "gpu", 1800))
# Calibration cell without published audit/accounting: make_job must refuse.
calib_cell = next(c["id"] for c in preflight["cells"] if c["mode"] == "calibration")
attempt("make_job calibration without audit", lambda: C.make_job(binding, calib_cell, evidence, fresh + "/probe_cal"))
print("recipe equal to review config:", J._read(ROOT / config_pin["path"]) is not None)
