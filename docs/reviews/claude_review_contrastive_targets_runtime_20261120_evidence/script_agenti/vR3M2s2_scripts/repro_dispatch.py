"""Dispatcher (contrastive_job.build_command) and runtime (contrastive_run.validate_release) with the fake binding."""
import copy, json, os, sys
from pathlib import Path
D = Path(sys.argv[1]).resolve(); W = Path(sys.argv[2]).resolve()
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
sys.path.insert(0, str(D / "src")); sys.path.insert(0, str(D / "scripts"))
os.chdir(D)
import contrastive_job as J, contrastive_run as RUN, contrastive_release as C
REL = D / "results/infra/contrastive_targets_20261118/release_sim"
real_job = json.loads((REL / "audit.job.json").read_text())
fake_job = json.loads((W / "audit_fake.job.json").read_text())
print("job falso: package_review.review_commit =", fake_job["package_review"]["review_commit"][:12], "path =", fake_job["package_review"]["original_review_path"])

def attempt(label, fn):
    try:
        out = fn(); print(label, "-> OK", str(out)[:160])
    except Exception as e:
        print(label, "->", type(e).__name__, str(e)[:160])

# Dispatcher: audit job lacks config/inputs (evidence vuota): a failure on 'config' means binding/cell checks passed.
attempt("DISPATCH job reale", lambda: J.build_command(real_job, "cpu", 7200))
attempt("DISPATCH job falso", lambda: J.build_command(fake_job, "cpu", 7200))
ctl = copy.deepcopy(fake_job); ctl["package_review"]["original_review_sha256"] = "0" * 64
attempt("DISPATCH controllo (original_review_sha256 manomesso)", lambda: J.build_command(ctl, "cpu", 7200))
ctl2 = copy.deepcopy(fake_job); ctl2["id"] = "CONTRAST-CALIB-targets-v3-20261118"
attempt("DISPATCH controllo (cella diversa)", lambda: J.build_command(ctl2, "cpu", 7200))

# Runtime: only the four input files are simulated (their sha), everything else real.
plan = json.loads(C.PLAN.read_text()); pins = plan["input_sha256"]
fake_inputs = {k: W / f"input_{k}.bin" for k in pins}
for p in fake_inputs.values(): p.write_bytes(b"x")
real_sha = RUN.sha
RUN.sha = lambda p: pins[[k for k, v in fake_inputs.items() if Path(v) == Path(p)][0]] if Path(p) in map(Path, fake_inputs.values()) else real_sha(p)
cfg = C.PLAN  # any pinned file: validate_release only compares config_sha256 at this point
binding_fake = json.loads((W / "binding_fake.json").read_text())
release = {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration", "source_sha256": C.source_digest(),
           "config_sha256": real_sha(cfg), "world_size": 4, "workers_per_rank": 6, "inputs": dict(pins),
           "review_pin": binding_fake["review_commit"], "package_review": binding_fake, "time_limit_s": 1770}
attempt("RUNTIME release con binding falso", lambda: RUN.validate_release(release, "calibration", cfg, fake_inputs, time_limit_s=1770))
r2 = copy.deepcopy(release); r2["review_pin"] = "3" * 40
attempt("RUNTIME controllo review_pin != review_commit", lambda: RUN.validate_release(r2, "calibration", cfg, fake_inputs, time_limit_s=1770))
r3 = copy.deepcopy(release); r3["package_review"]["original_review_sha256"] = "0" * 64
attempt("RUNTIME controllo original_review_sha256 manomesso", lambda: RUN.validate_release(r3, "calibration", cfg, fake_inputs, time_limit_s=1770))
