"""Independent reproduction for R3-N4 (skeptic 1). Read-only on the clone; fixtures in n4v1_data.

Usage: python -B repro.py <clone_root> <data_dir>
"""
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
DATA = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(CLONE / "src"))
sys.path.insert(0, str(CLONE / "scripts"))
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"

import contrastive_release as C  # noqa: E402
import contrastive_run as Rn  # noqa: E402
import contrastive_job as J  # noqa: E402
import contrastive_prepare as P  # noqa: E402

assert Path(C.__file__).resolve().is_relative_to(CLONE), C.__file__
assert Path(J.__file__).resolve().is_relative_to(CLONE), J.__file__
assert Path(Rn.__file__).resolve().is_relative_to(CLONE), Rn.__file__

results = []


def case(name, fn):
    try:
        out = fn()
        results.append((name, "ACCEPTED", out))
    except Exception as e:  # noqa: BLE001
        results.append((name, "REJECTED", f"{type(e).__name__}: {e}"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def fsha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if DATA.exists():
    shutil.rmtree(DATA)
DATA.mkdir(parents=True)

SRC = P.source_digest()
preflight = json.loads(C.PREFLIGHT.read_text())
AUDIT_ID = [c["id"] for c in preflight["cells"] if c["mode"] == "audit"][0]
CALIB_ID = [c["id"] for c in preflight["cells"] if c["mode"] == "calibration"][0]
FRESH = preflight["fresh_root"]
print("source_digest", SRC)
print("audit cell", AUDIT_ID, "calibration cell", CALIB_ID, "fresh_root", FRESH)

# ---------- audit + accounting fixtures (real preflight logical ID) ----------
audit = dump(DATA / "audit.json", {"source_sha256": SRC, "passed": True})


def make_accounting(name, job_id, elapsed=7000):
    tsv = DATA / f"sacct_{name}.tsv"
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n"
                   f"{job_id}|COMPLETED|0:0|{elapsed}|8|120\n")
    return dump(DATA / f"accounting_{name}.json",
                {"schema": C.AUDIT_SCHEMA, "source_sha256": SRC, "audit_sha256": fsha(audit),
                 "logical_id": AUDIT_ID, "job_id": job_id, "sacct": C.pin(tsv)})


acc = make_accounting("A", "111")
acc_pin = C.pin(acc)

# A: validate_audit_accounting itself
case("A1 accounting as string path", lambda: C.validate_audit_accounting(audit, str(acc))["job_id"])
case("A2 accounting as exact pin", lambda: C.validate_audit_accounting(audit, acc_pin)["job_id"])
case("A3 accounting pin with wrong sha", lambda: C.validate_audit_accounting(audit, {**acc_pin, "sha256": "0" * 64}))

# ---------- validate_release (calibration stage) with only unrelated heavy gates stubbed ----------
cfg = dump(DATA / "calibration.config.json", {"fixture": "config"})
inputs = {k: dump(DATA / f"in_{k}.json", {"k": k}) for k in ("manifest", "scales", "window_seconds", "pose_stats")}
pins = {k: fsha(p) for k, p in inputs.items()}
review_file = dump(DATA / "review.md.json", {"review": "fixture"})
binding = {"schema": C.BINDING_SCHEMA, "review_commit": "a" * 40, "review": C.pin(review_file),
           "fixture": True}

# Stubs: review binding (git), audit report content, input coverage pins -- none touch audit_sacct.
C.validate_review_binding = lambda b, **kw: {"source_sha256": SRC, "reviewed_commit": "a" * 40}
Rn.validate_audit = lambda *a, **k: {"passed": True}
Rn.validate_input_bindings = lambda p: None
J.validate_input_bindings = lambda p: None


def release_with(audit_sacct):
    return {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration", "source_sha256": SRC,
            "config_sha256": fsha(cfg), "world_size": 4, "workers_per_rank": 6, "inputs": dict(pins),
            "review_pin": "a" * 40, "package_review": binding, "time_limit_s": 1770,
            "audit_sha256": fsha(audit), "audit_sacct": audit_sacct}


def run_release(rel):
    return Rn.validate_release(rel, "calibration", cfg, inputs, None, 1770, audit_path=audit)


case("B0 control: release audit_sacct exact pin", lambda: run_release(release_with(acc_pin)))
case("B1 release audit_sacct STRING path (unpinned)", lambda: run_release(release_with(str(acc))))
case("B2 release audit_sacct pin with wrong sha", lambda: run_release(release_with({**acc_pin, "sha256": "0" * 64})))
# B3: after the release is written, the accounting bytes change to a different (valid) allocation:
acc_swapped = make_accounting("A_swap_tmp", "999")
rel_str = release_with(str(acc))
rel_pin = release_with(copy.deepcopy(acc_pin))
orig_bytes = acc.read_bytes()
acc.write_bytes(acc_swapped.read_bytes())
case("B3 string release, accounting swapped to job 999", lambda: (run_release(rel_str),
                                                                   C.validate_audit_accounting(audit, str(acc))["job_id"]))
case("B4 pinned release, accounting swapped to job 999", lambda: run_release(rel_pin))
acc.write_bytes(orig_bytes)

# ---------- dispatcher build_command (calibration) ----------
roots = [DATA / "root_work", DATA / "root_scratch"]
for r in roots:
    r.mkdir()
J.DATA_ROOTS = {r.resolve() for r in roots}


def calib_job(job_audit_sacct, release_audit_sacct, out=FRESH + "/calibration_n4v1"):
    rel = dump(DATA / "release_calibration.json", release_with(release_audit_sacct))
    return {"status": "APPROVED_FOR_PREFLIGHT", "mode": "calibration", "source_sha256": SRC, "id": CALIB_ID,
            "arm": None, "package_review": binding, "review": binding["review"], "out": out,
            "audit": C.pin(audit), "audit_sacct": job_audit_sacct, "config": C.pin(cfg),
            "inputs": {k: C.pin(p) for k, p in inputs.items()}, "roots": [str(r) for r in roots],
            "release": C.pin(rel)}


def build(job, kind="gpu", secs=1800):
    cmd = J.build_command(job, kind, secs)
    return "command ok; --release " + cmd[cmd.index("--release") + 1].split("/")[-1]


case("C0 control: job pin == release pin", lambda: build(calib_job(acc_pin, acc_pin)))
case("C1 job audit_sacct STRING", lambda: build(calib_job(str(acc), str(acc))))
case("C2 job pin, release STRING", lambda: build(calib_job(acc_pin, str(acc))))

# ---------- make_job: --result-out is copied without fresh_root check ----------
evidence_audit = {"config": C.pin(cfg), "inputs": {k: C.pin(p) for k, p in inputs.items()},
                  "roots": [str(r) for r in roots]}
# make_job checks evidence input pins against the science plan; give it the plan's pins via a patched PLAN read.
science = json.loads(C.PLAN.read_text())
evidence_audit_mj = copy.deepcopy(evidence_audit)
for k in evidence_audit_mj["inputs"]:
    evidence_audit_mj["inputs"][k]["sha256"] = science["input_sha256"][k]

outs = {"D1 out in another cohort": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/other_cohort/audit",
        "D2 out == fresh_root": FRESH,
        "D3 out outside WORK": "/tmp/n4v1_elsewhere/audit",
        "D4 out relative": "relative/audit",
        "D0 control: out under fresh_root": FRESH + "/audit_n4v1"}
made = {}
for name, out in outs.items():
    def mk(out=out, name=name):
        job = C.make_job(binding, AUDIT_ID, evidence_audit_mj, out)
        made[name] = job
        return "job.out=" + job["out"]
    case("make_job " + name, mk)

# Dispatch those jobs (restore fixture input pins so later checks are reachable for the control).
Rn.validate_recipe = lambda *a, **k: None
for name, job in made.items():
    job = copy.deepcopy(job)
    job["inputs"] = {k: C.pin(p) for k, p in inputs.items()}

    def disp(job=job):
        cmd = J.build_command(job, "cpu", 7200)
        return "command ok; --out " + cmd[cmd.index("--out") + 1]
    case("dispatch " + name, disp)

# D5: make_job for a non-audit cell accepts an UNPINNED audit_sacct string; dispatcher rejects it.
evidence_calib = {**copy.deepcopy(evidence_audit_mj), "audit": C.pin(audit), "audit_sacct": str(acc),
                  "release": C.pin(dump(DATA / "release_calibration.json", release_with(str(acc))))}
case("D5 make_job calibration with audit_sacct STRING",
     lambda: "job.audit_sacct=" + repr(C.make_job(binding, CALIB_ID, evidence_calib, FRESH + "/calib_n4v1")["audit_sacct"]))
case("D5b make_job calibration with audit_sacct pin wrong sha",
     lambda: C.make_job(binding, CALIB_ID, {**evidence_calib, "audit_sacct": {**acc_pin, "sha256": "0" * 64}}, FRESH + "/calib_n4v1"))

for name, status, detail in results:
    print(f"{name:55s} {status:9s} {detail}")
