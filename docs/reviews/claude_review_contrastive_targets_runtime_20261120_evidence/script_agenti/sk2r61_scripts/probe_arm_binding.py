"""Scettico 2, R6-1: il braccio della cella e' legato a config/release eseguiti?

Uso: python -B probe_arm_binding.py <clone temporaneo a 0efa4d45> <dir dati temporanea>
Binding di review REALE (bind_review + validate_review_binding con verifica Git), audit
accounting REALE, celle REALI del preflight_plan. Stub solo di DATA_ROOTS (radici dati di
Leonardo assenti sul Mac) e di validate_input_bindings (pin SHA256 dei dati reali).
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
TMP = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(CLONE / "src"))
sys.path.insert(0, str(CLONE / "scripts"))
import contrastive_prepare as P  # noqa: E402
import contrastive_release as C  # noqa: E402
import contrastive_job as J  # noqa: E402

assert J.ROOT == CLONE and C.ROOT == CLONE, (J.ROOT, C.ROOT)
print("ROOT", J.ROOT)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def git(*args):
    return subprocess.check_output(["git", "-c", "user.name=Probe", "-c", "user.email=probe@localhost.invalid", *args],
                                   cwd=CLONE, stderr=subprocess.PIPE).decode().strip()


# 1) Review reale con blocco di binding, committata SOLO nel clone temporaneo.
reviewed = git("rev-parse", "HEAD")
payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": reviewed,
           "source_sha256": P.source_digest(), "science_plan_sha256": C.file_sha(C.PLAN),
           "preflight_plan_sha256": C.file_sha(C.PREFLIGHT)}
review_rel = "docs/reviews/probe_sk2_r61_review.md"
review = CLONE / review_rel
if not review.exists():
    review.write_text("# Probe review\n\n```contrastive-release-review\n" + json.dumps(payload, indent=2) + "\n```\n")
    git("add", review_rel)
    git("commit", "-qm", "probe review (temporary clone only)")
review_commit = git("rev-parse", "HEAD")
binding = C.bind_review(review_commit, review_rel, review)
print("binding reale: validate_review_binding ->", C.validate_review_binding(binding)["status"])

# 2) Audit e accounting sintetici ma validati dal codice reale.
audit = dump(TMP / "audit/audit.json", {"source_sha256": P.source_digest(), "passed": True})
tsv = TMP / "audit/sacct.tsv"
tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n123|COMPLETED|0:0|7000|8|120\n")
accounting = dump(TMP / "audit/accounting.json", {"schema": C.AUDIT_SCHEMA, "source_sha256": P.source_digest(),
                  "audit_sha256": sha(audit), "logical_id": "CONTRAST-AUDIT-targets-v3-20261118",
                  "job_id": "123", "sacct": pin(tsv)})
print("audit accounting reale ->", C.validate_audit_accounting(audit, pin(accounting))["logical_id"])

# 3) Altre evidenze a monte (contenuto irrilevante per build_command: solo hash).
cuda = dump(TMP / "cuda/report.json", {"probe": "cuda"})
cohort = dump(TMP / "cohort/cohort.json", {"probe": "cohort"})
calib = dump(TMP / "calibration/report.json", {"probe": "calibration"})
calib_summary = dump(TMP / "calibration/run/summary.json", {"probe": "calibration summary"})
configs = {arm: dump(TMP / f"cohort/{arm}.json", {"max_steps": 576, "contrastive": {"arm": arm}}) for arm in ("C0", "CN", "CR")}
# Pin dei dati come li darebbe l'operatore (percorsi Leonardo, SHA del piano scientifico).
science = P._read(C.PLAN)
plan_inputs = {k: {"path": "/leonardo/" + k, "sha256": v} for k, v in science["input_sha256"].items()}
# Copie locali per far girare checked_file sul Mac (validate_input_bindings stub: richiede i dati reali).
local_inputs = {k: pin(dump(TMP / f"inputs/{k}.json", {"probe": k})) for k in ("manifest", "scales", "window_seconds", "pose_stats")}
roots = [TMP / "data1", TMP / "data2"]
for r in roots:
    r.mkdir(parents=True, exist_ok=True)
J.DATA_ROOTS = {r.resolve() for r in roots}
J.validate_input_bindings = lambda pins: None
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
fresh = Path(P._read(C.PREFLIGHT)["fresh_root"])


def release_for(arm, phase, extra=None):
    body = {"status": "APPROVED_FOR_PREPARATION", "stage": "benchmark", "phase": phase, "arm": arm,
            "config_sha256": sha(configs[arm]), "source_sha256": P.source_digest(), "time_limit_s": 1770,
            "world_size": 4, "workers_per_rank": 6, "review_pin": binding["review_commit"],
            "inputs": {k: v["sha256"] for k, v in local_inputs.items()}, "audit_sha256": sha(audit),
            "package_review": binding, "audit_sacct": pin(accounting), "cuda_report": pin(cuda),
            "cohort_sha256": sha(cohort), "calibration_sha256": sha(calib)}
    body.update(extra or {})
    return pin(dump(TMP / f"releases/{arm}_{phase}.json", body))


def evidence(arm, phase, extra=None, release_extra=None):
    ev = {"config": pin(configs[arm]), "release": release_for(arm, phase, release_extra), "inputs": plan_inputs,
          "roots": [str(r) for r in roots], "audit": pin(audit), "audit_sacct": pin(accounting), "cohort": pin(cohort),
          "calibration_report": pin(calib), "calibration_summary": pin(calib_summary), "cuda_report": pin(cuda)}
    ev.update(extra or {})
    return ev


def make(cell_id, ev, out_name):
    job = C.make_job(binding, cell_id, ev, str(fresh / out_name))   # make_job REALE
    job = copy.deepcopy(job)
    job["inputs"] = local_inputs                                    # solo per checked_file sul Mac
    return job


def arg(command, flag):
    return command[command.index(flag) + 1]


def show(label, job):
    try:
        command = J.build_command(job, "gpu", 1800)
    except Exception as error:  # noqa: BLE001
        print(f"{label}: RIFIUTATO -> {type(error).__name__}: {error}")
        return None
    cfg = json.loads(Path(arg(command, "--config")).read_text())
    rel = json.loads(Path(arg(command, "--release")).read_text())
    print(f"{label}: ACCETTATO; cella={job['id']} job.arm={job.get('arm')} job.repeat={job.get('repeat')} -> "
          f"config eseguito arm={cfg['contrastive']['arm']}, release arm={rel['arm']} phase={rel['phase']}; "
          f"script={Path(command[command.index('--config') - 1]).name}")
    return command


# A) Cella full C0 con config/release CN.
jobA = make("CONTRAST-BENCH-C0-targets-v3-20261118", evidence("CN", "full"), "probe_bench_C0")
show("A  BENCH-C0 + config/release CN", jobA)
# A0) Controllo positivo: stessa cella con il suo braccio.
show("A0 BENCH-C0 + config/release C0", make("CONTRAST-BENCH-C0-targets-v3-20261118", evidence("C0", "full"), "probe_bench_C0b"))
# A1) Controllo negativo: etichetta arm del job cambiata -> rifiuto (solo etichetta).
bad = copy.deepcopy(jobA); bad["arm"] = "CN"
show("A1 BENCH-C0 con job.arm=CN (etichetta)", bad)
# A2) make_job vieta di sovrascrivere arm dall'evidenza.
try:
    C.make_job(binding, "CONTRAST-BENCH-C0-targets-v3-20261118", dict(evidence("C0", "full"), arm="CN"), str(fresh / "x"))
    print("A2 make_job con evidence.arm: ACCETTATO")
except ValueError as error:
    print("A2 make_job con evidence.arm: RIFIUTATO ->", error)

# B) Cella reload CN-A con config/release CR e fork dal full CR.
fork_dir = TMP / "full_CR_run"
fork = dump(fork_dir / "checkpoint_step0000544.pt", {"probe": "fork CR"})
cont = dump(fork_dir / "summary.json", {"probe": "summary CR", "contrastive": {"config": {"arm": "CR"}}})
fork_pins = {"fork_checkpoint": pin(fork), "continuous_summary": pin(cont)}
jobB = make("CONTRAST-RELOAD-CN-A-targets-v3-20261118", evidence("CR", "resume_a", fork_pins, fork_pins), "probe_reload_CN_A")
show("B  RELOAD-CN-A + config/release CR (fork CR)", jobB)
# B1) Controllo: il repeat invece e' legato alla release (phase resume_b su cella A -> rifiuto).
jobB1 = make("CONTRAST-RELOAD-CN-A-targets-v3-20261118", evidence("CN", "resume_b", fork_pins, fork_pins), "probe_reload_CN_A2")
show("B1 RELOAD-CN-A + release CN phase resume_b", jobB1)
# B2) Cella reload CN-A con config/release C0 phase resume_a: build_command non lo vede; il runtime si'.
jobB2 = make("CONTRAST-RELOAD-CN-A-targets-v3-20261118", evidence("C0", "resume_a", fork_pins, fork_pins), "probe_reload_CN_A3")
show("B2 RELOAD-CN-A + config/release C0 phase resume_a", jobB2)
# C) Cella full CR con config/release C0 (attiverebbe observe_control del C0 sotto l'ID CR).
show("C  BENCH-CR + config/release C0", make("CONTRAST-BENCH-CR-targets-v3-20261118", evidence("C0", "full"), "probe_bench_CR"))
