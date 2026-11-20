"""r3 review: audit-accounting gate (direct, release, dispatcher, runtime, readout).

argv[1]: temporary clone positioned on the synthetic audit-only release commit created by r3_release.py.
"""
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TMP = Path(os.environ["R3TMP"]) / "r3_audit_work"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests/cpu"))
import contrastive_release as C  # noqa: E402
import contrastive_prepare as P  # noqa: E402
import contrastive_job as J  # noqa: E402
import contrastive_run as RUN  # noqa: E402
import contrastive_benchmark_readout as B  # noqa: E402

REL = ROOT / "results/infra/contrastive_targets_20261118/release_audit_r3"
binding = json.loads((REL / "binding.json").read_text())
SRC = P.source_digest()
AUDIT_ID = "CONTRAST-AUDIT-targets-v3-20261118"
HEADER = "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw"


def report(label, fn):
    try:
        value = fn()
        out = "ACCEPTED" + ("" if value is None else " " + json.dumps(value)[:120])
    except Exception as error:
        out = "REJECTED " + type(error).__name__ + ": " + str(error)[:140]
    print(f"{label:66s} {out}", flush=True)


counter = [0]


def make(rows=None, *, header=HEADER, audit_source=None, acc=None, sacct_text=None, job_id="64900001"):
    counter[0] += 1
    d = TMP / f"case{counter[0]:03d}"
    d.mkdir()
    audit = d / "audit.json"
    audit.write_text(json.dumps({"source_sha256": audit_source or SRC, "passed": True}))
    tsv = d / "sacct.tsv"
    if sacct_text is None:
        rows = rows if rows is not None else [f"{job_id}|COMPLETED|0:0|7000|8|120", f"{job_id}.batch|COMPLETED|0:0|7000|8|",
                                              f"{job_id}.extern|COMPLETED|0:0|7000|8|"]
        sacct_text = header + "\n" + "\n".join(rows) + "\n"
    tsv.write_text(sacct_text)
    accounting = {"schema": C.AUDIT_SCHEMA, "source_sha256": SRC, "audit_sha256": C.file_sha(audit),
                  "logical_id": AUDIT_ID, "job_id": job_id, "sacct": C.pin(tsv)}
    if acc:
        for key, value in acc.items():
            if value is None:
                accounting.pop(key)
            else:
                accounting[key] = value
    path = d / "accounting.json"
    path.write_text(json.dumps(accounting))
    return audit, path, tsv


J0 = "64900001"
print("== validate_audit_accounting diretto ==")
cases = {
    "G1 COMPLETED 0:0 7000 s 8 CPU 120 min + .batch/.extern": {},
    "G2 ElapsedRaw = 7200 (limite incluso)": {"rows": [f"{J0}|COMPLETED|0:0|7200|8|120"]},
    "G3 ElapsedRaw = 0": {"rows": [f"{J0}|COMPLETED|0:0|0|8|120"]},
    "G4 header con colonne in piu' e ordine diverso": {"sacct_text": "State|JobIDRaw|JobName|ExitCode|AllocCPUS|TimelimitRaw|ElapsedRaw\nCOMPLETED|" + J0 + "|contrastive_cpu|0:0|8|120|7100\n"},
    "G5 formato sacct -p (| finale)": {"sacct_text": HEADER + "|\n" + f"{J0}|COMPLETED|0:0|7100|8|120|\n"},
    "G6 altra allocazione nel TSV oltre alla propria": {"rows": ["64900002|TIMEOUT|0:0|7387|8|120", f"{J0}|COMPLETED|0:0|7100|8|120"]},
    "R1 ElapsedRaw 7201": {"rows": [f"{J0}|COMPLETED|0:0|7201|8|120"]},
    "R2 COMPLETED ma 7380 s (OverTimeLimit 180 s)": {"rows": [f"{J0}|COMPLETED|0:0|7380|8|120"]},
    "R3 TIMEOUT 0:0 (come 64652559) con .batch COMPLETED": {"rows": [f"{J0}|TIMEOUT|0:0|7387|8|120", f"{J0}.batch|COMPLETED|0:0|7387|8|"]},
    "R4 storico 64652559 letterale (TIMEOUT, 3787 s, 60 min)": {"rows": [f"{J0}|TIMEOUT|0:0|3787|8|60", f"{J0}.batch|COMPLETED|0:0|3787|8|"]},
    "R5 FAILED 1:0": {"rows": [f"{J0}|FAILED|1:0|100|8|120"]},
    "R6 COMPLETED exit 0:15": {"rows": [f"{J0}|COMPLETED|0:15|100|8|120"]},
    "R7 CANCELLED by 123": {"rows": [f"{J0}|CANCELLED by 123|0:0|100|8|120"]},
    "R8 OUT_OF_MEMORY 0:125": {"rows": [f"{J0}|OUT_OF_MEMORY|0:125|100|8|120"]},
    "R9 AllocCPUS 16": {"rows": [f"{J0}|COMPLETED|0:0|100|16|120"]},
    "R10 TimelimitRaw 60 (lancio senza --time=02:00:00)": {"rows": [f"{J0}|COMPLETED|0:0|100|8|60"]},
    "R11 TimelimitRaw 02:00:00 (non in minuti)": {"rows": [f"{J0}|COMPLETED|0:0|100|8|02:00:00"]},
    "R12 ElapsedRaw negativo": {"rows": [f"{J0}|COMPLETED|0:0|-5|8|120"]},
    "R13 ElapsedRaw 7000.0": {"rows": [f"{J0}|COMPLETED|0:0|7000.0|8|120"]},
    "R14 solo righe .batch/.extern (manca allocazione)": {"rows": [f"{J0}.batch|COMPLETED|0:0|7000|8|", f"{J0}.extern|COMPLETED|0:0|7000|8|"]},
    "R15 due righe di allocazione (duplicati/requeue)": {"rows": [f"{J0}|COMPLETED|0:0|7000|8|120", f"{J0}|COMPLETED|0:0|7000|8|120"]},
    "R16 header JobID invece di JobIDRaw": {"header": "JobID|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw"},
    "R17 TSV con tabulazioni": {"sacct_text": HEADER.replace("|", "\t") + "\n" + f"{J0}\tCOMPLETED\t0:0\t7000\t8\t120\n"},
    "R18 riga di data prima dell'header (come accounting.txt storici)": {"sacct_text": "2026-11-20T10:00:00+01:00\n" + HEADER + "\n" + f"{J0}|COMPLETED|0:0|7000|8|120\n"},
    "R19 TSV vuoto": {"sacct_text": ""},
    "R20 State 'COMPLETED ' (spazio)": {"rows": [f"{J0}|COMPLETED |0:0|7000|8|120"]},
    "R21 audit di sorgenti storici": {"audit_source": "0" * 64},
    "R22 accounting: schema diverso": {"acc": {"schema": "contrastive-audit-accounting-v0"}},
    "R23 accounting: audit_sha256 diverso": {"acc": {"audit_sha256": "0" * 64}},
    "R24 accounting: source diverso": {"acc": {"source_sha256": "0" * 64}},
    "R25 accounting: logical_id storico": {"acc": {"logical_id": "CONTRAST-AUDIT-20261116"}},
    "R26 accounting: job_id intero (non stringa)": {"acc": {"job_id": 64900001}},
    "R27 accounting: job_id diverso dal TSV": {"acc": {"job_id": "64900009"}},
    "R28 accounting: sacct pin senza sha256": {"acc": {"sacct": {"path": "x"}}},
    "R29 accounting: chiave sacct assente": {"acc": {"sacct": None}},
    "R30 accounting: job_id '' ": {"acc": {"job_id": ""}},
}
made = {}
for label, spec in cases.items():
    made[label] = make(**spec)
    audit, acc, tsv = made[label]
    report(label, lambda: C.validate_audit_accounting(audit, acc)["elapsed_seconds"])
audit, acc, tsv = made["G1 COMPLETED 0:0 7000 s 8 CPU 120 min + .batch/.extern"]
tsv.write_text(tsv.read_text() + "\n")
report("R31 TSV modificato dopo il pin", lambda: C.validate_audit_accounting(audit, acc))
audit, acc, tsv = make()
report("R32 expected_job_id diverso (parametro mai passato dai chiamanti)", lambda: C.validate_audit_accounting(audit, acc, expected_job_id="1"))
audit.write_text(audit.read_text() + " ")
report("R33 audit modificato dopo l'accounting", lambda: C.validate_audit_accounting(audit, acc))
report("R34 accounting assente (None)", lambda: C.validate_audit_accounting(audit, None))
audit, acc, tsv = make()
report("R35 accounting come pin dict con sha sbagliato", lambda: C.validate_audit_accounting(audit, {"path": str(acc), "sha256": "0" * 64}))
report("G7 accounting come pin dict corretto", lambda: C.validate_audit_accounting(audit, C.pin(acc))["job_id"])

# Callers. Good and TIMEOUT receipts.
GOOD = make()
BAD = make([f"{J0}|TIMEOUT|0:0|7387|8|120", f"{J0}.batch|COMPLETED|0:0|7387|8|"])

print("== make_job (release) per la calibrazione ==")
calib = "CONTRAST-CALIB-targets-v3-20261118"
for label, (audit, acc, _) in (("K1 make_job calibrazione con accounting valido", GOOD), ("K2 make_job calibrazione con TIMEOUT", BAD)):
    report(label, lambda: sorted(C.make_job(binding, calib, {"audit": C.pin(audit), "audit_sacct": C.pin(acc)},
                                             "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/calibration"))[:3])
report("K3 make_job calibrazione senza audit_sacct", lambda: C.make_job(binding, calib, {"audit": C.pin(GOOD[0])}, "/x"))
report("K4 make_job audit senza accounting (atteso: nessun gate)", lambda: C.make_job(binding, AUDIT_ID, {}, "/x")["mode"])

print("== build_command (dispatcher) per la calibrazione ==")
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
fresh = Path(P._read(C.PREFLIGHT)["fresh_root"])
for label, (audit, acc, _) in (("D1 calibrazione, accounting valido (si ferma dopo)", GOOD), ("D2 calibrazione, TIMEOUT", BAD)):
    job = C.make_job(binding, calib, {"audit": C.pin(audit), "audit_sacct": C.pin(acc)} if label.startswith("D1") else
                     {"audit": C.pin(GOOD[0]), "audit_sacct": C.pin(GOOD[1])}, str(fresh / ("calibration_" + label[:2])))
    if label.startswith("D2"):
        job["audit"], job["audit_sacct"] = C.pin(audit), C.pin(acc)
    report(label, lambda: J.build_command(job, "gpu", 1800))
job = C.make_job(binding, calib, {"audit": C.pin(GOOD[0]), "audit_sacct": C.pin(GOOD[1])}, str(fresh / "calibration_D3"))
job.pop("audit_sacct")
report("D3 calibrazione senza audit_sacct", lambda: J.build_command(job, "gpu", 1800))

print("== contrastive_run.validate_release (runtime calibrazione) ==")
RUN.validate_input_bindings = lambda pins: None
RUN.validate_audit = lambda *args: {"passed": True}
cfg = P.build_review_configs(P._read(P.BASELINE))["calibration"]
config_path = TMP / "calibration_config.json"
config_path.write_text(json.dumps(cfg))
inputs = {}
for k in ("manifest", "scales", "window_seconds", "pose_stats"):
    f = TMP / (k + ".json")
    f.write_text(json.dumps({"stand_in": k}))
    inputs[k] = f


def release(audit, acc, **over):
    value = {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration", "source_sha256": SRC,
             "config_sha256": RUN.sha(config_path), "world_size": 4, "workers_per_rank": 6,
             "inputs": {k: RUN.sha(p) for k, p in inputs.items()}, "review_pin": binding["review_commit"],
             "package_review": copy.deepcopy(binding), "time_limit_s": 1770, "audit_sha256": RUN.sha(audit),
             "audit_sacct": C.pin(acc)}
    for key, item in over.items():
        if item is None:
            value.pop(key)
        else:
            value[key] = item
    return value


def run_release(rel, audit):
    return RUN.validate_release(rel, "calibration", config_path, inputs, None, 1770, audit_path=audit)


ga, gacc, _ = GOOD
ba, bacc, _ = BAD
report("V1 release calibrazione valida", lambda: run_release(release(ga, gacc), ga))
report("V2 senza package_review", lambda: run_release(release(ga, gacc, package_review=None), ga))
report("V3 review_pin diverso dal review_commit del binding", lambda: run_release(release(ga, gacc, review_pin="3" * 40), ga))
report("V4 package_review di un altro pacchetto (payload alterato)", lambda: run_release(release(ga, gacc, package_review={**binding, "review_payload": {**binding["review_payload"], "preflight_plan_sha256": "0" * 64}}), ga))
report("V5 senza audit_sacct", lambda: run_release(release(ga, gacc, audit_sacct=None), ga))
report("V6 accounting TIMEOUT", lambda: run_release(release(ba, bacc), ba))
report("V7 audit_sacct come percorso stringa (non pin)", lambda: run_release(release(ga, gacc, audit_sacct=str(gacc)), ga))

print("== read_preflight (lettura finale) ==")
from test_contrastive_prepare import fixture_run  # noqa: E402
folder = TMP / "calibration"
cfg2, summary, _ = fixture_run(folder)
receipt = P.calibration_report(folder)
receipt_path = TMP / "calibration_receipt.json"
receipt_path.write_text(json.dumps(receipt))
cohort_dir = TMP / "cohort"
sys.argv = ["contrastive_prepare", "calibrated-configs", "--receipt", str(receipt_path), "--run-dir", str(folder), "--out", str(cohort_dir)]
P.main()
audit_payload = {"source_sha256": SRC, "inputs": receipt["input_sha256"], "steps": 1024, "world_size": 4, "workers_per_rank": 6,
                 "passed": True, "config_sha256": hashlib.sha256(json.dumps(cfg2, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                 "expected_datasets": summary["contrastive"]["expected_datasets"],
                 "calibration_panel": summary["contrastive"]["coverage_partial_block"]}


def readout_case(label, rows, no_accounting=False):
    audit, acc, _ = make(rows)
    audit.write_text(json.dumps(audit_payload))
    data = json.loads(acc.read_text())
    data["audit_sha256"] = C.file_sha(audit)
    acc.write_text(json.dumps(data))
    report(label, lambda: B.read_preflight(cohort_dir / "cohort.json", {a: TMP / a for a in ("C0", "CN", "CR")},
                                           {a: TMP / a for a in ("CN_A", "CN_B", "CR_A", "CR_B")}, TMP / "unread_cuda.json",
                                           cpu_audit_path=audit, calibration_path=receipt_path, calibration_run_dir=folder,
                                           cpu_audit_sacct_path=None if no_accounting else acc))


readout_case("Q1 lettura: accounting valido (si ferma dopo, sul CUDA mancante)", None)
readout_case("Q2 lettura: TIMEOUT", [f"{J0}|TIMEOUT|0:0|7387|8|120", f"{J0}.batch|COMPLETED|0:0|7387|8|"])
readout_case("Q3 lettura: COMPLETED 7201 s", [f"{J0}|COMPLETED|0:0|7201|8|120"])
readout_case("Q4 lettura: senza accounting (None)", None, no_accounting=True)
shutil.rmtree(TMP)
print("done")
