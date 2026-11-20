"""Scettico 2, R3-M3: il gate accetta una coppia accounting/TSV coerente di un job qualsiasi."""
import json, sys, hashlib
from pathlib import Path
CLONE = Path(sys.argv[1]); WORK = Path(sys.argv[2])
sys.path.insert(0, str(CLONE / "scripts"))
import contrastive_release as C
WORK.mkdir(parents=True, exist_ok=True)
assert C.ROOT == CLONE.resolve(), C.ROOT
src = C.source_digest()
plan = json.loads(C.PREFLIGHT.read_text())
logical = [c["id"] for c in plan["cells"] if c["mode"] == "audit"][0]
print("source_digest", src, "== preflight", src == plan["source_sha256"])
print("logical audit id", logical)

def make(tag, job_id, rows, audit_extra=None):
    d = WORK / tag; d.mkdir(exist_ok=True)
    audit = d / "audit.json"
    audit.write_text(json.dumps({"source_sha256": src, "passed": True, **(audit_extra or {})}) + "\n")
    tsv = d / "sacct.tsv"
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n" + "".join(r + "\n" for r in rows))
    acc = d / "accounting.json"
    acc.write_text(json.dumps({"schema": C.AUDIT_SCHEMA, "source_sha256": src, "audit_sha256": C.file_sha(audit),
                               "logical_id": logical, "job_id": job_id, "sacct": C.pin(tsv)}) + "\n")
    return audit, acc

def run(name, audit, acc, **kw):
    try:
        r = C.validate_audit_accounting(audit, acc, **kw)
        print(f"{name}: ACCETTATO job_id={r['job_id']} elapsed={r['elapsed_seconds']} state={r['state']}")
    except Exception as e:
        print(f"{name}: RESPINTO {type(e).__name__}: {e}")

# S1: l'audit "vero" (ipotetico 64900000) e' durato 7387 s (oltre 7200, dentro il margine SLURM) e ha scritto il report.
a, acc = make("S1_true_overrun", "64900000", ["64900000|COMPLETED|0:0|7387|8|120", "64900000.batch|COMPLETED|0:0|7387|8|120"])
run("S1 audit vero, sacct vero (7387 s)", a, acc)
# S2: stesso report dell'audit, accounting/TSV di un ALTRO job (64900001) con stessa forma 8 CPU/120 min, 3600 s.
a2, acc2 = make("S2_other_job", "64900001", ["64900001|COMPLETED|0:0|3600|8|120", "64900001.batch|COMPLETED|0:0|3600|8|120"])
# usa lo stesso report dell'audit S1 (byte identici) per mostrare che il legame e' solo sha256 del report
acc2_data = json.loads(acc2.read_text()); acc2_data["audit_sha256"] = C.file_sha(a); acc2.write_text(json.dumps(acc2_data) + "\n")
run("S2 stesso report, coppia accounting/TSV di 64900001", a, acc2)
# S3: con expected_job_id (mai passato in produzione) il gate respinge
run("S3 come S2 ma expected_job_id=64900000", a, acc2, expected_job_id="64900000")
# S4: incoerenza interna (job_id 64900001, TSV di 64900000) -> respinto (unico caso coperto dai test di produzione)
a4, acc4 = make("S4_incoherent", "64900001", ["64900000|COMPLETED|0:0|3600|8|120"])
run("S4 job_id e TSV incoerenti", a4, acc4)
# S5: TSV di un job di checkout (4 CPU, 4 min) o GPU (32 CPU, 30 min) -> respinto dalla forma
a5, acc5 = make("S5_checkout", "64900002", ["64900002|COMPLETED|0:0|60|4|4"])
run("S5 coppia di un checkout 4CPU/4min", a5, acc5)
a6, acc6 = make("S6_gpu", "64900003", ["64900003|COMPLETED|0:0|1500|32|30"])
run("S6 coppia di un job GPU 32CPU/30min", a6, acc6)
# S7: il report dell'audit non contiene alcun campo SLURM/orario da confrontare
print("S7 campi SLURM nel report scritto da contrastive_audit.main:",
      [l.strip() for l in (CLONE/"scripts/contrastive_audit.py").read_text().splitlines() if l.strip().startswith("report[")])
