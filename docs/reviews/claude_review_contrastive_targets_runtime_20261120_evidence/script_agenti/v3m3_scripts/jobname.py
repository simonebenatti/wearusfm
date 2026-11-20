import json, sys
from pathlib import Path
CL = Path(sys.argv[1]); W = Path(sys.argv[2])
sys.path.insert(0, str(CL / "scripts")); sys.path.insert(0, str(CL / "src"))
import contrastive_release as C
audit = W / "audit/report.json"
tsv = W / "jobname_other.tsv"
tsv.write_text("JobIDRaw|JobName|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw|Comment\n"
               "64900001|ALTRO-LAVORO-8CPU-2H|COMPLETED|0:0|7000|8|120|wearusfm:ALTRO-LAVORO-8CPU-2H\n")
acc = W / "jobname_other.json"
acc.write_text(json.dumps({"schema": C.AUDIT_SCHEMA, "source_sha256": C.source_digest(), "audit_sha256": C.file_sha(audit),
                           "logical_id": "CONTRAST-AUDIT-targets-v3-20261118", "job_id": "64900001", "sacct": C.pin(tsv)}))
try:
    r = C.validate_audit_accounting(audit, acc); print("JobName/Comment di un altro ID logico: ACCETTATO", r["job_id"], r["logical_id"])
except Exception as e:
    print("RESPINTO", e)
