"""Print the actual outcome (return or exception type/message) under a mutant."""
import json
import contrastive_release as C
from test_contrastive_release import package, accounting, dump  # noqa: F401


def outcome(call):
    try:
        result = call()
        return "ACCEPTED " + json.dumps({k: result[k] for k in ("job_id", "elapsed_seconds", "state")})
    except Exception as exc:  # report, do not judge
        return type(exc).__name__ + ": " + str(exc)


def test_outcomes(package):
    audit, receipt, tsv = accounting(package)
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS\n123|COMPLETED|0:0|7200|8\n")
    data = json.loads(receipt.read_text()); data["sacct"] = C.pin(tsv); dump(receipt, data)
    print("SK2_OUT missing_TimelimitRaw ->", outcome(lambda: C.validate_audit_accounting(audit, receipt)))
    audit, receipt, tsv = accounting(package)
    data = json.loads(receipt.read_text()); data["job_id"] = "123.batch"; dump(receipt, data)
    print("SK2_OUT job_id_123.batch_fixture_step_row(Timelimit 120) ->", outcome(lambda: C.validate_audit_accounting(audit, receipt)))
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n123|COMPLETED|0:0|7200|8|120\n123.batch|COMPLETED|0:0|7199|8|\n")
    data = json.loads(receipt.read_text()); data["sacct"] = C.pin(tsv); dump(receipt, data)
    print("SK2_OUT job_id_123.batch_leonardo_step_row(blank Timelimit) ->", outcome(lambda: C.validate_audit_accounting(audit, receipt)))
