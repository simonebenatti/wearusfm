"""G-6 skeptic check: production call signature (no expected_job_id) vs declared SLURM ID."""
import json
import pytest
from test_contrastive_release import package, accounting, dump, C  # noqa: F401  (fixture reuse)


def test_any_declared_allocation_passes_production_signature(package):
    audit, receipt, tsv = accounting(package)
    # Same audit report, receipt now names an unrelated allocation 999 with its own sacct row.
    tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n"
                   "999|COMPLETED|0:0|100|8|120\n")
    data = json.loads(receipt.read_text()); data["job_id"] = "999"; data["sacct"] = C.pin(tsv); dump(receipt, data)
    result = C.validate_audit_accounting(audit, receipt)  # signature used by job:81, run:195, release:242
    assert result["job_id"] == "999" and result["audit_sha256"] == C.file_sha(audit)
    # The only path that would reject it is the hook no production caller uses.
    with pytest.raises(ValueError, match="job ID"):
        C.validate_audit_accounting(audit, receipt, expected_job_id="123")


def test_resource_fields_narrow_admissible_mixups(package):
    audit, receipt, tsv = accounting(package)
    for row in ("999|COMPLETED|0:0|100|4|4", "999|COMPLETED|0:0|100|32|30", "999|COMPLETED|0:0|100|8|60"):
        tsv.write_text("JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n" + row + "\n")
        data = json.loads(receipt.read_text()); data["job_id"] = "999"; data["sacct"] = C.pin(tsv); dump(receipt, data)
        with pytest.raises(ValueError, match="COMPLETED 0:0"):
            C.validate_audit_accounting(audit, receipt)
