"""Apply one named mutation (or a '+'-joined combination) to a temporary clone.
Each edit must match exactly once in the file; otherwise abort."""
import sys
from pathlib import Path

J = "scripts/contrastive_job.py"
R = "scripts/contrastive_run.py"
L = "scripts/contrastive_release.py"
C = "scripts/contrastive_cuda_check.py"

M = {
 # dispatcher
 "J_bind_call": [(J, "    validate_review_binding(job.get('package_review'))\n", "    pass  # MUT\n")],
 "J_job_review": [(J, "    if job.get('review') != job['package_review'].get('review'):\n        raise ValueError('job review differs from its bound package review')\n", "")],
 "J_cell": [(J, "    if (len(cells) != 1 or cells[0]['mode'] != mode or cells[0].get('arm') != job.get('arm')\n            or cells[0].get('repeat') != job.get('repeat') or cells[0]['seconds'] != allocated_seconds):\n        raise ValueError('job differs from the reviewed fixed preflight cell')\n", "")],
 "J_accounting": [(J, "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))\n", "        pass  # MUT\n")],
 "J_fresh_root": [(J, "not out.is_relative_to(cohort_root) or out == cohort_root or ", "")],
 "J_release_eq": [(J, "    if payload.get('package_review') != job['package_review'] or payload.get('audit_sacct') != job['audit_sacct']:\n        raise ValueError('job review/audit accounting differs from execution release')\n", "")],
 "J_resume_pins": [(J, " or payload.get('fork_checkpoint') != job.get('fork_checkpoint') or payload.get('continuous_summary') != job.get('continuous_summary'):", ":")],
 # runtime
 "R_bind_call": [(R, "    validate_review_binding(release.get(\"package_review\"))\n", "    pass  # MUT\n")],
 "R_review_pin": [(R, "    if release[\"review_pin\"] != release[\"package_review\"][\"review_commit\"]:\n        raise ValueError(\"release review commit differs from its bound package review\")\n", "")],
 "R_sacct_presence": [(R, "    if not release.get(\"audit_sacct\"):\n        raise ValueError(\"calibration requires completed new audit accounting\")\n", "")],
 "R_accounting_call": [(R, "    validate_audit_accounting(audit_path, release[\"audit_sacct\"])\n", "    pass  # MUT\n")],
 "R_cuda_prereq": [(R, "        validate_cuda_prerequisite(release.get(\"cuda_report\"), cohort, pins, calibration_run_dir)\n", "        pass  # MUT\n")],
 "R_phase_arm": [(R, "        if phase not in (\"full\", \"resume_a\", \"resume_b\") or raw[\"max_steps\"] != 576:\n            raise ValueError(\"benchmark phase must be full576 or common544 resume_a/resume_b\")\n        if phase != \"full\" and release.get(\"arm\") not in (\"CN\", \"CR\"):\n            raise ValueError(\"only CN/CR have active common-checkpoint resume tests\")\n", "")],
 "R_target_fork": [(R, "        validate_target_fork(release, cfg, cohort)\n", "        pass  # MUT\n")],
 "R_barrier_sha": [(R, "                if sha(a.out_dir / \"checkpoint.pt\") != release[\"fork_checkpoint\"][\"sha256\"]:\n                    raise ValueError(\"reload checkpoint bytes changed after common-fork copy\")\n", "")],
 "R_numeric_profile": [(R, "            flags = stack.enter_context(numeric_profile(require_cuda=True))\n", "            flags = {}  # MUT\n")],
 # release.py
 "L_makejob_accounting": [(L, "    if cell[\"mode\"] != \"audit\":\n        audit = checked_pin(job[\"audit\"])\n        validate_audit_accounting(audit, job[\"audit_sacct\"])\n", "")],
 "L_cuda_prereq_body": [(L, "    if cuda.get(\"inputs\", {}).get(\"calibration_checkpoint\") != file_sha(Path(calibration_run_dir) / \"checkpoint.pt\"):\n        raise ValueError(\"CUDA receipt checkpoint differs from the calibrated cohort\")\n", ""),
                        (L, "    if file_sha(rank_exit) != cuda.get(\"rank_exit_report_sha256\"):\n        raise ValueError(\"CUDA rank-exit prerequisite pin changed\")\n    validate_receipt(rank_exit, expected_source_sha256=cohort[\"source_sha256\"])\n", "")],
 # cuda_check
 "C_profile_ctx": [(C, "    with numeric_profile(require_cuda=True):\n        return _main(argv)\n", "    return _main(argv)\n")],
 "C_rank_flags": [(C, ", \"numeric_profile\": assert_numeric_profile(require_cuda=True)})", "})")],
 # positive controls (must be killed)
 "CTRL_J_cuda": [(J, "        if payload.get('cuda_report') != job.get('cuda_report') or not job.get('cuda_report'):\n            raise ValueError('benchmark requires its pinned CUDA prerequisite')\n", "")],
 "CTRL_R_time": [(R, "    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit or time_limit_s != limit:\n        raise ValueError(\"missing/mismatched bounded time allocation\")\n", "")],
 "CTRL_B_readout_accounting": [("scripts/contrastive_benchmark_readout.py", "    accounting = validate_audit_accounting(cpu_audit_path, cpu_audit_sacct_path, expected_source=cohort['source_sha256'])\n", "    accounting = {}  # MUT\n")],
}

def apply(root, name):
    for part in name.split("+"):
        for rel, old, new in M[part]:
            path = Path(root) / rel
            text = path.read_text()
            n = text.count(old)
            if n != 1:
                raise SystemExit(f"{part}: expected exactly one match in {rel}, found {n}")
            path.write_text(text.replace(old, new))

if __name__ == "__main__":
    root, name = sys.argv[1], sys.argv[2]
    if name == "LIST":
        print(" ".join(M)); raise SystemExit(0)
    if name == "ALL":
        name = "+".join(M)
    apply(root, name)
    print("applied", name)
