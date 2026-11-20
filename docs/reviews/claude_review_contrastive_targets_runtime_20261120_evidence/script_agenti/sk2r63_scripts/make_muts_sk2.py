import json, sys
from pathlib import Path
clone, out = Path(sys.argv[1]), Path(sys.argv[2])
B, RN, J, C = "scripts/contrastive_benchmark_readout.py", "scripts/contrastive_run.py", "scripts/contrastive_job.py", "scripts/contrastive_cuda_check.py"
M = [
 ("B01", B, "resume FAILED_REQUIRES_REVIEW branch disabled (elif False)", "            elif not check.get('passed'):", "            elif False:"),
 ("B02", B, "resume INCONCLUSIVE reason dropped",
  "                result['inconclusive_reasons'].append(arm+': common-checkpoint comparison inconclusive')", "                pass"),
 ("B02x", B, "resume INCONCLUSIVE misclassified as failure",
  "                result['inconclusive_reasons'].append(arm+': common-checkpoint comparison inconclusive')",
  "                result['failure_reasons'].append(arm+': common-checkpoint comparison inconclusive')"),
 ("B04", B, "reload origin suffix comparison disabled",
  "                if [r['origin'] for r in reload_records] != [r['origin'] for r in records[544:]]:", "                if False:"),
 ("B05", B, "both reloads read with expected_phase resume_a",
  "expected_phase=phase.replace('reload_', 'resume_'))", "expected_phase='resume_a')"),
 ("B11", B, "CUDA per-rank numeric profile assert removed",
  "        assert_numeric_flags(rank.get('numeric_profile', {}))\n", ""),
 ("B12", B, "cross-rank flag equality disabled",
  "    if any(row['flags'] != numeric[0]['flags'] for row in numeric[1:]):", "    if False:"),
 ("B13", B, "per-receipt flag assert removed in _run_evidence",
  "        assert_numeric_flags(row.get('flags', {}))\n        numeric.append(row)", "        numeric.append(row)"),
 ("B14", B, "wrong-profile receipts not counted missing",
  "                missing.append(key + ': required profile unavailable: ' + str(exc))", "                pass"),
 ("B16", B, "descriptive drift turned into a failure",
  "        result['trajectory_drift_descriptive'][arm] = trajectory_drift(streams['C0'][0][:256], streams[arm][0][:256])\n",
  "        result['trajectory_drift_descriptive'][arm] = trajectory_drift(streams['C0'][0][:256], streams[arm][0][:256])\n"
  "        if result['trajectory_drift_descriptive'][arm]['fields']['total']['outside_count']: result['failure_reasons'].append('drift total')\n"),
 ("R01", RN, "validate_release: review binding call removed", '    validate_review_binding(release.get("package_review"))\n', ""),
 ("R02", RN, "validate_release: review_pin vs bound commit disabled",
  '    if release["review_pin"] != release["package_review"]["review_commit"]:', "    if False:"),
 ("R03", RN, "validate_release: audit accounting call removed", '    validate_audit_accounting(audit_path, release["audit_sacct"])', "    pass"),
 ("R04", RN, "validate_release: CUDA prerequisite call removed",
  '        validate_cuda_prerequisite(release.get("cuda_report"), cohort, pins, calibration_run_dir)', "        pass"),
 ("R05", RN, "validate_release: C0 reload allowed", '        if phase != "full" and release.get("arm") not in ("CN", "CR"):', "        if False:"),
 ("R06", RN, "main: validate_target_fork skipped", "        validate_target_fork(release, cfg, cohort)", "        pass"),
 ("R07", RN, "validate_target_fork: checkpoint name not checked", ' or checkpoint.name != "checkpoint_step0000544.pt" or', " or"),
 ("R08", RN, "validate_target_fork: provenance not checked", '    if contrastive.get("provenance") != provenance:', "    if False:"),
 ("R09", RN, "main: reload output may pre-exist",
  '        if a.out_dir.exists():\n            raise ValueError("each common-checkpoint restart', '        if False:\n            raise ValueError("each common-checkpoint restart'),
 ("R10", RN, "main: numeric receipt phase hard-coded full", '"phase": release.get("phase"),', '"phase": "full",'),
 ("R11", RN, "main: numeric profile context not entered", "            flags = stack.enter_context(numeric_profile(require_cuda=True))", "            flags = {}"),
 ("J01", J, "dispatcher: review binding call removed", "    validate_review_binding(job.get('package_review'))\n", ""),
 ("J02", J, "dispatcher: job review vs bound review disabled", "    if job.get('review') != job['package_review'].get('review'):", "    if False:"),
 ("J03", J, "dispatcher: cell arm not compared",
  "    if (len(cells) != 1 or cells[0]['mode'] != mode or cells[0].get('arm') != job.get('arm')", "    if (len(cells) != 1 or cells[0]['mode'] != mode"),
 ("J03r", J, "dispatcher: cell repeat/seconds not compared",
  "            or cells[0].get('repeat') != job.get('repeat') or cells[0]['seconds'] != allocated_seconds):", "            ):"),
 ("J04", J, "dispatcher: audit accounting call removed",
  "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))", "        pass"),
 ("J05", J, "dispatcher: out not bound to cohort root", "not out.is_relative_to(cohort_root) or out == cohort_root or ", ""),
 ("J06", J, "dispatcher: release package_review/audit_sacct not compared",
  "    if payload.get('package_review') != job['package_review'] or payload.get('audit_sacct') != job['audit_sacct']:", "    if False:"),
 ("J07", J, "dispatcher: resume fork/continuous pins not compared",
  " or payload.get('fork_checkpoint') != job.get('fork_checkpoint') or payload.get('continuous_summary') != job.get('continuous_summary'):", ":"),
 ("J08", J, "dispatcher: resume phase fixed to resume_a",
  "    expected_phase = 'resume_' + str(job.get('repeat', '')).lower() if mode == 'resume' else mode", "    expected_phase = 'resume_a' if mode == 'resume' else mode"),
 ("C01", C, "cuda_check main without numeric profile", "    with numeric_profile(require_cuda=True):\n        return _main(argv)", "    return _main(argv)"),
 ("C02", C, "cuda_check rank numeric profile not gathered", ', "numeric_profile": assert_numeric_profile(require_cuda=True)})', "})"),
]
rows = []
for mid, f, desc, old, new in M:
    n = (clone / f).read_text().count(old)
    assert n == 1, (mid, n)
    rows.append({"id": mid, "file": f, "desc": desc, "old": old, "new": new})
out.write_text(json.dumps(rows, indent=1))
print(len(rows), "mutations ok")
