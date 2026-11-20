"""Independent mutation list (sk1r63). Each mutation replaces exactly one occurrence.

Usage: muts.py CLONE OUT_JSON  -> validates uniqueness in CLONE and writes OUT_JSON.
"""
import json
import sys
from pathlib import Path

B = "scripts/contrastive_benchmark_readout.py"
RN = "scripts/contrastive_run.py"
J = "scripts/contrastive_job.py"
CU = "scripts/contrastive_cuda_check.py"

M = [
    # read_preflight / readout
    ("B01", B, "resume FAILED: failure reason not appended",
     "                result['failure_reasons'].append(arm+': common-checkpoint historical bounds or invariants failed')",
     "                result['failure_reasons'].extend([])"),
    ("B02", B, "resume INCONCLUSIVE: reason not appended",
     "                result['inconclusive_reasons'].append(arm+': common-checkpoint comparison inconclusive')",
     "                result['inconclusive_reasons'].extend([])"),
    ("B04", B, "reload origin suffix check: raise removed",
     "                    raise ValueError('reloaded stream differs from continuous suffix')",
     "                    pass"),
    ("B05", B, "reload expected_phase forced to full",
     "expected_phase=phase.replace('reload_', 'resume_'))",
     "expected_phase='full')"),
    ("B05b", B, "reload expected_phase passed as reload_x (realistic slip)",
     "expected_phase=phase.replace('reload_', 'resume_'))",
     "expected_phase=phase)"),
    ("B11", B, "_cuda_evidence: per-rank numeric profile not asserted",
     "        assert_numeric_flags(rank.get('numeric_profile', {}))\n",
     "        pass\n"),
    ("B12", B, "_run_evidence: cross-rank flag equality raise removed",
     "        raise ValueError('production numeric flags differ across ranks')",
     "        pass"),
    ("B13", B, "_run_evidence: per-receipt assert_numeric_flags removed",
     "        assert_numeric_flags(row.get('flags', {}))\n        numeric.append(row)",
     "        numeric.append(row)"),
    ("B14", B, "numeric_context: wrong profile not marked missing",
     "                missing.append(key + ': required profile unavailable: ' + str(exc))",
     "                pass"),
    ("B16", B, "trajectory drift turned into a failure gate",
     "        result['trajectory_drift_descriptive'][arm] = trajectory_drift(streams['C0'][0][:256], streams[arm][0][:256])\n",
     "        result['trajectory_drift_descriptive'][arm] = trajectory_drift(streams['C0'][0][:256], streams[arm][0][:256])\n"
     "        if any(f['outside_count'] for f in result['trajectory_drift_descriptive'][arm]['fields'].values()):\n"
     "            result['failure_reasons'].append('drift')\n"),
    # contrastive_run
    ("R01", RN, "validate_release: binding call removed",
     "    validate_review_binding(release.get(\"package_review\"))\n",
     "    pass\n"),
    ("R02", RN, "validate_release: review_pin vs binding raise removed",
     "        raise ValueError(\"release review commit differs from its bound package review\")",
     "        pass"),
    ("R03", RN, "validate_release: audit accounting call removed",
     "    validate_audit_accounting(audit_path, release[\"audit_sacct\"])\n",
     "    pass\n"),
    ("R04", RN, "validate_release: CUDA prerequisite call removed",
     "        validate_cuda_prerequisite(release.get(\"cuda_report\"), cohort, pins, calibration_run_dir)\n",
     "        pass\n"),
    ("R05", RN, "validate_release: C0 reload raise removed",
     "            raise ValueError(\"only CN/CR have active common-checkpoint resume tests\")",
     "            pass"),
    ("R06", RN, "main: validate_target_fork call removed",
     "        validate_target_fork(release, cfg, cohort)\n",
     "        pass\n"),
    ("R07", RN, "validate_target_fork: parent/name condition removed",
     "            checkpoint.parent != summary_path.parent or checkpoint.name != \"checkpoint_step0000544.pt\" or\n",
     ""),
    ("R08", RN, "validate_target_fork: provenance raise removed",
     "        raise ValueError(\"common544 source provenance differs from released cohort\")",
     "        pass"),
    ("R09", RN, "main: reload fresh-output raise removed",
     "            raise ValueError(\"each common-checkpoint restart requires a fresh output directory\")",
     "            pass"),
    ("R10", RN, "main: receipt phase hard-coded full",
     "\"phase\": release.get(\"phase\"),",
     "\"phase\": \"full\","),
    ("R11", RN, "main: numeric_profile context not entered",
     "            flags = stack.enter_context(numeric_profile(require_cuda=True))",
     "            flags = {}"),
    # contrastive_job (dispatcher)
    ("J01", J, "build_command: binding call removed",
     "    validate_review_binding(job.get('package_review'))\n",
     "    pass\n"),
    ("J02", J, "build_command: review vs binding raise removed",
     "        raise ValueError('job review differs from its bound package review')",
     "        pass"),
    ("J03", J, "build_command: preflight cell raise removed",
     "        raise ValueError('job differs from the reviewed fixed preflight cell')",
     "        pass"),
    ("J04", J, "build_command: audit accounting call removed",
     "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))",
     "        pass"),
    ("J05", J, "build_command: out need not be under cohort root",
     "not out.is_relative_to(cohort_root) or out == cohort_root or ",
     ""),
    ("J06", J, "build_command: release review/accounting raise removed",
     "        raise ValueError('job review/audit accounting differs from execution release')",
     "        pass"),
    ("J07", J, "build_command: resume repeat/fork raise removed",
     "            raise ValueError('resume requires the reviewed repeat and exact common checkpoint')",
     "            pass"),
    ("J08", J, "build_command: resume phase always resume_a",
     "    expected_phase = 'resume_' + str(job.get('repeat', '')).lower() if mode == 'resume' else mode",
     "    expected_phase = 'resume_a' if mode == 'resume' else mode"),
    # cuda_check
    ("C01", CU, "main: numeric profile context removed",
     "    with numeric_profile(require_cuda=True):\n        return _main(argv)",
     "    return _main(argv)"),
    ("C02", CU, "rank gather: numeric_profile key removed",
     ", \"numeric_profile\": assert_numeric_profile(require_cuda=True)})",
     "})"),
    # null
    ("NULL", B, "null change (comment only)",
     "if __name__ == '__main__': main()",
     "if __name__ == '__main__': main()  # null"),
]

clone, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for mid, f, desc, old, new in M:
    text = (clone / f).read_text()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{mid}: old occurs {n} times")
    line = text[:text.index(old)].count("\n") + 1
    rows.append({"id": mid, "file": f, "line": line, "desc": desc, "old": old, "new": new})
out.write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r["id"], r["file"] + ":" + str(r["line"]), r["desc"])
