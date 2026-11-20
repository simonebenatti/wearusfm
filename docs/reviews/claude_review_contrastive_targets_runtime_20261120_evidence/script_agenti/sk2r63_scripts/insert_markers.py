"""Insert non-behavioural execution markers at the 30 surviving mutation sites (my own anchors). Usage: insert_markers.py CLONE MARKFILE"""
import sys
from pathlib import Path
clone, mark = Path(sys.argv[1]), sys.argv[2]
def mk(expr):
    return ("__import__('pathlib').Path(%r).open('a').write(str(%s)+'|'+__import__('os').environ.get('PYTEST_CURRENT_TEST','?')+'\\n')" % (mark, expr))
B, RN, J, C = "scripts/contrastive_benchmark_readout.py", "scripts/contrastive_run.py", "scripts/contrastive_job.py", "scripts/contrastive_cuda_check.py"
sites = [
 (B, "                result['failure_reasons'].append(arm+': common-checkpoint historical bounds or invariants failed')", "before", [mk("'B01_kill'")]),
 (B, "                result['inconclusive_reasons'].append(arm+': common-checkpoint comparison inconclusive')", "before", [mk("'B02_kill'")]),
 (B, "                    raise ValueError('reloaded stream differs from continuous suffix')", "before", [mk("'B04_kill'")]),
 (B, "                _, _, reload_records, rp = _run_evidence(paths[phase], arm, cohort, expected[arm], provenance, first_update=545, expected_phase=phase.replace('reload_', 'resume_'))", "before",
     [mk("'B05_reach:'+str(getattr(_run_evidence,'__module__',None))+':'+str(getattr(compare_three_paths,'__module__',None))")]),
 (B, "        assert_numeric_flags(rank.get('numeric_profile', {}))", "before",
     ["try: assert_numeric_flags(rank.get('numeric_profile', {})); " + mk("'B11_reach'"), "except Exception: " + mk("'B11_kill'")]),
 (B, "        raise ValueError('production numeric flags differ across ranks')", "before", [mk("'B12_kill'")]),
 (B, "        assert_numeric_flags(row.get('flags', {}))", "before",
     ["try: assert_numeric_flags(row.get('flags', {})); " + mk("'B13_reach'"), "except Exception: " + mk("'B13_kill'")]),
 (B, "                missing.append(key + ': required profile unavailable: ' + str(exc))", "before", [mk("'B14_kill'")]),
 (B, "        result['trajectory_drift_descriptive'][arm] = trajectory_drift(streams['C0'][0][:256], streams[arm][0][:256])", "after",
     [mk("'B16_kill' if any(f['outside_count'] for f in result['trajectory_drift_descriptive'][arm]['fields'].values()) else 'B16_reach'")]),
 (RN, '    validate_review_binding(release.get("package_review"))', "before", [mk("'R01_reach:'+str(getattr(validate_review_binding,'__module__',None))")]),
 (RN, '        raise ValueError("release review commit differs from its bound package review")', "before", [mk("'R02_kill'")]),
 (RN, '    validate_audit_accounting(audit_path, release["audit_sacct"])', "before", [mk("'R03_reach:'+str(getattr(validate_audit_accounting,'__module__',None))")]),
 (RN, '        validate_cuda_prerequisite(release.get("cuda_report"), cohort, pins, calibration_run_dir)', "before", [mk("'R04_reach:'+str(getattr(validate_cuda_prerequisite,'__module__',None))")]),
 (RN, '            raise ValueError("only CN/CR have active common-checkpoint resume tests")', "before", [mk("'R05_kill'")]),
 (RN, '        validate_target_fork(release, cfg, cohort)', "before", [mk("'R06_reach'")]),
 (RN, '    checkpoint = checked_pin(release["fork_checkpoint"])', "before", [mk("'R07_reach'")]),
 (RN, '        raise ValueError("common544 source provenance differs from released cohort")', "before", [mk("'R08_kill'")]),
 (RN, '            raise ValueError("each common-checkpoint restart requires a fresh output directory")', "before", [mk("'R09_kill'")]),
 (RN, '            with (a.out_dir / f"numeric_profile.rank{ctx.rank}.json").open("x") as stream:', "before", [mk("'R10_reach'")]),
 (RN, '            flags = stack.enter_context(numeric_profile(require_cuda=True))', "before", [mk("'R11_reach'")]),
 (J, "    validate_review_binding(job.get('package_review'))", "before", [mk("'J01_reach:'+str(getattr(validate_review_binding,'__module__',None))")]),
 (J, "        raise ValueError('job review differs from its bound package review')", "before", [mk("'J02_kill'")]),
 (J, "        raise ValueError('job differs from the reviewed fixed preflight cell')", "before", [mk("'J03_kill'")]),
 (J, "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))", "before", [mk("'J04_reach:'+str(getattr(validate_audit_accounting,'__module__',None))")]),
 (J, "    if not out.is_relative_to(results) or not out.is_relative_to(cohort_root) or out == cohort_root or out.exists():", "before",
     [mk("'J05_kill' if (out.is_relative_to(results) and not out.exists() and (not out.is_relative_to(cohort_root) or out == cohort_root)) else 'J05_reach'")]),
 (J, "        raise ValueError('job review/audit accounting differs from execution release')", "before", [mk("'J06_kill'")]),
 (J, "        if job.get('repeat') not in ('a', 'b') or payload.get('fork_checkpoint') != job.get('fork_checkpoint') or payload.get('continuous_summary') != job.get('continuous_summary'):", "before",
     [mk("'J07_kill' if (job.get('repeat') not in ('a', 'b') or payload.get('fork_checkpoint') != job.get('fork_checkpoint') or payload.get('continuous_summary') != job.get('continuous_summary')) else 'J07_reach'")]),
 (J, "    expected_phase = 'resume_' + str(job.get('repeat', '')).lower() if mode == 'resume' else mode", "after",
     [mk("'J08_kill' if (mode == 'resume' and expected_phase != 'resume_a') else 'J08_reach'")]),
 (C, "    with numeric_profile(require_cuda=True):", "before", [mk("'C01_reach'")]),
 (C, '        torch.distributed.all_gather_object(ranks, {"rank": ctx.rank, "passed": own_passed, "arms": report["arms"], "stop_checks": report["stop_checks"], "numeric_profile": assert_numeric_profile(require_cuda=True)})', "before", [mk("'C02_reach'")]),
]
files = {}
for f, anchor, pos, code in sites:
    lines = files.setdefault(f, (clone / f).read_text().split("\n"))
    idx = [i for i, l in enumerate(lines) if l == anchor]
    assert len(idx) == 1, (f, anchor[:60], len(idx))
    i = idx[0]
    indent = anchor[:len(anchor) - len(anchor.lstrip())]
    new = [indent + c for c in code]
    if pos == "before": lines[i:i] = new
    else: lines[i+1:i+1] = new
for f, lines in files.items():
    (clone / f).write_text("\n".join(lines))
    compile((clone / f).read_text(), f, "exec")
print("inserted", len(sites), "markers in", sorted(files))
