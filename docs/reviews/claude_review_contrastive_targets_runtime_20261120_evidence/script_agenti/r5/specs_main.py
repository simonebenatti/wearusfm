G_RUN = ["tests/cpu/test_contrastive_targets_guards.py", "tests/cpu/test_contrastive_run.py", "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_historical_callsites.py"]
G_READ = ["tests/cpu/test_contrastive_targets_readout.py", "tests/cpu/test_contrastive_benchmark_readout.py", "tests/cpu/test_contrastive_targets_guards.py", "tests/cpu/test_contrastive_historical_callsites.py"]
G_PREP = ["tests/cpu/test_contrastive_targets_guards.py", "tests/cpu/test_contrastive_prepare.py", "tests/cpu/test_contrastive_historical_calibration.py", "tests/cpu/test_contrastive_run.py"]
G_PAR = ["tests/cpu/test_contrastive_target_parity.py", "tests/cpu/test_contrastive_targets_readout.py"]
G_RES = ["tests/cpu/test_contrastive_target_resume.py", "tests/cpu/test_contrastive_targets_readout.py"]
G_REL = ["tests/cpu/test_contrastive_release.py", "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_run.py", "tests/cpu/test_contrastive_targets_readout.py", "tests/cpu/test_contrastive_historical_callsites.py"]
RUN, READ, PREP = 'scripts/contrastive_run.py', 'scripts/contrastive_benchmark_readout.py', 'scripts/contrastive_prepare.py'
PAR, RES, REL, JOB = 'scripts/contrastive_target_parity.py', 'scripts/contrastive_target_resume.py', 'scripts/contrastive_release.py', 'scripts/contrastive_job.py'
def S(i, f, old, new, tests): return {'id': i, 'edits': [(f, old, new)], 'tests': tests}
SPECS = [
 # (b) M-1
 S('M1a_validate_audit_config', RUN, 'or report.get("config_sha256") != hashlib.sha256(json.dumps(calibration_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):', 'or False):', G_RUN),
 S('M1b_read_preflight_config', READ, "audit.get('workers_per_rank') != 6 or audit.get('config_sha256') != canonical_calibration):", "audit.get('workers_per_rank') != 6 or False):", G_READ),
 S('M1c_authenticate_calibration_receipt', PREP, '    if _jsonable(receipt) != comparable:\n', '    if False:\n', G_PREP),
 # (b) N-4
 S('N4a_PLAN_pointer_old', PREP, 'PLAN = ROOT / "results/infra/contrastive_targets_20261118/plan.json"', 'PLAN = ROOT / "results/infra/contrastive_comparison_20261116/plan.json"', ["tests/cpu/test_contrastive_targets_guards.py"]),
 S('N4b_readout_text_old', READ, "'interpretation': 'Descriptive historical reference only: target supports changed from RMS25/fast50 to RMS100/fast125, so even update 1 need not match. Subsequent historical weights trained while calibration weights stay fixed.'}", "'interpretation': 'Only update 1 may match exactly; subsequent historical weights trained while calibration weights stay fixed.'}", ["tests/cpu/test_contrastive_targets_guards.py"]),
 S('N4c1_builder_rms25', PREP, '    base["loader"]["rms_window_ms"] = 100.\n', '    base["loader"]["rms_window_ms"] = 25.\n', ["tests/cpu/test_contrastive_targets_guards.py"]),
 S('N4c2_plan_cli_args_CN', 'results/infra/contrastive_targets_20261118/plan.json', '"rnc_weight": 0.0,\n      "rms_window_ms": 100.0,\n      "ms_fast_window_ms": 125.0,\n      "target_cli_args": [\n        "--rms-window-ms",\n        "100",\n        "--ms-fast-window-ms",\n        "125"\n      ]\n    },\n    "CR"', '"rnc_weight": 0.0,\n      "rms_window_ms": 100.0,\n      "ms_fast_window_ms": 125.0,\n      "target_cli_args": [\n        "--rms-window-ms",\n        "25",\n        "--ms-fast-window-ms",\n        "125"\n      ]\n    },\n    "CR"', ["tests/cpu/test_contrastive_targets_guards.py"]),
 # (c) parity
 S('C1a_parity_byte_identity', PAR, '    if hashes and len(set(hashes)) != 1 and not missing:\n', '    if False:\n', G_PAR),
 S('C1b_parity_probe_equals_live', PAR, '        if any(reference.get("before", {}).get(key) != row.get("live_before", {}).get(key) for key in MATCH_FIELDS):\n', '        if False:\n', G_PAR),
 S('C1c_parity_cross_rank_identity', PAR, '    if replicated_inputs and len(set(replicated_inputs)) != 1:\n', '    if False:\n', G_PAR),
 S('C1d_parity_live_restored_forced_true', PAR, '            row["live_state_restored"] = before == after and state_sha256(saved_grads) == state_sha256(_gradient_state(student))\n', '            row["live_state_restored"] = True\n', G_PAR),
 S('C1e_parity_no_rng_restore_in_finally', PAR, '            restore_rng(saved_rng, device)\n            # Probe clones', '            pass\n            # Probe clones', G_PAR),
 S('C1f_parity_batch_index_check', PAR, '        if batch.index != runtime.step:\n', '        if False:\n', G_PAR),
 # (c) resume
 S('C2a_resume_model_1pct', RES, '        if relative[part] > RELATIVE_LIMIT:\n', '        if False:\n', G_RES),
 S('C2b_resume_per_tensor_failures_dropped', RES, '    failures.extend(f"{name}: floating tensor exceeds historical 1%" for name in floating_errors)\n', '    pass\n', G_RES),
 S('C2c_resume_per_tensor_bound_100pct', RES, '        _nested_equal(left[part], right[part], part, RELATIVE_LIMIT, floating_errors)\n', '        _nested_equal(left[part], right[part], part, 1.0, floating_errors)\n', G_RES),
 S('C2d_resume_boundary_state', RES, '                if trace[0].get("boundary", {}).get(field) != value:\n', '                if False:\n', G_RES),
 S('C2e_resume_boundary_rng', RES, '                if trace[0].get(f"rng_{domain}_before_sha256") != digest:\n', '                if False:\n', G_RES),
 S('C2f_resume_start_json', RES, '            if (start.get("protocol") != PROTOCOL or start.get("phase") != phase\n', '            if False and (start.get("protocol") != PROTOCOL or start.get("phase") != phase\n', G_RES),
 S('C2g_resume_active_weights', RES, '                if weights != expected_weights or not any(v > 0 for v in weights):\n', '                if False:\n', G_RES),
 S('C2h_resume_trace_vs_metrics', RES, '                    if not _finite_number(row[name]) or row[name] != metric.get(name):\n', '                    if False:\n', G_RES),
 # (c) release binding
 S('C3a_rel_no_source_git_check', REL, '        verify_reviewed_sources(declared["reviewed_commit"], declared["source_sha256"])\n', '        pass\n', G_REL),
 S('C3b_rel_no_plan_git_check', REL, '            if git_bytes(declared["reviewed_commit"], str(path.relative_to(ROOT))) != path.read_bytes():\n', '            if False:\n', G_REL),
 S('C3c_rel_no_payload_copy_check', REL, '    if binding.get("review_payload") != declared or binding.get("original_review_sha256") != file_sha(review):\n', '    if False:\n', G_REL),
 S('C3d_bind_no_byte_identity', REL, '    if original != local_copy.read_bytes():\n', '    if False:\n', G_REL),
 S('C3e_rel_status_not_required', REL, 'payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":', 'payload["schema"] != REVIEW_SCHEMA:', G_REL),
 S('C3f_rel_multiple_blocks', REL, '    if len(blocks) != 1:\n', '    if len(blocks) < 1:\n', G_REL),
 S('C3g_rel_other_cohort_plans', REL, '    if science != PLAN.resolve() or preflight != PREFLIGHT.resolve():\n', '    if False:\n', G_REL),
 S('C3h_rel_preflight_binding', REL, '    if (plan.get("source_sha256") != declared["source_sha256"] or\n', '    if False and (plan.get("source_sha256") != declared["source_sha256"] or\n', G_REL),
 S('C3i_job_no_binding_call', JOB, "    validate_review_binding(job.get('package_review'))\n", '    pass\n', G_REL),
 S('C3j_run_no_binding_call', RUN, '    validate_review_binding(release.get("package_review"))\n', '    pass\n', G_REL),
 S('C3k_job_review_vs_binding', JOB, "    if job.get('review') != job['package_review'].get('review'):\n", '    if False:\n', G_REL),
 S('C3l_run_review_pin_vs_binding', RUN, '    if release["review_pin"] != release["package_review"]["review_commit"]:\n', '    if False:\n', G_REL),
 S('C3m_job_release_binding_equal', JOB, "    if payload.get('package_review') != job['package_review'] or payload.get('audit_sacct') != job['audit_sacct']:\n", '    if False:\n', G_REL),
 S('C3n_makejob_forbidden_override', REL, '    if forbidden & set(evidence):\n', '    if False:\n', G_REL),
 S('C3o_job_cell_matching', JOB, "    if (len(cells) != 1 or cells[0]['mode'] != mode or cells[0].get('arm') != job.get('arm')\n", "    if False and (len(cells) != 1 or cells[0]['mode'] != mode or cells[0].get('arm') != job.get('arm')\n", G_REL),
 S('C3p_makejob_no_binding_validation', REL, '    declared = validate_review_binding(binding)\n    plan = _read(PREFLIGHT)\n', '    declared = binding["review_payload"]\n    plan = _read(PREFLIGHT)\n', G_REL),
 # (c) audit sacct gate
 S('C4a1_sacct_state', REL, 'if (row["State"] != "COMPLETED" or row["ExitCode"]', 'if (row["ExitCode"]', G_REL),
 S('C4a2_sacct_exitcode', REL, ' or row["ExitCode"] != "0:0" or not 0 <= elapsed', ' or not 0 <= elapsed', G_REL),
 S('C4a3_sacct_elapsed_grace', REL, 'not 0 <= elapsed <= 7200', 'not 0 <= elapsed <= 7560', G_REL),
 S('C4a4_sacct_cpus', REL, '            or cpus != 8 or minutes != 120):', '            or minutes != 120):', G_REL),
 S('C4a5_sacct_timelimit', REL, '            or cpus != 8 or minutes != 120):', '            or cpus != 8):', G_REL),
 S('C4b_sacct_unique_row', REL, '    if len(allocation) != 1:\n', '    if len(allocation) < 1:\n', G_REL),
 S('C4c_acct_audit_sha', REL, 'accounting.get("schema") != AUDIT_SCHEMA or accounting.get("audit_sha256") != file_sha(audit_path)', 'accounting.get("schema") != AUDIT_SCHEMA', G_REL),
 S('C4d_acct_audit_source_only', REL, 'or accounting.get("source_sha256") != source or audit.get("source_sha256") != source):', 'or accounting.get("source_sha256") != source):', G_REL),
 S('C4e_acct_logical_id', REL, '    if len(expected_logical) != 1 or accounting.get("logical_id") != expected_logical[0]:\n', '    if len(expected_logical) != 1:\n', G_REL),
 S('C4f1_run_no_accounting_call', RUN, '    validate_audit_accounting(audit_path, release["audit_sacct"])\n', '    pass\n', G_REL),
 S('C4f2_job_no_accounting_call', JOB, "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))\n", '        pass\n', G_REL),
 S('C4f3_makejob_no_accounting_call', REL, '        validate_audit_accounting(audit, job["audit_sacct"])\n', '        pass\n', G_REL),
 S('C4f4_readout_no_accounting_call', READ, "    accounting = validate_audit_accounting(cpu_audit_path, cpu_audit_sacct_path, expected_source=cohort['source_sha256'])\n", '    accounting = {}\n', G_READ),
 S('C4f5_run_sacct_presence', RUN, '    if not release.get("audit_sacct"):\n', '    if False:\n', G_REL),
 # (c) numeric profile 28 receipts
 S('C5a_numeric_differences_ignored', READ, '            if changed:\n', '            if False:\n', G_READ),
 S('C5b_numeric_missing_ignored', READ, "                missing.append(key + ': missing profile receipt')\n", '                pass\n', G_READ),
 S('C5c_numeric_profile_assert_skipped', READ, "                assert_numeric_flags(flags)\n            except NumericProfileUnavailable as exc:", "                pass\n            except NumericProfileUnavailable as exc:", G_READ),
 S('C5d_numeric_no_early_return', READ, "    if numerical['status'] != 'PASSED':\n", '    if False:\n', G_READ),
 S('C5e_numeric_three_ranks', READ, "        for rank in range(4):\n            key = name + '/rank' + str(rank)", "        for rank in range(3):\n            key = name + '/rank' + str(rank)", G_READ),
 S('C5f_run_evidence_rank_flags_consistency', READ, "    if any(row['flags'] != numeric[0]['flags'] for row in numeric[1:]):\n", '    if False:\n', G_READ),
]
