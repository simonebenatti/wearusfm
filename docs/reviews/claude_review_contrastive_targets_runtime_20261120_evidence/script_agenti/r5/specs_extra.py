G_RUN = ["tests/cpu/test_contrastive_targets_guards.py", "tests/cpu/test_contrastive_run.py", "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_historical_callsites.py"]
G_READ = ["tests/cpu/test_contrastive_targets_readout.py", "tests/cpu/test_contrastive_benchmark_readout.py", "tests/cpu/test_contrastive_targets_guards.py", "tests/cpu/test_contrastive_historical_callsites.py"]
G_GUARD = ["tests/cpu/test_contrastive_targets_guards.py"]
G_PAR = ["tests/cpu/test_contrastive_target_parity.py", "tests/cpu/test_contrastive_targets_readout.py"]
G_RES = ["tests/cpu/test_contrastive_target_resume.py", "tests/cpu/test_contrastive_targets_readout.py"]
G_REL = ["tests/cpu/test_contrastive_release.py", "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_run.py", "tests/cpu/test_contrastive_targets_readout.py", "tests/cpu/test_contrastive_historical_callsites.py"]
G_NUM = ["tests/cpu/test_contrastive_numeric_profile.py", "tests/cpu/test_contrastive_target_parity.py", "tests/cpu/test_contrastive_targets_readout.py", "tests/cpu/test_contrastive_benchmark_readout.py", "tests/cpu/test_contrastive_cuda_check.py"]
RUN, READ, PREP = 'scripts/contrastive_run.py', 'scripts/contrastive_benchmark_readout.py', 'scripts/contrastive_prepare.py'
PAR, RES, REL, JOB = 'scripts/contrastive_target_parity.py', 'scripts/contrastive_target_resume.py', 'scripts/contrastive_release.py', 'scripts/contrastive_job.py'
NUM, CUDA, LV = 'scripts/contrastive_numeric_profile.py', 'scripts/contrastive_cuda_check.py', 'scripts/contrastive_local_validation.py'
PLANJ = 'results/infra/contrastive_targets_20261118/plan.json'
def S(i, f, old, new, tests): return {'id': i, 'edits': [(f, old, new)], 'tests': tests}
SPECS = [
 # N-4 extra
 S('N4c3_builder_fast50', PREP, '    base["loader"]["ms_fast_window_ms"] = 125.\n', '    base["loader"]["ms_fast_window_ms"] = 50.\n', G_GUARD),
 S('N4c4_builder_version_v2', PREP, '    base["target_version"] = target_version_for_fast_window(125.)\n', '    base["target_version"] = target_version_for_fast_window(100.)\n', G_GUARD),
 S('N4c5_plan_C0_fast50', PLANJ, '      "rnc_weight": 0.0,\n      "rms_window_ms": 100.0,\n      "ms_fast_window_ms": 125.0,\n      "target_cli_args": [\n        "--rms-window-ms",\n        "100",\n        "--ms-fast-window-ms",\n        "125"\n      ]\n    },\n    "CN"', '      "rnc_weight": 0.0,\n      "rms_window_ms": 100.0,\n      "ms_fast_window_ms": 50.0,\n      "target_cli_args": [\n        "--rms-window-ms",\n        "100",\n        "--ms-fast-window-ms",\n        "125"\n      ]\n    },\n    "CN"', G_GUARD),
 # parity extra
 S('C1g_parity_two_hashes_allowed', PAR, '    if hashes and len(set(hashes)) != 1 and not missing:\n', '    if hashes and len(set(hashes)) > 2 and not missing:\n', G_PAR),
 S('C1h_parity_only_rank0_hashes', PAR, '            hashes.append(digest)\n', '            hashes.append(digest) if row.get("rank") == 0 else None\n', G_PAR),
 S('C1i_parity_nonzero_weight', PAR, '            if case.get("nce_weight") != 0.0 or case.get("rnc_weight") != 0.0:\n', '            if False:\n', G_PAR),
 S('C1j_parity_global_batch', PAR, '        if row.get("global_batch_size") != 32:\n', '        if False:\n', G_PAR),
 S('C1l_parity_production_cfg', PAR, '        if cfg.batch_size != 32 or cfg.accum_steps != 1 or cfg.amp_bf16 is not True or ctx.world_size != 4:\n', '        if False:\n', G_PAR),
 S('C1m_parity_cfg_change_between_points', PAR, '        if self.config_sha256 not in (None, cfg_digest):\n', '        if False:\n', G_PAR),
 S('C1n_parity_duplicate_update', PAR, '        if update in [p["update"] for p in self.points]:\n', '        if False:\n', G_PAR),
 S('C1q_parity_report_profile', PAR, '    if report.get("schema") != 1 or report.get("profile") != PROFILE:\n', '    if False:\n', G_PAR),
 S('C1r_parity_report_both_updates', PAR, '    if [p.get("update") for p in points] != list(UPDATES):\n', '    if False:\n', G_PAR),
 S('C1t_parity_measure_raise_on_unrestored', PAR, '        if not all(r["live_state_restored"] for r in point["ranks"]):\n', '        if False:\n', G_PAR),
 # resume extra
 S('C2i_resume_limit_3pct', RES, 'RELATIVE_LIMIT = .01\n', 'RELATIVE_LIMIT = .03\n', G_RES),
 S('C2j_resume_pair_trace_differences_ignored', RES, '        pair["passed"] = pair["passed"] and not differences and not boundary_errors and not support_missing\n', '        pair["passed"] = pair["passed"] and not boundary_errors and not support_missing\n', G_RES),
 S('C2k_resume_support_missing_ignored', RES, '                if active_support <= 0:\n', '                if False:\n', G_RES),
 S('C2l_resume_fork_rank_rng', RES, '    if ranks > 1:\n        rank_rng = state.get("rank_rng")', '    if False:\n        rank_rng = state.get("rank_rng")', G_RES),
 S('C2m_resume_copy_identity', RES, '    if file_sha(target) != expected_sha or file_sha(source_checkpoint) != expected_sha:\n', '    if False:\n', G_RES),
 S('C2n_resume_endpoint_checks', RES, '        if (summary.get("steps") != end_step or summary.get("stopped") != "max_steps"\n', '        if False and (summary.get("steps") != end_step or summary.get("stopped") != "max_steps"\n', G_RES),
 S('C2o_resume_validate_fork_state', RES, '    if (state.get("step") != fork_step or state.get("world_size", 1) != ranks\n', '    if False and (state.get("step") != fork_step or state.get("world_size", 1) != ranks\n', G_RES),
 S('C2p_resume_rows_order', RES, '    if [row.get("update", row.get("step")) for row in values] != list(range(start + 1, end + 1)):\n', '    if False:\n', G_RES),
 S('C2r_resume_trace_row_invariants', RES, '                if (row.get("rank") != rank or row.get("phase") != phase\n', '                if False and (row.get("rank") != rank or row.get("phase") != phase\n', G_RES),
 # release extra
 S('C3q_rel_git_path_guard', REL, '    if not _full_sha(commit) or not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:\n', '    if False:\n', G_REL),
 S('C3r_rel_review_extra_keys', REL, '    if set(payload) != keys or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":\n', '    if not keys <= set(payload) or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":\n', G_REL),
 S('C3s_rel_hash_format', REL, '        if not isinstance(payload[name], str) or re.fullmatch("[a-f0-9]{64}", payload[name]) is None:\n', '        if False:\n', G_REL),
 S('C3t_rel_pin_inside_checkout', REL, '    if repository and not path.is_relative_to(ROOT.resolve()):\n', '    if False:\n', G_REL),
 S('C3v_makejob_inputs_vs_science', REL, '    if evidence.get("inputs") is not None and {k: v.get("sha256") for k, v in evidence["inputs"].items()} != science["input_sha256"]:\n', '    if False:\n', G_REL),
 S('C3w_rel_binding_full_review_commit', REL, '    if not _full_sha(binding.get("review_commit")):\n', '    if False:\n', G_REL),
 S('C3x_cuda_prereq_checkpoint', REL, '    if cuda.get("inputs", {}).get("calibration_checkpoint") != file_sha(Path(calibration_run_dir) / "checkpoint.pt"):\n', '    if False:\n', G_REL),
 S('C3y_cuda_prereq_rank_exit_pin', REL, '    if file_sha(rank_exit) != cuda.get("rank_exit_report_sha256"):\n', '    if False:\n', G_REL),
 S('C3z_cuda_prereq_rank_exit_receipt', REL, '    validate_receipt(rank_exit, expected_source_sha256=cohort["source_sha256"])\n    return cuda', '    pass\n    return cuda', G_REL),
 # accounting extra
 S('C4g_acct_required_columns', REL, '    if not records or not required <= set(records[0]):\n', '    if not records:\n', G_REL),
 S('C4h_acct_job_id_digits', REL, '    if not isinstance(job_id, str) or not job_id.isdigit() or (expected_job_id is not None and job_id != str(expected_job_id)):\n', '    if (expected_job_id is not None and job_id != str(expected_job_id)):\n', G_REL),
 # dispatcher extra
 S('C8a_job_fresh_cohort_root', JOB, '    if not out.is_relative_to(results) or not out.is_relative_to(cohort_root) or out == cohort_root or out.exists():\n', '    if not out.is_relative_to(results) or out.exists():\n', G_REL),
 S('C8b_job_cuda_prereq', JOB, "        if payload.get('cuda_report') != job.get('cuda_report') or not job.get('cuda_report'):\n", '        if False:\n', G_REL),
 S('C8c_job_resume_fork_pins', JOB, "        if job.get('repeat') not in ('a', 'b') or payload.get('fork_checkpoint') != job.get('fork_checkpoint') or payload.get('continuous_summary') != job.get('continuous_summary'):\n", '        if False:\n', G_REL),
 S('C8d_job_expected_phase', JOB, "    expected_phase = 'resume_' + str(job.get('repeat', '')).lower() if mode == 'resume' else mode\n", "    expected_phase = payload.get('phase')\n", G_REL),
 S('C8e_job_readout_arm_set', JOB, "            if set(job[key]) != set(arms): raise ValueError('readout arm set differs')\n", "            if False: raise ValueError('readout arm set differs')\n", G_REL),
 # runtime extra
 S('C6a_run_no_target_fork_call', RUN, '        validate_target_fork(release, cfg, cohort)\n', '        pass\n', G_REL),
 S('C6b_run_target_fork_summary_checks', RUN, '    if (summary.get("steps") != 576 or summary.get("stopped") != "max_steps" or\n            summary.get("world_size") != 4 or contrastive.get(', '    if False and (summary.get("steps") != 576 or summary.get("stopped") != "max_steps" or\n            summary.get("world_size") != 4 or contrastive.get(', G_REL),
 S('C6c_run_no_cuda_prereq_call', RUN, '        validate_cuda_prerequisite(release.get("cuda_report"), cohort, pins, calibration_run_dir)\n', '        pass\n', G_REL),
 S('C6d_run_benchmark_phase', RUN, '        if phase not in ("full", "resume_a", "resume_b") or raw["max_steps"] != 576:\n', '        if False:\n', G_REL),
 S('C6e_run_resume_arm', RUN, '        if phase != "full" and release.get("arm") not in ("CN", "CR"):\n', '        if False:\n', G_REL),
 S('C6g_run_reload_copy_sha', RUN, '                if sha(a.out_dir / "checkpoint.pt") != release["fork_checkpoint"]["sha256"]:\n', '                if False:\n', G_REL),
 S('C6h_run_cohort_keep_checkpoints', RUN, '        if raw["max_steps"] in (5000, 19618):\n            split_expected["schedule"]["keep_checkpoints"] = [5000, 19618]\n', '        if False:\n            split_expected["schedule"]["keep_checkpoints"] = [5000, 19618]\n', G_REL),
 # readout extra
 S('C5h_readout_parity_binding', READ, "        if (parity.get('source_sha256') != cohort['source_sha256'] or parity.get('config_sha256') != canonical_cfg or\n", "        if False and (parity.get('source_sha256') != cohort['source_sha256'] or parity.get('config_sha256') != canonical_cfg or\n", G_READ),
 S('C5i_readout_parity_failed_ignored', READ, "        elif status != 'PASSED':\n            result['failure_reasons'].append('production zero-weight gradients/invariants not byte-identical')", "        elif False:\n            result['failure_reasons'].append('production zero-weight gradients/invariants not byte-identical')", G_READ),
 S('C5j_readout_reload_stream', READ, "                if [r['origin'] for r in reload_records] != [r['origin'] for r in records[544:]]:\n", '                if False:\n', G_READ),
 S('C5l_readout_gpu_memory', READ, "    if type(summary.get('gpu_max_memory_gib')) not in (int,float) or not math.isfinite(summary['gpu_max_memory_gib']) or summary['gpu_max_memory_gib'] <= 0:\n", '    if False:\n', G_READ),
 S('C5m_cuda_evidence_numeric_profile', READ, "        assert_numeric_flags(rank.get('numeric_profile', {}))\n        if rank.get('passed') is not True", "        pass\n        if rank.get('passed') is not True", G_READ),
 S('C5o_readout_resume_failure_ignored', READ, "            elif not check.get('passed'):\n                result['failure_reasons'].append(arm+': common-checkpoint historical bounds or invariants failed')", "            elif False:\n                result['failure_reasons'].append(arm+': common-checkpoint historical bounds or invariants failed')", G_READ),
 S('C5p_numeric_differences_as_inconclusive', READ, "    status = 'FAILED' if differences else 'INCONCLUSIVE' if missing else 'PASSED'\n    return {'status': status, 'required_runs'", "    status = 'INCONCLUSIVE' if differences or missing else 'PASSED'\n    return {'status': status, 'required_runs'", G_READ),
 # numeric profile extra
 S('C7b_profile_cudnn_sdp', NUM, '    if flags.get("cudnn_sdp_enabled") not in (False, None):\n', '    if False:\n', G_NUM),
 S('C7c_profile_require_cuda', NUM, '    if require_cuda and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):\n', '    if False:\n', G_NUM),
 S('C7d_profile_require_autocast', NUM, '    if require_autocast and (flags.get("autocast_cuda_enabled") is not True or\n', '    if False and (flags.get("autocast_cuda_enabled") is not True or\n', G_NUM),
 S('C7e_cuda_check_no_profile_context', CUDA, '    with numeric_profile(require_cuda=True):\n        return _main(argv)', '    if True:\n        return _main(argv)', G_NUM),
 S('C7f_cuda_check_rank_profile_empty', CUDA, '"numeric_profile": assert_numeric_profile(require_cuda=True)})', '"numeric_profile": {}})', G_NUM),
 S('C7g_run_no_profile_context', RUN, '            flags = stack.enter_context(numeric_profile(require_cuda=True))\n', '            flags = {}\n', G_REL),
 # local validation
 S('C9a_lv_verify_source', LV, "        if receipt['source_sha256']!=source_digest(): raise ValueError('validation source changed')\n", "        if False: raise ValueError('validation source changed')\n", ["tests/cpu/test_contrastive_targets_plan.py", "tests/cpu/test_contrastive_targets_guards.py"]),
]
