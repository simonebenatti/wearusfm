SC = 'scripts/contrastive_source_compat.py'
G_HIST = ["tests/cpu/test_contrastive_historical_calibration.py", "tests/cpu/test_contrastive_historical_callsites.py", "tests/cpu/test_contrastive_source_compat.py"]
SPECS = [
 {'id': 'M1d_historical_identity_barrier', 'edits': [(SC, '    _require(actual == manifest["current_sources"], "current execution source inventory/pins differ")\n', '    pass\n')], 'tests': G_HIST},
]
ns_main = {}
exec(open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/r5/specs_main.py').read(), ns_main)
_by = {s['id']: s for s in ns_main['SPECS']}
SPECS.append({'id': 'M1_all_three_barriers__ALL', 'edits': _by['M1a_validate_audit_config']['edits'] + _by['M1b_read_preflight_config']['edits'] + _by['M1c_authenticate_calibration_receipt']['edits'], 'tests': 'ALL'})
SPECS.append({'id': 'N4_plan_pointer_and_text_old__ALL', 'edits': _by['N4a_PLAN_pointer_old']['edits'] + _by['N4b_readout_text_old']['edits'], 'tests': 'ALL'})
REL = 'scripts/contrastive_release.py'
G_REL = ["tests/cpu/test_contrastive_release.py", "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_run.py", "tests/cpu/test_contrastive_targets_readout.py", "tests/cpu/test_contrastive_historical_callsites.py"]
SPECS.append({'id': 'C3aa_rel_declared_plan_hashes_ignored', 'edits': [(REL, '    if (declared["source_sha256"] != source_digest() or declared["science_plan_sha256"] != file_sha(science)\n            or declared["preflight_plan_sha256"] != file_sha(preflight)):\n', '    if (declared["source_sha256"] != source_digest()):\n')], 'tests': G_REL})
SPECS.append({'id': 'C3ab_rel_declared_source_ignored', 'edits': [(REL, '    if (declared["source_sha256"] != source_digest() or declared["science_plan_sha256"] != file_sha(science)\n', '    if (declared["science_plan_sha256"] != file_sha(science)\n')], 'tests': G_REL})
