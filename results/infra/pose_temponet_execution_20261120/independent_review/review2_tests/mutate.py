"""Reviewer TESTS: new mutations (different from the author's 8). Runs only in the reviewer's own detached worktree."""
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
ONLY = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
assert "wf2_tests_tree" in str(ROOT), "refuse to mutate anything but my own worktree"
PROV = "scripts/pose_temponet/provenance.py"
EXEC = "scripts/pose_temponet/execution.py"
METR = "scripts/pose_temponet/metrics.py"
RUNN = "scripts/pose_temponet/runner.py"
T = lambda *names: ["tests/cpu/test_pose_temponet_%s.py" % n for n in names]
CORE = T("provenance", "execution", "preflight_real_layout")

FIRST = '    require(cache_users == producer["users"], "cache user population differs from original producer")\n'
SECOND = ('    require(sorted(storage_directory(user) for user in cache_users) == producer["storage_directories"],\n'
          '            "cache users do not map onto the producer storage directories")\n')
LOOP = '    for key in ("targets","key_time_valid_sha256","dim","n_recordings","n_users","n_windows"):\n'
VAL = ('    val = read_verified_inputs(plan,"val",stage="preflight",depth="post")\n'
       '    result = historical_reference_gate(val,D.read_predictions(plan["inputs"]["post_ridge_predictions"]),\n'
       '                                      D.read_predictions(plan["inputs"]["trained_head_predictions"]))\n')
VALREAD = '    val = read_verified_inputs(plan,"val",stage="preflight",depth="post")\n'
COMPARE = '        require({k:v for k,v in entry.items() if k != "receipt"} == expected, "historical producer identity was rewritten")\n'

MUTATIONS = [
 ("P1", "historical_producer returns storage_directories unsorted (reverse order)", [(PROV,
    '"storage_directories":sorted(directories),', '"storage_directories":sorted(directories,reverse=True),')], CORE),
 ("P2", "historical_producer returns users unsorted (reverse order)", [(PROV,
    '    users = sorted(cache_user(directory) for directory in directories)\n',
    '    users = sorted((cache_user(directory) for directory in directories),reverse=True)\n')], CORE),
 ("P4", "read_verified_inputs first check compares sets instead of sorted lists", [(PROV, FIRST,
    '    require(set(cache_users) == set(producer["users"]), "cache user population differs from original producer")\n')], CORE),
 ("P5", "read_verified_inputs second check compares sets (ignores order and duplicates)", [(PROV, SECOND,
    '    require({storage_directory(user) for user in cache_users} == set(producer["storage_directories"]),\n'
    '            "cache users do not map onto the producer storage directories")\n')], CORE),
 ("P6", "first check weakened to subset (a cache with fewer users passes it)", [(PROV, FIRST,
    '    require(set(cache_users) <= set(producer["users"]), "cache user population differs from original producer")\n')], CORE),
 ("P6b", "both user checks weakened to subset", [(PROV, FIRST,
    '    require(set(cache_users) <= set(producer["users"]), "cache user population differs from original producer")\n'),
    (PROV, SECOND,
    '    require({storage_directory(user) for user in cache_users} <= set(producer["storage_directories"]),\n'
    '            "cache users do not map onto the producer storage directories")\n')], CORE),
 ("P7", "first user check skipped for the val split", [(PROV, FIRST,
    '    require(split == "val" or cache_users == producer["users"], "cache user population differs from original producer")\n')], CORE),
 ("P7b", "both user checks skipped for the val split", [(PROV, FIRST,
    '    require(split == "val" or cache_users == producer["users"], "cache user population differs from original producer")\n'),
    (PROV, SECOND,
    '    require(split == "val" or sorted(storage_directory(user) for user in cache_users) == producer["storage_directories"],\n'
    '            "cache users do not map onto the producer storage directories")\n')], CORE),
 ("P7c", "both user checks skipped for the train split", [(PROV, FIRST,
    '    require(split == "train" or cache_users == producer["users"], "cache user population differs from original producer")\n'),
    (PROV, SECOND,
    '    require(split == "train" or sorted(storage_directory(user) for user in cache_users) == producer["storage_directories"],\n'
    '            "cache users do not map onto the producer storage directories")\n')], CORE),
 ("P8a", "six details comparisons skipped entirely", [(PROV, LOOP, '    for key in ():\n')], CORE),
 ("P8b", "details comparison reduced: targets dropped", [(PROV, LOOP,
    '    for key in ("key_time_valid_sha256","dim","n_recordings","n_users","n_windows"):\n')], CORE),
 ("P8c", "details comparison reduced: key_time_valid_sha256 dropped", [(PROV, LOOP,
    '    for key in ("targets","dim","n_recordings","n_users","n_windows"):\n')], CORE),
 ("P8d", "details comparison reduced to the three counts and dim (both digests dropped)", [(PROV, LOOP,
    '    for key in ("dim","n_recordings","n_users","n_windows"):\n')], CORE),
 ("P8e", "details comparison skipped for the val split only", [(PROV, LOOP,
    '    for key in (("targets","key_time_valid_sha256","dim","n_recordings","n_users","n_windows") if split == "train" else ()):\n')], CORE),
 ("P9", "cache_user uses re.match instead of re.fullmatch", [(PROV,
    're.fullmatch(r"u[0-9A-Za-z.@-]+",directory)', 're.match(r"u[0-9A-Za-z.@-]+",directory)')], CORE),
 ("P10", "cache_user regex restricted to lowercase hex", [(PROV, 'r"u[0-9A-Za-z.@-]+"', 'r"u[0-9a-f]+"')], CORE),
 ("P11", "storage_directory uses another rule (emg2qwerty regex, no @)", [(PROV,
    '    return "u"+_STORAGE_UNSAFE.sub("_",user)\n', '    return "u"+re.sub(r"[^0-9A-Za-z_.-]","_",user)\n')], CORE),
 ("P12", "storage_directory accepts empty and non-string IDs", [(PROV,
    '    require(isinstance(user,str) and user, "invalid cache user ID")\n', '')], CORE),
 ("P13", "historical_producer no longer checks the number of user directories", [(PROV,
    '    require(len(users) == details["n_users"] == expected_users and\n',
    '    require(details["n_users"] == expected_users and\n')], CORE),
 ("P14", "validate_plan stops comparing the producer entry with the receipt", [(PROV, COMPARE, '')], CORE),
 ("P15", "validate_plan ignores users and storage_directories when comparing the producer entry", [(PROV, COMPARE,
    '        require({k:v for k,v in entry.items() if k not in ("receipt","users","storage_directories")} ==\n'
    '                {k:v for k,v in expected.items() if k not in ("users","storage_directories")}, "historical producer identity was rewritten")\n')], CORE),
 ("P16", "preflight returns before reading val (no val read, no R-1 gate)", [(PROV, VAL, '    result = {"passed":True}\n')], CORE),
 ("P16b", "preflight reads val through D.read_cache_pin only (no user and no details check on val)", [(PROV, VALREAD,
    '    val = D.read_cache_pin(plan["inputs"]["cache_val"],producer=plan["producers"]["val"]["metadata"],\n'
    '                           reference_pin=plan["inputs"]["target_reference_val"],depth="post")\n')], CORE),
 ("P16c", "preflight runs the R-1 gate on the train split instead of val", [(PROV, VALREAD,
    '    val = read_verified_inputs(plan,"train",stage="preflight",depth="post")\n')], CORE),
 ("P17", "R-1 tolerance loosened from 1e-12 to 1e-6", [(METR,
    'abs(value-HISTORICAL_REFERENCE[reader+"_E_dyn"]) <= 1e-12', 'abs(value-HISTORICAL_REFERENCE[reader+"_E_dyn"]) <= 1e-6')],
    CORE + T("metrics")),
 ("P17b", "R-1 E_dyn comparison removed", [(METR,
    '        require(value is not None and abs(value-HISTORICAL_REFERENCE[reader+"_E_dyn"]) <= 1e-12,\n'
    '                "historical E_dyn parity failed: " + reader)\n', '')], CORE + T("metrics")),
 ("P18", "R-1 constants altered (E_dyn last digits, users 15 -> 16)", [(METR,
    '    "post_ridge_E_dyn": 0.7859020004638595,\n', '    "post_ridge_E_dyn": 0.7859020004638695,\n'),
    (METR, '    "dynamic_users": 15,\n', '    "dynamic_users": 16,\n')],
    T("benchmark", "cli", "cli_guards", "controls", "data", "execution", "heads", "linear", "metrics", "operational",
      "preflight_real_layout", "provenance", "report", "resources", "runner", "runtime_fixes", "scientific_integration",
      "selection", "train", "workflow")),
 ("E1", "job_id drops the delivery suffix for one stage (collection keeps 20261119)", [(EXEC,
    '    parts.append(DELIVERY)\n', '    parts.append(DELIVERY if stage != "collection" else "20261119")\n')], T("execution", "operational")),
 ("E2", "CAMPAIGN diverges from the ID suffix (literal 20261119)", [(EXEC,
    'CAMPAIGN = "pose_temponet_" + DELIVERY\n', 'CAMPAIGN = "pose_temponet_20261119"\n')], T("execution", "operational", "cli")),
 ("E3", "CODE_DIR diverges from CAMPAIGN", [(EXEC,
    'CODE_DIR = "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/" + CAMPAIGN\n',
    'CODE_DIR = "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261119"\n')], T("execution", "operational", "cli")),
 ("E4", "submit_command records the old campaign in the ledger (--campagna)", [(EXEC,
    '"--campagna", CAMPAIGN,', '"--campagna", "pose_temponet_20261119",')], T("execution", "operational", "cli", "cli_guards")),
 ("E5", "preflight export passes the old campaign name to the checkout wrapper", [(EXEC,
    'export += f",CAMPAIGN={Path(code_dir).name},CAMPAIGN_SHA={checkout_sha}"',
    'export += f",CAMPAIGN=pose_temponet_20261119,CAMPAIGN_SHA={checkout_sha}"')], T("execution", "operational", "cli", "cli_guards")),
 ("R1a", "runner gives the benchmarks the storage directories as allowed users", [(RUNN,
    '                      "allowed_train_users": plan["producers"]["train"]["users"],\n                      "input_identity": _input_identity(plan)}\n',
    '                      "allowed_train_users": plan["producers"]["train"]["storage_directories"],\n                      "input_identity": _input_identity(plan)}\n')],
    T("runner", "runtime_fixes", "benchmark", "resources", "e2e_chain")),
 ("R1b", "runner gives calibration the storage directories as allowed users", [(RUNN,
    'controls.build_positive_control_fold(cache, fit, held, plan["producers"]["train"]["users"],',
    'controls.build_positive_control_fold(cache, fit, held, plan["producers"]["train"]["storage_directories"],')],
    T("runner", "e2e_chain")),
 ("P14all", "validate_plan stops comparing the producer entry with the receipt (20 TCN files, all but e2e_chain)", [(PROV, COMPARE, '')],
    T("benchmark", "cli", "cli_guards", "controls", "data", "execution", "heads", "linear", "metrics", "operational",
      "preflight_real_layout", "provenance", "report", "resources", "runner", "runtime_fixes", "scientific_integration",
      "selection", "train", "workflow")),
 ("P16ball", "preflight reads val through D.read_cache_pin only (20 TCN files, all but e2e_chain)", [(PROV, VALREAD,
    '    val = D.read_cache_pin(plan["inputs"]["cache_val"],producer=plan["producers"]["val"]["metadata"],\n'
    '                           reference_pin=plan["inputs"]["target_reference_val"],depth="post")\n')],
    T("benchmark", "cli", "cli_guards", "controls", "data", "execution", "heads", "linear", "metrics", "operational",
      "preflight_real_layout", "provenance", "report", "resources", "runner", "runtime_fixes", "scientific_integration",
      "selection", "train", "workflow")),
 ("A0", "first-delivery behaviour (author's mutation a), here ONLY against the real_layout file the author did not mutation-check", [(PROV,
    '    users = sorted(cache_user(directory) for directory in directories)\n', '    users = sorted(directories)\n')],
    T("preflight_real_layout")),
 ("A1", "author's mutation b (second check removed) ONLY against the real_layout file", [(PROV, SECOND, '')], T("preflight_real_layout")),
 ("A2", "author's mutation f (first check removed) ONLY against the real_layout file", [(PROV, FIRST, '')], T("preflight_real_layout")),
]

ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def run(tests):
    command = [str(Path.home()/".venvs/wearusfm-tok/bin/python"), "-c",
               "import sys; sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', "
               "'--tb=line', '-o', 'tmp_path_retention_policy=none', " + ", ".join(repr(t) for t in tests) + "]))"]
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True)
    text = result.stdout
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)))
    summary = [line for line in text.splitlines() if re.search(r"\d+ (passed|failed|error)", line)]
    lines = [line for line in text.splitlines() if re.match(r"^(/|E  |tests/).*(Error|Failed|assert|DID NOT)", line)]
    return {"returncode": result.returncode, "failed": failed, "summary": summary[-1] if summary else text[-400:],
            "reasons": lines[:12], "seconds": round(time.monotonic()-started, 1)}


def git_clean():
    return subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == ""


assert git_clean(), "worktree not clean"
report = json.loads(OUT.read_text()) if OUT.exists() else []
done = {r["id"] for r in report}
OVERRIDE = os.environ.get("WF2_TESTS_OVERRIDE")
for ident, name, edits, tests in MUTATIONS:
    if ONLY is not None and ident not in ONLY:
        continue
    if OVERRIDE:
        ident, tests = ident+"+gaptests", OVERRIDE.split(",")
    if ident in done:
        continue
    originals = {}
    try:
        for path, old, new in edits:
            file = ROOT/path
            originals.setdefault(file, file.read_text())
            text = file.read_text()
            assert text.count(old) == 1, (ident, path, text.count(old), old)
            file.write_text(text.replace(old, new))
        outcome = run(tests)
    finally:
        for file, text in originals.items():
            file.write_text(text)
    assert git_clean(), "restore failed"
    row = {"id": ident, "mutation": name, "tests": [t.split("/")[-1][:-3].replace("test_pose_temponet_", "") for t in tests], **outcome,
           "caught": bool(outcome["failed"])}
    report.append(row)
    OUT.write_text(json.dumps(report, indent=1))
    print(ident, "CAUGHT" if row["caught"] else "SURVIVED", "|", outcome["summary"], "|", outcome["seconds"], "s |", name, flush=True)
    for f in outcome["failed"]:
        print("     ", f.split("::", 1)[1] if "::" in f else f, flush=True)
