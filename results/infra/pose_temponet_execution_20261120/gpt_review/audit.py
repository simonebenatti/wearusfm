"""Independent local review of the second TCN delivery; no release or remote I/O.

Run with the wearusfm-tok Python from the repository root. Assertions compare
committed plans, original receipts and archived metadata observations. They do
not certify the unread full cache arrays or replace the production R-1 gate.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from pose_temponet import provenance as P, execution as E

BASE = "31877ad85c619ec69214cd5e9fb53df50e452148"
TIP = "0e548bf8218f15ba07fc731efcc2ead512e65034"
CODE = "5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba"
HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plan.json"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    new = read(PLAN)
    old = json.loads(git("show", BASE + ":results/infra/pose_temponet_execution_20261119/plan.json"))
    expected_source = "37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de"
    assert sha(PLAN) == "1e8e2b46b046072c706738995e745f608f009d33406d970729a3eb22a73f563e"
    assert P.source_identity() == new["reader"]["source_sha256"] == expected_source
    assert len(P.source_manifest()) == 246
    assert new["status"] == "REVIEW_REQUIRED" and new["submission_released"] is False
    assert git("diff", "--name-only", CODE, TIP, "--", "src", "scripts") == b""
    changed = git("diff", "--name-only", BASE, TIP, "--", "src", "scripts").decode().splitlines()
    assert changed == ["scripts/pose_temponet/execution.py", "scripts/pose_temponet/provenance.py"]
    for path, digest in P.source_manifest().items():
        assert hashlib.sha256(git("show", CODE + ":" + path)).hexdigest() == digest
    fixed = ("proposal", "recipe_sha256", "inputs", "resource_policy", "output_root", "numeric_pins")
    assert all(new[key] == old[key] for key in fixed)
    normalized_jobs = json.loads(json.dumps(new["jobs"]).replace("-20261120", "-20261119"))
    assert normalized_jobs == old["jobs"] and new["jobs"] == E.job_graph()
    assert len(new["jobs"]) == 24
    assert not {job["id"] for job in new["jobs"]} & {job["id"] for job in old["jobs"]}
    assert P.canonical_bytes(P.build_plan(new["proposal"]["path"], jobs=E.job_graph())) == P.canonical_bytes(new)
    try:
        P.validate_plan(old)
    except ValueError as exc:
        old_plan_rejection = str(exc)
    else:
        raise AssertionError("old plan accepted by new source")
    users = read(ROOT / "tests/cpu/data/pose_temponet_cache_users_20261120.json")
    operator = read(ROOT / "results/infra/pose_temponet_execution_20261119/launcher_preflight_20261120/user_format_diagnosis.json")
    populations = {}
    for split, count in (("train", 40), ("val", 15)):
        entry = new["producers"][split]
        receipt = read(ROOT / P.HISTORICAL_RECEIPTS[split]["local_copy"])
        directories = sorted({Path(name).parts[Path(name).parts.index("emg2pose") + 1]
                              for name in receipt["raw_inputs"]["file_states"] if "emg2pose" in Path(name).parts})
        assert all(re.fullmatch(r"u[0-9a-f]{10}", directory) for directory in directories)
        # Independent inverse for the observed population: no call to cache_user.
        decoded = [directory[1:] for directory in directories]
        assert decoded == entry["users"] == users[split]["users"] and len(decoded) == count
        assert directories == entry["storage_directories"] == old["producers"][split]["users"]
        assert hashlib.sha256(json.dumps(decoded).encode()).hexdigest() == operator[split]["actual_users_sha256"]
        assert hashlib.sha256(json.dumps(directories).encode()).hexdigest() == operator[split]["producer_users_sha256"]
        assert entry["metadata"] == old["producers"][split]["metadata"]
        assert entry["details"] == old["producers"][split]["details"]
        populations[split] = {"users": count, "windows": users[split]["windows"], "receipt_matches_cache_and_operator": True}
    assert not set(new["producers"]["train"]["users"]) & set(new["producers"]["val"]["users"])
    pinned = read(HERE.parent / "delivery_pin.json")
    verified_pins = []
    for name, pin in pinned.items():
        if isinstance(pin, dict) and {"path", "sha256", "bytes"} <= pin.keys():
            path = ROOT / pin["path"]
            assert sha(path) == pin["sha256"] and path.stat().st_size == pin["bytes"], name
            verified_pins.append(name)
    headers = subprocess.check_output([sys.executable, str(HERE.parent / "real_input_headers/check_headers.py")], cwd=ROOT)
    header_result = json.loads(headers)
    assert header_result == {"checks": 93, "passed": 93, "failed": []}
    report = {"reviewed_tip": TIP, "reviewed_code": CODE, "base": BASE,
              "source_sha256": expected_source, "source_files": 246, "changed_sources": changed,
              "plan_sha256": sha(PLAN), "fixed_plan_fields": list(fixed), "same_normalized_24_job_graph": True,
              "old_plan_rejection": old_plan_rejection, "populations": populations,
              "delivery_pins_verified": verified_pins, "archived_header_checks": header_result,
              "recipe_changed": False, "release_performed": False, "remote_operations": False,
              "limit": "Archived metadata only; full array digests, real R-1, timing and memory await the new preflight."}
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
