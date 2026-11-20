"""Skeptic R5-1: own mutation driver (one textual mutation at a time, then git checkout).

Usage: mutants.py <clone> <out.jsonl> <tests: SUBSET|ALL> [ids,comma,separated]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
TMP = Path("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp")
REL = "scripts/contrastive_release.py"
SUBSET = ["tests/cpu/test_contrastive_release.py", "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_run.py",
          "tests/cpu/test_contrastive_historical_callsites.py", "tests/cpu/test_contrastive_targets_readout.py"]

M = {
    # central barriers named in R5-1
    "S01_status": [('or payload["status"] != "APPROVED":', ':')],
    "S02_one_block": [("if len(blocks) != 1:", "if len(blocks) < 1:")],
    "S03_cohort_plans": [("if science != PLAN.resolve() or preflight != PREFLIGHT.resolve():", "if False:")],
    "S04_declared_plan_hashes": [('if (declared["source_sha256"] != source_digest() or declared["science_plan_sha256"] != file_sha(science)\n'
                                  '            or declared["preflight_plan_sha256"] != file_sha(preflight)):',
                                  'if (declared["source_sha256"] != source_digest()):')],
    "S05_preflight_internal": [('if (plan.get("source_sha256") != declared["source_sha256"] or\n'
                                '            plan.get("science_plan", {}).get("sha256") != declared["science_plan_sha256"]):',
                                'if False:')],
    "S06_reviewed_sources": [('        verify_reviewed_sources(declared["reviewed_commit"], declared["source_sha256"])\n',
                              '        pass\n')],
    "S07_plan_bytes_git": [('if git_bytes(declared["reviewed_commit"], str(path.relative_to(ROOT))) != path.read_bytes():',
                            'if False:')],
    "S08_copy_payload": [('if binding.get("review_payload") != declared or binding.get("original_review_sha256") != file_sha(review):',
                          'if False:')],
    "S09_make_job_no_validation": [("    declared = validate_review_binding(binding)\n    plan = _read(PREFLIGHT)",
                                    '    declared = binding["review_payload"]\n    plan = _read(PREFLIGHT)')],
    "S10_make_job_inputs": [('if evidence.get("inputs") is not None and {k: v.get("sha256") for k, v in evidence["inputs"].items()} != science["input_sha256"]:',
                             'if False:')],
    # form checks named in R5-1
    "S11_pin_inside_checkout": [("if repository and not path.is_relative_to(ROOT.resolve()):", "if False:")],
    "S12_git_bytes_form": [('if not _full_sha(commit) or not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:',
                            'if False:')],
    "S13_extra_keys": [("if set(payload) != keys or", "if not keys <= set(payload) or")],
    "S14_hash_format": [('if not isinstance(payload[name], str) or re.fullmatch("[a-f0-9]{64}", payload[name]) is None:', 'if False:')],
    "S15_review_commit_full": [('if not _full_sha(binding.get("review_commit")):', 'if False:')],
    # controls: these SHOULD be killed (sanity of the driver)
    "K01_source_declared": [('if (declared["source_sha256"] != source_digest() or declared["science_plan_sha256"] != file_sha(science)',
                             'if (declared["science_plan_sha256"] != file_sha(science)')],
    "K02_bind_byte_identity": [("if original != local_copy.read_bytes():", "if False:")],
    "K03_override": [("if forbidden & set(evidence):", "if False:")],
}


def apply(clone, mid):
    path = clone / REL
    text = path.read_text()
    for old, new in M[mid]:
        if text.count(old) != 1:
            raise SystemExit(f"{mid}: {text.count(old)} occurrences of {old!r}")
        text = text.replace(old, new)
    path.write_text(text)


def main():
    clone, out, which = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    ids = sys.argv[4].split(",") if len(sys.argv) > 4 else list(M)
    if which == "ALL":
        receipt = json.loads((clone / "results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json").read_text())
        tests = [x["path"] for x in receipt["test_files"]]
    else:
        tests = SUBSET
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{clone}/src:{clone}/tests/cpu:{clone}/scripts"}
    env.pop("SK1_TRACE", None)
    for mid in ids:
        if subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout:
            raise SystemExit("clone not clean before " + mid)
        if mid != "NONE":
            for part in mid.split("+"):
                apply(clone, part)
        diff = subprocess.run(["git", "-C", str(clone), "diff", "--numstat"], capture_output=True, text=True).stdout.strip()
        bt = TMP / ("sk1_r51_bt_" + str(abs(hash(mid))) + "_" + which)
        t0 = time.time()
        proc = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                               "-rfE", "--basetemp=" + str(bt), *tests], cwd=clone, env=env, capture_output=True, text=True)
        text = proc.stdout + proc.stderr
        subprocess.run(["git", "-C", str(clone), "checkout", "--", REL], check=True)
        clean = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout == ""
        shutil.rmtree(bt, ignore_errors=True)
        summary = [l for l in text.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
        failed = [l[:240] for l in text.splitlines() if l.startswith(("FAILED", "ERROR"))]
        row = {"id": mid, "tests": which, "files": len(tests), "diff": diff, "rc": proc.returncode,
               "seconds": round(time.time() - t0, 1), "summary": summary[-1] if summary else text[-400:],
               "failed": failed, "clean_after": clean}
        with out.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
