"""Skeptic-2 mutation driver for R5-1 (independent from r5 scripts).

usage: python -B mut.py <clone> <out.jsonl> <set: targeted|full|release> <mutation-id> [<mutation-id> ...]
Mutation ids joined with '+' are applied together in one run.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REL = "scripts/contrastive_release.py"
PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
TMP = Path("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp")

M = {
    # central barriers
    "M01_status": ('    if set(payload) != keys or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":\n',
                   '    if set(payload) != keys or payload["schema"] != REVIEW_SCHEMA:\n'),
    "M02_block_ge1": ('    if len(blocks) != 1:\n', '    if len(blocks) < 1:\n'),
    "M02b_block_removed": ('    if len(blocks) != 1:\n', '    if False:\n'),
    "M03_other_cohort": ('    if science != PLAN.resolve() or preflight != PREFLIGHT.resolve():\n', '    if False:\n'),
    "M04_declared_plan_hashes": ('    if (declared["source_sha256"] != source_digest() or declared["science_plan_sha256"] != file_sha(science)\n'
                                 '            or declared["preflight_plan_sha256"] != file_sha(preflight)):\n',
                                 '    if (declared["source_sha256"] != source_digest()):\n'),
    "M05_preflight_internal": ('    if (plan.get("source_sha256") != declared["source_sha256"] or\n'
                               '            plan.get("science_plan", {}).get("sha256") != declared["science_plan_sha256"]):\n',
                               '    if False:\n'),
    "M06_reviewed_sources_call": ('        verify_reviewed_sources(declared["reviewed_commit"], declared["source_sha256"])\n', '        pass\n'),
    "M06b_source_bytes_cmp": ('        if original != path.read_bytes():\n'
                              '            raise ValueError("execution sources differ from reviewed commit: " + relative)\n', ''),
    "M06c_source_inventory": ('    if digest.hexdigest() != expected or expected != source_digest():\n', '    if False:\n'),
    "M07_plan_bytes_git": ('            if git_bytes(declared["reviewed_commit"], str(path.relative_to(ROOT))) != path.read_bytes():\n'
                           '                raise ValueError("reviewed plan bytes differ from executable package")\n',
                           '            pass\n'),
    "M08_copy_payload": ('    if binding.get("review_payload") != declared or binding.get("original_review_sha256") != file_sha(review):\n',
                         '    if False:\n'),
    "M08a_payload_only_off": ('    if binding.get("review_payload") != declared or binding.get("original_review_sha256") != file_sha(review):\n',
                              '    if binding.get("original_review_sha256") != file_sha(review):\n'),
    "M08b_origsha_only_off": ('    if binding.get("review_payload") != declared or binding.get("original_review_sha256") != file_sha(review):\n',
                              '    if binding.get("review_payload") != declared:\n'),
    "M09_makejob_no_validation": ('    declared = validate_review_binding(binding)\n', '    declared = binding["review_payload"]\n'),
    "M10_makejob_inputs": ('    if evidence.get("inputs") is not None and {k: v.get("sha256") for k, v in evidence["inputs"].items()} != science["input_sha256"]:\n',
                           '    if False:\n'),
    # form checks
    "M11_pin_inside_checkout": ('    if repository and not path.is_relative_to(ROOT.resolve()):\n', '    if False:\n'),
    "M12_git_bytes_form": ('    if not _full_sha(commit) or not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:\n',
                           '    if False:\n'),
    "M13_extra_keys": ('    if set(payload) != keys or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":\n',
                       '    if not keys <= set(payload) or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":\n'),
    "M14_hash_format": ('        if not isinstance(payload[name], str) or re.fullmatch("[a-f0-9]{64}", payload[name]) is None:\n', '        if False:\n'),
    "M15_review_commit_full": ('    if not _full_sha(binding.get("review_commit")):\n', '    if False:\n'),
    # controls
    "M00_neutral": ('if __name__ == "__main__":\n', '# skeptic-2 neutral comment\nif __name__ == "__main__":\n'),
    # combined line 96 (status + extra keys) for the all-in-one run
    "M01M13_line96": ('    if set(payload) != keys or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":\n',
                      '    if not keys <= set(payload) or payload["schema"] != REVIEW_SCHEMA:\n'),
}

TARGETED = ["tests/cpu/test_contrastive_release.py", "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_run.py",
            "tests/cpu/test_contrastive_targets_readout.py", "tests/cpu/test_contrastive_historical_callsites.py"]
RELEASE_ONLY = ["tests/cpu/test_contrastive_release.py"]


def receipt_files(clone):
    data = json.loads((clone / "results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json").read_text())
    return [row["path"] for row in data["test_files"]]


def run(clone, out, which, combo):
    clone = Path(clone).resolve()
    if subprocess.run(["git", "status", "--porcelain"], cwd=clone, capture_output=True, text=True).stdout.strip():
        raise SystemExit("dirty clone before " + combo)
    path = clone / REL
    text = path.read_text()
    for mid in ([] if combo == "NONE" else combo.split("+")):
        old, new = M[mid]
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{mid}: expected one occurrence, found {count}")
        text = text.replace(old, new)
    path.write_text(text)
    diff = subprocess.run(["git", "diff", "--stat"], cwd=clone, capture_output=True, text=True).stdout.strip()
    probe = str(Path(__file__).resolve().parent / "test_sk2_release_probes.py")
    files = {"targeted": TARGETED, "release": RELEASE_ONLY, "full": receipt_files(clone), "probe": [probe]}[which]
    bt = TMP / f"sk2_r51_bt_{os.getpid()}_{int(time.time()*1000)}"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{clone}/src:{clone}/tests/cpu:{clone}/scripts")
    start = time.time()
    proc = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                           "--tb=line", "-rf", f"--basetemp={bt}", *files], cwd=clone, env=env, capture_output=True, text=True)
    elapsed = time.time() - start
    subprocess.run(["git", "checkout", "-q", "--", REL], cwd=clone, check=True)
    shutil.rmtree(bt, ignore_errors=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=clone, capture_output=True, text=True).stdout.strip()
    lines = proc.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    failed = [line for line in lines if line.startswith("FAILED ")]
    record = {"mutation": combo, "set": which, "nfiles": len(files), "rc": proc.returncode, "summary": summary,
              "failed": failed[:20], "elapsed_s": round(elapsed, 1), "diff": diff, "dirty_after": dirty,
              "stderr_tail": proc.stderr.strip().splitlines()[-3:]}
    with open(out, "a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(combo, which, "->", summary, f"({elapsed:.0f}s)", "FAILED:" + str(len(failed)) if failed else "", flush=True)
    if dirty:
        raise SystemExit("dirty clone after " + combo)


if __name__ == "__main__":
    clone, out, which, *combos = sys.argv[1:]
    for combo in combos:
        run(clone, out, which, combo)
