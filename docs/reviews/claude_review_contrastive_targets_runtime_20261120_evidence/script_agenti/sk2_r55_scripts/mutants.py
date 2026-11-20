"""Skeptic 2 / R5-5: own mutation driver (no reuse of the reviewer's scripts).

usage: mutants.py <clone> <results.jsonl> <id>[,<id>...]
Each id is applied alone (or COMBINED applies all), then: my isolating probes,
then the receipt's 29 test files exactly as listed by local_validation.json.
The tracked file is restored with git checkout after every mutant.
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
READ = "scripts/contrastive_benchmark_readout.py"
MUTANTS = {
    "C4d": (REL, 'or accounting.get("source_sha256") != source or audit.get("source_sha256") != source):',
            'or accounting.get("source_sha256") != source):'),
    "C4g": (REL, "    if not records or not required <= set(records[0]):\n", "    if not records:\n"),
    "C4h": (REL, "    if not isinstance(job_id, str) or not job_id.isdigit() or (expected_job_id is not None and job_id != str(expected_job_id)):\n",
            "    if (expected_job_id is not None and job_id != str(expected_job_id)):\n"),
    "C5c": (READ, "                assert_numeric_flags(flags)\n            except NumericProfileUnavailable as exc:\n",
            "                pass\n            except NumericProfileUnavailable as exc:\n"),
    "C5f": (READ, "    if any(row['flags'] != numeric[0]['flags'] for row in numeric[1:]):\n", "    if False:\n"),
    "C5m": (READ, "        assert_numeric_flags(rank.get('numeric_profile', {}))\n", "        pass\n"),
}


def pytest(clone, files, tag, extra_path=""):
    bt = TMP / ("sk2_r55_bt_" + tag)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{clone}/src:{extra_path}{clone}/scripts"}
    t0 = time.time()
    proc = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                           "-rfE", "--tb=line", "--basetemp=" + str(bt), *files], cwd=clone, env=env, capture_output=True, text=True)
    shutil.rmtree(bt, ignore_errors=True)
    out = proc.stdout + proc.stderr
    summary = [line for line in out.splitlines() if re.search(r"\d+ (passed|failed)", line)]
    failed = {}
    for line in out.splitlines():
        match = re.match(r"^(?:FAILED|ERROR) (\S+)(?: - (.*))?$", line)
        if match:
            failed[match.group(1)] = (match.group(2) or "")[:240]
    return {"rc": proc.returncode, "seconds": round(time.time() - t0, 1),
            "summary": summary[-1] if summary else out[-400:], "failed": failed}


def main():
    clone, results, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
    receipt = json.loads((clone / "results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json").read_text())
    files = [row["path"] for row in receipt["test_files"]]
    with results.open("a") as stream:
        for mid in ids:
            chosen = list(MUTANTS) if mid == "COMBINED" else [] if mid == "NONE" else [mid]
            touched, record = set(), {"id": mid, "applied": chosen}
            for name in chosen:
                rel, old, new = MUTANTS[name]
                path = clone / rel
                text = path.read_text()
                record.setdefault("occurrences", {})[name] = text.count(old)
                if text.count(old) != 1:
                    raise SystemExit(f"{name}: occurrences {text.count(old)}")
                path.write_text(text.replace(old, new))
                touched.add(rel)
            record["diff_stat"] = subprocess.run(["git", "-C", str(clone), "diff", "--numstat"], capture_output=True, text=True).stdout.strip()
            record["probes"] = pytest(clone, ["tests/cpu/test_zz_sk2_r55.py"], mid + "_probe", extra_path=f"{clone}/tests/cpu:")
            record["receipt_set"] = pytest(clone, files, mid + "_receipt")
            if touched:
                subprocess.run(["git", "-C", str(clone), "checkout", "--", *sorted(touched)], check=True)
            record["clean_after"] = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
            stream.write(json.dumps(record) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
