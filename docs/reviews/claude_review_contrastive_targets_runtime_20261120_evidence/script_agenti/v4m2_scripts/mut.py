"""Independent mutation driver (skeptic 2 of R3-M4).

argv: out_tsv clone1 [clone2 ...]
Each worker owns one clone at 0efa4d45; mutations are distributed round-robin.
Every mutation runs the FULL 29-file validation list of the package receipt; the
failing tests are then classified as inside/outside the reviewer's 7 files.
"""
import concurrent.futures as cf
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
SEVEN = {"test_contrastive_release.py", "test_contrastive_targets_guards.py", "test_contrastive_job.py",
         "test_contrastive_run.py", "test_contrastive_benchmark_readout.py", "test_contrastive_targets_readout.py",
         "test_contrastive_historical_callsites.py"}
REL, JOB, RUN = "scripts/contrastive_release.py", "scripts/contrastive_job.py", "scripts/contrastive_run.py"

# (id, file, old, new) -- my own phrasing of each guard removal/relaxation.
MUTS = [
    # controls
    ("K00_trivial_comment_release", REL, 'if __name__ == "__main__":\n    main()\n', 'if __name__ == "__main__":\n    main()\n# skeptic no-op\n'),
    ("K04_no_verify_git(MU4)", REL, "    if verify_git:\n", "    if verify_git and False:\n"),
    ("K13_timeout_ok(MU13)", REL, 'row["State"] != "COMPLETED"', 'row["State"] not in {"COMPLETED", "TIMEOUT"}'),
    # surviving set of the finding
    ("A01_first_block_only(MU1)", REL, "    if len(blocks) != 1:\n", "    if len(blocks) == 0:\n"),
    ("A02_no_status(MU2)", REL, ' or payload["status"] != "APPROVED":', ':'),
    ("A03_superset_keys(MU3)", REL, 'if set(payload) != keys or', 'if not keys <= set(payload) or'),
    ("A05_plans_not_vs_commit(MU5)", REL, '            if git_bytes(declared["reviewed_commit"], str(path.relative_to(ROOT))) != path.read_bytes():\n                raise ValueError("reviewed plan bytes differ from executable package")\n', '            pass\n'),
    ("A06a_payload_unchecked(MU6)", REL, 'if binding.get("review_payload") != declared or ', 'if '),
    ("A06b_orig_sha_unchecked(MU6)", REL, ' or binding.get("original_review_sha256") != file_sha(review):', ':'),
    ("A08_other_cohort(MU8)", REL, "    if science != PLAN.resolve() or preflight != PREFLIGHT.resolve():\n", "    if science != science or preflight != preflight:\n"),
    ("A09_no_preflight_link(MU9)", REL, '    if (plan.get("source_sha256") != declared["source_sha256"] or\n', '    if False and (plan.get("source_sha256") != declared["source_sha256"] or\n'),
    ("A19_audit_report_source(MU19)", REL, ' or audit.get("source_sha256") != source)', ')'),
    ("A21_make_job_no_gate(MU21)", REL, '        validate_audit_accounting(audit, job["audit_sacct"])\n', '        pass\n'),
    ("A32_review_outside_checkout(MU32)", REL, 'review = checked_pin(binding["review"], repository=True)', 'review = checked_pin(binding["review"], repository=False)'),
    ("A33_no_sacct_header(MU33)", REL, ' or not required <= set(records[0]):', ':'),
    ("A35_no_digest_compare(MU35)", REL, '    if digest.hexdigest() != expected or expected != source_digest():\n', '    if False:\n'),
    ("A36_git_bytes_uncontained(MU36)", REL, ' or Path(relative).is_absolute() or ".." in Path(relative).parts:', ':'),
    ("A37_job_id_any_string(MU37)", REL, 'not isinstance(job_id, str) or not job_id.isdigit() or ', ''),
    ("A39_no_byte_compare(MU39)", REL, '        if original != path.read_bytes():\n            raise ValueError("execution sources differ from reviewed commit: " + relative)\n', ''),
    ("B22_dispatch_no_binding(MU22)", JOB, "    validate_review_binding(job.get('package_review'))\n", ""),
    ("B23_dispatch_review_unlinked(MU23)", JOB, "    if job.get('review') != job['package_review'].get('review'):\n        raise ValueError('job review differs from its bound package review')\n", ""),
    ("B24_dispatch_no_cell(MU24)", JOB, "        raise ValueError('job differs from the reviewed fixed preflight cell')\n", "        pass\n"),
    ("B25_dispatch_no_gate(MU25)", JOB, "    if mode != 'audit':\n        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))\n", ""),
    ("B26_dispatch_out_any(MU26)", JOB, " or not out.is_relative_to(cohort_root) or out == cohort_root", ""),
    ("B27_dispatch_release_unlinked(MU27)", JOB, "        raise ValueError('job review/audit accounting differs from execution release')\n", "        pass\n"),
    ("C28_runtime_no_binding(MU28)", RUN, '    validate_review_binding(release.get("package_review"))\n', ''),
    ("C29_runtime_review_pin(MU29)", RUN, '        raise ValueError("release review commit differs from its bound package review")\n', '        pass\n'),
    ("C30_runtime_no_gate(MU30)", RUN, '    validate_audit_accounting(audit_path, release["audit_sacct"])\n', ''),
    ("C34_runtime_no_cuda(MU34)", RUN, '        validate_cuda_prerequisite(release.get("cuda_report"), cohort, pins, calibration_run_dir)\n', ''),
]


def test_files(root):
    receipt = json.loads((root / "results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json").read_text())
    return [item["path"] for item in receipt["test_files"]]


def run_one(root, mid, rel, old, new):
    path = root / rel
    text = path.read_text()
    if text.count(old) != 1:
        return mid, "NOT_APPLIED count=%d" % text.count(old), "", 0.0
    path.write_text(text.replace(old, new))
    base = TMP / ("v4m2_pt_" + re.sub(r"[^A-Za-z0-9_]", "_", mid))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{root}/src:{root}/tests/cpu:{root}/scripts")
    start = time.monotonic()
    try:
        proc = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf", "--basetemp=" + str(base),
                               *test_files(root)], cwd=root, env=env, capture_output=True, text=True, timeout=1500)
    finally:
        subprocess.run(["git", "checkout", "-q", "--", rel], cwd=root, check=True)
        shutil.rmtree(base, ignore_errors=True)
    elapsed = time.monotonic() - start
    lines = proc.stdout.strip().splitlines()
    summary = lines[-1] if lines else proc.stderr[-300:]
    failed = re.findall(r"^FAILED (\S+)", proc.stdout, re.M)
    errors = re.findall(r"^ERROR (\S+)", proc.stdout, re.M)
    inside = [f for f in failed + errors if f.split("::")[0].split("/")[-1] in SEVEN]
    outside = [f for f in failed + errors if f.split("::")[0].split("/")[-1] not in SEVEN]
    verdict = "SURVIVED" if proc.returncode == 0 else ("KILLED_IN7" if inside else "KILLED_OUTSIDE7_ONLY")
    detail = "in7=" + ",".join(x.split("::")[-1] for x in inside)[:250] + " | out7=" + ",".join(x.split("/")[-1] for x in outside)[:250]
    return mid, verdict + " rc=%d" % proc.returncode, summary + " | " + detail, elapsed


def worker(root, items):
    rows = []
    for mid, rel, old, new in items:
        rows.append(run_one(root, mid, rel, old, new))
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True).stdout
        if status.strip():
            rows.append(("DIRTY_AFTER_" + mid, status.strip()[:200], "", 0.0))
            break
    return rows


def main():
    out = Path(sys.argv[1])
    roots = [Path(p).resolve() for p in sys.argv[2:]]
    only = os.environ.get("ONLY")
    muts = [m for m in MUTS if not only or m[0].split("_")[0] in only.split(",")]
    queues = [muts[i::len(roots)] for i in range(len(roots))]
    with cf.ThreadPoolExecutor(len(roots)) as pool:
        futures = [pool.submit(worker, root, queue) for root, queue in zip(roots, queues)]
        results = [row for future in futures for row in future.result()]
    order = {m[0]: i for i, m in enumerate(MUTS)}
    results.sort(key=lambda r: order.get(r[0], 999))
    with out.open("w") as stream:
        for mid, verdict, detail, elapsed in results:
            stream.write(f"{mid}\t{verdict}\t{elapsed:.0f}s\t{detail}\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
