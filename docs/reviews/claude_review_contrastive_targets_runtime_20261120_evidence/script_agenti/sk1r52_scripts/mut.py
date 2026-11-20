"""Independent mutation harness (skeptic 1, R5-2). Usage:
  python mut.py <clone> <set: targeted|full> <out.jsonl> <ID>[+<ID>...] ...
Each argument is one run; IDs joined with '+' are applied together.
Every mutation must match exactly once; files are restored with git checkout.
"""
import json, os, re, subprocess, sys, time, shutil
from pathlib import Path

JOB, RUN, REL, CUDA = "scripts/contrastive_job.py", "scripts/contrastive_run.py", "scripts/contrastive_release.py", "scripts/contrastive_cuda_check.py"
M = {
    # neutral control (changes source digest only)
    "NEUTRAL": (JOB, '"""Execute one reviewed preflight cell; fixed commands, no submission or retries."""\n',
                '"""Execute one reviewed preflight cell; fixed commands, no submission or retries."""\n# neutral\n'),
    # positive controls (expected killed)
    "POS_job_cuda_prereq": (JOB, "        if payload.get('cuda_report') != job.get('cuda_report') or not job.get('cuda_report'):\n", "        if False:\n"),
    "POS_makejob_override": (REL, '    if forbidden & set(evidence):\n', '    if False:\n'),
    # dispatcher
    "J_binding_call": (JOB, "    validate_review_binding(job.get('package_review'))\n", "    pass\n"),
    "J_review_vs_binding": (JOB, "    if job.get('review') != job['package_review'].get('review'):\n", "    if False:\n"),
    "J_cell": (JOB, "    if (len(cells) != 1 or cells[0]['mode'] != mode or cells[0].get('arm') != job.get('arm')\n",
               "    if False and (len(cells) != 1 or cells[0]['mode'] != mode or cells[0].get('arm') != job.get('arm')\n"),
    "J_accounting_call": (JOB, "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))\n", "        pass\n"),
    "J_cohort_root": (JOB, "    if not out.is_relative_to(results) or not out.is_relative_to(cohort_root) or out == cohort_root or out.exists():\n",
                      "    if not out.is_relative_to(results) or out.exists():\n"),
    "J_release_eq_job": (JOB, "    if payload.get('package_review') != job['package_review'] or payload.get('audit_sacct') != job['audit_sacct']:\n", "    if False:\n"),
    "J_resume_fork_pins": (JOB, "        if job.get('repeat') not in ('a', 'b') or payload.get('fork_checkpoint') != job.get('fork_checkpoint') or payload.get('continuous_summary') != job.get('continuous_summary'):\n", "        if False:\n"),
    # runtime
    "R_binding_call": (RUN, '    validate_review_binding(release.get("package_review"))\n', '    pass\n'),
    "R_review_pin": (RUN, '    if release["review_pin"] != release["package_review"]["review_commit"]:\n', '    if False:\n'),
    "R_sacct_presence": (RUN, '    if not release.get("audit_sacct"):\n', '    if False:\n'),
    "R_accounting_call": (RUN, '    validate_audit_accounting(audit_path, release["audit_sacct"])\n', '    pass\n'),
    "R_cuda_prereq_call": (RUN, '        validate_cuda_prerequisite(release.get("cuda_report"), cohort, pins, calibration_run_dir)\n', '        pass\n'),
    "R_phase": (RUN, '        if phase not in ("full", "resume_a", "resume_b") or raw["max_steps"] != 576:\n', '        if False:\n'),
    "R_resume_arm": (RUN, '        if phase != "full" and release.get("arm") not in ("CN", "CR"):\n', '        if False:\n'),
    "R_target_fork_call": (RUN, '        validate_target_fork(release, cfg, cohort)\n', '        pass\n'),
    "R_target_fork_summary": (RUN, '    if (summary.get("steps") != 576 or summary.get("stopped") != "max_steps" or\n            summary.get("world_size") != 4 or contrastive.get("config", {}).get("arm") != release["arm"] or\n',
                              '    if False and (summary.get("steps") != 576 or summary.get("stopped") != "max_steps" or\n            summary.get("world_size") != 4 or contrastive.get("config", {}).get("arm") != release["arm"] or\n'),
    "R_target_fork_prov": (RUN, '    if contrastive.get("provenance") != provenance:\n        raise ValueError("common544 source provenance differs from released cohort")\n',
                           '    if False:\n        raise ValueError("common544 source provenance differs from released cohort")\n'),
    "R_copy_sha": (RUN, '                if sha(a.out_dir / "checkpoint.pt") != release["fork_checkpoint"]["sha256"]:\n', '                if False:\n'),
    "R_profile_ctx": (RUN, '            flags = stack.enter_context(numeric_profile(require_cuda=True))\n', '            flags = {}\n'),
    # release helpers
    "L_makejob_accounting": (REL, '        validate_audit_accounting(audit, job["audit_sacct"])\n', '        pass\n'),
    "L_cuda_ckpt": (REL, '    if cuda.get("inputs", {}).get("calibration_checkpoint") != file_sha(Path(calibration_run_dir) / "checkpoint.pt"):\n', '    if False:\n'),
    "L_cuda_rankexit_pin": (REL, '    if file_sha(rank_exit) != cuda.get("rank_exit_report_sha256"):\n', '    if False:\n'),
    "L_cuda_rankexit_receipt": (REL, '    validate_receipt(rank_exit, expected_source_sha256=cohort["source_sha256"])\n    return cuda', '    pass\n    return cuda'),
    # cuda check
    "K_profile_ctx": (CUDA, '    with numeric_profile(require_cuda=True):\n        return _main(argv)', '    if True:\n        return _main(argv)'),
    "K_rank_profile": (CUDA, '"numeric_profile": assert_numeric_profile(require_cuda=True)})', '"numeric_profile": {}})'),
}

def receipt_files(clone):
    data = json.loads((Path(clone) / "results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json").read_text())
    return [row["path"] for row in data["test_files"]]

TARGETED = ["tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_run.py", "tests/cpu/test_contrastive_release.py",
            "tests/cpu/test_contrastive_targets_guards.py", "tests/cpu/test_contrastive_cuda_check.py",
            "tests/cpu/test_contrastive_historical_callsites.py", "tests/cpu/test_contrastive_numeric_profile.py",
            "tests/cpu/test_contrastive_target_resume.py"]

def run(clone, which, out, ids, base):
    clone = Path(clone)
    touched = set()
    for mid in ids:
        path, old, new = M[mid]
        f = clone / path
        text = f.read_text()
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{mid}: expected exactly one match, found {n}")
        f.write_text(text.replace(old, new))
        touched.add(path)
    diff = subprocess.run(["git", "-C", str(clone), "diff", "--stat"], capture_output=True, text=True).stdout.strip()
    files = receipt_files(clone) if which == "full" else TARGETED
    basetemp = Path(base) / ("pt_" + re.sub(r"[^A-Za-z0-9]+", "_", "+".join(ids))[:80])
    env = dict(os.environ, PYTHONPATH=f"{clone}/src:{clone}/scripts", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    proc = subprocess.run(["/Users/simonebenatti/.venvs/wearusfm-tok/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-o", "tmp_path_retention_policy=none", f"--basetemp={basetemp}", "-rfE", *files],
                          cwd=clone, env=env, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)
    lines = [l for l in proc.stdout.splitlines() if re.search(r"\b(passed|failed|error)", l)]
    summary = lines[-1] if lines else proc.stdout[-300:]
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
    for path in touched:
        subprocess.run(["git", "-C", str(clone), "checkout", "--", path], check=True)
    clean = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    shutil.rmtree(basetemp, ignore_errors=True)
    row = {"ids": ids, "set": which, "n_files": len(files), "rc": proc.returncode, "summary": summary, "failed": failed[:12],
           "elapsed_s": elapsed, "diffstat": diff.splitlines()[-1] if diff else "", "clean_after": clean == ""}
    with open(out, "a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)

if __name__ == "__main__":
    clone, which, out = sys.argv[1:4]
    base = Path(out).parent / "sk1r52_bt"
    base.mkdir(exist_ok=True)
    for arg in sys.argv[4:]:
        run(clone, which, out, arg.split("+"), base)
