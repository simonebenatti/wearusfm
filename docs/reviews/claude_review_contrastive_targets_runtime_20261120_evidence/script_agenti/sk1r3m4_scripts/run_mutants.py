"""Apply each mutant in a throwaway clone, run a test selection, restore. Skeptic 1, R3-M4.

argv: clone, out.tsv, selection ("seven" | "full"), comma list of mutant ids or "all".
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutants import MUTANTS

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
SELECTION = sys.argv[3]
WANTED = None if sys.argv[4] == "all" else set(sys.argv[4].split(","))
TMP = Path("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp")
PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
SEVEN = ["test_contrastive_release.py", "test_contrastive_targets_guards.py", "test_contrastive_job.py",
         "test_contrastive_run.py", "test_contrastive_benchmark_readout.py", "test_contrastive_targets_readout.py",
         "test_contrastive_historical_callsites.py"]
# Exactly the 29 files of validation_final/local_validation.json
FULL = ["test_contrastive_audit.py", "test_contrastive_benchmark_readout.py", "test_contrastive_callsite_oracle.py",
        "test_contrastive_common_checkpoint.py", "test_contrastive_common_state_cuda.py", "test_contrastive_compare.py",
        "test_contrastive_core.py", "test_contrastive_cuda_check.py", "test_contrastive_diagnostics.py",
        "test_contrastive_historical_calibration.py", "test_contrastive_historical_callsites.py", "test_contrastive_job.py",
        "test_contrastive_memory.py", "test_contrastive_mmap_compatibility.py", "test_contrastive_numeric_profile.py",
        "test_contrastive_prepare.py", "test_contrastive_rank_exit.py", "test_contrastive_release.py",
        "test_contrastive_run.py", "test_contrastive_runtime.py", "test_contrastive_source_compat.py",
        "test_contrastive_target_parity.py", "test_contrastive_target_resume.py", "test_contrastive_targets_guards.py",
        "test_contrastive_targets_plan.py", "test_contrastive_targets_readout.py", "test_ms_fast_target_window.py",
        "test_rms_target_window.py", "test_target_windows_fix.py"]
TESTS = ["tests/cpu/" + name for name in (SEVEN if SELECTION == "seven" else FULL)]


def run_tests(tag):
    base = TMP / ("sk1r3m4_pt_" + SELECTION + "_" + tag)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{ROOT}/src:{ROOT}/tests/cpu:{ROOT}/scripts")
    start = time.monotonic()
    proc = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                           "--basetemp=" + str(base), *TESTS], cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    shutil.rmtree(base, ignore_errors=True)
    lines = proc.stdout.strip().splitlines()
    tail = lines[-1] if lines else proc.stderr[-300:].replace("\n", " ")
    failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", proc.stdout, re.M)
    return proc.returncode, tail, failed, elapsed


def clean():
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


assert clean() == "", "clone not clean before start"
with OUT.open("a") as out:
    if WANTED is None or "BASE" in WANTED:
        code, tail, failed, elapsed = run_tests("BASE")
        out.write(f"BASE\tbaseline\t{'PASS' if code == 0 else 'FAIL'}\t{tail}\t{elapsed:.0f}s\t{' '.join(failed)[:300]}\n")
        out.flush()
    for ident, label, rel, old, new in MUTANTS:
        if WANTED is not None and ident not in WANTED:
            continue
        path = ROOT / rel
        text = path.read_text()
        count = text.count(old)
        if count != 1:
            out.write(f"{ident}\t{label}\tNOT_APPLIED(count={count})\n"); out.flush()
            continue
        mutated = text.replace(old, new)
        try:
            ast.parse(mutated)
        except SyntaxError as error:
            out.write(f"{ident}\t{label}\tSYNTAX_ERROR {error}\n"); out.flush()
            continue
        path.write_text(mutated)
        try:
            code, tail, failed, elapsed = run_tests(ident)
        finally:
            subprocess.run(["git", "checkout", "-q", "--", rel], cwd=ROOT, check=True)
        assert clean() == "", "clone not clean after " + ident
        verdict = "KILLED" if code != 0 else "SURVIVED"
        names = " ".join(f.split("::", 1)[-1] for f in failed)
        out.write(f"{ident}\t{label}\t{verdict}\t{tail}\t{elapsed:.0f}s\t{names[:400]}\n")
        out.flush()
    out.write("final status: " + (clean() or "clean") + "\n")
