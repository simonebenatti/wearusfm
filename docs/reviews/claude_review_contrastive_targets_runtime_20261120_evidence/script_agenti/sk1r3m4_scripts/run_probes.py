"""Apply each mutant in the probe clone and run the skeptic's behavioural probes. argv: clone, out.tsv, ids|all."""
import os, re, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutants import MUTANTS
ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2])
WANTED = None if sys.argv[3] == "all" else set(sys.argv[3].split(","))
TMP = Path("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp")
PROBES = Path(__file__).resolve().parent / "probes" / "test_sk1_probes.py"
PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
def clean():
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
assert clean() == ""
with OUT.open("a") as out:
    for ident, label, rel, old, new in MUTANTS:
        if WANTED is not None and ident not in WANTED: continue
        path = ROOT / rel; text = path.read_text()
        assert text.count(old) == 1, ident
        path.write_text(text.replace(old, new))
        base = TMP / ("sk1r3m4_pt_probe_" + ident)
        env = dict(os.environ, SK1_CLONE=str(ROOT), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{ROOT}/src:{ROOT}/tests/cpu:{ROOT}/scripts")
        try:
            proc = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                                   "--basetemp=" + str(base), str(PROBES)], cwd=ROOT, env=env, capture_output=True, text=True)
        finally:
            subprocess.run(["git", "checkout", "-q", "--", rel], cwd=ROOT, check=True)
            shutil.rmtree(base, ignore_errors=True)
        assert clean() == "", ident
        lines = proc.stdout.strip().splitlines(); tail = lines[-1] if lines else proc.stderr[-200:]
        failed = [f.split("::", 1)[-1] for f in re.findall(r"^(?:FAILED|ERROR) (\S+)", proc.stdout, re.M)]
        out.write(f"{ident}\t{label}\t{'OBSERVABLE' if proc.returncode else 'NOT_OBSERVED'}\t{tail}\t{' '.join(failed)}\n"); out.flush()
    out.write("final status: " + (clean() or "clean") + "\n")
