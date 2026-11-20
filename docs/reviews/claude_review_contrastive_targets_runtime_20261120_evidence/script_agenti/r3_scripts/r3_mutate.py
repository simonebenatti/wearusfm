"""Targeted mutations of the release guards; each mutation runs the release-related test files.

argv[1]: temporary mutation clone at 0efa4d45; argv[2]: output TSV.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
TMP = Path(os.environ["R3TMP"])
PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
TESTS = ["tests/cpu/test_contrastive_release.py", "tests/cpu/test_contrastive_targets_guards.py",
         "tests/cpu/test_contrastive_job.py", "tests/cpu/test_contrastive_run.py",
         "tests/cpu/test_contrastive_benchmark_readout.py", "tests/cpu/test_contrastive_targets_readout.py",
         "tests/cpu/test_contrastive_historical_callsites.py"]
REL, JOB, RUN, RD = "scripts/contrastive_release.py", "scripts/contrastive_job.py", "scripts/contrastive_run.py", "scripts/contrastive_benchmark_readout.py"
M = [
    ("MU1 accetta piu' blocchi (usa il primo)", REL, "if len(blocks) != 1:", "if len(blocks) < 1:"),
    ("MU2 nessun controllo di status APPROVED", REL, 'or payload["schema"] != REVIEW_SCHEMA or payload["status"] != "APPROVED":', 'or payload["schema"] != REVIEW_SCHEMA:'),
    ("MU3 chiavi del blocco non esatte", REL, 'if set(payload) != keys or payload["schema"]', 'if payload["schema"]'),
    ("MU4 nessuna verifica Git del commit rivisto", REL, "    if verify_git:\n", "    if False:\n"),
    ("MU5 piani non confrontati al commit rivisto", REL, 'if git_bytes(declared["reviewed_commit"], str(path.relative_to(ROOT))) != path.read_bytes():', "if False:"),
    ("MU6 payload/sha originale del binding non controllati", REL, 'if binding.get("review_payload") != declared or binding.get("original_review_sha256") != file_sha(review):', "if False:"),
    ("MU7 bind senza identita' byte della copia", REL, "if original != local_copy.read_bytes():", "if False:"),
    ("MU8 binding verso un'altra coorte ammesso", REL, "if science != PLAN.resolve() or preflight != PREFLIGHT.resolve():", "if False:"),
    ("MU9 legame interno preflight->source/plan rimosso", REL, 'if (plan.get("source_sha256") != declared["source_sha256"] or\n            plan.get("science_plan", {}).get("sha256") != declared["science_plan_sha256"]):', "if False:"),
    ("MU10 hash dichiarati non confrontati con source/piani correnti", REL, 'if (declared["source_sha256"] != source_digest() or declared["science_plan_sha256"] != file_sha(science)\n            or declared["preflight_plan_sha256"] != file_sha(preflight)):', "if False:"),
    ("MU11 durata fino a 7380 s (OverTimeLimit)", REL, "not 0 <= elapsed <= 7200", "not 0 <= elapsed <= 7380"),
    ("MU12 durata < 7200 (limite escluso)", REL, "not 0 <= elapsed <= 7200", "not 0 <= elapsed < 7200"),
    ("MU13 eccezione TIMEOUT", REL, 'row["State"] != "COMPLETED"', 'row["State"] not in ("COMPLETED", "TIMEOUT")'),
    ("MU14 nessun controllo AllocCPUS", REL, "or cpus != 8 or minutes != 120", "or minutes != 120"),
    ("MU15 nessun controllo TimelimitRaw", REL, "or cpus != 8 or minutes != 120", "or cpus != 8"),
    ("MU16 ExitCode non controllato", REL, ' or row["ExitCode"] != "0:0"', ""),
    ("MU17 righe di allocazione duplicate ammesse", REL, "if len(allocation) != 1:", "if not allocation:"),
    ("MU18 ID logico non controllato", REL, 'or accounting.get("logical_id") != expected_logical[0]:', ":"),
    ("MU19 source del report audit non controllata", REL, ' or audit.get("source_sha256") != source)', ")"),
    ("MU20 audit_sha256 dell'accounting non controllato", REL, ' or accounting.get("audit_sha256") != file_sha(audit_path)', ""),
    ("MU21 make_job senza gate accounting", REL, '    if cell["mode"] != "audit":\n        audit = checked_pin(job["audit"])', '    if False:\n        audit = checked_pin(job["audit"])'),
    ("MU22 dispatcher senza validate_review_binding", JOB, "    validate_review_binding(job.get('package_review'))\n", "    pass\n"),
    ("MU23 dispatcher: review del job non legata al binding", JOB, "    if job.get('review') != job['package_review'].get('review'):", "    if False:"),
    ("MU24 dispatcher senza controllo della cella", JOB, "    if (len(cells) != 1 or cells[0]['mode'] != mode", "    if False and (len(cells) != 1 or cells[0]['mode'] != mode"),
    ("MU25 dispatcher senza gate accounting", JOB, "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))", "        pass"),
    ("MU26 dispatcher: out fuori da fresh_root", JOB, "not out.is_relative_to(cohort_root) or out == cohort_root or ", ""),
    ("MU27 dispatcher: release senza stesso binding/accounting", JOB, "    if payload.get('package_review') != job['package_review'] or payload.get('audit_sacct') != job['audit_sacct']:", "    if False:"),
    ("MU28 runtime senza validate_review_binding", RUN, '    validate_review_binding(release.get("package_review"))\n', "    pass\n"),
    ("MU29 runtime: review_pin non legato al binding", RUN, '    if release["review_pin"] != release["package_review"]["review_commit"]:', "    if False:"),
    ("MU30 runtime senza gate accounting", RUN, '    validate_audit_accounting(audit_path, release["audit_sacct"])\n', "    pass\n"),
    ("MU31 lettura finale senza gate accounting", RD, "    accounting = validate_audit_accounting(cpu_audit_path, cpu_audit_sacct_path, expected_source=cohort['source_sha256'])", "    accounting = {}"),
]


def run_tests(tag):
    base = TMP / ("r3m_pt_" + tag)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{ROOT}/src:{ROOT}/tests/cpu:{ROOT}/scripts")
    proc = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp=" + str(base), *TESTS],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    shutil.rmtree(base, ignore_errors=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
    failed = re.findall(r"^FAILED (\S+)", proc.stdout, re.M)
    return proc.returncode, tail, failed


with OUT.open("w") as out:
    for label, rel, old, new in M:
        path = ROOT / rel
        text = path.read_text()
        if text.count(old) != 1:
            out.write(f"{label}\tNOT_APPLIED (count={text.count(old)})\n"); out.flush()
            continue
        path.write_text(text.replace(old, new))
        try:
            code, tail, failed = run_tests(label.split()[0])
        finally:
            subprocess.run(["git", "checkout", "-q", "--", rel], cwd=ROOT, check=True)
        verdict = "KILLED" if code != 0 else "SURVIVED"
        out.write(f"{label}\t{verdict}\t{tail}\t{' '.join(f.split('::')[-1] for f in failed)[:300]}\n")
        out.flush()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True).stdout
    out.write("final status: " + (status.strip() or "clean") + "\n")
