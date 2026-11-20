"""R5-9 skeptic 1: independent reproduction in a disposable clone (no shared state).

Usage: python -B repro_r59.py <clone_dir> <out_dir>
Builds a real review commit inside the disposable clone, runs the real CLI
bind-review -> check-binding -> make-job on several evidence variants.
"""
import json
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

W = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "PYTHONPATH": f"{W}/src:{W}/scripts"}


def py(*args):
    p = subprocess.run([PY, "-B", *args], cwd=W, env=ENV, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def git(*args):
    return subprocess.check_output(["git", "-c", "user.name=sk1-r59", "-c", "user.email=sk1-r59@example.invalid", *args],
                                   cwd=W, text=True).strip()


head = git("rev-parse", "HEAD")
rc, so, se = py("-c", "import json, contrastive_release as R; print(json.dumps({'source': R.source_digest(), "
                "'science': R.file_sha(R.PLAN), 'preflight': R.file_sha(R.PREFLIGHT), 'root': str(R.ROOT)}))")
assert rc == 0, se
dig = json.loads(so)
print("digests", dig, "head", head)
payload = {"schema": "contrastive-target-release-review-v1", "status": "APPROVED", "reviewed_commit": head,
           "source_sha256": dig["source"], "science_plan_sha256": dig["science"], "preflight_plan_sha256": dig["preflight"]}
rel = "docs/reviews/sk1_r59_review.md"
(W / rel).write_text("# Review di prova sk1-r59 (clone usa e getta)\n\n```contrastive-release-review\n"
                     + json.dumps(payload, indent=2) + "\n```\n")
git("add", rel)
git("commit", "-qm", "sk1-r59 disposable review")
review_commit = git("rev-parse", "HEAD")
copy = W / "results/infra/contrastive_targets_20261118/release/sk1_r59_copy.md"
copy.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(W / rel, copy)
binding = OUT / "binding.json"
rc, so, se = py("scripts/contrastive_release.py", "bind-review", "--review-commit", review_commit, "--review-path", rel,
                "--review-copy", str(copy.relative_to(W)), "--out", str(binding))
print("bind-review rc", rc, (se.strip().splitlines() or [""])[-1])
assert rc == 0
rc, so, se = py("scripts/contrastive_release.py", "check-binding", "--binding", str(binding))
print("check-binding rc", rc, so.strip()[:120])
assert rc == 0

science = json.loads((W / "results/infra/contrastive_targets_20261118/plan.json").read_text())
inputs = {k: {"path": "/leonardo_work/IscrB_WearUsFM/sk1/" + k, "sha256": v} for k, v in science["input_sha256"].items()}
audit_file = OUT / "audit_fixture.json"
audit_file.write_text('{"fixture": true}\n')
rc, so, se = py("-c", "import json,sys, contrastive_release as R; print(json.dumps(R.pin(sys.argv[1])))", str(audit_file))
audit_pin = json.loads(so)

cases = {
    "calib_empty": ("CONTRAST-CALIB-targets-v3-20261118", {}),
    "calib_inputs_only": ("CONTRAST-CALIB-targets-v3-20261118", {"inputs": inputs}),
    "cuda_empty": ("CONTRAST-CUDA-targets-v3-20261118", {}),
    "readout_empty": ("CONTRAST-READOUT-targets-v3-20261118", {}),
    "calib_audit_no_sacct": ("CONTRAST-CALIB-targets-v3-20261118", {"inputs": inputs, "audit": audit_pin}),
    "calib_audit_null": ("CONTRAST-CALIB-targets-v3-20261118", {"inputs": inputs, "audit": None}),
    "calib_audit_sacct_null": ("CONTRAST-CALIB-targets-v3-20261118", {"inputs": inputs, "audit": audit_pin, "audit_sacct": None}),
    "audit_control": ("CONTRAST-AUDIT-targets-v3-20261118", {"inputs": inputs}),
}
for name, (cell, ev) in cases.items():
    evp = OUT / f"evidence_{name}.json"
    evp.write_text(json.dumps(ev))
    job = OUT / f"job_{name}.json"
    rc, so, se = py("scripts/contrastive_release.py", "make-job", "--binding", str(binding), "--cell-id", cell,
                    "--evidence", str(evp), "--result-out",
                    f"/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/sk1-{name}",
                    "--out", str(job))
    lines = se.strip().splitlines()
    frame = [l.strip() for l in lines if "contrastive_release.py" in l]
    print(f"{name}: rc={rc} out_exists={job.exists()} last={lines[-1] if lines else ''!r} frames={frame[-2:]}")
