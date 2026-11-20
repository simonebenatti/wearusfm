"""Integration (disposable clone): real bind-review -> check-binding -> make-job -> contrastive_job.build_command."""
import json, os, shutil, subprocess, sys
from pathlib import Path
W = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
PY = '/Users/simonebenatti/.venvs/wearusfm-tok/bin/python'
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'PYTHONPATH': f'{W}/src:{W}/scripts'}
def run(*args, **kw):
    p = subprocess.run([PY, '-B', *args], cwd=W, env=env, capture_output=True, text=True, **kw)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1:] if (p.stdout + p.stderr).strip() else []
def git(*args):
    return subprocess.check_output(['git', '-c', 'user.name=R5', '-c', 'user.email=r5@example.invalid', *args], cwd=W, text=True).strip()
payload = {"schema": "contrastive-target-release-review-v1", "status": "APPROVED",
           "reviewed_commit": "fbfde797b48644744c36547533acf0e8a7793535",
           "source_sha256": "40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0",
           "science_plan_sha256": "b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a",
           "preflight_plan_sha256": "e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92"}
results = {}
def review_text(p):
    return ("# Review di prova R5 (clone usa e getta)\n\n```bash\necho esempio\n```\n\n```contrastive-release-review\n"
            + json.dumps(p, indent=2) + "\n```\n\nFine.\n")
cases = {'content_commit': payload,
         'pin_commit': {**payload, 'reviewed_commit': git('rev-parse', 'HEAD')},
         'base_commit_wrong': {**payload, 'reviewed_commit': git('rev-parse', '438c8d65')},
         'status_review_required': {**payload, 'status': 'REVIEW_REQUIRED'}}
for name, p in cases.items():
    rel = f'docs/reviews/r5_{name}.md'
    (W / rel).write_text(review_text(p))
    git('add', rel); git('commit', '-qm', 'R5 disposable review ' + name)
    commit = git('rev-parse', 'HEAD')
    copy = W / f'results/infra/contrastive_targets_20261118/release/r5_{name}_copy.md'
    copy.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(W / rel, copy)
    binding = OUT / f'binding_{name}.json'
    rc, tail = run('scripts/contrastive_release.py', 'bind-review', '--review-commit', commit, '--review-path', rel,
                   '--review-copy', str(copy.relative_to(W)), '--out', str(binding))
    results[name] = {'bind_rc': rc, 'bind_tail': tail}
    if rc == 0:
        rc2, tail2 = run('scripts/contrastive_release.py', 'check-binding', '--binding', str(binding))
        results[name].update(check_rc=rc2, check_tail=[t[:160] for t in tail2])
# make-job from the valid binding
ev = OUT / 'evidence_audit.json'
ev.write_text(json.dumps({"inputs": {k: {"path": "/leonardo_work/IscrB_WearUsFM/x/" + k, "sha256": v} for k, v in json.loads((W/'results/infra/contrastive_targets_20261118/plan.json').read_text())['input_sha256'].items()}}))
job_path = OUT / 'job_audit.json'
rc, tail = run('scripts/contrastive_release.py', 'make-job', '--binding', str(OUT / 'binding_content_commit.json'), '--cell-id',
               'CONTRAST-AUDIT-targets-v3-20261118', '--evidence', str(ev), '--result-out',
               '/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/audit-r5', '--out', str(job_path))
results['make_job_audit'] = {'rc': rc, 'tail': tail}
# feed real make-job output to the real dispatcher
code = r'''
import json, os, sys
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
import contrastive_job as J
job = json.load(open(sys.argv[1]))
try:
    J.build_command(job, "cpu", 7200); print("BUILD_OK")
except Exception as e:
    print("BUILD_STOPPED", type(e).__name__, str(e)[:200])
'''
rc, tail = run('-c', code, str(job_path))
results['build_command_audit'] = {'rc': rc, 'tail': tail}
# make-job for calibration without audit evidence
rc, tail = run('scripts/contrastive_release.py', 'make-job', '--binding', str(OUT / 'binding_content_commit.json'), '--cell-id',
               'CONTRAST-CALIB-targets-v3-20261118', '--evidence', str(ev), '--result-out',
               '/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/calib-r5', '--out', str(OUT / 'job_cal.json'))
results['make_job_calibration_without_audit'] = {'rc': rc, 'tail': tail}
print(json.dumps(results, indent=1))
