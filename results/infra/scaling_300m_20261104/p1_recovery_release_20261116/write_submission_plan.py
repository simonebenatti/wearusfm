"""Write the two approved commands from the published package; never submit."""
from pathlib import Path
import hashlib
import json
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
REL = BASE.relative_to(ROOT)
CODE_DIR = '/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_p1_recovery_20261116_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(code):
    assert re.fullmatch('[0-9a-f]{40}', code)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == code
    tests = json.loads((BASE / 'cpu_tests.json').read_text())
    assert tests['passed'] is True and tests['counts']['passed'] == 163
    for path, expected in tests['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
    relative_manifest = str(REL / 'recovery_manifest.json')
    manifest_sha = sha(BASE / 'recovery_manifest.json')
    common = ['python3', str(ROOT / 'scripts/submit_job.py'), '--agente', 'gpt',
              '--campagna', 'scaling-300m-20261104', '--ssh',
              'ssh leonardo -o ProxyCommand=none -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15']
    checkout_id = 'SCALING-P1-recovery-checkout-20261116'
    p1_id = 'SCALING-300M-eval19618-p1-r2-20261116'
    checkout = common + ['--id', checkout_id, '--gpu-ore', '0', '--script-stdin',
        'scripts/slurm/campaign_checkout.sbatch', '--', '--partition=lrd_all_serial',
        '--cpus-per-task=2', '--mem=4G', '--time=00:05:00',
        '--export=ALL,CAMPAIGN=scaling_p1_recovery_20261116_v1,CAMPAIGN_SHA=' + code, '/dev/stdin']
    p1 = common + ['--id', p1_id, '--gpu-ore', '2.05', '--dopo', checkout_id,
        '--script-stdin', 'scripts/slurm/scaling_p1_recovery.sbatch', '--',
        '--export=ALL,CODE_DIR=' + CODE_DIR + ',EXPECTED_SHA=' + code,
        '/dev/stdin', relative_manifest, manifest_sha]
    plan = dict(protocol='scaling-p1-recovery-submission-20261116-v2', code_sha=code,
        approved=True, approval_sha256=sha(BASE/'criterion_amendment.json'),
        code_dir=CODE_DIR, deploy_ref='refs/heads/deploy/scaling_p1_recovery_20261116_v1',
        manifest_sha256=manifest_sha, budget_sha256=sha(BASE/'budget.json'),
        cwd=str(ROOT), checkout=dict(id=checkout_id, argv=checkout,
            stdin_sha256=sha(ROOT/'scripts/slurm/campaign_checkout.sbatch')),
        p1=dict(id=p1_id, argv=p1, stdin_sha256=sha(ROOT/'scripts/slurm/scaling_p1_recovery.sbatch')),
        stop_on_non_submitted=True, no_retry=True, max_new_p1_jobs=1)
    for name, data in [('submission_plan.json', json.dumps(plan, indent=2)+'\n'),
                       ('submit_lines.txt', shlex.join(checkout)+'\n'+shlex.join(p1)+'\n')]:
        with (BASE/name).open('x') as out:
            out.write(data)
    print(json.dumps({'code_sha':code,'manifest_sha256':manifest_sha,
                      'plan_sha256':sha(BASE/'submission_plan.json')},indent=2))

if __name__ == '__main__':
    main(sys.argv[1])
