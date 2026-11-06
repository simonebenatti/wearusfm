"""Independent local reproduction: no SSH, deploy or submission. Uses release's fixture repository."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path('/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests/cpu'))
spec = importlib.util.spec_from_file_location('gates_fixture', ROOT / 'tests/cpu/test_scaling_gates19618.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
S = G.S
out = {'reviewed_commit': 'f6dc8ec0fb33ae1bcf6d34018194c2e9c71f7cb5', 'cases': []}
for waiting in (0, 9, 10):
    temp = Path(tempfile.mkdtemp(prefix=f'eval19618-review-{waiting}-'))
    repo, ledger, reviewed, code, documents = G.chain.__wrapped__(temp)
    G.ledger_entry(ledger, S.checkout_job('plans')['id'], 701, ['sbatch', G.export('plans', documents)])
    request, commits, published, table, submitted = S.gate_request('pose-benchmark', repo, ledger)
    queue = ''.join(f'{9000+n}|other-line|PENDING|8|Priority\n' for n in range(waiting))
    reading = G.live(commits, published, table, commands__squeue={'stdout': queue})
    S.remote_read = lambda **kw: reading
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exit_code = S.run_gate('pose-benchmark', root=repo, ledger=ledger)
    lines = S.stage_lines('pose-benchmark', code, root=repo)
    record = json.loads(next((repo / S.PACKAGE / 'evidence/gates').glob('*.json')).read_text())
    out['cases'].append({'queue_jobs': waiting, 'gate_exit': exit_code,
                         'all_checks_passed': record['passed'],
                         'submittable_now': record['facts']['submittable_now'],
                         'default_lines_emitted': len(lines), 'gate_stdout': stream.getvalue(),
                         'first_submission': lines[0],
                         'checkout_jobs_requested_from_sacct': list(submitted.values())})
resources_path = repo / S.PACKAGE / S.DOCUMENTS['probe_resources'][1]
resources_path.write_bytes(S.dumps(G.resources_fixture()))
out['aggregate_command'] = {'queue_jobs': 0, 'default_lines_emitted': len(S.stage_lines('probe-aggregate', code, root=repo)),
                            'published_line1_simultaneous_limit': 4,
                            'explanation': 'evaluation_chain_plan section 9 emits all six aggregates; section 6 says at most four submitted line1 jobs.'}
Path('/private/tmp/eval19618-gates-review-repro.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
