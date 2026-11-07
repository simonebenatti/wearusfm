"""Read-only release review with local fixture repos/ledgers and simulated remote observations.
Exercises the actual main(['gate',...]) -> main(['lines',...]) path. No SSH, no jobs, no source changes.
"""
import calendar
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path('/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests/cpu')]
spec = importlib.util.spec_from_file_location('closure_fixture', ROOT / 'tests/cpu/test_scaling_gates19618.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
S = G.S
results = {'commit': '40797b1ccb640b8b986f30539f05a73ac0a8f485', 'cases': []}

def setup():
    S.ROOT = ROOT
    temp = Path(tempfile.mkdtemp(prefix='eval19618-b1-closure-'))
    repo, ledger, reviewed, code, documents = G.chain.__wrapped__(temp)
    G.ledger_entry(ledger, S.checkout_job('plans')['id'], 701, ['sbatch', G.export('plans', documents)])
    G.collected(repo, 'plans', 701)
    S.ROOT, S.LEDGER = repo, ledger
    return repo, ledger, code

def call(argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            status = S.main(argv)
            exception = None
        except Exception as e:
            status, exception = None, type(e).__name__ + ': ' + str(e)
    return {'status': status, 'exception': exception, 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue()}

def queued(mine, others):
    return ''.join(f'{1000+n}|SCALING-x-{n}|PENDING|8|Priority\n' for n in range(mine)) + ''.join(
        f'{2000+n}|LOGSQ-x-{n}|PENDING|8|Priority\n' for n in range(others))

def at(h, m, s=0):
    return calendar.timegm((2026, 11, 8, h, m, s))

for mine, others, limit in [(0, 0, None), (0, 9, None), (0, 10, None), (4, 0, None), (2, 7, None), (0, 0, 6)]:
    repo, ledger, code = setup()
    _, commits, published, table, submitted = S.gate_request('probe-benchmark', repo, ledger)
    reading = G.live(commits, published, table, commands__squeue={'stdout': queued(mine, others)})
    S.remote_read = lambda **kw: reading
    S._now = lambda: at(10, 0, 30)
    gate = call(['gate', 'probe-benchmark'] + ([] if limit is None else ['--line-limit', str(limit)]))
    lines = call(['lines', 'probe-benchmark'])
    results['cases'].append({'name': 'tranche', 'mine': mine, 'others': others, 'line_limit': limit,
                             'gate_status': gate['status'], 'lines_status': lines['status'],
                             'command_count': len(lines['stdout'].splitlines()), 'stderr': lines['stderr']})

for mutation in ('missing_gate', 'old_gate', 'max_expands', 'max_shortens', 'ledger_changed', 'code_changed', 'later_failed_gate'):
    repo, ledger, code = setup()
    _, commits, published, table, submitted = S.gate_request('pose-benchmark', repo, ledger)
    reading = G.live(commits, published, table, commands__squeue={'stdout': queued(0, 0)})
    S._now = lambda: at(10, 0, 30)
    S.remote_read = lambda **kw: reading
    if mutation != 'missing_gate':
        assert call(['gate', 'pose-benchmark'])['status'] == 0
    args = ['lines', 'pose-benchmark']
    if mutation == 'old_gate':
        S._now = lambda: at(10, 30, 1)
    elif mutation == 'max_expands':
        args += ['--max', '4']
    elif mutation == 'max_shortens':
        args += ['--max', '2']
    elif mutation == 'ledger_changed':
        G.ledger_entry(ledger, table[0]['id'], 900)
    elif mutation == 'code_changed':
        statepath = ledger / S.checkout_job('code')['id'] / 'state.json'
        state = json.loads(statepath.read_text())
        state['sbatch'] = ['sbatch', G.export('code', 'e' * 40)]
        statepath.write_text(json.dumps(state))
    elif mutation == 'later_failed_gate':
        reading = G.live(commits, published, table, commands__squeue={'stdout': queued(0, 10)}, time_utc='2026-11-08T10:01:00Z')
        assert call(['gate', 'pose-benchmark'])['status'] == 1
        S._now = lambda: at(10, 1, 30)
    lines = call(args)
    results['cases'].append({'name': mutation, 'lines_status': lines['status'],
                             'command_count': len(lines['stdout'].splitlines()), 'stderr': lines['stderr']})

# A successful remote-time-stamped gate, followed by a refusal stamped by a permitted 180s-lagging Mac clock.
repo, ledger, code = setup()
_, commits, published, table, submitted = S.gate_request('pose-benchmark', repo, ledger)
reading = G.live(commits, published, table, commands__squeue={'stdout': queued(0, 0)}, time_utc='2026-11-08T10:03:00Z')
S._now = lambda: at(10, 0)
S.remote_read = lambda **kw: reading
first = call(['gate', 'pose-benchmark'])
S._now = lambda: at(10, 0, 1)
def failed_read(**kw):
    raise ValueError('remote read failed after the preceding successful gate')
S.remote_read = failed_read
second = call(['gate', 'pose-benchmark'])
lines = call(['lines', 'pose-benchmark'])
results['cases'].append({'name': 'later_refusal_with_supported_clock_skew', 'first_gate': first,
                         'second_gate': second, 'lines_after_refusal': lines,
                         'record_names': sorted(p.name for p in (repo / S.PACKAGE / 'evidence/gates').glob('pose-benchmark_*.json')),
                         'latest_gate_passed': S.latest_gate('pose-benchmark', repo)['passed']})

# The prescribed shell control flow is short-circuit AND: an exception/nonzero gate does not invoke lines.
# Exercise that decision separately from the standalone diagnostic call above.
S._now = lambda: at(10, 0, 2)
sequence_gate = call(['gate', 'pose-benchmark'])
sequence_lines = call(['lines', 'pose-benchmark']) if sequence_gate['status'] == 0 else None
assert sequence_lines is None
results['cases'].append({'name': 'prescribed_gate_and_lines_control_flow',
                         'gate_exception': sequence_gate['exception'], 'lines_invocations': 0,
                         'note': 'Matches gate && lines: gate exception has nonzero process status, so lines is not invoked.'})

Path('/private/tmp/eval19618-b1-closure-repro.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
