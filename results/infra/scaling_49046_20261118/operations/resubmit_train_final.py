"""Operator adapter for the resubmission of the second 300M segment (39.237 -> 49.046). No sender.

Job 65188321 (SCALING-49046-train-49046-20261118) was CANCELLED in the queue on 23/11/2026 without ever starting. The submitter
never reuses a logical ID, so the same job goes out under a new one. The operator files of the campaign must stay byte-identical to
the frozen code commit (scaling_operations49046.attestation), therefore this file changes nothing on disk: it replaces that one ID in
memory and then runs the frozen controller with the same arguments.

Use it INSTEAD OF scripts/scaling_plans49046.py for every command from `gate train-final` to `compare`: later stages look up the
training job in the ledger.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import scaling_plans49046 as S
import scaling_terminal_builders as T

OLD_ID = 'SCALING-49046-train-49046-20261118'
OLD_JOB = '65188321'
NEW_ID = 'SCALING-49046-train-49046-r2-20261126'
require = S.require


def install():
    """Rename the second training segment, in this process only. Idempotent."""
    if getattr(S.ledger_job, 'resubmission_alias', False):
        return
    training_job, ledger_job = T.training_job, S.ledger_job

    def renamed_job(module, end, root):
        job = training_job(module, end, root)
        if job['id'] == OLD_ID:
            job['id'] = NEW_ID
        return job

    def renamed_ledger_job(job_id, *args, **kwargs):
        # the extraction plan builder of the frozen code names the old ID literally
        return ledger_job(NEW_ID if job_id == OLD_ID else job_id, *args, **kwargs)

    renamed_ledger_job.resubmission_alias = True
    T.training_job, S.ledger_job = renamed_job, renamed_ledger_job


def replaced_job_check(answer):
    """The replaced job can never run: its single accounting row is CANCELLED, with no start and no elapsed time."""
    row = S._sacct_parent(answer, OLD_JOB)
    require(row is not None, 'no accounting row for the replaced job ' + OLD_JOB)
    require(row.get('State', '').split(' ')[0] == 'CANCELLED' and row.get('Start') in ('None', 'Unknown') and
            row.get('ElapsedRaw') == '0', 'replaced job is not "cancelled before starting": %s' % row)
    return row


def self_check(root=ROOT):
    """This adapter runs only as committed."""
    relative = str(Path(__file__).resolve().relative_to(root))
    blob = subprocess.run(['git', '-C', str(root), 'show', 'HEAD:' + relative], capture_output=True)
    require(blob.returncode == 0 and blob.stdout == Path(__file__).read_bytes(), 'commit the adapter before using it')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    self_check()
    if argv[:2] in (['gate', 'train-final'], ['lines', 'train-final']):
        reading = S.remote_read(slurm_jobs=[OLD_JOB])
        row = replaced_job_check(reading['sacct'][OLD_JOB])
        print('PASS  replaced job %s: %s, never started (read %s)' %
              (OLD_JOB, row['State'], time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        print(json.dumps({'replaced': {OLD_ID: OLD_JOB}, 'new_id': NEW_ID}))
    install()
    return S.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
