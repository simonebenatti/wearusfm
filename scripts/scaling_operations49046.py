"""Mac-side closed-stage prerequisites and frozen operator attestation. No sender."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import calendar

PREREQUISITES={
 'checkout-code':(), 'train-predecay':(), 'train-final':('train-predecay',),
 'archive':('train-final',), 'checkout-extraction':('archive',),
 'extractions':('archive','train-final'), 'checkout-plans':('extractions',),
 'pose-validate':('extractions',), 'pose-benchmark':('pose-validate',),
 'probe-benchmark':('extractions',),
 'checkout-budgets':('pose-validate','pose-benchmark','probe-benchmark'),
 'pose-ridge':('pose-validate','pose-benchmark'), 'probe-fit':('pose-validate','probe-benchmark'),
 'probe-aggregate':('probe-fit',), 'checkout-inputs':('probe-aggregate','pose-ridge'),
 'compare':('probe-aggregate','pose-ridge'),
}
OPERATOR_FILES=('scripts/scaling_plans49046.py','scripts/scaling_operations49046.py',
                'scripts/scaling_terminal_builders.py','scripts/scaling_build49046.py',
                'results/infra/scaling_49046_20261118/evaluation/training/training_39237.json',
                'results/infra/scaling_49046_20261118/evaluation/training/training_49046.json',
                'results/infra/scaling_49046_20261118/evaluation/training/resource_policy.json',
                'results/infra/scaling_49046_20261118/evaluation/training/archive_plan.json',
                'results/infra/scaling_49046_20261118/launcher/training_39237.json',
                'results/infra/scaling_49046_20261118/launcher/training_49046.json',
                'results/infra/scaling_49046_20261118/launcher/archive_plan.json',
                'scripts/submit_job.py','scripts/slurm/campaign_checkout.sbatch','scripts/slurm/lib_code_dir.sh',
                'scripts/scaling_continue49046.py','scripts/scaling_preserve49046.py','scripts/scaling_finalize49046.py',
                'scripts/scaling_dose_interaction.py','scripts/scaling_derive49046.py',
                'scripts/scaling_evaluation49046.py','scripts/scaling_probe49046.py','scripts/scaling_pose_readout49046.py',
                'scripts/scaling_compare49046.py','scripts/scaling_reading49046.py',
                'scripts/slurm/scaling_evaluation49046.sbatch','scripts/slurm/scaling_readers49046.sbatch',
                'scripts/slurm/scaling_continue49046.sbatch','scripts/slurm/scaling_preserve49046.sbatch',
                'results/infra/scaling_49046_20261118/launcher/evaluation_freeze.json',
                'results/infra/scaling_49046_20261118/launcher/resource_policy.json')
DEADLINE_UTC='2026-12-19T00:00:00Z'


def attestation(S,commit,root):
    S.ST.check_source_contract(Path(root)/S.INFRA/'training_source_contract.json')
    out={}
    for relative in OPERATOR_FILES:
        data=(Path(root)/relative).read_bytes()
        sha=hashlib.sha256(data).hexdigest()
        S.require(S._blob_sha256(commit,relative,root)==sha,'operator differs from frozen code: '+relative)
        out[relative]=sha
    return {'commit':commit,'files':out}


def dependencies(S,stage,root,ledger):
    S.require(stage in PREREQUISITES,'unregistered operational stage: '+stage)
    S.require(time.time()<calendar.timegm(time.strptime('2026-12-19','%Y-%m-%d')),
              'retention horizon expired: stop and re-establish preservation')
    result={}
    for previous in PREREQUISITES[stage]:
        files,manifest=S.evidence(previous,root)
        table=S.stage_table(previous,root)
        S.require(set(manifest['jobs'])=={j['id'] for j in table},'incomplete prerequisite collection: '+previous)
        for job in table:
            slurm=S.ledger_job(job['id'],ledger)
            row=manifest['jobs'][job['id']]
            S.require(row['job_slurm']==slurm,'prerequisite ledger mismatch')
            local=job['local']+'/' if job['local'] else ''
            raw=(S.evidence_dir(previous,root)/(local+'sacct.txt')).read_text()
            measured=S.accounting(raw,slurm,job['cpus'],job['seconds'],job['gpus'],need_rss=previous in S.RSS_GATED)
            recorded=json.loads((S.evidence_dir(previous,root)/(local+'accounting.json')).read_text())
            S.require(recorded==measured,'prerequisite accounting differs from raw sacct')
            result[job['id']]=slurm
    return result


def resource_check(S):
    policy=json.loads((S.ROOT/S.INFRA/'launcher/resource_policy.json').read_text())
    budget=S.budget_table()
    training=[S.stage_table(stage)[0] for stage in ('train-predecay','train-final')]
    return (sum(j['gpus']*j['seconds'] for j in training)/3600==119 and
            [j['seconds'] for j in training]==[71100,36000] and
            all(j['gpus']==4 and j['cpus']==32 and j['memory']=='480G' for j in training) and
            budget['within_stage_caps'] and budget['within_total_cap'] and
            budget['cap']==policy['evaluation']['reader_cpu_only_cap_core_hours']==400 and
            sum(j['gpus']*j['seconds'] for j in S.jobs('extractions'))/3600==14 and
            policy['training']['gpu_hours_requested']==119 and policy['storage']['minimum_free_run_bytes']==20*2**30)


def context(S,stage,root,ledger):
    jobs=dependencies(S,stage,root,ledger)
    files={}
    for previous in PREREQUISITES[stage]:
        folder=S.evidence_dir(previous,root)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'dependencies':jobs,'evidence_sha256':files}
