"""Explicit Mac-only bridge from accepted mixed P1-r2 extractions to frozen readers.

No submission, remote access, runtime patches, retries or numerical code changes.
The controller is passed explicitly: CLI execution as __main__ keeps one context.
"""
from pathlib import Path
import json
import re

PROTOCOL = 'scaling-p1-reader-bridge-20261117-v1'
BASE = 'results/infra/scaling_300m_20261104/p1_recovery_readout_20261117'
ACCEPTANCE_SHA256 = '0b1f538757dc8f39e31425dfc20d6b7ac039820c2bc34c5c65e1351fce46c9ce'
CODE = 'df2de35c05381e7a5a69edde409171b210a3b5d8'
P1_CODE = 'e4ebdbfa709c3f83cb5b1b11bdd261d132ccae3a'
EXTRACTIONS = {
    'SCALING-300M-eval19618-pose-20261106': ('64571999', 'claude'),
    'SCALING-300M-eval19618-p1-r2-20261116': ('64620434', 'gpt'),
    'SCALING-300M-eval19618-p2-20261106': ('64572001', 'claude'),
}
PREVIOUS = {
    'pose-validate': (), 'pose-benchmark': ('pose-validate',), 'probe-benchmark': ('pose-validate',),
    'pose-ridge': ('pose-validate', 'pose-benchmark', 'probe-benchmark'),
    'probe-fit': ('pose-validate', 'pose-benchmark', 'probe-benchmark'),
    'probe-aggregate': ('probe-fit',), 'compare': ('pose-ridge', 'probe-aggregate'),
}
MAC_SOURCES = ('scripts/scaling_plans19618.py', 'scripts/scaling_p1_reader_bridge.py',
               'scripts/scaling_p1_reader_budget.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cpu_job_id(job_id):
    return re.fullmatch(r'SCALING-19618-(?:plans-checkout|budgets-checkout|inputs-checkout|pose-validate|'
                       r'pose-bench-(?:30m-natural|300m-natural|300m-matched)|pose-ridge|'
                       r'probe-bench-(?:30m-natural|300m-natural|300m-matched)-(?:ninapro-db[236]|epn612|uci-emg)|'
                       r'probe-fit-(?:0[1-9]|1[0-5])|probe-aggregate-0[1-6]|compare)-20261106', job_id) is not None


def allowed_stage(stage):
    require(stage in PREVIOUS or stage in ('checkout-plans', 'checkout-budgets', 'checkout-inputs'),
            'P1 reader profile permits only the remaining CPU reader stages/checkouts')


def accepted(S, root):
    base = Path(root)/BASE
    raw = (base/'acceptance.json').read_bytes()
    require(S.digest(raw) == ACCEPTANCE_SHA256, 'accepted P1-r2 evidence changed')
    result = json.loads(raw)
    require(result.get('accepted') is True and result.get('job_id') == '64620434' and
            result.get('scheduler_state') == 'COMPLETED' and result.get('exit_code') == '0:0' and
            result.get('code_sha') == P1_CODE, 'P1-r2 not accepted')
    for name, digest in result['evidence_sha256'].items():
        require(S.digest((base/name).read_bytes()) == digest, 'accepted evidence changed: '+name)
    return result


def plans(S, root):
    import scaling_p1_recovery_plans as M
    acceptance = accepted(S, root)
    base = Path(root)/BASE
    snapshot = json.loads((base/'evidence/snapshot.json').read_bytes())
    manifest = json.loads((base/'evidence/release/recovery_manifest.json').read_bytes())
    extraction = json.loads((base/'evidence/release/extraction_plan.json').read_bytes())
    receipts, pins, provenances = {}, {}, {}
    for task, entry in snapshot['tasks'].items():
        key = entry['receipt_key']
        row = snapshot['files'][key]
        receipts[task] = (base/'evidence'/key).read_bytes()
        pins[task] = {k: row[k] for k in ('path', 'sha256', 'bytes')}
        provenances[task] = {name: (base/'evidence'/item['provenance_key']).read_bytes()
                             for name, item in entry['outputs'].items()}
    jobs = {task: {'job_id': job, 'state': 'COMPLETED', 'exit_code': '0:0',
                   'code_sha': P1_CODE if task == 'p1' else CODE}
            for task, job in (('pose', '64571999'), ('p1', '64620434'), ('p2', '64572001'))}
    pose, probe, lineage = M.mixed_candidate_caches(receipts, pins, provenances, jobs,
        extraction['checkpoints'][S.ARM], manifest['extraction_plan']['sha256'], P1_CODE,
        S.digest((base/'evidence/release/recovery_manifest.json').read_bytes()), manifest['source_sha256'])
    original = Path(root)/S.INFRA/'cpu_readers_20261105'
    control_dir = Path(root)/S.PACKAGE/'control19618_identity'
    inventory = S._load('results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json', root)
    control = S.control_probe_caches(inventory,
        {ds: (control_dir/(ds+'.npz.provenance.json')).read_bytes() for ds in S.DATASETS},
        S._load(S.RECOVERY['plan'], root)['control']['checkpoint'])
    pose_plan = S.pose_plan(json.loads((original/'pose_plan.json').read_bytes()),
        S._load('results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json', root), pose)
    probe_plan = S.probe_plan(json.loads((original/'probe_plan.json').read_bytes()), control, probe)
    for name, value in (('pose_plan', pose_plan), ('probe_plan', probe_plan), ('mixed_lineage', lineage)):
        require(value == json.loads((base/'reader_inputs'/(name+'.json')).read_bytes()),
                'rebuilt reader inputs differ from accepted P1-r2: '+name)
    return pose_plan, probe_plan, lineage


def build_reader_plans(S, code_sha, root):
    require(code_sha == CODE, 'readers must use the original frozen numerical checkout')
    pose, probe, lineage = plans(S, root)
    S._write(S.DOCUMENTS['pose_plan'][1], S.dumps(pose), root)
    S._write(S.DOCUMENTS['probe_plan'][1], S.dumps(probe), root)
    S._write('readers/p1_r2_lineage.json', S.dumps(lineage), root)


def attestation(S, root):
    accepted(S, root)
    return {'protocol': PROTOCOL, 'accepted_p1_job': '64620434', 'acceptance_sha256': ACCEPTANCE_SHA256,
            'reader_code_sha': CODE, 'p1_producer_sha': P1_CODE,
            'mac_sources': {name: S.digest((Path(root)/name).read_bytes()) for name in MAC_SOURCES}}


def _ledger(S, job_id, ledger, expected_agent=None):
    state = json.loads((Path(ledger)/job_id/'state.json').read_text())
    require(state.get('stato') == 'sottomesso' and state.get('campagna') == S.CAMPAIGN and
            (state.get('agente') == expected_agent if expected_agent else state.get('agente') in ('claude', 'gpt')) and
            str(state.get('job_slurm', '')).isdigit(), 'unaccepted predecessor ledger: '+job_id)
    return str(state['job_slurm'])


def collected_accounting(S, stage, job, root, ledger):
    directory = S.evidence_dir(stage, root)/(job['local'] or '')
    record = json.loads((directory/'accounting.json').read_text())
    slurm = S.ledger_job(job['id'], ledger)
    raw = (directory/'sacct.txt').read_text()
    actual = S.accounting(raw, slurm, job['cpus'], job['seconds'], job['gpus'], need_rss=stage in S.RSS_GATED)
    require(record == actual, 'collected accounting differs from raw sacct: '+job['id'])
    return actual


def dependencies(S, stage, root, ledger):
    allowed_stage(stage)
    accepted(S, root)
    result = {}
    for name, (job, agent) in EXTRACTIONS.items():
        require(_ledger(S, name, ledger, agent) == job, 'only the accepted P1-r2/pose/P2 extractions are predecessors')
        result[name] = job
    for previous in PREVIOUS.get(stage, ()):
        table = S.stage_table(previous, root)
        files, collection = S.evidence(previous, root)
        require(set(collection['jobs']) == {j['id'] for j in table}, 'predecessor stage is incomplete: '+previous)
        for job in table:
            slurm = S.ledger_job(job['id'], ledger)
            require(str(collection['jobs'][job['id']]['job_slurm']) == slurm, 'predecessor collection/ledger differ')
            measured = collected_accounting(S, previous, job, root, ledger)
            if previous == 'pose-validate':
                report = json.loads(files['report.json'][0])
                plan = S._load(f"{S.PACKAGE}/{S.DOCUMENTS['pose_plan'][1]}", root)
                S._pose_report(report, plan, S.document('pose_plan', root)['sha256'], CODE, S.P.PROTOCOL, S.STEP)
                require(report.get('status') == 'validated_input_metadata' and
                        report.get('scientific_metrics_computed') is False and report.get('slurm_job_id') == slurm and
                        set(report.get('inspections', {})) == {a+'/'+s for a in S.ARMS for s in ('train', 'val')} and
                        measured.get('MaxRSS_bytes') is not None and 0 < measured['MaxRSS_bytes'] <= 30*2**30,
                        'complete accepted four-cache input validation required before benchmarks')
            result[job['id']] = slurm
    return result


def budget(S, root, ledger):
    from scaling_p1_reader_budget import evaluate_budget
    tables = {'checkout-'+name: [S.checkout_job(name)] for name in ('plans', 'budgets', 'inputs')}
    for stage in PREVIOUS:
        if stage == 'probe-fit':
            ready = (Path(root)/S.PACKAGE/S.DOCUMENTS['probe_resources'][1]).exists()
            tables[stage] = S.stage_table(stage, root) if ready else []
        elif stage == 'probe-aggregate':
            tables[stage] = S.jobs(stage, {'protocol': S.C.RESOURCE_PROTOCOL, 'shards': []})
        else:
            tables[stage] = S.stage_table(stage, root)
    submissions, accounting = {}, {}
    # A missing fit plan must not hide an already used fit ID. Unknown IDs in
    # this remaining CPU namespace are passed to the budget validator and stop.
    for folder in Path(ledger).glob('SCALING-19618-*'):
        if folder.name != S.checkout_job('code')['id']:
            submissions[folder.name] = json.loads((folder/'state.json').read_text())
    for stage, table in tables.items():
        for job in table:
            folder = Path(ledger)/job['id']
            if folder.exists():
                submissions[job['id']] = json.loads((folder/'state.json').read_text())
            if (S.evidence_dir(stage, root)/(job['local'] or '')/'accounting.json').exists():
                accounting[job['id']] = collected_accounting(S, stage, job, root, ledger)
    value = evaluate_budget(tables, submissions, accounting, fit_plan_ready=ready)
    require(value['within_total_cap'] is True and value['within_stage_caps'] is True,
            'CPU-only cumulative envelope exceeds 351 core-hours or a frozen stage cap')
    return value


def context(S, stage, root, ledger):
    allowed_stage(stage)
    pose, probe, lineage = plans(S, root)
    for name, expected in (('pose_plan', pose), ('probe_plan', probe)):
        require(S._load(f'{S.PACKAGE}/{S.DOCUMENTS[name][1]}', root) == expected,
                'published reader plan differs from accepted mixed provenance: '+name)
    return {**attestation(S, root), 'predecessors': dependencies(S, stage, root, ledger),
            'lineage_sha256': S.digest(S.dumps(lineage)), 'budget': budget(S, root, ledger)}


def checks(S, stage, reading, root, ledger):
    value = context(S, stage, root, ledger)
    return [('accepted mixed extractions and previous CPU stages remain COMPLETED 0:0',
             all(S._completed(reading['sacct'].get(job, {'exit': 1, 'stdout': ''}), job)
                 for job in value['predecessors'].values())),
            ('CPU-only cumulative envelope stays within 351 core-hours', value['budget']['within_total_cap'])], value
