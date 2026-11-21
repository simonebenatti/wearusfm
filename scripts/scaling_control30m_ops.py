"""Separate Mac release of the three already-budgeted 30M terminal extractions.

No sender. Original training, extraction controller and CPU numerical sources stay
byte-identical to BASE_CODE. A separate checkout carries only an input adapter;
the common ledger and the original three output paths prevent duplicate work.
"""
from pathlib import Path
import argparse
import calendar
import json
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import scaling_plans49046 as S
import scaling_operations49046 as O
import scaling_control30m_bridge as B

BASE_CODE = '87da975bc0fd4824684e166659bb996e118ff107'
PROTOCOL = 'scaling-control30m-early-20261121-v1'
BASE = S.PACKAGE + '/evaluation/control30m_early_20261121'
PLAN = BASE + '/plan.json'
POLICY = BASE + '/resource_policy.json'
CAMPAIGN = 'scaling_49046_control30m_20261121_v1'
CODE_DIR = S.WORK + '/wearusfm_campaigns/' + CAMPAIGN
CHECKOUT_ID = 'SCALING-49046-control30m-checkout-20261121'
STAGES = ('checkout', 'extractions')
NEW_SOURCES = (
    'scripts/scaling_control30m_ops.py', 'scripts/scaling_control30m_bridge.py',
    'scripts/scaling_control30m_evaluation.py',
    'scripts/slurm/scaling_control30m_evaluation.sbatch',
)
require = S.require


def table(stage):
    require(stage in STAGES, 'unregistered control stage')
    if stage == 'checkout':
        return [S._job('checkout', CHECKOUT_ID, 2, 300, CODE_DIR, 'control30m',
                       files={}, memory='4G', log='campaign_checkout')]
    return [j for j in S.jobs('extractions') if j['flags'][0] == S.ARMS[0]]


def policy():
    old = S.budget_table()
    extra = 2 * (300 + 180) / 3600
    return {'protocol': PROTOCOL, 'base_code': BASE_CODE,
            'existing_extraction_ids': [j['id'] for j in table('extractions')],
            'control_gpu_hours_requested': 7.0, 'control_gpu_hours_with_margin': 7.15,
            'all_extractions_gpu_hours_requested': 14.0,
            'additional_checkout_id': CHECKOUT_ID, 'additional_checkout_core_hours_with_margin': extra,
            'checkout_count': 6, 'checkout_stage_cap_with_margin': 1.6,
            'cpu_only_worst_case_with_margin': old['core_hours_with_scheduler_margin'] + extra,
            'cpu_only_cap': 400, 'other_stage_caps_unchanged': True,
            'no_duplicate_extractions': True}


def write_new(path, value):
    path = Path(path)
    data = S.dumps(value)
    if path.exists():
        require(path.read_bytes() == data, 'existing document differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def prepare(root=ROOT):
    write_new(Path(root) / PLAN, B.build_control_plan(S, root))
    write_new(Path(root) / POLICY, policy())


def plan_pin(root=ROOT):
    return S.pin(CODE_DIR + '/' + PLAN, (Path(root) / PLAN).read_bytes())


def attestation(commit, root=ROOT):
    require(S.ST.digest(commit, 40), 'full reviewed control code SHA required')
    require(O.attestation(S, BASE_CODE, root)['commit'] == BASE_CODE, 'frozen base differs')
    sources = {}
    for relative in (*NEW_SOURCES, PLAN, POLICY):
        data = (Path(root) / relative).read_bytes()
        sources[relative] = S.digest(data)
        require(S._blob_sha256(commit, relative, root) == sources[relative],
                'local release differs from control checkout: ' + relative)
    require(json.loads((Path(root) / PLAN).read_bytes()) == B.build_control_plan(S, root),
            'control plan differs from pinned historical inputs')
    require(json.loads((Path(root) / POLICY).read_bytes()) == policy(), 'resource amendment differs')
    require(policy()['cpu_only_worst_case_with_margin'] <= 400, 'amended CPU cap exceeded')
    require(time.time() < calendar.timegm(time.strptime('2026-12-19', '%Y-%m-%d')),
            'retention horizon expired')
    require(S.checkout_commit('code') == BASE_CODE, 'training/CPU checkout must remain frozen')
    return {'protocol': PROTOCOL, 'commit': commit, 'base_code': BASE_CODE, 'sha256': sources}


def control_commit(ledger=S.LEDGER):
    S.ledger_job(CHECKOUT_ID, ledger)
    state = json.loads((Path(ledger) / CHECKOUT_ID / 'state.json').read_text())
    values = [v for v in state['sbatch'] if v.startswith('--export=')]
    require(len(values) == 1, 'ambiguous control checkout export')
    fields = dict(x.split('=', 1) for x in values[0][9:].split(',') if '=' in x)
    require(fields.get('CAMPAIGN') == CAMPAIGN and fields.get('CAMPAIGN_SHA') == fields.get('EXPECTED_SHA')
            and S.ST.digest(fields.get('CAMPAIGN_SHA', ''), 40), 'wrong control checkout receipt')
    return fields['CAMPAIGN_SHA']


def evidence_dir(stage, root=ROOT):
    return Path(root) / S.PACKAGE / 'evidence' / ('control30m-' + stage)


def accounting(stage, root=ROOT, ledger=S.LEDGER):
    manifest_path = evidence_dir(stage, root) / 'collection.json'
    manifest = json.loads(manifest_path.read_text())
    jobs = table(stage)
    require(set(manifest['jobs']) == {j['id'] for j in jobs}, 'incomplete control collection')
    result = {}
    for job in jobs:
        slurm = S.ledger_job(job['id'], ledger)
        require(manifest['jobs'][job['id']]['job_slurm'] == slurm, 'control collection/ledger mismatch')
        folder = evidence_dir(stage, root) / job['local']
        measured = S.accounting((folder / 'sacct.txt').read_text(), slurm,
                               job['cpus'], job['seconds'], job['gpus'], need_rss=False)
        require(json.loads((folder / 'accounting.json').read_text()) == measured,
                'control accounting differs from raw sacct')
        result[job['id']] = slurm
    return result


def state_snapshot(stage, commit, root=ROOT, ledger=S.LEDGER):
    base_job = S.checkout_job('code')['id']
    require(S._collected_checkout('code', ledger, root), 'base checkout not collected successfully')
    dependencies = {base_job: S.ledger_job(base_job, ledger)}
    if stage == 'extractions':
        require(control_commit(ledger) == commit, 'wrong producer commit')
        dependencies.update(accounting('checkout', root, ledger))
    submitted = {}
    for job in table(stage):
        if (Path(ledger) / job['id']).exists():
            submitted[job['id']] = S.ledger_job(job['id'], ledger)
    files = {}
    for folder in (S.evidence_dir('checkout-code', root),
                   evidence_dir('checkout', root) if stage == 'extractions' else None):
        if folder is not None:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    files[str(path.relative_to(root))] = S.digest(path.read_bytes())
    return {'dependencies': dependencies, 'submitted': submitted, 'evidence_sha256': files}


def request(stage, commit, root=ROOT, ledger=S.LEDGER):
    snapshot = state_snapshot(stage, commit, root, ledger)
    commands = {'squeue': S.QUEUE, 'quota': S.QUOTA}
    dirs = {'base': S.checkout('code')}
    if stage == 'extractions':
        dirs['control'] = CODE_DIR
    for name, directory in dirs.items():
        commands['head:' + name] = S._git(directory, 'rev-parse', 'HEAD')
        commands['status:' + name] = S._git(directory, 'status', '--porcelain', '--untracked-files=all')
    return snapshot, {'slurm_jobs': list(dict.fromkeys([*snapshot['dependencies'].values(),
                                                       *snapshot['submitted'].values()])),
                      'commands': commands, 'exists': [j['out'] for j in table(stage)],
                      'hashed': [plan_pin(root)['path']] if stage == 'extractions' else []}


def checks(stage, commit, reading, snapshot, root=ROOT, ledger=S.LEDGER):
    lane = S.lane(reading)  # failed or boolean exit codes are rejected before counting
    commands = reading['commands']
    expected = {'base': BASE_CODE, **({'control': commit} if stage == 'extractions' else {})}
    tests = []
    for name, sha in expected.items():
        head, clean = commands['head:' + name], commands['status:' + name]
        tests.append((name + ' checkout has frozen HEAD and clean tree',
                      type(head.get('exit')) is int and head['exit'] == 0 and head['stdout'].strip() == sha and
                      type(clean.get('exit')) is int and clean['exit'] == 0 and not clean['stdout'].strip()))
    tests += [('predecessors COMPLETED 0:0', all(S._completed(reading['sacct'][n], n)
                                               for n in snapshot['dependencies'].values())),
              ('submitted jobs COMPLETED 0:0', all(S._completed(reading['sacct'][n], n)
                                                  for n in snapshot['submitted'].values())),
              ('WORK quota at least30GiB', lane['work_free_gib'] >= 30)]
    todo = [j for j in table(stage) if j['id'] not in snapshot['submitted']]
    tests += [('unused outputs absent', all(reading['exists'][j['out']] is False for j in todo)),
              ('at least one unused logical ID', bool(todo))]
    if stage == 'checkout':
        tests += [('serial queue has room', lane['serial_room'] >= 1 and lane['line_jobs'] < 4)]
    else:
        p = plan_pin(root)
        actual = {f['path']: f for f in reading['files']}.get(p['path'], {})
        tests += [('published plan present and byte-identical', actual.get('canonical') is True and
                   actual.get('sha256') == p['sha256'] and actual.get('bytes') == p['bytes'])]
    p = json.loads((Path(root) / PLAN).read_bytes())
    pins = S.checkout_pins(p, {'code': BASE_CODE}, 'code', root)
    tests += [('input pins inside historical checkout match its commit', all(a == b for _, a, b in pins))]
    return tests, {**lane, 'tranche': [j['id'] for j in todo]}


def gate_path(stage, root=ROOT):
    return Path(root) / BASE / 'gates' / stage


def run_gate(stage, commit, reviewed=None, root=ROOT, ledger=S.LEDGER):
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    try:
        frozen = attestation(commit, root)
        if stage == 'checkout':
            require(reviewed is not None, 'reviewed release required for checkout')
            S.release_check(reviewed, commit, root)
            r = subprocess.run(['git', '-C', str(root), 'ls-remote', 'leonardo',
                                'refs/heads/deploy/' + CAMPAIGN], capture_output=True, text=True, timeout=300)
            require(r.returncode == 0, 'cannot check deployment reference')
            refs = [x.split()[0] for x in r.stdout.splitlines() if x.strip()]
            require(refs in ([], [commit]), 'deployment reference points to other code')
        snapshot, payload = request(stage, commit, root, ledger)
        reading = S.remote_read(**payload)
        tests, facts = checks(stage, commit, reading, snapshot, root, ledger)
        record = {'protocol': PROTOCOL, 'stage': stage, 'commit': commit, 'reviewed': reviewed,
                  'read_at_utc': reading['time_utc'], 'passed': all(v for _, v in tests),
                  'checks': tests, 'facts': facts, 'snapshot': snapshot, 'attestation': frozen,
                  'reading': reading}
        write_new(gate_path(stage, root) / (stamp + '.json'), record)
        print(json.dumps(record, indent=2))
        return 0 if record['passed'] else 1
    except Exception as error:
        write_new(gate_path(stage, root) / (stamp + '_refused.json'),
                  {'protocol': PROTOCOL, 'stage': stage, 'passed': False, 'error': str(error)})
        raise


def raw_lines(stage, commit, ids, root=ROOT):
    if stage == 'checkout':
        job = table(stage)[0]
        require(ids == [CHECKOUT_ID], 'wrong checkout tranche')
        flags = ['--cpus-per-task=2', '--mem=4G', '--time=00:05:00',
                 f'--export=ALL,CAMPAIGN={CAMPAIGN},CAMPAIGN_SHA={commit},EXPECTED_SHA={commit}', '/dev/stdin']
        return [shlex.join(['git', '-C', str(root), 'push', 'leonardo', commit + ':refs/heads/deploy/' + CAMPAIGN]),
                S._submit(job['id'], 0, flags, stdin=str(Path(root) / 'scripts/slurm/campaign_checkout.sbatch'), root=root)]
    output = []
    for job in table(stage):
        if job['id'] not in ids:
            continue
        plan = plan_pin(root)
        flags = [f"--time={S.clock(job['seconds'])}",
                 f"--export=ALL,CODE_DIR={CODE_DIR},EXPECTED_SHA={commit},SCALING_EXTRACTION_PLAN_SHA={plan['sha256']}",
                 CODE_DIR + '/scripts/slurm/scaling_control30m_evaluation.sbatch', plan['path'],
                 *job['flags'], job['out']]
        output.append(S._submit(job['id'], job['seconds'] / 3600, flags, root=root))
    require(len(output) == len(ids), 'tranche contains foreign IDs')
    return output


def admitted_lines(stage, commit, root=ROOT, ledger=S.LEDGER, now=None):
    paths = sorted(gate_path(stage, root).glob('*.json'))
    require(paths, 'no fresh passed control gate')
    record = json.loads(paths[-1].read_text())
    require(record.get('protocol') == PROTOCOL and record.get('passed') is True and
            record.get('stage') == stage and record.get('commit') == commit, 'latest gate did not pass for this release')
    read_at = calendar.timegm(time.strptime(record['read_at_utc'], '%Y-%m-%dT%H:%M:%SZ'))
    age = (time.time() if now is None else now) - read_at
    require(-300 <= age <= 1800, 'gate expired')
    require(record['attestation'] == attestation(commit, root), 'release changed after gate')
    require(record['snapshot'] == state_snapshot(stage, commit, root, ledger), 'ledger/evidence changed after gate')
    ids = record['facts']['tranche']
    require(ids and not any((Path(ledger) / i).exists() for i in ids), 'tranche already submitted')
    return raw_lines(stage, commit, ids, root)


def collect(stage, root=ROOT):
    jobs = table(stage)
    ids = {j['id']: S.ledger_job(j['id']) for j in jobs}
    reading = S.remote_read([p for j in jobs for p in j['files'].values()] +
                            [S.log_path(j, ids[j['id']]) for j in jobs], ids.values(),
                            listings=[j['folder'] for j in jobs])
    result = S.collect('control30m-' + stage, jobs, ids, reading, evidence_dir(stage, root))
    if stage == 'extractions':
        accept_control(root)
    print(json.dumps(result, indent=2))


def accept_control(root=ROOT):
    commit = control_commit()
    attestation(commit, root)
    jobs = accounting('extractions', root)
    files, manifest = S.evidence('control30m-extractions', root)
    plan = json.loads((Path(root) / PLAN).read_bytes())
    arm = S.ARMS[0]
    expected = {'checkpoint': plan['checkpoints'][arm], 'code_sha': commit,
                'plan_sha256': plan_pin(root)['sha256']}
    pose_prefix = arm + '/pose/'
    pose = S.candidate_pose_caches(json.loads(files[pose_prefix + 'receipt.json'][0]),
                                  files[pose_prefix + 'receipt.json'][1],
                                  {split: files[pose_prefix + name + '.npz.provenance.json'][0]
                                   for split, name in (('train', 'train_subset'), ('val', 'val'))}, expected, arm=arm)
    tasks = {dataset: task for task, datasets in S.TASKS.items() for dataset in datasets}
    probe = S.candidate_probe_caches(
        {task: json.loads(files[arm + '/' + task + '/receipt.json'][0]) for task in S.TASKS},
        {task: files[arm + '/' + task + '/receipt.json'][1] for task in S.TASKS},
        {dataset: files[arm + '/' + tasks[dataset] + '/' + dataset + '.npz.provenance.json'][0]
         for dataset in S.DATASETS}, expected, arm=arm)
    acceptance = {'protocol': PROTOCOL, 'accepted_receipts': True, 'code_sha': commit,
                  'plan_sha256': expected['plan_sha256'], 'jobs': jobs,
                  'pose_caches': pose, 'probe_caches': probe,
                  'file_pins': {name: value[1] for name, value in files.items()},
                  'cache_bytes_rechecked_by_frozen_readers_before_fits': True}
    write_new(Path(root) / BASE / 'acceptance.json', acceptance)
    return acceptance


def reader_plans(campaign_root, root=ROOT):
    campaign_root = Path(campaign_root).resolve(strict=True)
    commit = control_commit()
    attestation(commit, root)
    acceptance = accept_control(root)
    early, _ = S.evidence('control30m-extractions', root)
    complete, _ = S.evidence('extractions', campaign_root)
    for name, (data, pin) in early.items():
        require(name in complete and complete[name][1] == pin and complete[name][0] == data,
                'early control evidence changed in final collection: ' + name)
    plan = json.loads((Path(root) / PLAN).read_bytes())
    pose, probe, lineage = B.mixed_reader_plans(S, BASE_CODE, campaign_root,
                                               plan, plan_pin(root)['sha256'], commit)
    # The original campaign receives only documents, so its frozen numerical
    # checkout and its original metadata-only release checks remain untouched.
    write_new(campaign_root / S.PACKAGE / S.DOCUMENTS['pose_plan'][1], pose)
    write_new(campaign_root / S.PACKAGE / S.DOCUMENTS['probe_plan'][1], probe)
    write_new(campaign_root / S.PACKAGE / 'readers/control30m_early_lineage.json', lineage)
    write_new(campaign_root / S.PACKAGE / 'readers/control30m_early_plan.json', plan)
    write_new(campaign_root / S.PACKAGE / 'readers/control30m_early_resource_policy.json', policy())
    write_new(campaign_root / S.PACKAGE / 'readers/control30m_early_acceptance.json', acceptance)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('budget')
    for name in ('gate', 'lines'):
        q = sub.add_parser(name)
        q.add_argument('stage', choices=STAGES)
        q.add_argument('--sha', required=True)
        if name == 'gate':
            q.add_argument('--reviewed')
    sub.add_parser('collect').add_argument('stage', choices=STAGES)
    sub.add_parser('reader-plans').add_argument('--campaign-root', type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == 'prepare': prepare()
    elif a.command == 'budget': print(json.dumps(policy(), indent=2))
    elif a.command == 'gate': return run_gate(a.stage, a.sha, a.reviewed)
    elif a.command == 'lines': print('\n'.join(admitted_lines(a.stage, a.sha)))
    elif a.command == 'collect': collect(a.stage)
    elif a.command == 'reader-plans': reader_plans(a.campaign_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
