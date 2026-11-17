"""Read-only audit of actual mixed P1-r2/pose/P2 metadata and frozen readers.

No remote access, cache opening, fits, scientific score reads, or repo writes.
The audit JSON and two candidate-plan JSON files are written alongside this script.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-p1-recovery'))
    p.add_argument('--output', type=Path, default=Path('/private/tmp/p1-r2-provenance-review-20261117.json'))
    a = p.parse_args()
    root = a.root.resolve()
    base = root/'results/infra/scaling_300m_20261104'
    evidence = base/'p1_recovery_readout_20261117/evidence'
    sys.path.insert(0, str(root/'scripts'))
    sys.path.insert(0, str(root/'src'))
    import scaling_p1_recovery_plans as M
    S, R = M.S, M.R
    checks = []
    def check(name, ok, details=None):
        checks.append(dict(name=name, passed=bool(ok), details=details))
        if not ok:
            raise ValueError(name)
    def read(path):
        return json.loads(Path(path).read_bytes())
    snap = read(evidence/'snapshot.json')
    def small(key):
        row = snap['files'][key]
        data = (evidence/key).read_bytes()
        check('collected bytes '+key, data == row['text'].encode() and len(data) == row['bytes'] and sha(data) == row['sha256'])
        return data, {k: row[k] for k in ('path', 'sha256', 'bytes')}
    for key, value in snap['commands'].items():
        check('remote command '+key, type(value['exit']) is int and value['exit'] == 0)
    release = read(base/'p1_recovery_release_20261116/submission_plan.json')
    recovery_code = release['code_sha']
    check('actual new code checkout', snap['commands']['code_head']['stdout'].strip() == recovery_code and
          snap['commands']['code_status']['stdout'].strip() == '')
    check('actual historical code checkout', snap['commands']['old_code_head']['stdout'].strip() == M.HISTORICAL_CODE_SHA and
          snap['commands']['old_code_status']['stdout'].strip() == '')
    raw_manifest, manifest_pin = small('release/recovery_manifest.json')
    manifest = json.loads(raw_manifest)
    check('deployed recovery manifest equals published release', manifest_pin['sha256'] == release['manifest_sha256'] and
          raw_manifest == (base/'p1_recovery_release_20261116/recovery_manifest.json').read_bytes())
    raw_extract, extract_pin = small('release/extraction_plan.json')
    extraction = json.loads(raw_extract)
    check('original extraction plan preserved', extract_pin == manifest['extraction_plan'] and
          raw_extract == (base/'continuation19618_20261106/evaluation/extraction_plan.json').read_bytes())
    raw_contract, contract_pin = small('release/training_source_contract.json')
    contract = json.loads(raw_contract)
    check('scientific source contract preserved', raw_contract == (base/'training_source_contract.json').read_bytes())
    remote_sources = snap['sources']
    py_sources = {n: d for n, d in contract['scientific_source_sha256'].items() if n.endswith('.py')}
    check('frozen numerical Python source bytes', all(remote_sources.get(n) == d for n, d in py_sources.items()), len(py_sources))
    check('recovery source pins in deployed checkout', all(remote_sources.get(n) == d for n, d in manifest['source_sha256'].items()
          if n.startswith('scripts/')))
    check('unchanged output guard since diagnostic', remote_sources['scripts/scaling_output_identity.py'] ==
          '717362c25cb99151ab058b4f212cc73611a36899f67251952ca613a71058e656')
    rows = {row[0]: row for row in (line.split('|') for line in snap['commands']['sacct']['stdout'].splitlines()) if row[0].isdigit()}
    jobs = {'pose': '64571999', 'p1': '64620434', 'p2': '64572001'}
    completed = {}
    for task, job in jobs.items():
        row = rows[job]
        check('accepted scheduler job '+task, row[1:3] == ['COMPLETED', '0:0'] and row[4] == '8' and 'gres/gpu=1' in row[5])
        completed[task] = {'job_id': job, 'state': row[1], 'exit_code': row[2],
                           'code_sha': recovery_code if task == 'p1' else M.HISTORICAL_CODE_SHA}
    check('historical failed attempts remain failed', all(rows[j][1:3] == ['FAILED', '1:0'] for j in ('64572000', '64596915')))
    receipt_bytes, receipt_pins, provenances = {}, {}, {}
    for task in ('pose', 'p1', 'p2'):
        entry = snap['tasks'][task]
        receipt_bytes[task], receipt_pins[task] = small(entry['receipt_key'])
        receipt = json.loads(receipt_bytes[task])
        provenances[task] = {}
        for name, row in entry['outputs'].items():
            data, pp = small(row['provenance_key'])
            provenances[task][name] = data
            prov, output = json.loads(data), receipt['outputs'][name]
            check('live cache canonical size '+task+'/'+name,
                  snap['cache_states'][output['path']]['size'] == output['bytes'] and
                  not snap['cache_states'][output['path']]['symlink'] and
                  snap['cache_states'][output['path']]['canonical'] == output['path'])
            for key in ('protocol', 'arm', 'which', 'checkpoint_step', 'checkpoint_sha256', 'code_sha',
                        'plan_sha256', 'training_source_contract_sha256', 'model_config'):
                check('sidecar identity '+task+'/'+name+'/'+key, prov.get(key) == receipt.get(key))
    args = dict(receipt_bytes=receipt_bytes, receipt_pins=receipt_pins, provenances=provenances,
                completed_jobs=completed, checkpoint=extraction['checkpoints'][R.ARM],
                extraction_plan_sha256=extract_pin['sha256'], recovery_code_sha=recovery_code,
                recovery_manifest_sha256=manifest_pin['sha256'], recovery_sources=manifest['source_sha256'])
    pose, probes, lineage = M.mixed_candidate_caches(**args)
    check('mixed builder accepts actual 3 receipts and 7 provenance files', len(pose) == 2 and len(probes) == 5)
    check('P1 uses actual new producer and P2 uses historical producer', all(probes[ds]['extraction_code_sha'] ==
          (recovery_code if task == 'p1' else M.HISTORICAL_CODE_SHA) for task, datasets in S.TASKS.items() for ds in datasets))
    check('pose keeps historical producer', all(v['extraction_code_sha'] == M.HISTORICAL_CODE_SHA for v in pose.values()))
    check('new criterion does not claim reproduction', lineage['release_criterion'] == R.RELEASE_CRITERION and
          lineage['diagnostic_outcome'] == 'NOT_REPRODUCED' and lineage['diagnostic_report'] == manifest['diagnostic_report'])
    control_inventory = read(root/'results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json')
    control_checkpoint = read(root/S.RECOVERY['plan'])['control']['checkpoint']
    control_provenances = {ds: (base/'continuation19618_20261106/control19618_identity'/f'{ds}.npz.provenance.json').read_bytes()
                           for ds in S.DATASETS}
    control = S.control_probe_caches(control_inventory, control_provenances, control_checkpoint)
    probe_plan = S.probe_plan(read(base/'cpu_readers_20261105/probe_plan.json'), control, probes)
    pose_inventory = read(root/'results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json')
    pose_plan = S.pose_plan(read(base/'cpu_readers_20261105/pose_plan.json'), pose_inventory, pose)
    check('frozen reader plan validators accept mixed provenance', probe_plan['step'] == pose_plan['step'] == 19618)
    check('30M pose control byte-for-byte metadata preserved', pose_plan['caches'][S.ARMS[0]] == pose_inventory['caches']['T100-G'])
    for ds, cache in control.items():
        old = control_inventory['caches']['T100-G'][ds]
        check('30M classification anchor '+ds, all(cache[k] == old[k] for k in old if k in cache) and
              cache['checkpoint_sha256'] == S.E.CONTROL_CHECKPOINT_SHA256)
    check('classification recipe, references, folds remain registered',
          probe_plan['datasets'] == read(base/'cpu_readers_20261105/probe_plan.json')['datasets'] and
          probe_plan['references'] == read(base/'cpu_readers_20261105/probe_plan.json')['references'])
    check('pose reference support and folds remain registered',
          pose_plan['references'] == pose_inventory['references'] and pose_plan['fold_users'] == pose_inventory['fold_users'])
    source_sets = {**S.C.source_hashes(), **S.P.source_hashes()}
    frozen_mismatches = []
    for name, digest in source_sets.items():
        old_bytes = subprocess.check_output(['git', '-C', str(root), 'show', M.HISTORICAL_CODE_SHA+':'+name])
        if sha(old_bytes) != digest or (name.endswith('.py') and remote_sources.get(name) != digest):
            frozen_mismatches.append(name)
    check('all frozen reader source bytes unchanged', not frozen_mismatches, len(source_sets))
    single_producer_rejections = []
    for alleged_code in (M.HISTORICAL_CODE_SHA, recovery_code):
        try:
            S.candidate_probe_caches({t: json.loads(receipt_bytes[t]) for t in S.TASKS},
                {t: receipt_pins[t] for t in S.TASKS},
                {ds: provenances[t][ds] for t, datasets in S.TASKS.items() for ds in datasets},
                {'checkpoint': args['checkpoint'], 'code_sha': alleged_code, 'plan_sha256': extract_pin['sha256']})
            rejected = False
        except ValueError as error:
            rejected = True
            single_producer_rejections.append({'claimed_code_sha': alleged_code, 'reason': str(error)})
        check('single-producer historical builder correctly refuses mixed evidence '+alleged_code[:8], rejected)
    mutant_results = []
    for kind in ('failed_p1_adoption', 'wrong_p1_code', 'wrong_checkpoint', 'missing_criterion', 'wrong_diagnostic'):
        bad = copy.deepcopy(args)
        if kind == 'failed_p1_adoption':
            bad['completed_jobs']['p1']['job_id'] = '64596915'
        elif kind == 'wrong_p1_code':
            bad['completed_jobs']['p1']['code_sha'] = M.HISTORICAL_CODE_SHA
        elif kind == 'wrong_checkpoint':
            bad['checkpoint']['sha256'] = '0'*64
        else:
            r = json.loads(bad['receipt_bytes']['p1'])
            if kind == 'missing_criterion': r.pop('release_criterion')
            else: r['diagnostic_report']['sha256'] = '0'*64
            data = json.dumps(r).encode()
            bad['receipt_bytes']['p1'] = data
            bad['receipt_pins']['p1'] = {**bad['receipt_pins']['p1'], 'bytes': len(data), 'sha256': sha(data)}
        try:
            M.mixed_candidate_caches(**bad)
            rejected = False
        except ValueError:
            rejected = True
        check('mixed provenance mutant rejected '+kind, rejected)
        mutant_results.append({'mutation': kind, 'rejected': rejected})
    report = {'protocol': 'p1-r2-mixed-provenance-review-20261117-v1', 'passed': True,
              'evidence_sha256': sha((evidence/'snapshot.json').read_bytes()),
              'reviewer_code_sha': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              'checks': checks, 'lineage': lineage, 'pose_cache_pins': pose, 'probe_cache_pins': probes,
              'candidate_pose_plan': pose_plan, 'candidate_probe_plan': probe_plan,
              'mutations': mutant_results, 'old_single_producer_rejections': single_producer_rejections,
              'old_controller_p1_job': S.RELAUNCH['id'],
              'payload_hash_verification': 'Not rerun here: producer receipts contain full hashes; collection checks live canonical metadata/size only. Reader validation will rehash payloads on compute.',
              'residual_gates': [
                'Publish mixed lineage and complete pose/probe plans with actual receipt/cache pins, retaining original 30M anchors.',
                'Do not call historical build_reader_plans: its table still points to failed p1-r1 and assumes one producer code_sha.',
                'Freeze/document the mixed orchestration bridge; do not satisfy old release_check/rereview_check by changing historical evidence or source pins.',
                'Fresh completed checkout, published-pin, output absence, ledger, serial capacity and quota gates before the registered reader stage.',
                'Run frozen pose validation and train-only pose/probe benchmarks before fitting; rehash actual inputs and retain unchanged final guards.',
                'Compute and publish runtime/RSS packing against the approved cumulative CPU envelope before any fit; stop on first gate failure/timeout, no retry.'],
              'scientific_readout': 'Not performed; no fitting or model-quality conclusions in this audit.'}
    for kind, plan in (('pose', pose_plan), ('probe', probe_plan)):
        destination = a.output.parent / ('p1-r2-'+kind+'-plan-20261117.json')
        destination.write_text(json.dumps(plan, indent=2, sort_keys=True)+'\n')
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'output': str(a.output), 'candidate_caches': 7, 'control_caches': 7}))


if __name__ == '__main__':
    main()
