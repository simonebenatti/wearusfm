#!/usr/bin/env python3
"""Audit archived Git evidence locally; writes only --out, outside the repo.

No remote queries, no scientific payload reads, and no job operations.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

DELIVERY = '2930d7d6feb6a054ae0ce9fded98ac5ac1b12abe'
SCRIPT = '29dd8ebc829c3db10c4928defbf276df8c76d59d'
SUMMARY = 'c109bfef5a3f80cc4b2334e3554c2171db938467'
BASE = 'results/infra/consolidation_long_20261105'
PRES = BASE + '/preservation'
TAG = 'wearusfm_runs_results_tokenizer_probe_20261103_19618_T100-G'
REL = 'wearusfm_runs/results/tokenizer_probe_20261103/19618/T100-G'
SRC = '/leonardo_scratch/large/userexternal/sbenatti/' + REL
DST = '/leonardo_work/IscrB_WearUsFM/' + REL
LOGICAL_ID = 'LONG-PRESERVE-WORK-T100G-20261107'
JOB = 63785874

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', required=True, type=pathlib.Path)
parser.add_argument('--out', required=True, type=pathlib.Path)
parser.add_argument('--ledger', type=pathlib.Path,
                    default=pathlib.Path.home() / 'wearusfm_local/jobs')
args = parser.parse_args()
root, out = args.repo.resolve(), args.out.resolve()
assert out != root and root not in out.parents, 'Output must be outside repo'
out.mkdir(parents=True, exist_ok=True)
checks = []


def check(ok, name):
    checks.append({'check': name, 'passed': bool(ok)})
    if not ok:
        raise AssertionError(name)


def get(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=root)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(raw, path):
    return {'git_path': path, 'sha256': sha(raw), 'bytes': len(raw)}


def normalize(path):
    check(path == '.' or path.startswith('./'), 'relative_path:' + path)
    name = path[2:] if path.startswith('./') else ''
    check(not pathlib.PurePosixPath(name).is_absolute()
          and '..' not in pathlib.PurePosixPath(name).parts, 'safe_path:' + path)
    return name


manifest_types = ('source.sizes', 'copy.sizes', 'source.dirs', 'copy.dirs',
                  'source.sha256', 'copy.sha256', 'receipt.txt')
raw = {kind: get(DELIVERY, PRES + '/' + TAG + '.' + kind)
       for kind in manifest_types}
receipt = raw['receipt.txt'].decode()
log_path = PRES + '/preserve_long_to_work_63785874.log'
log_raw = get(DELIVERY, log_path)
log = log_raw.decode()
operator_raw = get(DELIVERY, PRES + '/inventory_check_t100g.json')
operator = json.loads(operator_raw)
for kind in ('sizes', 'dirs', 'sha256'):
    check(raw['source.' + kind] == raw['copy.' + kind],
          'source_copy_' + kind + '_byte_equal')

receipt_hashes = {name: digest for digest, name in
                  re.findall(r'^([a-f0-9]{64})  (\S+)$', receipt, re.M)}
check(set(receipt_hashes) == {TAG + '.' + kind for kind in manifest_types
                            if kind != 'receipt.txt'}, 'receipt_exact_six_manifest_names')
for name, digest in receipt_hashes.items():
    check(sha(raw[name[len(TAG) + 1:]]) == digest, 'receipt_hash:' + name)

sizes, hashes, dirs = {}, {}, []
for line in raw['copy.sizes'].decode().splitlines():
    size, path = line.split(' ', 1)
    name = normalize(path)
    check(name not in sizes, 'unique_size:' + name)
    sizes[name] = int(size)
for line in raw['copy.sha256'].decode().splitlines():
    digest, path = line.split('  ', 1)
    name = normalize(path)
    check(name not in hashes and bool(re.fullmatch('[a-f0-9]{64}', digest)),
          'unique_valid_hash:' + name)
    hashes[name] = digest
for line in raw['copy.dirs'].decode().splitlines():
    dirs.append(normalize(line))

check(set(sizes) == set(hashes) and len(sizes) == 26, '26_files_sizes_hashes')
check(len(dirs) == len(set(dirs)) == 6 and '' in dirs, '6_dirs_root_included')
check(sum(sizes.values()) == 23981927, '23981927_total_bytes')
check(all(('' if str(pathlib.PurePosixPath(name).parent) == '.'
           else str(pathlib.PurePosixPath(name).parent)) in dirs
          for name in sizes), 'all_file_parents_listed')
check(SRC in receipt and DST in receipt and SCRIPT in receipt
      and 'job: 63785874' in receipt, 'receipt_expected_roots_script_job')
check('file: 26; byte: 23981927; sha256 identici' in receipt
      and 'riuscite=1 problemi=0 su 1 voci' in log,
      'receipt_and_log_completion_claim')

plan_path = BASE + '/execution/summary_plan.json'
summary_path = BASE + '/execution/receipts/LONG-SUMMARY-20261105.json'
plan_raw, summary_raw = get(SUMMARY, plan_path), get(SUMMARY, summary_path)
plan, summary = json.loads(plan_raw), json.loads(summary_raw)
historical = plan['historical_classification_reports']
check(set(historical) == {'p1', 'p2'}, 'summary_plan_exact_p1_p2_pins')
check(summary['classification_result']['historical_reports']
      == [[historical['p1']], [historical['p2']]],
      'executed_summary_reuses_same_historical_pins')
for task in ('p1', 'p2'):
    entry, name = historical[task], task + '.json'
    check(entry['path'] == SRC + '/' + name and entry['bytes'] == sizes[name]
          and entry['sha256'] == hashes[name], 'historical_pin_preserved:' + name)

check(operator['job'] == JOB and operator['logical_id'] == LOGICAL_ID
      and operator['receipt_manifest_hashes_match'] is True
      and operator['source_equals_copy'] == dict.fromkeys(('sizes', 'dirs', 'sha256'), True)
      and operator['copy_files'] == len(sizes)
      and operator['copy_bytes'] == sum(sizes.values())
      and operator['copy_dirs'] == len(dirs)
      and operator['files'] == sorted(sizes)
      and operator['summary_plan_historical_reports_present'] == {'p1.json': True, 'p2.json': True}
      and operator['verdict'] == 'OK', 'operator_inventory_check_reproduced')

script_path = 'scripts/slurm/preserve_long_to_work.sbatch'
script = get(SCRIPT, script_path)
# The delivery branch carries evidence; the script has a separate pinned commit.
state_raw = (args.ledger.expanduser() / LOGICAL_ID / 'state.json').read_bytes()
state = json.loads(state_raw)
check(state['script_stdin_sha256'] == sha(script) and state['codice_sha'] == SCRIPT,
      'ledger_stdin_hash_matches_pinned_script')
check(state['id'] == LOGICAL_ID and state['job_slurm'] == JOB
      and state['stato'] == 'sottomesso' and state['agente'] == 'claude'
      and state['campagna'] == 'consolidation_long_20261105'
      and [event['stato'] for event in state['storia']]
      == ['prenotato', 'in_sottomissione', 'sottomesso'], 'local_ledger_single_submission')
check(all(flag in state['sbatch'] for flag in (
    '--no-requeue', '--cpus-per-task=1', '--mem=1G', '--time=00:10:00',
    '--export=ALL,PRESERVE_MODE=copy,PRESERVE_SET=t100g-19618,EXPECTED_SHA=' + SCRIPT)),
    'submission_resources_and_explicit_t100g_copy_set')
check(subprocess.run(['bash', '-n'], input=script, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE).returncode == 0, 'bash_syntax')
text = script.decode()
check('cp -r --preserve=mode,timestamps -- "$src" "$dst"' in text,
      'script_requests_mode_and_timestamp_preservation')
check('SET="${PRESERVE_SET:-long-root}"' in text
      and 't100g-19618) ITEMS=( "' + REL + '" ) ;;' in text
      and 't100g-19618:' + REL + ') ;;' in text,
      'closed_t100g_set_and_original_default')

limits = [
    'Local review of immutable Git evidence and the local submitter ledger only; no remote files, quota, permissions, metadata or scheduler state were queried.',
    'Archived source/copy manifests agree on names, sizes, directory tree and every file SHA. Scientific payload bytes were not independently reread or hashed.',
    'p1/p2 manifest hashes and byte counts equal the pins used by the frozen long summary. This checks preservation of those dependencies, not a new statistical evaluation.',
    'cp requests mode/timestamp preservation, but no stat metadata manifests prove their equality. Hash reads may update atime; no explicit original content write, rename or deletion occurs in the production copy path.',
    'EXPECTED_SHA is printed, not verified against a checkout by the stdin script. The local ledger binds the submitted script through script_stdin_sha256.',
    'COMPLETED 0:0, 8 elapsed seconds and 2 allocated CPUs are operator-reported. The ledger proves 1 requested CPU and one submission; no fresh or archived sacct export was independently checked here.',
    'WORK avoids the stated SCRATCH 40-day purge; this is not a guarantee of permanent retention.'
]
result = {
    'verdict': 'PASS_ARCHIVED_T100G_PRESERVATION_EVIDENCE',
    'delivery_commit': DELIVERY, 'script_commit': SCRIPT, 'long_summary_commit': SUMMARY,
    'check_count': len(checks), 'checks': checks,
    'preservation': {'source': SRC, 'destination': DST, 'files': len(sizes),
                     'directories_including_root': len(dirs), 'bytes': sum(sizes.values()),
                     'MiB': sum(sizes.values()) / 1024**2,
                     'source_copy_manifests_byte_identical': True,
                     'oof_npz_files': sum(name.endswith('.oof.npz') for name in sizes),
                     'historical_report_pins': historical},
    'archived_manifest_pins': {kind: pin(content, PRES + '/' + TAG + '.' + kind)
                               for kind, content in raw.items()},
    'other_evidence_pins': {
        'log': pin(log_raw, log_path),
        'operator_inventory': pin(operator_raw, PRES + '/inventory_check_t100g.json'),
        'summary_plan': pin(plan_raw, plan_path), 'summary': pin(summary_raw, summary_path),
        'script': pin(script, script_path),
        'local_submitter_state': {'logical_id': LOGICAL_ID, 'sha256': sha(state_raw),
                                 'bytes': len(state_raw)}},
    'accounting': {'job': JOB, 'scheduler_reported_state': 'COMPLETED 0:0',
                   'elapsed_seconds_operator_reported': 8,
                   'cpus_requested_verified_in_ledger': 1,
                   'cpus_allocated_operator_reported': 2,
                   'allocated_core_hours_conditional_on_reported_values': 8 * 2 / 3600,
                   'gpu_hours': 0, 'requested_walltime_seconds': 600,
                   'allocation_cap_core_hours_if2CPUs': 600 * 2 / 3600,
                   'sacct_independently_read': False},
    'limits': limits
}
(out / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
(out / 'README.md').write_text(
    '# Verifica locale della conservazione T100-G\n\n'
    '**Esito: nessuna discrepanza nei manifest o nei pin storici.**\n\n'
    f'Consegna `{DELIVERY}`, script `{SCRIPT}`. Verificati 26 file, 6 directory '
    '(radice inclusa), 23.981.927 byte. Le tre coppie di manifest sorgente/copia '
    'sono identiche e i sei hash della ricevuta coincidono. SHA e dimensioni '
    'di `p1.json` e `p2.json` sono quelli pinnati nel piano e nel report finale '
    'della valutazione lunga. Presenti sette OOF NPZ (cinque shard e due aggregate).\n\n'
    'Il ledger locale documenta un solo invio del job 63785874 e lega lo stdin '
    'allo script esatto. Il costo 0,004444 coreh deriva da 8 s × 2 CPU allocate, '
    'valori riportati dall’operatore; il ledger conferma una CPU richiesta. '
    'Nessuna verifica sacct o lettura remota eseguita.\n\n'
    'I manifest provano la coerenza documentata di contenuti e struttura, '
    'non lo stato remoto attuale. Modi e date sono richiesti dal comando di copia '
    'ma non confrontati indipendentemente.\n\n'
    'Riproduzione: `python3 audit.py --repo <repo-con-gli-oggetti-Git> '
    '--out <directory-esterna> [--ledger <ledger-locale>]`. '
    'Altri limiti in `audit.json`.\n')
print(json.dumps({key: result[key] for key in
                  ('verdict', 'check_count', 'preservation', 'accounting')}, indent=2))
