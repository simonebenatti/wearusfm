#!/usr/bin/env python3
"""Rebuild or check the input-compatibility manifest of amendment A1.

Everything comes from Git objects of the two original phase commits and from the
receipts, audits and cache manifest already published under execution/. No
cluster access, no submission, nothing rewritten. Run from anywhere:

    python3 results/infra/consolidation_long_20261105/amendment_a1/build_inputs.py --check
    python3 results/infra/consolidation_long_20261105/amendment_a1/build_inputs.py --write   (refuses to overwrite)
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import consolidation_long_amendment as A  # noqa: E402  (light module: no torch, no data)

PROTOCOL = 'consolidation-long-evaluation-20261105-v1'
CAMPAIGN = 'results/infra/consolidation_long_20261105'
TARGET = HERE / 'input_compatibility.json'
FIELDS = {
    'extract-classification': ('checkpoint_step', 'checkpoint_sha256', 'task', 'artifacts'),
    'extract-pose-train': ('checkpoint_step', 'checkpoint_sha256', 'split', 'artifacts'),
    'extract-pose-val': ('checkpoint_step', 'checkpoint_sha256', 'split', 'artifacts'),
    'benchmark-classification': ('checkpoint_step', 'checkpoint_sha256', 'dataset', 'view',
                                 'extraction_receipt', 'artifacts'),
    'benchmark-pose': ('checkpoint_step', 'checkpoint_sha256', 'cache_manifest', 'train_cache', 'artifacts'),
}


def need(ok, message):
    if not ok:
        raise SystemExit('build_inputs: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def covered(path):
    return (path.startswith('src/') and path.endswith('.py')
            or path.startswith('scripts/') and path.count('/') == 1 and path.endswith('.py')
            or path in (A.LAUNCHER, 'scripts/slurm/lib_code_dir.sh'))


def source_map(commit, cache={}):
    """Path -> SHA-256 of every covered file of a commit, as the runtime enumerates a clean checkout."""
    found = {}
    for line in git('ls-tree', '-r', commit, '--', 'src', 'scripts').decode().splitlines():
        meta, path = line.split('\t', 1)
        kind, oid = meta.split()[1:3]
        if kind == 'blob' and covered(path):
            if oid not in cache:
                cache[oid] = hashlib.sha256(git('cat-file', 'blob', oid)).hexdigest()
            found[path] = cache[oid]
    return dict(sorted(found.items()))


def show(commit, path):
    return git('show', f'{commit}:{path}')


def build():
    phases = A.ORIGINAL_PHASES
    maps = {name: source_map(phase['code_sha']) for name, phase in phases.items()}
    need(maps['extract'] == maps['benchmark'], 'the two original phases do not share one source map')
    files = maps['benchmark']
    need(digest(files) == A.ORIGINAL_SOURCE, 'original source identity is not reproduced from Git')
    for name, phase in phases.items():
        raw = show(phase['code_sha'], CAMPAIGN + '/plan.json')
        need(hashlib.sha256(raw).hexdigest() == phase['plan_sha256'], name + ': phase plan hash differs')
        old = json.loads(raw)
        need(old['source_sha256'] == A.ORIGINAL_SOURCE and old['recipe_sha256'] == A.ORIGINAL_RECIPE,
             name + ': phase plan identities differ')
    commit = phases['benchmark']['code_sha']
    cache_pin = json.loads(show(commit, CAMPAIGN + '/plan.json'))['cache_manifest']
    cache_raw = (ROOT / cache_pin['path']).read_bytes()
    need(hashlib.sha256(cache_raw).hexdigest() == cache_pin['sha256'] and len(cache_raw) == cache_pin['bytes']
         and cache_raw == show(commit, cache_pin['path']), 'cache manifest differs from the benchmark phase')
    caches = json.loads(cache_raw)
    need(caches['source_sha256'] == A.ORIGINAL_SOURCE and caches['recipe_sha256'] == A.ORIGINAL_RECIPE,
         'cache manifest identities differ')

    texts = {path: show(commit, path).decode() for path in A.CHANGED_PATHS}
    definitions = {path: A.definitions(texts[path]) for path in A.AMENDED_DEFINITIONS}
    order = {path: list(value) for path, value in definitions.items()}
    for path, text in texts.items():
        if path in A.AMENDED_DEFINITIONS:
            parts = list(A.regions(text).values())
            need('\n'.join(parts[:-1]) + parts[-1] == text, path + ': the regions do not rebuild the original file')
    blocks = {path: A.blocks(texts[path], spec) for path, spec in A.FROZEN_BLOCKS.items()}
    lines = texts[A.LAUNCHER].split('\n')
    first, last = A.LAUNCHER_REPLACED_LINES
    frame = {'path': A.LAUNCHER, 'lines': len(lines) - 1, 'replaced_lines': [first, last],
             'prefix_lines': first - 1, 'suffix_lines': len(lines) - last}
    frame.update(A.launcher_frame(texts[A.LAUNCHER], frame['prefix_lines'], frame['suffix_lines']))

    launch = json.loads((ROOT / CAMPAIGN / 'launch_plan.json').read_text())
    audits = {}
    for name in phases:
        audits.update(json.loads((ROOT / CAMPAIGN / f'execution/audit/receipts_{name}.json').read_text())['receipts'])
    receipts = []
    for row in launch['jobs']:
        if row['phase'] not in phases:
            continue
        raw = (ROOT / CAMPAIGN / 'execution/receipts' / (row['id'] + '.json')).read_bytes()
        audit, report, phase = audits[row['id']], json.loads(raw), phases[row['phase']]
        need(audit['remote_path'] == row['out'] + '/report.json'
             and hashlib.sha256(raw).hexdigest() == audit['sha256'] and len(raw) == audit['bytes'],
             row['id'] + ': collected copy differs from the audited remote receipt')
        need(report['complete'] is True and report['protocol'] == PROTOCOL and report['kind'] == row['stage']
             and report['source_sha256'] == A.ORIGINAL_SOURCE and report['recipe_sha256'] == A.ORIGINAL_RECIPE
             and report['code_sha'] == phase['code_sha'] and report['plan_sha256'] == phase['plan_sha256'],
             row['id'] + ': receipt provenance differs')
        receipts.append({'id': row['id'], 'phase': row['phase'], 'kind': report['kind'],
                         'path': audit['remote_path'], 'sha256': audit['sha256'], 'bytes': audit['bytes'],
                         'fields': {key: report[key] for key in FIELDS[report['kind']]}})
    listed = {(r['path'], r['sha256'], r['bytes']): r for r in receipts}
    payloads = []
    for task in ('classification', 'pose'):
        for step, entries in caches[task].items():
            for entry in entries.values():
                pin = entry['receipt']
                receipt = listed.get((pin['path'], pin['sha256'], pin['bytes']))
                need(receipt is not None and receipt['phase'] == 'extract'
                     and receipt['fields']['checkpoint_step'] == int(step)
                     and entry['cache'] in receipt['fields']['artifacts'].values(),
                     'cache manifest entry is not covered by a listed extraction receipt')
                payloads.append(entry['cache']['sha256'])
    need(len(payloads) == len(set(payloads)) == 21, 'cache manifest does not hold 21 distinct payloads')

    value = {
        'protocol': PROTOCOL, 'amendment': A.AMENDMENT, 'kind': 'input-compatibility',
        'scope': 'exception to R-2(i) limited to the 28 receipts listed here: they were written by the original '
                 'source and are read, never rewritten, by the amended source; no other source, receipt or input '
                 'is admitted',
        'source_sha256': A.ORIGINAL_SOURCE, 'recipe_sha256': A.ORIGINAL_RECIPE, 'phases': phases,
        'cache_manifest': cache_pin, 'cache_payloads': len(payloads),
        'receipts': receipts, 'launcher': frame, 'blocks': blocks, 'definitions': definitions, 'order': order,
        'source_files': files,
    }
    need(len(files) == 227, 'original source map does not hold 227 files')
    A.require(len(receipts) == 28, 'the closed list does not hold 28 receipts')
    return json.dumps(value, indent=1) + '\n'


if __name__ == '__main__':
    mode = sys.argv[1:] or ['--check']
    need(mode in (['--check'], ['--write']), 'usage: build_inputs.py [--check|--write]')
    text = build()
    if mode == ['--write']:
        with TARGET.open('x') as stream:
            stream.write(text)
    need(TARGET.read_text() == text, 'committed manifest differs from the one rebuilt from Git and receipts')
    raw = TARGET.read_bytes()
    pin = {'path': str(TARGET.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    need(mode == ['--write'] or pin == A.INPUT_MANIFEST, 'manifest pin differs from the one frozen in the module')
    print(json.dumps(pin))
