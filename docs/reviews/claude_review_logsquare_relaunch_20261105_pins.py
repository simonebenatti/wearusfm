"""Recompute every path/sha256 pin found in JSON files, against the content of one commit.

usage: check_pins.py <repository> <commit> <json path in the commit>...
"""
import hashlib
import json
import subprocess
import sys

repo, commit, targets = sys.argv[1], sys.argv[2], sys.argv[3:]


def blob(path):
    run = subprocess.run(['git', '-C', repo, 'show', f'{commit}:{path}'], capture_output=True)
    return run.stdout if run.returncode == 0 else None


def pins(node, where=''):
    if isinstance(node, dict):
        if isinstance(node.get('path'), str) and isinstance(node.get('sha256'), str):
            yield where, node['path'], node['sha256'], node.get('bytes')
        for key, value in node.items():
            if key.endswith('sha256') and isinstance(value, dict):
                for path, digest in value.items():
                    if isinstance(digest, str) and len(digest) == 64:
                        yield f'{where}/{key}', path, digest, None
            yield from pins(value, f'{where}/{key}')
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from pins(value, f'{where}[{index}]')


for target in targets:
    data = blob(target)
    if data is None:
        print(f'{target}: absent in {commit}')
        continue
    seen, good, remote, bad = set(), 0, 0, []
    for where, path, digest, size in pins(json.loads(data)):
        if (where, path) in seen:
            continue
        seen.add((where, path))
        if path.startswith('/'):
            remote += 1
            continue
        content = blob(path)
        if content is None:
            bad.append((where, path, 'absent'))
        elif hashlib.sha256(content).hexdigest() != digest:
            bad.append((where, path, 'sha256 differs: pinned ' + digest[:10] + ' file ' + hashlib.sha256(content).hexdigest()[:10]))
        elif size is not None and size != len(content):
            bad.append((where, path, f'bytes differ: pinned {size} file {len(content)}'))
        else:
            good += 1
    print(f'{target} @ {commit[:7]}: {good} pins match, {remote} remote paths not checked, {len(bad)} do not match')
    for item in bad:
        print('    ', item)
