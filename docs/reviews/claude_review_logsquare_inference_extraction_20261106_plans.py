"""Leaf-by-leaf comparison of JSON plans taken from commits, plus the benchmark-context hash.

usage: claude_review_logsquare_inference_extraction_20261106_plans.py <repository> <commitA:path> <commitB:path> [A B ...]

The context hash is rewritten from `benchmark_context` in scripts/logsquare_evaluation.py: it must be the same for the
plan the benchmark ran on and for the later extraction plan, and equal to `benchmark_context_sha256` in the receipt.
"""
import hashlib
import json
import subprocess
import sys

repo = sys.argv[1]


def load(ref):
    return json.loads(subprocess.run(['git', '-C', repo, 'show', ref], capture_output=True, check=True).stdout)


def leaves(node, prefix=''):
    if isinstance(node, dict):
        if not node:
            yield prefix.rstrip('.'), {}
        for key, value in node.items():
            yield from leaves(value, prefix + str(key) + '.')
    elif isinstance(node, list):
        yield prefix.rstrip('.'), json.dumps(node, sort_keys=True)
    else:
        yield prefix.rstrip('.'), node


def context_sha256(plan):
    keys = ('protocol', 'arm', 'inputs', 'paths', 'pose_references', 'classification_references',
            'training_plan', 'candidate_config', 'source_sha256')
    value = {key: plan[key] for key in keys}
    value['checkpoint_5000'] = plan['checkpoints']['5000']
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


pairs = sys.argv[2:]
for a_ref, b_ref in zip(pairs[::2], pairs[1::2]):
    a, b = dict(leaves(load(a_ref))), dict(leaves(load(b_ref)))
    same = [k for k in a if k in b and a[k] == b[k]]
    changed = [k for k in a if k in b and a[k] != b[k]]
    removed = [k for k in a if k not in b]
    added = [k for k in b if k not in a]
    print(f'=== {a_ref}  ->  {b_ref}')
    print(f'    {len(a)} -> {len(b)} leaves: {len(same)} identical, {len(changed)} changed, {len(removed)} removed, {len(added)} added')
    for k in changed:
        print(f'    CHANGED {k}: {str(a[k])[:70]} -> {str(b[k])[:90]}')
    for k in removed:
        print(f'    REMOVED {k}: {str(a[k])[:90]}')
    groups = {}
    for k in added:
        groups.setdefault(k.split('.')[0], []).append(k)
    for head, keys in groups.items():
        if len(keys) > 12:
            print(f'    ADDED {head}.*: {len(keys)} leaves')
        else:
            for k in keys:
                print(f'    ADDED {k}: {str(b[k])[:90]}')
    for ref in (a_ref, b_ref):
        plan = load(ref)
        if all(key in plan for key in ('inputs', 'pose_references', 'checkpoints', 'source_sha256')) and plan['checkpoints'].get('5000'):
            print(f'    benchmark context sha256 of {ref.split(":")[0]}: {context_sha256(plan)}')
