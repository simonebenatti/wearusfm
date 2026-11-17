"""Reviewer cross-check: proposal.json pins vs archived inventory in GPT tree (no cluster access)."""
import json, sys
from pathlib import Path
snap = Path(sys.argv[1])
prop = json.loads((snap/'results/infra/pose_temponet_proposal_20261117/proposal.json').read_text())
inv_path = snap/'results/infra/consolidation_long_20261105/gpt_review_20261107/preservation_inventory.json'
inv = json.loads(inv_path.read_text())
print('inventory top keys:', list(inv.keys())[:20] if isinstance(inv, dict) else type(inv))
def walk(o, path=''):
    if isinstance(o, dict):
        if 'sha256' in o and ('path' in o or 'relative_path' in o or 'name' in o):
            yield path, o
        for k, v in o.items():
            yield from walk(v, path + '/' + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + f'[{i}]')
entries = list(walk(inv))
print('entries with sha256:', len(entries))
bysha = {}
for p, e in entries:
    bysha.setdefault(e['sha256'], []).append((p, e))
wanted = {
 'caches.train': prop['caches']['train'], 'caches.val': prop['caches']['val'],
 **{'references.'+k: v for k, v in prop['references'].items()},
}
for name, pin in wanted.items():
    hit = bysha.get(pin['sha256'])
    if not hit:
        print(name, 'SHA NOT FOUND in inventory', pin['sha256'][:16]); continue
    for p, e in hit:
        b = e.get('bytes', e.get('size'))
        print(name, 'FOUND', 'bytes_equal=' + str(b == pin['bytes']), b, '|', e.get('path') or e.get('relative_path') or e.get('name'))
for k in ('checkpoint_sha256', 'stats_sha256'):
    hit = bysha.get(prop[k])
    print(k, 'FOUND' if hit else 'NOT FOUND in inventory', [ (e.get('path') or e.get('relative_path') or e.get('name')) for _, e in (hit or [])][:3])
