import json
R = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/r5/'
ns = {}; exec(open(R + 'specs_extra.py').read(), ns)
nm = {}; exec(open(R + 'specs_m1d.py').read(), nm)
by = {s['id']: s for s in ns['SPECS'] + nm['SPECS']}
rows = [json.loads(l) for f in ('results_extra.jsonl', 'results_m1d.jsonl') for l in open(R + f)]
surv = [r['id'] for r in rows if r.get('rc') == 0 and not r['id'].endswith('__ALL')]
edits = [e for i in surv for e in by[i]['edits']]
SPECS = [{'id': 'EXTRA_SURVIVORS_COMBINED__ALL', 'edits': edits, 'tests': 'ALL', 'members': surv}]
