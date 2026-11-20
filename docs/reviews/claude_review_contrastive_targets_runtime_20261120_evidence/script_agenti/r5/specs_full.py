import json
ns = {}
exec(open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/r5/specs_main.py').read(), ns)
rows = [json.loads(l) for f in ('results_a.jsonl', 'results_b.jsonl') for l in open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/r5/' + f)]
surv = {r['id'] for r in rows if r.get('rc') == 0}
SPECS = [{**s, 'id': s['id'] + '__ALL', 'tests': 'ALL'} for s in ns['SPECS'] if s['id'] in surv]
