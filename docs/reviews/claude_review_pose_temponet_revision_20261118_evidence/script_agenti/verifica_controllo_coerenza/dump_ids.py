import json, sys
d = json.load(open(sys.argv[1]))
want = set(sys.argv[2].split(','))
r = d['result']
for p in r:
    for rv in p['reviews']:
        for f in rv['findings']:
            if f['id'] in want:
                print('=== FINDING', f['id'], 'dim', rv['dimension'])
                print(json.dumps(f, ensure_ascii=False, indent=1))
    for v in p['verify']['verdicts']:
        if v['finding_id'] in want:
            print('=== VERDICT', v['finding_id'])
            print(json.dumps(v, ensure_ascii=False, indent=1))
    for nf in p['verify']['new_findings']:
        if nf['id'] in want:
            print('=== NEW', nf['id'])
            print(json.dumps(nf, ensure_ascii=False, indent=1))
    for sc in p['verify']['seeded_claims']:
        if sc['claim_id'] in want:
            print('=== SEED', sc['claim_id'])
            print(json.dumps(sc, ensure_ascii=False, indent=1))
