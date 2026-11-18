import json, sys
d = json.load(open(sys.argv[1]))
r = d['result']
rows = []
for p in r:
    rev = {}
    for rv in p['reviews']:
        for f in rv['findings']:
            rev[f['id']] = (rv['dimension'], f['severity'], f.get('kind'), f['title'])
    ver = {v['finding_id']: v for v in p['verify']['verdicts']}
    for fid, (dim, sev, kind, title) in rev.items():
        v = ver.get(fid, {})
        rows.append((fid, 'rev', sev, kind, v.get('verdict'), v.get('severity'), v.get('kind'), title[:90]))
    for fid in ver:
        if fid not in rev:
            print('VERDICT WITHOUT FINDING', fid)
    for nf in p['verify']['new_findings']:
        rows.append((nf['id'], 'new', None, None, 'NUOVO', nf['severity'], nf.get('kind'), nf['title'][:90]))
    for sc in p['verify']['seeded_claims']:
        rows.append((sc['claim_id'], 'seed', None, None, sc['verdict'], sc['severity'], sc.get('kind'), sc.get('reason','')[:90]))
    print('POINT VERDICTS', p['pair'], [(pv['point'][:40], pv['verdict']) for pv in p['verify']['point_verdicts']])
for row in rows:
    print('\t'.join(str(x) for x in row))
from collections import Counter
fin = [x for x in rows if x[1] in ('rev','new')]
print('N findings', len(fin))
print('final severity', Counter(x[5] for x in fin))
print('verdicts rev', Counter(x[4] for x in rows if x[1]=='rev'))
print('seed verdicts', Counter(x[4] for x in rows if x[1]=='seed'))
print('final kind', Counter(x[6] for x in fin))
