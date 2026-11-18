import json,re
d=json.load(open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/tasks/w785grg3y.output'))
bozza=open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo/bozza_controllo.md').read().split('\n')
rows=[]
for p in d['result']:
    v=p['verify']
    for x in v['verdicts']: rows.append((x['finding_id'],x['verdict'],x['severity'],x['kind']))
    for x in v['new_findings']: rows.append((x['id'],'NUOVO',x['severity'],x['kind']))
    for x in v['seeded_claims']: rows.append((x['claim_id'],x['verdict'],x['severity'],x['kind']))
for i,ver,s,k in rows:
    pat=re.compile(r'(?<![\w:-])'+re.escape(i)+r'(?![\w-])')
    lines=sorted(set(n+1 for n,l in enumerate(bozza) if pat.search(l.replace('32e5452:'+i,''))))
    print(f"{i:28s} {ver:10s} {s:10s} {k:16s} righe {lines}")
