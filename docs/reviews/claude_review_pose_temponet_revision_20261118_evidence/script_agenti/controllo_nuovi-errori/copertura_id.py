import json,re
d=json.load(open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/tasks/w785grg3y.output'))
bozza=open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo/bozza_controllo.md').read().split('\n')
ids=[]
for p in d['result']:
    v=p['verify']
    for x in v['verdicts']: ids.append((x['finding_id'],x['severity']))
    for x in v['new_findings']: ids.append((x['id'],x['severity']))
    for x in v['seeded_claims']: ids.append((x['claim_id'],x['severity']+'/'+x['verdict']))
for i,s in ids:
    # match backticked id or bare id not preceded by '32e5452:'
    pat=re.compile(r'(?<![\w:-])'+re.escape(i)+r'(?![\w-]|\d)')
    lines=[n+1 for n,l in enumerate(bozza) if pat.search(l) and ('32e5452:'+i) not in l or (pat.search(l.replace('32e5452:'+i,'')) )]
    lines=sorted(set(n+1 for n,l in enumerate(bozza) if pat.search(l.replace('32e5452:'+i,''))))
    print(f"{i:28s} {s:22s} righe {lines}")
