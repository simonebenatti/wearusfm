import json, sys
d=json.load(open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/tasks/w785grg3y.output'))
r=d['result']
rows=[]
for p in r:
    rev={}
    for rv in p['reviews']:
        for f in rv.get('findings',[]):
            rev[f['id'] if 'id' in f else f.get('finding_id')]=f
    v=p['verify']
    print('=====',p['pair'])
    print('point_verdicts:')
    for pv in v['point_verdicts']:
        print('  ',json.dumps(pv,ensure_ascii=False)[:600])
    for vd in v['verdicts']:
        fid=vd['finding_id']
        f=rev.get(fid,{})
        print(f"{fid:28s} REV {f.get('severity','?')}/{f.get('kind','?')} -> VER {vd['verdict']} {vd['severity']}/{vd['kind']} :: {f.get('title','')[:110]}")
    print('new_findings:')
    for nf in v['new_findings']:
        print('  ',nf.get('id'),nf.get('severity'),nf.get('kind'),'::',nf.get('title','')[:150])
    print('seeded_claims:')
    for sc in v['seeded_claims']:
        print('  ',json.dumps(sc,ensure_ascii=False)[:400])
