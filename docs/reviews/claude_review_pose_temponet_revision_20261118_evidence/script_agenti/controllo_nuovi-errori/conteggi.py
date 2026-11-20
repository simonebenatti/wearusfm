import json
from collections import Counter
d=json.load(open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/tasks/w785grg3y.output'))
r=d['result']
nrev=0; verd=Counter(); sev_after=Counter(); seeded=Counter(); newf=[]
sev_list=[]
for p in r:
    ids=[]
    for rv in p['reviews']:
        for f in rv.get('findings',[]):
            nrev+=1; ids.append(f['id'])
    v=p['verify']
    vids=[x['finding_id'] for x in v['verdicts']]
    assert sorted(vids)==sorted(ids),(p['pair'],set(ids)^set(vids))
    for x in v['verdicts']:
        verd[x['verdict']]+=1
        if x['verdict']!='RESPINTO':
            sev_after[x['severity']]+=1; sev_list.append((x['finding_id'],x['severity'],x['kind']))
    for nf in v['new_findings']:
        newf.append(nf['id']); sev_after[nf['severity']]+=1; sev_list.append((nf['id'],nf['severity'],nf['kind']))
    for sc in v['seeded_claims']:
        seeded[sc['verdict']]+=1
print('review findings',nrev)
print('verdicts',dict(verd))
print('new findings',len(newf),newf)
print('seeded',dict(seeded))
print('sev after (63)',dict(sev_after), sum(sev_after.values()))
print('bloccanti',[s for s in sev_list if s[1]=='bloccante'])
print('minori',[s[0] for s in sev_list if s[1]=='minore'])
print('n minori',len([s for s in sev_list if s[1]=='minore']))
print('note',len([s for s in sev_list if s[1]=='nota']))
for p in r:
    for pv in p['verify']['point_verdicts']:
        print('POINT',pv['point'][:80],'=>',pv['verdict'])
    for sc in p['verify']['seeded_claims']:
        print('SEED',sc['claim_id'],sc['verdict'],sc['severity'],sc['kind'])
