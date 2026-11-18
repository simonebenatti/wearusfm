import json,sys,collections
F=sys.argv[1]
d=json.load(open(F))
r=d['result']
findings={}
orig_sev={}
for x in r:
    for rv in x['reviews']:
        for f in rv['findings']:
            findings[f['id']]=f
            orig_sev[f['id']]=f['severity']
print('review findings:',len(findings))
print('orig severities:',collections.Counter(orig_sev.values()))
verd={}
post={}
for x in r:
    vf=x['verify']
    for v in vf['verdicts']:
        verd[v['finding_id']]=v['verdict']
        post[v['finding_id']]=v['severity']
print('verdicts:',len(verd),collections.Counter(verd.values()))
missing=set(findings)-set(verd)
print('findings without verdict:',missing)
extra=set(verd)-set(findings)
print('verdicts w/o finding:',extra)
newf={}
for x in r:
    for nf in x['verify']['new_findings']:
        newf[nf['id']]=nf['severity']
print('new findings',newf)
seeded={}
for x in r:
    for s in x['verify']['seeded_claims']:
        seeded[s['claim_id']]=(s['verdict'],s['severity'])
print('seeded',seeded, collections.Counter(v[0] for v in seeded.values()))
allpost=dict(post); allpost.update(newf)
print('post-verify severities over', len(allpost), collections.Counter(allpost.values()))
for k in sorted(allpost):
    if allpost[k]!='nota':
        print('  ',k,orig_sev.get(k,'NEW'),'->',allpost[k], verd.get(k,'NEW'))
print('changes of severity:')
for k in sorted(post):
    if post[k]!=orig_sev[k]:
        print('  ',k,orig_sev[k],'->',post[k],verd[k])
# kinds
kinds={}
for x in r:
    for v in x['verify']['verdicts']:
        kinds[v['finding_id']]=(findings[v['finding_id']]['kind'],v['kind'])
print('kind changes:')
for k,(a,b) in sorted(kinds.items()):
    if a!=b: print('  ',k,a,'->',b)
for x in r:
    for pv in x['verify']['point_verdicts']:
        print('POINT', pv['point'][:60], pv['verdict'])
    for rv in x['reviews']:
        print('REVIEWPOINT', rv['dimension'], rv['point_verdict'])
