# Rifà i conteggi della sezione «Materiali e metodo» dal JSON del workflow.
import json, collections, sys
P = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/tasks/w785grg3y.output'
d = json.load(open(P))
r = d['result']
rev_findings = []
for pair in r:
    for rv in pair['reviews']:
        for f in rv['findings']:
            rev_findings.append((pair['pair'], rv['dimension'], f['id'], f['severity'], f.get('kind')))
print('Rilievi dei revisori:', len(rev_findings))
c = collections.Counter(x[3] for x in rev_findings)
print('  severita dei revisori:', dict(c))
for dim in sorted(set(x[1] for x in rev_findings)):
    print('   ', dim, sum(1 for x in rev_findings if x[1]==dim), dict(collections.Counter(x[3] for x in rev_findings if x[1]==dim)))
verd = []
seeded = []
newf = []
for pair in r:
    v = pair['verify']
    for x in v['verdicts']:
        verd.append((pair['pair'], x['finding_id'], x['verdict'], x['severity'], x.get('kind')))
    for x in v['seeded_claims']:
        seeded.append((pair['pair'], x['claim_id'], x['verdict'], x['severity'], x.get('kind')))
    for x in v['new_findings']:
        newf.append((pair['pair'], x['id'], x['severity'], x.get('kind')))
print('Verdetti dei verificatori:', len(verd), dict(collections.Counter(x[2] for x in verd)))
ids_rev = set(x[2] for x in rev_findings)
ids_ver = [x[1] for x in verd]
print('  rilievi senza verdetto:', sorted(ids_rev - set(ids_ver)))
print('  verdetti senza rilievo:', sorted(set(ids_ver) - ids_rev))
print('  duplicati nei verdetti:', [k for k,v in collections.Counter(ids_ver).items() if v>1])
print('Sospetti:', len(seeded), dict(collections.Counter(x[2] for x in seeded)))
for s in sorted(seeded, key=lambda t:t[1]): print('   ', s)
print('Nuovi rilievi:', len(newf), dict(collections.Counter(x[2] for x in newf)))
for s in newf: print('   ', s)
# severita dopo la verifica sui 58 + 5
sev_after = collections.Counter(x[3] for x in verd) + collections.Counter(x[2] for x in newf)
print('Severita dopo la verifica (58 verdetti + nuovi):', dict(sev_after), 'totale', sum(sev_after.values()))
print('  solo verdetti:', dict(collections.Counter(x[3] for x in verd)))
print('  bloccanti:', [x for x in verd if x[3]=='bloccante'] + [x for x in newf if x[2]=='bloccante'])
print('  minori:', sorted([x[1] for x in verd if x[3]=='minore'] + [x[1] for x in newf if x[2]=='minore']))
# severita dei sospetti
print('Severita dei sospetti:', dict(collections.Counter(x[3] for x in seeded)))
# kind
print('Kind dopo verifica:', dict(collections.Counter(x[4] for x in verd) + collections.Counter(x[3] for x in newf)))
# confronto severita revisore -> verificatore
sev_rev = {x[2]: x[3] for x in rev_findings}
chg = [(x[1], sev_rev.get(x[1]), x[3], x[2]) for x in verd if sev_rev.get(x[1]) != x[3]]
print('Cambi di severita:', len(chg))
for c_ in chg: print('   ', c_)
