import json,sys
import numpy as np
E=sys.argv[1]; W=sys.argv[2]
for arm in ['CN','CR']:
    C={d['step']:d for d in map(json.loads,open(f'{E}/benchmark_execution_20261121/{arm}/metrics.jsonl'))}
    q=np.load(f'{W}/queue_{arm}.npy')
    # simulate tail from updates 545..576
    tail=[]  # (update, gidx, ident)
    odd=[]
    for u in range(545,577):
        org=[tuple(o) for o in C[u]['origin']]
        el=C[u]['contrast_nce_eligible']
        if el!=len(org): odd.append((u,el,len(org)))
        seen=set(); cur=[]
        for i,o in enumerate(org):
            if o not in seen: cur.append((u,i,o)); seen.add(o)
        ids={c[2] for c in cur}
        tail=[e for e in tail if e[2] not in ids]+cur
    K=len(tail)
    print(arm,'windows with eligible<32:',odd,'kept tail entries K =',K,'nonzero in queue =',int((q>0).sum()), 'first nonzero idx', int(np.nonzero(q)[0][0]))
    start=len(q)-K
    print('  tail start index', start, 'zeros within tail', int((q[start:]==0).sum()), 'nonzeros before tail', int((q[:start]>0).sum()))
    per={}
    for j,(u,i,o) in enumerate(tail):
        per.setdefault(u,[]).append(q[start+j])
    out=[]
    for u in sorted(per):
        a=np.array(per[u]); out.append((u,len(a),int((a==0).sum()),float(np.median(a)),float(a.max())))
    for r in out: print('   upd %d n=%d zero=%d median=%.3e max=%.3e'%r)
    json.dump(out,open(f'{W}/queue_by_update_{arm}.json','w'))
