import json,sys,re
import numpy as np
d=json.load(open(sys.argv[1]))
for arm in ['CN','CR']:
    for pair in ['continuous__reload_a','continuous__reload_b','reload_a__reload_b']:
        f=d['arms'][arm]['resume']['pairs'][pair]['floating_tensor_relative_to_left']
        groups={'exp_avg':[], 'exp_avg_sq':[], 'queue':[]}
        idx={'exp_avg':[], 'exp_avg_sq':[], 'queue':[]}
        for k,v in f.items():
            m=re.match(r'optimizer\.state\.(\d+)\.(exp_avg_sq|exp_avg)$',k)
            if m: groups[m.group(2)].append(v); idx[m.group(2)].append(int(m.group(1))); continue
            m=re.match(r'contrastive_state\.queue\.entries\.(\d+)\.key$',k)
            if m: groups['queue'].append(v); idx['queue'].append(int(m.group(1))); continue
            print('other',k)
        print(f'== {arm} {pair}')
        for g in groups:
            a=np.array(groups[g]); 
            if len(a)==0: continue
            nz=a[a>0]
            q=np.quantile(a,[0,.1,.5,.9,.99,1]) if len(a) else []
            print(f'  {g}: n={len(a)} zero={int((a==0).sum())} quantiles(0,10,50,90,99,100)%=' + ' '.join('%.2e'%x for x in q) + f' >1e-2: {int((a>1e-2).sum())} >5e-3: {int((a>5e-3).sum())}')
            if g!='queue':
                order=np.argsort(-a)
                print('   top10:', [(idx[g][i], '%.3e'%a[i]) for i in order[:10]])
                print('   zero ids:', sorted(idx[g][i] for i in range(len(a)) if a[i]==0)[:40])
        if pair=='continuous__reload_a':
            ea=dict(zip(idx['exp_avg'],groups['exp_avg'])); es=dict(zip(idx['exp_avg_sq'],groups['exp_avg_sq']))
            print('  266-269 exp_avg', [ '%.4e'%ea[i] for i in range(266,270)], 'exp_avg_sq', ['%.3e'%es[i] for i in range(266,270)])
            ra=np.array(groups['exp_avg']); print('  rank of 266-269 among exp_avg:', [int((ra>ea[i]).sum())+1 for i in range(266,270)])
            q=np.array(groups['queue'])[np.argsort(idx['queue'])]
            np.save(sys.argv[2]+f'/queue_{arm}.npy', q)
            z=np.nonzero(q)[0]; print('  queue: first nonzero idx', z[:3] if len(z) else None, 'n nonzero', len(z), 'last zero idx before tail', (np.where(q==0)[0].max() if (q==0).any() else None))
