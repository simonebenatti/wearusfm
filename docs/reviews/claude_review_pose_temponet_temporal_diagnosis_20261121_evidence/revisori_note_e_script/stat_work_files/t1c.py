import json
base='results/infra/pose_temponet_execution_20261121_A1/launcher/'
J=lambda n: json.load(open(base+n))
cal=[J('calibration_0.json'), J('calibration_1.json')]
def E(st): return st['dynamic_sse']/st['dynamic_target_ss']
def pool(rs): return {k: sum(r[k] for r in rs) for k in ('mse_sum','window_count','dynamic_sse','dynamic_target_ss')}
ref={}
for f in (0,1):
    t=cal[f]['outer_reference_statistics']['temporal']
    ref['fold%d'%f]={k:t[k] for k in ('constant','clean_ls','oracle')}
ref['pooled']={k:pool([ref['fold0'][k],ref['fold1'][k]]) for k in ('constant','clean_ls','oracle')}
cl,orc=E(ref['pooled']['clean_ls']),E(ref['pooled']['oracle'])
rec=lambda e:(cl-e)/(cl-orc)
for h in ('TCN_post','L0','MLP'):
    s=J('selection_temporal_%s.json'%h)
    print('=====',h)
    c0=s['curves'][0]
    print('curve keys',list(c0.keys()),'point keys',list(c0['points'][0].keys()))
    print('first points',c0['points'][:5])
    allpts=[]
    for c in s['curves']:
        pts=c['points']
        adm=[p for p in pts if p.get('exposure',0)>0]
        bm=min(adm,key=lambda p:(p['mse'],p['exposure'])); be=min(adm,key=lambda p:(p['edyn'],p['exposure']))
        last=pts[-1]
        print('lr',c['lr'],'horizon',c['horizon'],'n_points',len(pts),
              '| min MSE: exp',bm['exposure'],'mse %.6f'%bm['mse'],'edyn %.6f'%bm['edyn'],'rec %.2f%%'%(100*rec(bm['edyn'])),
              '| min Edyn: exp',be['exposure'],'edyn %.6f'%be['edyn'],'rec %.2f%%'%(100*rec(be['edyn'])),
              '| last: exp',last['exposure'],'mse %.6f'%last['mse'],'edyn %.6f'%last['edyn'],'rec %.2f%%'%(100*rec(last['edyn'])))
        allpts+= [(p['edyn'],c['lr'],p['exposure'],p['mse']) for p in adm]
    b=min(allpts); print('GLOBAL best E_dyn over all admitted points:',b,'recovery %.4f%%'%(100*rec(b[0])), 'n points',len(allpts))
    bm=min(allpts,key=lambda t:(t[3],t[2],t[1])); print('GLOBAL best MSE:',bm,'recovery %.4f%%'%(100*rec(bm[0])))
    if h=='TCN_post':
        c=[c for c in s['curves'] if c['lr']==0.003][0]
        for p in c['points']:
            if p['exposure'] in (0,0.25,0.5,0.75,1,2,3,5,8,10,15,20,25,27,29,31,33,35,40,50,60,75,100,125,150):
                print('  lr0.003 exp',p['exposure'],'mse %.6f'%p['mse'],'edyn %.6f'%p['edyn'],'rec %.2f%%'%(100*rec(p['edyn'])), {k:v for k,v in p.items() if k not in ('mse','edyn','exposure')})
