import json, sys
from fractions import Fraction
base='results/infra/pose_temponet_execution_20261121_A1/launcher/'
J=lambda n: json.load(open(base+n))
cal=[J('calibration_0.json'), J('calibration_1.json')]
sel={h:J('selection_temporal_%s.json'%h) for h in ('TCN_post','L0','MLP')}
gate=J('positive_controls_gate_20261121.json')

def E(st): return st['dynamic_sse']/st['dynamic_target_ss']
def M(st): return st['mse_sum']/st['window_count']
def pool(rs): return {k: sum(r[k] for r in rs) for k in ('mse_sum','window_count','dynamic_sse','dynamic_target_ss')}

ref={}
for f in (0,1):
    t=cal[f]['outer_reference_statistics']['temporal']
    ref['fold%d'%f]={k:t[k] for k in ('constant','clean_ls','oracle')}
ref['pooled']={k:pool([ref['fold0'][k],ref['fold1'][k]]) for k in ('constant','clean_ls','oracle')}

print('== reference (held) from calibration_{0,1}.json outer_reference_statistics.temporal ==')
for p in ('fold0','fold1','pooled'):
    r=ref[p]
    print(p, 'windows', r['clean_ls']['window_count'], 'dyn_target_ss', r['clean_ls']['dynamic_target_ss'],
          'E_const',E(r['constant']), 'E_cleanLS',E(r['clean_ls']), 'E_oracle',E(r['oracle']), 'margin',E(r['clean_ls'])-E(r['oracle']),
          'gap_frac',(E(r['clean_ls'])-E(r['oracle']))/E(r['clean_ls']))
    print('    MSE const',M(r['constant']),'cleanLS',M(r['clean_ls']),'oracle',M(r['oracle']))

def rec(p, st):
    r=ref[p]
    return (E(r['clean_ls'])-E(st))/(E(r['clean_ls'])-E(r['oracle']))
def recM(p, st):
    r=ref[p]
    return (M(r['clean_ls'])-M(st))/(M(r['clean_ls'])-M(r['oracle']))
print()
print('== heads at selected checkpoint (held, observed y): selected_outer_statistics ==')
for h in ('TCN_post','MLP','L0'):
    s=sel[h]
    print(h,'selected',s['selected'],'alarms',s['optimizer_alarms'])
    for p in ('fold0','fold1','pooled'):
        st=s['selected_outer_statistics'][p]
        # check same denominators
        assert abs(st['dynamic_target_ss']-ref[p]['clean_ls']['dynamic_target_ss'])<1e-6*st['dynamic_target_ss'], (h,p)
        print('   ',p,'E_dyn %.6f'%E(st),'MSE %.6f'%M(st),'recovery_Edyn %.4f%%'%(100*rec(p,st)),'recovery_MSE %.4f%%'%(100*recM(p,st)),
              'target_ss equal:', st['dynamic_target_ss']==ref[p]['clean_ls']['dynamic_target_ss'])
print()
print('gate json temporal_head_gate:', json.dumps(gate['temporal_head_gate'], indent=None)[:900])
