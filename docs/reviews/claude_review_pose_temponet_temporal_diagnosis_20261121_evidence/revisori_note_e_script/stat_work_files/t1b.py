import json, sys
base='results/infra/pose_temponet_execution_20261121_A1/launcher/'
J=lambda n: json.load(open(base+n))
cal=[J('calibration_0.json'), J('calibration_1.json')]
sel={h:J('selection_temporal_%s.json'%h) for h in ('TCN_post','L0','MLP')}
gate=J('positive_controls_gate_20261121.json')['gate']
def E(st): return st['dynamic_sse']/st['dynamic_target_ss']
def M(st): return st['mse_sum']/st['window_count']
def pool(rs): return {k: sum(r[k] for r in rs) for k in ('mse_sum','window_count','dynamic_sse','dynamic_target_ss')}
print('gate temporal_head_gate:', json.dumps(gate['temporal_head_gate']))
print('gate reference temporal_records:', json.dumps(gate['reference_gate']['temporal_records']))
print('gate reference temporal_gate:', json.dumps(gate['reference_gate']['temporal_gate']))
print('selected_panels temporal:', json.dumps(gate['selected_panels']['temporal']))

ref={}
for f in (0,1):
    t=cal[f]['outer_reference_statistics']['temporal']
    ref['fold%d'%f]={k:t[k] for k in ('constant','clean_ls','oracle')}
ref['pooled']={k:pool([ref['fold0'][k],ref['fold1'][k]]) for k in ('constant','clean_ls','oracle')}
def rec(p, e): 
    r=ref[p]; return (E(r['clean_ls'])-e)/(E(r['clean_ls'])-E(r['oracle']))

# closed-form L0 / L13 fitted on NOISY y: worlds.temporal.{fitting,held}.statistics
print()
print('== closed-form WLS fitted on noisy y (linear_diagnostics.<L0|L13>.worlds.temporal) ==')
cf={}
for name in ('L0','L13'):
    cf[name]={}
    for split in ('fitting','held'):
        sts=[cal[f]['outer_reference_statistics']['linear_diagnostics'][name]['worlds']['temporal'][split]['statistics'] for f in (0,1)]
        cf[name][split]={'fold0':sts[0],'fold1':sts[1],'pooled':pool(sts)}
        for p in ('fold0','fold1','pooled'):
            st=cf[name][split][p]
            extra=''
            if split=='held':
                extra=' recovery %.4f%%'%(100*rec(p,E(st)))
            print(name,split,p,'windows',st['window_count'],'E_dyn %.6f'%E(st),'MSE %.6f'%M(st),'target_ss',st['dynamic_target_ss'],extra)
# clean-fit on fitting, scored on y, fitting half
print()
print('== clean-fit (y*) scored on y, FITTING half: temporal_margin_clean_fit_statistics ==')
cfit={}
for name in ('L0','L13'):
    sts=[cal[f]['outer_reference_statistics']['linear_diagnostics'][name]['temporal_margin_clean_fit_statistics'] for f in (0,1)]
    cfit[name]={'fold0':sts[0],'fold1':sts[1],'pooled':pool(sts)}
    for f,p in enumerate(('fold0','fold1')):
        print(name,p,'E_dyn %.6f'%E(sts[f]),'MSE %.6f'%M(sts[f]), 'recovery_fitting(json)', cal[f]['outer_reference_statistics']['linear_diagnostics'][name]['temporal_margin_recovery_fitting'],
              'T_unread', cal[f]['outer_reference_statistics']['linear_diagnostics'][name]['T_unread_fraction_fitting'])
    print(name,'pooled','E_dyn %.6f'%E(cfit[name]['pooled']),'MSE %.6f'%M(cfit[name]['pooled']))
print()
print('== TCN fitting diagnostics (selected checkpoint, fitting half, observed y) ==')
for f in ('0','1'):
    st=sel['TCN_post']['selected_fitting_diagnostics'][f]
    print('fold',f,'E_dyn %.6f'%E(st),'MSE %.6f'%M(st),'windows',st['window_count'],'target_ss',st['dynamic_target_ss'])
for h in ('L0','MLP'):
    for f in ('0','1'):
        st=sel[h]['selected_fitting_diagnostics'][f]
        print(h,'fold',f,'fitting E_dyn %.6f'%E(st),'MSE %.6f'%M(st))
print()
print('== held clean-target scores (selected_outer_clean_statistics) ==')
for h in ('TCN_post','MLP','L0'):
    for p in ('fold0','fold1','pooled'):
        st=sel[h]['selected_outer_clean_statistics'][p]
        print(h,p,'clean E_dyn %.6f'%E(st),'clean MSE %.6f'%M(st),'clean dyn sse',st['dynamic_sse'],'clean target ss',st['dynamic_target_ss'])
print()
print('hidden_active_fraction_fitting', [c.get('components',{}).get('hidden_active_fraction_fitting') for c in cal])
print('calibration selected temporal', [ {k:v for k,v in c['calibration']['selected']['temporal'].items() if k!='alpha_curve'} for c in cal])
print('E_real', [c['calibration']['E_real'] for c in cal])
print('components keys', [list(c.get('components',{}).keys()) for c in cal])
