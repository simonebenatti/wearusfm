import json, math
base='results/infra/pose_temponet_execution_20261121_A1/launcher/'
J=lambda n: json.load(open(base+n))
cal=[J('calibration_0.json'), J('calibration_1.json')]
sel={h:J('selection_temporal_%s.json'%h) for h in ('TCN_post','L0','MLP')}
def E(st): return st['dynamic_sse']/st['dynamic_target_ss']
def M(st): return st['mse_sum']/st['window_count']
def pool(rs): return {k: sum(r[k] for r in rs) for k in ('mse_sum','window_count','dynamic_sse','dynamic_target_ss')}
P=('fold0','fold1','pooled')
ref={}
for f in (0,1):
    t=cal[f]['outer_reference_statistics']['temporal']
    ref['fold%d'%f]={k:t[k] for k in ('constant','clean_ls','oracle')}
ref['pooled']={k:pool([ref['fold0'][k],ref['fold1'][k]]) for k in ('constant','clean_ls','oracle')}
# relative difference of denominators between head scoring and reference scoring
for h in sel:
    for p in P:
        a=sel[h]['selected_outer_statistics'][p]['dynamic_target_ss']; b=ref[p]['clean_ls']['dynamic_target_ss']
        print('denominator rel diff',h,p,(a-b)/b)
print()
# SSE-domain quantities on held
print('== SSE-domain (held, dynamic windows, centred), per panel ==')
out={}
for p in P:
    r=ref[p]; SS=r['clean_ls']['dynamic_target_ss']
    S_or=r['oracle']['dynamic_sse']; S_cl=r['clean_ls']['dynamic_sse']
    margin=S_cl-S_or
    print(p,'target SS %.1f'%SS,'oracle SSE %.1f'%S_or,'cleanLS SSE %.1f'%S_cl,'margin SSE %.1f'%margin,'(E units %.6f)'%(margin/SS))
    out[p]=dict(SS=SS,S_or=S_or,S_cl=S_cl,margin=margin)
    for h in ('TCN_post','MLP','L0'):
        st=sel[h]['selected_outer_statistics'][p]; ct=sel[h]['selected_outer_clean_statistics'][p]
        cross=st['dynamic_sse']-ct['dynamic_sse']-S_or
        print('   %-8s noisy SSE %.1f clean SSE %.1f  cross-term %.1f (%.3f%% of margin)  excess-over-oracle %.1f = %.2f%% of margin; clean SSE/margin %.2f%%; noisy rec %.2f%%; clean-domain rec %.2f%%'%(
            h,st['dynamic_sse'],ct['dynamic_sse'],cross,100*cross/margin,st['dynamic_sse']-S_or,100*(st['dynamic_sse']-S_or)/margin,100*ct['dynamic_sse']/margin,
            100*(S_cl-st['dynamic_sse'])/margin, 100*(1-ct['dynamic_sse']/margin)))
        out[p][h]=dict(noisy=st['dynamic_sse'],clean=ct['dynamic_sse'],cross=cross)
    print('   clean target SS (held dyn) %.1f ; cleanLS clean E_dyn approx (margin/cleanSS) %.4f ; 50%% bar clean E_dyn approx %.4f'%(
        sel['TCN_post']['selected_outer_clean_statistics'][p]['dynamic_target_ss'], margin/sel['TCN_post']['selected_outer_clean_statistics'][p]['dynamic_target_ss'], 0.5*margin/sel['TCN_post']['selected_outer_clean_statistics'][p]['dynamic_target_ss']))
    print('   noise SS/clean SS (dyn held) = %.3f ; clean/noisy target SS = %.4f'%(S_or/sel['TCN_post']['selected_outer_clean_statistics'][p]['dynamic_target_ss'], sel['TCN_post']['selected_outer_clean_statistics'][p]['dynamic_target_ss']/SS))
print()
print('== closed-form L0/L13 fitted on noisy y: held excess vs cleanLS (noise-driven estimation cost) ==')
cf={}
for name in ('L0','L13'):
    cf[name]={}
    for split in ('fitting','held'):
        sts=[cal[f]['outer_reference_statistics']['linear_diagnostics'][name]['worlds']['temporal'][split]['statistics'] for f in (0,1)]
        cf[name][split]={'fold0':sts[0],'fold1':sts[1],'pooled':pool(sts)}
cfit={}
for name in ('L0','L13'):
    sts=[cal[f]['outer_reference_statistics']['linear_diagnostics'][name]['temporal_margin_clean_fit_statistics'] for f in (0,1)]
    cfit[name]={'fold0':sts[0],'fold1':sts[1],'pooled':pool(sts)}
npar={'L0':384*20,'L13':4980*20}
for p in P:
    m=out[p]['margin']; SS=out[p]['SS']
    exL0=cf['L0']['held'][p]['dynamic_sse']-out[p]['S_cl']
    print(p,'L0(y) held excess over cleanLS: SSE %.1f = %.6f E = %.2f%% of margin; per scalar param (7680): %.3f SSE'%(exL0,exL0/SS,100*exL0/m,exL0/7680))
    # in-sample optimism for L0: fit-on-y vs fit-on-y* on fitting, both scored on y
    opt=cfit['L0'][p]['dynamic_sse']-cf['L0']['fitting'][p]['dynamic_sse']
    SSf=cf['L0']['fitting'][p]['dynamic_target_ss']
    print('    L0 in-sample optimism (clean-fit minus noisy-fit, fitting, on y): SSE %.1f = %.6f E ; held excess / in-sample optimism = %.2f'%(opt,opt/SSf,(exL0/SS)/(opt/SSf)))
    exL13_lo=cf['L13']['held'][p]['dynamic_sse']-out[p]['S_cl']
    optL13=cfit['L13'][p]['dynamic_sse']-cf['L13']['fitting'][p]['dynamic_sse']
    print('    L13(y) held: SSE over cleanLS %.1f = %.6f E = %.2f%% of margin ; over oracle %.2f%% of margin'%(exL13_lo,exL13_lo/SS,100*exL13_lo/m,100*(cf['L13']['held'][p]['dynamic_sse']-out[p]['S_or'])/m))
    print('    L13 in-sample optimism: SSE %.1f = %.6f E (L13 clean-fit on fitting %.6f, noisy-fit on fitting %.6f)'%(optL13,optL13/SSf,E(cfit['L13'][p]),E(cf['L13']['fitting'][p])))
    for r in (0.0,0.10,0.20,0.27):
        ex=exL13_lo+r*m
        print('      if L13 clean-fit held recovery were %.2f: noise-driven excess = %.1f SSE = %.6f E = %.1f%% of margin ; per scalar param (99600) %.3f'%(r,ex,ex/SS,100*ex/m,ex/99600))
    print('    ratio in-sample optimism L13/L0 = %.2f ; param ratio = %.2f'%(optL13/opt, 4980/384))
print()
print('== fitting-half references reconstructed (oracle on fitting from L13 clean-fit recovery) ==')
for f,p in enumerate(('fold0','fold1')):
    d=cal[f]['outer_reference_statistics']['linear_diagnostics']
    base_=E(d['L0']['temporal_margin_clean_fit_statistics']); l13=E(d['L13']['temporal_margin_clean_fit_statistics']); r=d['L13']['temporal_margin_recovery_fitting']
    margin_fit=(base_-l13)/r; orc=base_-margin_fit
    tcn=E(sel['TCN_post']['selected_fitting_diagnostics'][str(f)])
    held=out[p]
    print(p,'fitting: cleanLS %.6f oracle %.6f margin %.6f | TCN fitting %.6f rec_fit %.2f%% | TCN below oracle? %s by %.6f'%(base_,orc,margin_fit,tcn,100*(base_-tcn)/margin_fit, tcn<orc, orc-tcn))
    for h in ('L0','MLP'):
        e=E(sel[h]['selected_fitting_diagnostics'][str(f)])
        print('    %s neural fitting %.6f rec_fit %.2f%% ; held rec %.2f%%'%(h,e,100*(base_-e)/margin_fit,100*(held['S_cl']-held[h]['noisy'])/held['margin']))
    e=E(cf['L0']['fitting'][p]); print('    closed L0(y) fitting %.6f rec_fit %.2f%%'%(e,100*(base_-e)/margin_fit))
    e=E(cf['L13']['fitting'][p]); print('    closed L13(y) fitting %.6f rec_fit %.2f%%'%(e,100*(base_-e)/margin_fit))
    # TCN gap in E units
    tcn_held=held['TCN_post']['noisy']/held['SS']
    print('    TCN: fitting excess over oracle_fit %.6f ; held excess over oracle_held %.6f ; gap (held-fit) in excess %.6f = %.1f%% of held margin'%(tcn-orc, tcn_held-held['S_or']/held['SS'], (tcn_held-held['S_or']/held['SS'])-(tcn-orc), 100*((tcn_held-held['S_or']/held['SS'])-(tcn-orc))/(held['margin']/held['SS'])))
