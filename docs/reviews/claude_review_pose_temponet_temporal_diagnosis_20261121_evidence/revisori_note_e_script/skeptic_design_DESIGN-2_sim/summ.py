import json,glob,sys
W='/Users/simonebenatti/wearusfm_local/tcn_temporal_diag_review/skeptic_design_DESIGN-2_sim'
for f in sorted(glob.glob(W+'/run_*.json')):
    try: d=json.load(open(f))
    except Exception as e: print(f,'unreadable',e); continue
    r=d['rows']; ref=d['refs']
    cl,orc=ref['held']['E_cleanLS_noisy'],ref['held']['E_oracle_noisy']
    clf,orf=ref['fit']['E_cleanLS_noisy'],ref['fit']['E_oracle_noisy']
    rec=lambda e:(cl-e)/(cl-orc)
    mn=min(r,key=lambda x:x['held_mse_noisy']); mc=min(r,key=lambda x:x['held_mse_clean']); me=min(r,key=lambda x:x['held_edyn_noisy'])
    print('==',f.split('/')[-1],'epochs done',len(r),'refs held cleanLS %.6f oracle %.6f | fit cleanLS %.6f oracle %.6f | cleanLS clean held E %.6f mse %.6f'%(cl,orc,clf,orf,ref['held']['E_cleanLS_clean'],ref['held']['mse_cleanLS_clean']))
    print('   argmin held NOISY mse: epoch %d mse %.6f edyn %.6f recovery %.4f'%(mn['epoch'],mn['held_mse_noisy'],mn['held_edyn_noisy'],rec(mn['held_edyn_noisy'])))
    print('   argmin held noisy edyn: epoch %d edyn %.6f recovery %.4f'%(me['epoch'],me['held_edyn_noisy'],rec(me['held_edyn_noisy'])))
    print('   argmin held CLEAN mse: epoch %d mse %.6f'%(mc['epoch'],mc['held_mse_clean']))
    for e in (1,5,10,20,31,40,60,80,100,120,140,145,146,147,148,149,150):
        if e<=len(r):
            x=r[e-1]
            extra=''
            if 'fit_mse_clean' in x:
                extra=' | FIT mse_clean %.5f edyn_clean %.5f edyn_noisy %.5f fit-rec(noisy) %.3f'%(x['fit_mse_clean'],x['fit_edyn_clean'],x['fit_edyn_noisy'],(clf-x['fit_edyn_noisy'])/(clf-orf))
            print('   ep %3d held mse_noisy %.5f mse_clean %.5f edyn_noisy %.5f edyn_clean %.5f rec %.3f train_loss %.5f%s'%(e,x['held_mse_noisy'],x['held_mse_clean'],x['held_edyn_noisy'],x['held_edyn_clean'],rec(x['held_edyn_noisy']),x['train_loss'],extra))
