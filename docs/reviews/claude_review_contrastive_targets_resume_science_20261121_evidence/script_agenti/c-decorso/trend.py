import json,sys
import numpy as np
W=sys.argv[1]
rd=json.load(open(f'{W}/reldiff.json'))
keys=['grad_norm','contrast_projector_grad_norm','contrast_global_grad_base','contrast_global_grad_nce','contrast_global_grad_rnc','contrast_global_cos_nce_base','contrast_global_cos_rnc_base','total','jepa','pose','contrast_nce','contrast_rnc','log_rms','band_shape']
steps=np.arange(545,577)
for arm in ['CN','CR']:
    print('==',arm)
    for k in keys:
        r=rd[arm].get(k)
        if r is None: continue
        a=np.array([np.nan if x is None else x for x in r],float)
        if np.nanmax(a)==0: print(f'  {k}: all zero'); continue
        def fit(lo,hi):
            m=(steps>=lo)&(steps<=hi)&(a>0)&np.isfinite(a)
            if m.sum()<3: return (np.nan,np.nan)
            b=np.polyfit(steps[m],np.log10(a[m]),1); return b[0], m.sum()
        s1=fit(548,576); s2=fit(548,561); s3=fit(562,576)
        med=lambda lo,hi: np.nanmedian(np.where(a>0,a,np.nan)[(steps>=lo)&(steps<=hi)])
        print(f'  {k}: med548-556 {med(548,556):.2e} med557-566 {med(557,566):.2e} med567-576 {med(567,576):.2e}  slope log10/step 548-576 {s1[0]:+.4f} (548-561 {s2[0]:+.4f}, 562-576 {s3[0]:+.4f})')
