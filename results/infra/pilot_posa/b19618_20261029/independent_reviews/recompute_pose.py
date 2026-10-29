import json, hashlib, math
from pathlib import Path
import numpy as np

root=Path('/private/tmp/wearusfm-b19618-results-20261029')
repo=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science')
p=json.loads((root/'reports/pose_comparison.json').read_text())
manifest=json.loads((root/'manifest.json').read_text())
checks=[]
def check(name,a,b):
    if isinstance(a,dict): assert a==b,name
    elif np.asarray(a).dtype.kind in 'OUSb': assert np.array_equal(a,b),name
    else: np.testing.assert_allclose(a,b,rtol=3e-12,atol=3e-12,err_msg=name)
    checks.append(name)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def boot(num,den):
    num=np.asarray(num,float);den=np.asarray(den,float)
    d=np.random.default_rng(20261009).integers(0,len(num),(1000,len(num)))
    total=den[d].sum(1);assert np.all(np.isfinite(total)) and np.all(total>0)
    return float(num.sum()/den.sum()),np.percentile(num[d].sum(1)/total,[2.5,97.5]).tolist()
def signs(vals):
    v=np.asarray(vals,float)
    return {'negative':int((v<0).sum()),'zero':int((v==0).sum()),'positive':int((v>0).sum())}
for r in manifest['files']:
    f=root/r['local_path'];check('hash '+r['local_path'],sha(f),r['remote_sha256']);check('bytes '+r['local_path'],f.stat().st_size,r['bytes'])
models={'B5000':p['head_dynamics']['B5000'],'B19618':p['head_dynamics']['B19618'],'B_ridge':p['accessibility']['B']['analysis'],'R3c_ridge':p['accessibility']['R3c']['analysis']}
result={'source':str(root/'reports/pose_comparison.json'),'source_sha256':sha(root/'reports/pose_comparison.json'),'method':'Independent recomputation from per-user sums, no project analysis imports, no refit or jobs.','models':{},'comparisons':{}}
for name,a in models.items():
    e=a['e1_dynamic_control'];rows=[e['per_user'][u] for u in sorted(e['per_user'])]
    err=np.array([r['error_sum_rad'] for r in rows]);con=np.array([r['constant_sum_rad'] for r in rows]);n=np.array([r['n_valid_elements'] for r in rows]);ratio,ci=boot(err,con)
    for key,value in [('ratio',ratio),('ratio_ci95_users',ci),('mae',err.sum()/n.sum()),('mae_const',con.sum()/n.sum()),('median_user_ratio',np.median(err/con))]: check(name+' E1 '+key,value,e[key])
    result['models'][name]={'E1':{k:e[k] for k in ('ratio','ratio_ci95_users','mae','mae_const','median_user_ratio','n_valid_elements','n_windows','n_users')},'groups':{}}
    for group,g in a['groups'].items():
        c=g['centered_support_n_ge_2'];us=sorted(c['per_user']);rs=[c['per_user'][u] for u in us];n=np.array([r['n_valid_elements'] for r in rs]);sums={k:sum(r['sums'][k] for r in rs) for k in c['sums']}
        for k,v in sums.items(): check(name+group+' sum '+k,v,c['sums'][k]);check(name+group+' mean '+k,v/n.sum(),c['means'][k])
        check(name+group+' total identity',sums['total'],sums['centered']+sums['mean_error']);check(name+group+' centered identity',sums['centered'],sums['vary']+sums['varp']-2*sums['cov']);check(name+group+' gain identity',sums['total_gain'],sums['gain_mean']+sums['gain_centered'])
        ratio,ci=boot([r['sums']['centered'] for r in rs],[r['sums']['vary'] for r in rs]);check(name+group+' ratio',ratio,c['centered_mse_ratio']);check(name+group+' CI',ci,g['bootstrap_centered']['ratio_ci95'])
        _,gci=boot([r['sums']['vary']-r['sums']['centered'] for r in rs],n);check(name+group+' gain CI',gci,g['bootstrap_centered']['gain_ci95'])
        vr=sums['varp']/sums['vary'];rho=sums['cov']/math.sqrt(sums['vary']*sums['varp']);check(name+group+' variance',vr,c['variance_ratio']);check(name+group+' correlation',rho,c['pooled_centered_correlation'])
        result['models'][name]['groups'][group]={'centered_ratio':ratio,'centered_ci95':ci,'variance_ratio':vr,'correlation':rho,'means':{k:sums[k]/n.sum() for k in sums},'centered_gain_signs':signs([r['sums']['gain_centered'] for r in rs]),'per_user':{u:{'centered_ratio':r['sums']['centered']/r['sums']['vary'],'variance_ratio':r['sums']['varp']/r['sums']['vary'],'correlation':r['sums']['cov']/math.sqrt(r['sums']['varp']*r['sums']['vary']),'n_valid_elements':r['n_valid_elements'],'n_windows':r['n_windows']} for u,r in zip(us,rs)}}
for label,refname,candname,reported in [('A','B5000','B19618',p['head_evolution']),('C','R3c_ridge','B_ridge',p['accessibility']['B_minus_R3c'])]:
    ref,cand=models[refname],models[candname];a=ref['e1_dynamic_control']['per_user'];b=cand['e1_dynamic_control']['per_user'];us=sorted(a);check(label+' E1 users',us,sorted(b))
    for u in us:
        for k in ('n_valid_elements','n_windows','constant_sum_rad'):check(label+' E1 support '+u+k,a[u][k],b[u][k])
    num=np.array([b[u]['error_sum_rad']-a[u]['error_sum_rad'] for u in us]);den=np.array([a[u]['constant_sum_rad'] for u in us]);delta,ci=boot(num,den);check(label+' E1 delta',delta,reported['E1']['delta']);check(label+' E1 CI',ci,reported['E1']['ci95'])
    result['comparisons'][label]={'reference':refname,'candidate':candname,'E1':{'delta':delta,'ci95':ci,'signs':signs(num),'mae_change_rad':cand['e1_dynamic_control']['mae']-ref['e1_dynamic_control']['mae'],'relative_error_reduction':1-cand['e1_dynamic_control']['mae']/ref['e1_dynamic_control']['mae'],'per_user_delta':{u:float(n/d) for u,n,d in zip(us,num,den)}},'centered':{}}
    for group in ('dynamic','all','static'):
        a=ref['groups'][group]['centered_support_n_ge_2']['per_user'];b=cand['groups'][group]['centered_support_n_ge_2']['per_user'];us=sorted(a);check(label+group+' users',us,sorted(b))
        for u in us:
            for k in ('n_valid_elements','n_window_joints','n_windows'):check(label+group+' support '+u+k,a[u][k],b[u][k])
            check(label+group+' variance sum '+u,a[u]['sums']['vary'],b[u]['sums']['vary'])
        num=np.array([b[u]['sums']['centered']-a[u]['sums']['centered'] for u in us]);den=np.array([a[u]['sums']['vary'] for u in us]);delta,ci=boot(num,den);check(label+group+' delta',delta,reported['centered'][group]['delta_ratio']);check(label+group+' CI',ci,reported['centered'][group]['delta_ratio_ci95'])
        x=result['models'][refname]['groups'][group]['means'];y=result['models'][candname]['groups'][group]['means'];dm=y['mean_error']-x['mean_error'];dc=y['centered']-x['centered'];dt=y['total']-x['total']
        result['comparisons'][label]['centered'][group]={'delta_ratio':delta,'ci95':ci,'signs':signs(num),'relative_centered_error_reduction':1-y['centered']/x['centered'],'mean_error_change':dm,'centered_error_change':dc,'total_error_change':dt,'mean_error_reduction_share':dm/dt,'centered_error_reduction_share':dc/dt,'per_user_delta':{u:float(n/d) for u,n,d in zip(us,num,den)},'mean_error_change_signs':signs([b[u]['sums']['mean_error']-a[u]['sums']['mean_error'] for u in us])}
e2=[]
for name,key in [('B5000','baseline_head'),('B19618','candidate_head')]:
    e=p[key]['E2'];r=e['paired_support'];us=sorted(r['per_user']);rows=[r['per_user'][u] for u in us];n=np.array([v['n_valid_elements'] for v in rows]);ali=np.array([v['mae_aligned'] for v in rows]);mis=np.array([v['mae_misaligned'] for v in rows]);gap,ci=boot((mis-ali)*n,n)
    check(name+' E2 gap',gap,e['gap_with_ci']['delta']);check(name+' E2 CI',ci,e['gap_with_ci']['ci95']);check(name+' E2 aligned',(ali*n).sum()/n.sum(),r['mae_aligned']);check(name+' E2 misaligned',(mis*n).sum()/n.sum(),r['mae_misaligned'])
    result['models'][name]['E2']={'aligned_mae':r['mae_aligned'],'misaligned_mae':r['mae_misaligned'],'gap':gap,'ci95':ci,'n_users':len(us),'n_windows':r['n_windows'],'n_valid_elements':int(n.sum()),'gap_signs':signs(mis-ali),'pairing':e['pairing']};e2.append((us,n,mis-ali))
check('E2 users',e2[0][0],e2[1][0]);check('E2 weights',e2[0][1],e2[1][1]);check('E2 pairing',p['baseline_head']['E2']['pairing'],p['candidate_head']['E2']['pairing']);delta,ci=boot((e2[1][2]-e2[0][2])*e2[0][1],e2[0][1]);check('E2 paired delta',delta,p['head_evolution']['E2_gap']['delta']);check('E2 paired CI',ci,p['head_evolution']['E2_gap']['ci95'])
result['comparisons']['A']['E2_gap']={'delta':delta,'ci95':ci,'delta_signs':signs(e2[1][2]-e2[0][2]),'omitted_user':sorted(set(models['B19618']['e1_dynamic_control']['per_user'])-set(e2[0][0]))}
result['selection']={};result['m2']={k:p['accessibility'][k]['m2'] for k in ('B','R3c')};valusers=set(models['B19618']['e1_dynamic_control']['per_user'])
for name in ('B','R3c'):
    s=p['accessibility'][name]['selection'];a,b=s['fold_users'];assert len(a)==len(b)==20 and set(a).isdisjoint(b) and (set(a)|set(b)).isdisjoint(valusers)
    best=min(s['grid'],key=lambda r:(r['mse'],r['alpha']));check(name+' alpha',best['alpha'],s['alpha']);check(name+' cv',best['mse'],s['cv_mse']);assert not s['boundary_unresolved'] and s['grid'][0]['alpha']<s['alpha']<s['grid'][-1]['alpha'];result['selection'][name]={k:v for k,v in s.items() if k!='fold_users'}
check('same CV folds',p['accessibility']['B']['selection']['fold_users'],p['accessibility']['R3c']['selection']['fold_users'])
for split in ('train','val'):
    m=p['accessibility']['B']['m2'][split];assert m['passed'] and m['applicable'] and m['max_abs']<0.005 and m['rmse']<0.0005;assert not p['accessibility']['R3c']['m2'][split]['applicable']
for model in ('B','R3c'):
    for split,fn in [('train','train_subset.npz.provenance.json'),('val','val.npz.provenance.json')]:
        f=root/'extraction_metadata'/model/'pose'/fn;v=json.loads(f.read_text());rp=p['provenance'][model+'_'+split];rr=f.parent/'extraction_receipt.json';rec=json.loads(rr.read_text())
        check(model+split+' provenance hash',sha(f),rp['provenance_sha256']);check(model+split+' receipt hash',sha(rr),rp['extraction_receipt_sha256']);check(model+split+' provenance',{k:rp[k] for k in v},v);check(model+split+' cache hash',rec['outputs'][split]['sha256'],v['cache_sha256']);check(model+split+' step',v['checkpoint_step'],19618);check(model+split+' support attested',v['same_metadata_target_mask_std_pose_var_as_B5000'],True)
        if model=='B':check(split+' M2 count',p['accessibility']['B']['m2'][split]['n'],v['n_valid_elements'])
check('report source hash',sha(repo/'scripts/pose_b19618_report.py'),p['provenance']['source_sha256'])
# Also reproduce the legacy float32-input head report's secondary E1 blocks and
# E2 misaligned/constant ratio from its stored per-user MAEs and support counts.
for name,key in [('B5000','baseline_head'),('B19618','candidate_head')]:
    result['models'][name]['head_report_E1']={}
    for group in ('dynamic','all','static'):
        r=p[key]['E1'][group];us=sorted(r['per_user']);weights=models[name]['groups'][group]['total_support_n_ge_1']['per_user']
        n=np.array([weights[u]['n_valid_elements'] for u in us]);a=np.array([r['per_user'][u]['mae'] for u in us]);b=np.array([r['per_user'][u]['mae_const'] for u in us]);ratio,ci=boot(a*n,b*n)
        check(name+group+' head-report E1 ratio',ratio,r['ratio']);check(name+group+' head-report E1 CI',ci,r['ratio_ci95_users']);check(name+group+' head-report E1 median',np.median(a/b),r['median_user_ratio'])
        result['models'][name]['head_report_E1'][group]={'ratio':ratio,'ci95':ci,'mae':r['mae'],'median_user_ratio':r['median_user_ratio']}
    e=p[key]['E2'];us=sorted(e['per_user']);n=np.array([e['paired_support']['per_user'][u]['n_valid_elements'] for u in us]);a=np.array([e['per_user'][u]['mae'] for u in us]);b=np.array([e['per_user'][u]['mae_const'] for u in us]);ratio,ci=boot(a*n,b*n)
    check(name+' misaligned E2 ratio',ratio,e['ratio']);check(name+' misaligned E2 CI',ci,e['ratio_ci95_users']);result['models'][name]['E2']['misaligned_constant_ratio']=ratio;result['models'][name]['E2']['misaligned_constant_ratio_ci95']=ci
for model in ('B','R3c'):
    a,b=p['accessibility'][model]['selection']['fold_users'];us=sorted(a+b);check(model+' alternating fold A',a,us[::2]);check(model+' alternating fold B',b,us[1::2])
    actual=sorted({r['user'] for r in p['provenance'][model+'_train']['records']});check(model+' CV vs train records',us,actual)
    actual=sorted({r['user'] for r in p['provenance'][model+'_val']['records']});check(model+' val records users',actual,sorted(valusers))
result['checks']={'passed':len(checks),'failed':0,'names':checks}
result['limitations']=['Aggregate sums and provenance checked; no raw-cache reprocessing, ridge refit or M2 re-execution.','M2 per-block errors not stored in report; all-block pass attested by checked source and report.','C pertains only to mean-pooled then stateless-normalized latents; not full encoder information.','One seed and previously examined val users: no plateau, saturation, endpoint gate or FM success claim.','Shares partition additive error improvements, not causal mechanisms.']
out=Path('/private/tmp/b19618_pose_review_20261029.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
compact={'output':str(out),'checks':len(checks),'A':{k:({q:v for q,v in row.items() if q!='per_user_delta'} if k!='centered' else {g:{q:v for q,v in r.items() if q!='per_user_delta'} for g,r in row.items()}) for k,row in result['comparisons']['A'].items() if isinstance(row,dict)},'C':{k:({q:v for q,v in row.items() if q!='per_user_delta'} if k!='centered' else {g:{q:v for q,v in r.items() if q!='per_user_delta'} for g,r in row.items()}) for k,row in result['comparisons']['C'].items() if isinstance(row,dict)}}
print(json.dumps(compact,indent=2))
