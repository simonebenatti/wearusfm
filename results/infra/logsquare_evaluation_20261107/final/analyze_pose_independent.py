"""Independent sufficient-statistic pose readout; no production imports."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'results/infra/logsquare_evaluation_20261107'
FINAL = BASE/'final'
ARMS = ('T25-L', 'T100-L', 'T100-G', 'T100-LQ')
CONTROLS = ('T100-L', 'T25-L', 'T100-G')
GROUPS = ('dynamic', 'all', 'static')
CODE = '9a53c7a65f59de683be093a4ad1ac450a473bdf2'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def same(actual, expected, label):
    assert np.allclose(actual, expected, atol=2e-12, rtol=2e-12), label

def pins_in(obj, expected_path):
    found=[]
    if isinstance(obj,dict):
        if obj.get('path')==expected_path and 'sha256' in obj:
            found.append(obj)
        for value in obj.values(): found.extend(pins_in(value,expected_path))
    elif isinstance(obj,list):
        for value in obj: found.extend(pins_in(value,expected_path))
    return found

def analyze(dose):
    directory=FINAL/f'pose{dose}'
    raw=gzip.decompress((directory/'report.json.gz').read_bytes())
    report=json.loads(raw)
    expected_path=f'/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/logsquare_cpu_20261107/pose/{dose}/compare/report.json'
    pins=[]
    for f in directory.glob('*.json'):
        pins.extend(pins_in(json.loads(f.read_text()),expected_path))
    assert pins, 'no archived remote pin for '+expected_path
    assert all(p['sha256']==digest(raw) and ('bytes' not in p or p['bytes']==len(raw)) for p in pins)
    plan_path=BASE/'cpu_fit_release'/f'pose_readout_plan_{dose}.json'
    plan=json.loads(plan_path.read_text())
    assert report['status']=='compared' and report['complete'] and report['stage']=='compare'
    assert report['slurm_job_id']==('63786157' if dose==5000 else '63786158')
    accounting=json.loads((directory/'verification.json').read_text())['accounting']
    assert accounting['State']=='COMPLETED' and accounting['ExitCode']=='0:0' and accounting['JobIDRaw']==report['slurm_job_id']
    assert accounting['AllocCPUS']==2 and 0<accounting['ElapsedRaw']<=1800
    assert report['runtime']['code_sha']==CODE and report['plan_sha256']==digest(plan_path.read_bytes())
    eval_raw=json.dumps({k:v for k,v in plan.items() if k not in ('status','fit_release')},sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert report['evaluation_sha256']==digest(eval_raw)
    assert report['recipe']==plan['recipe'] and report['input_pins']==plan['caches'] and report['reference_pins']==plan['references']
    for source,h in report['runtime']['sources'].items(): assert digest((ROOT/source).read_bytes())==h
    histpin=plan['historical']['report'];histbytes=(ROOT/histpin['path']).read_bytes()
    assert digest(histbytes)==histpin['sha256'];historical=json.loads(histbytes)
    assert set(report['cells'])==set(ARMS)
    for arm in CONTROLS: assert report['cells'][arm]==historical['cells'][arm]
    output={'checkpoint_step':dose,'original_report':{'path':expected_path,'sha256':digest(raw),'bytes':len(raw)},
            'original_pin_verified':True,'accounting':accounting,'historical_cells_bit_preserved':True,'groups':{},'alpha_selection':{}}
    for arm in ARMS:
        cell=report['cells'][arm];selection=cell['fit']['selection']
        assert cell['complete'] and cell['which']=='teacher' and cell['checkpoint_step']==dose and cell['arm']==arm
        assert selection['boundary_unresolved'] is False
        assert selection['fold_users']==plan['fold_users']
        chosen=min(selection['grid'],key=lambda r:(r['mse'],r['alpha']))
        assert selection['alpha']==chosen['alpha'] and selection['cv_mse']==chosen['mse']
        assert all(0<=x<=3 for x in selection['extensions'].values())
        assert min(x['alpha'] for x in selection['grid'])<selection['alpha']<max(x['alpha'] for x in selection['grid'])
        output['alpha_selection'][arm]={'alpha':selection['alpha'],'cv_mse':selection['cv_mse'],
             'extensions':selection['extensions'],'boundary_unresolved':False,'selected_grid_minimum':True,'interior_to_evaluated_grid':True}
    for group in GROUPS:
        groupout={'arms':{},'contrasts':{}}
        ref=report['cells']['T100-L']['analysis']
        rows={a:report['cells'][a]['analysis']['groups'][group]['centered']['per_user'] for a in ARMS}
        users=sorted(rows['T100-L']);assert len(users)==15
        draws=np.random.default_rng(20261102).integers(0,len(users),size=(2000,len(users)))
        counts=np.asarray([np.bincount(row,minlength=len(users)) for row in draws])
        den=np.asarray([rows['T100-L'][u]['sums']['vary'] for u in users],dtype=np.float64)
        bden=counts@den;assert np.all(bden>0)
        errors={}
        for arm in ARMS:
            an=report['cells'][arm]['analysis']; ag=an['groups'][group]; r=rows[arm]
            assert an['support_sha256']==ref['support_sha256'] and an['targets_sha256']==ref['targets_sha256']
            assert sorted(r)==users and ag['n_selected_windows']==ref['groups'][group]['n_selected_windows']
            for u in users:
                for k in ('n_windows','n_window_joints','n_valid_elements'):assert r[u][k]==rows['T100-L'][u][k]
                assert r[u]['sums']['vary']==rows['T100-L'][u]['sums']['vary']
            sums={k:math.fsum(r[u]['sums'][k] for u in users) for k in r[users[0]]['sums']}
            n=sum(r[u]['n_valid_elements'] for u in users)
            means={k:v/n for k,v in sums.items()}
            errors[arm]=np.asarray([r[u]['sums']['centered'] for u in users])
            edyn=sums['centered']/sums['vary'];vr=sums['varp']/sums['vary'];corr=sums['cov']/math.sqrt(sums['varp']*sums['vary'])
            ci=np.quantile((counts@errors[arm])/bden,[.025,.975]).tolist()
            same(edyn,ag['centered']['centered_mse_ratio'],'E_dyn')
            same(vr,ag['centered']['variance_ratio'],'var ratio')
            same(corr,ag['centered']['pooled_centered_correlation'],'corr')
            same(means['total'],means['mean_error']+means['centered'],'MSE decomposition')
            same(edyn,1+vr-2*corr*math.sqrt(vr),'centered moment identity')
            same(ci,report['paired'][group]['cell_ratio_ci95'][arm],'E_dyn bootstrap')
            mae_rows=ag['mae_per_user']
            mae=math.fsum(v['error_sum_rad'] for v in mae_rows.values())/sum(v['n_valid_elements'] for v in mae_rows.values())
            same(mae,ag['mae_rad'],'MAE aggregation')
            user_ratios={u:r[u]['sums']['centered']/r[u]['sums']['vary'] for u in users}
            groupout['arms'][arm]={'E_dyn':edyn,'E_dyn_ci95_users':ci,'correlation':corr,'variance_ratio':vr,
               'amplitude_ratio':math.sqrt(vr),'MAE_rad':mae,'MAE_deg':mae*180/math.pi,
               'MSE_standardized':means['total'],'MSE_level_standardized':means['mean_error'],
               'MSE_centered_standardized':means['centered'],'target_temporal_variance':means['vary'],
               'prediction_temporal_variance':means['varp'],'gain_level':means['gain_mean'],
               'gain_centered':means['gain_centered'],'n_users':len(users),'n_selected_windows':ag['n_selected_windows'],
               'n_valid_elements_centered':n,'median_user_E_dyn':float(np.median(list(user_ratios.values()))),
               'users_better_than_centered_constant':sum(x<1 for x in user_ratios.values()),'per_user_E_dyn':user_ratios}
        for control in CONTROLS:
            dif=errors['T100-LQ']-errors[control]
            delta=float(np.sum(dif)/np.sum(den));ci=np.quantile((counts@dif)/bden,[.025,.975]).tolist()
            rel=float(-np.sum(dif)/np.sum(errors[control]));flag=rel>=.02 and ci[1]<0
            stored=report['paired'][group]['contrasts']['T100-LQ_minus_'+control]
            same(delta,stored['delta_ratio'],'delta');same(ci,stored['ci95'],'delta bootstrap');same(rel,stored['relative_error_reduction'],'relative gain')
            assert flag==stored['signal_2percent_and_ci_below_zero']
            groupout['contrasts'][control]={'role':'primary' if control=='T100-L' else 'secondary',
                'delta_E_dyn_LQ_minus_control':delta,'ci95':ci,'relative_error_reduction':rel,
                'signal_2percent_and_ci_below_zero':flag,'users_LQ_lower_error':int(np.sum(dif<0)),
                'users_LQ_equal_error':int(np.sum(dif==0)),'users_LQ_higher_error':int(np.sum(dif>0)),
                'per_user_delta_E_dyn':{u:float(dif[i]/den[i]) for i,u in enumerate(users)}}
        output['groups'][group]=groupout
    diag=report['log_weight_diagnostics']
    output['log_weights_descriptive']={kind:{'families':diag[kind]['families'],
       'nonzero_columns':int(np.count_nonzero(diag[kind]['column_l2_norms']))} for kind in ('student','teacher')}
    output['all_reported_group_statistics_and_bootstrap_reproduced']=True
    return output

def main():
    doses={str(d):analyze(d) for d in (5000,19618)}
    progression={}
    for arm in ARMS:
        first=doses['5000']['groups']['dynamic']['arms'][arm];last=doses['19618']['groups']['dynamic']['arms'][arm]
        progression[arm]={'E_dyn_delta_19618_minus_5000':last['E_dyn']-first['E_dyn'],
             'E_dyn_relative_reduction':1-last['E_dyn']/first['E_dyn'],
             'MAE_relative_reduction':1-last['MAE_rad']/first['MAE_rad'],
             'MSE_level_relative_reduction':1-last['MSE_level_standardized']/first['MSE_level_standardized'],
             'MSE_centered_relative_reduction':1-last['MSE_centered_standardized']/first['MSE_centered_standardized']}
    result={'protocol':'logsquare-pose-independent-readout-20261107','complete':True,
       'method':'Independent aggregation from per-user sufficient statistics, bootstrap multiplicities; no production imports.',
       'bootstrap':{'unit':'user','n_replicates':2000,'seed':20261102,'ci':'percentile 2.5/97.5','dropped_draws':0},
       'scope':'Single seed, reused development validation, frozen teacher mean-K64 ridge; uncertainty is across users, not trainings.',
       'primary_control':'T100-L','secondary_controls':['T25-L','T100-G'],'doses':doses,
       'progression_dynamic_descriptive_only':progression}
    output=FINAL/'pose_independent_analysis.json';output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for dose,data in doses.items():
        print('DOSE',dose)
        for arm,v in data['groups']['dynamic']['arms'].items():print(arm,{k:v[k] for k in ('E_dyn','E_dyn_ci95_users','correlation','variance_ratio','MAE_rad','MSE_level_standardized','MSE_centered_standardized','users_better_than_centered_constant')})
        print('CONTRASTS',{k:{x:y for x,y in v.items() if x!='per_user_delta_E_dyn'} for k,v in data['groups']['dynamic']['contrasts'].items()})
        print('ALPHAS',data['alpha_selection'])
    print('PROGRESSION',progression)

if __name__=='__main__':main()
