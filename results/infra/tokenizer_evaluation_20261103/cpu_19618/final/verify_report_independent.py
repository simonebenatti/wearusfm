from pathlib import Path
import hashlib
import json
import math
import numpy as np

root = Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science')
base = root / 'results/infra/tokenizer_evaluation_20261103'
dest = base / 'cpu_19618/final'
raw = (dest / 'report.json').read_bytes()
report = json.loads(raw)
plan = json.loads((base / 'pose_readout_plan_19618.json').read_text())
old = json.loads((base / 'cpu_5000/final/report.json').read_text())
remote = {Path(line.split(None, 1)[1].strip()).name: line.split(None, 1)[0]
          for line in (dest / 'remote_sha256.txt').read_text().splitlines()}
assert hashlib.sha256(raw).hexdigest() == remote['report.json']
assert report['complete'] and report['status'] == 'completed'
assert report['step'] == 19618 and report['slurm_job_id'] == '63495299'
assert report['runtime']['code_sha'] == 'b315dadfedc09c522deeb56af829d34ebc04f4b4'
assert report['plan_sha256'] == hashlib.sha256((base/'pose_readout_plan_19618.json').read_bytes()).hexdigest()
assert report['recipe'] == plan['recipe'] == old['recipe']
assert report['input_pins'] == plan['caches']
assert report['reference_pins'] == plan['references'] == old['reference_pins']
assert report['runtime']['sources'] == plan['sources'] == old['runtime']['sources']
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest() == h for p,h in plan['sources'].items())
arms = ['T25-L', 'T100-L', 'T100-G']
assert set(report['cells']) == set(arms)
def eq(a, b):
    assert np.allclose(a,b,rtol=1e-12,atol=1e-14), (a,b)

answer = {'report_sha256': hashlib.sha256(raw).hexdigest(), 'groups': {}}
for arm in arms:
    c = report['cells'][arm]
    assert c['which']=='teacher' and c['checkpoint_step']==19618 and c['complete']
    assert c['checkpoint_sha256']==plan['caches'][arm]['train']['checkpoint_sha256']
    sel=c['fit']['selection'];assert not sel['boundary_unresolved']
    assert sel['fold_users']==plan['fold_users']
    assert sel['alpha']==sorted(sel['grid'],key=lambda r:(r['mse'],r['alpha']))[0]['alpha']
    assert max(sel['extensions'].values())<=3
    for kind in ('coefficients','predictions'):
        assert remote[Path(c[kind]['path']).name]==c[kind]['sha256']
    for split in ('train','val'):
        inspect=report['inspections'][arm+'/'+split]
        assert inspect['targets']==report['inspections'][arms[0]+'/'+split]['targets']
        assert inspect['users']==old['inspections'][arm+'/'+split]['users']
        assert inspect['targets']==old['inspections'][arm+'/'+split]['targets']
    assert not set(report['inspections'][arm+'/train']['users']) & set(report['inspections'][arm+'/val']['users'])

for group in ('dynamic','all','static'):
    pg=report['paired'][group]
    users=sorted(report['cells'][arms[0]]['analysis']['groups'][group]['centered']['per_user'])
    assert len(users)==15 and users==pg['users']
    moments={}
    errors=[];denom=[];point=[];cellmeta={}
    for arm in arms:
        g=report['cells'][arm]['analysis']['groups'][group]
        cg=g['centered'];rows=cg['per_user'];assert sorted(rows)==users
        for key,stated in cg['sums'].items():eq(math.fsum(rows[u]['sums'][key] for u in users),stated)
        sums=cg['sums'];mse=sums['centered']/sums['vary'];vr=sums['varp']/sums['vary']
        corr=sums['cov']/math.sqrt(sums['varp']*sums['vary'])
        eq(cg['centered_mse_ratio'],mse);eq(cg['variance_ratio'],vr);eq(cg['pooled_centered_correlation'],corr)
        eq(mse,1+vr-2*corr*math.sqrt(vr))
        eq(g['mae_rad'],sum(v['error_sum_rad'] for v in g['mae_per_user'].values())/sum(v['n_valid_elements'] for v in g['mae_per_user'].values()))
        errors.append([rows[u]['sums']['centered'] for u in users])
        denom.append([rows[u]['sums']['vary'] for u in users]);point.append(mse)
        moments[arm]=rows
        oldg=old['cells'][arm]['analysis']['groups'][group]['centered']
        eq(oldg['sums']['vary'],sums['vary']);assert oldg['n_valid_elements']==cg['n_valid_elements']
        cellmeta[arm]={'E':mse,'corr':corr,'var_ratio':vr,'mae_rad':g['mae_rad'],'median_user':float(np.median(np.array(errors[-1])/np.array(denom[-1]))),'E_5000':oldg['centered_mse_ratio'],'relative_dose_reduction':1-mse/oldg['centered_mse_ratio']}
    eq(denom[0],denom[1]);eq(denom[0],denom[2])
    for u in users:
        for k in ('n_windows','n_window_joints','n_valid_elements'):
            assert len({moments[a][u][k] for a in arms})==1
    draw=np.random.default_rng(20261102).integers(0,15,size=(2000,15))
    drawhash=hashlib.sha256(json.dumps([draw.dtype.str,draw.shape]).encode()+draw.tobytes()).hexdigest()
    assert drawhash==pg['draws_sha256'] and pg['seed']==20261102 and pg['n_bootstrap']==2000
    weights=np.stack([np.bincount(row,minlength=15) for row in draw])
    denominator=weights@np.array(denom[0]);assert np.all(denominator>0)
    drawscores=(weights@np.array(errors).T)/denominator[:,None]
    for i,a in enumerate(arms):eq(np.quantile(drawscores[:,i],[.025,.975]),pg['cell_ratio_ci95'][a])
    contrasts={}
    for i,j in ((1,0),(2,1),(2,0)):
        name=arms[i]+'_minus_'+arms[j];expected=pg['contrasts'][name]
        delta=point[i]-point[j];ci=np.quantile(drawscores[:,i]-drawscores[:,j],[.025,.975])
        relative=-delta/point[j]
        eq(delta,expected['delta_ratio']);eq(ci,expected['ci95']);eq(relative,expected['relative_error_reduction'])
        assert expected['signal_2percent_and_ci_below_zero']==bool(relative>=.02 and ci[1]<0)
        better=sum(e1<e0 for e1,e0 in zip(errors[i],errors[j]))
        contrasts[name]={'delta':delta,'ci95':ci.tolist(),'relative_reduction':relative,'better_users':better,'signal':expected['signal_2percent_and_ci_below_zero']}
    answer['groups'][group]={'cells':cellmeta,'contrasts':contrasts}
print(json.dumps(answer,indent=2))
