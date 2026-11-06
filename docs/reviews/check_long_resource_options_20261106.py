"""Independent local proposal arithmetic from published immutable Git receipts.
No project module imported, no runtime/worktree edits, no network or submission.
The output is a proposal, not an executable launch plan or resource authorization.
"""
import csv, hashlib, io, json, math, subprocess
from collections import Counter
from pathlib import Path
ROOT=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-consolidation')
COMMIT='2a20316a0bf77c93e354d4ce3f5acee94db459df'
BASE='results/infra/consolidation_long_20261105/'
OUT=Path('/private/tmp/long_benchmark_options_20261106.json')
def blob(path): return subprocess.check_output(['git','show',COMMIT+':'+BASE+path],cwd=ROOT)
def read(path): return json.loads(blob(path))
def sha(x): return hashlib.sha256(x).hexdigest()
def canonical_sha(x): return sha(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
plan=read('plan.json');launch=read('launch_plan.json');audit=read('execution/audit/receipts_benchmark.json');owner=read('execution/audit/owner_checks_benchmark.json')
reports={};receipt_pins={}
for ident,pin in audit['receipts'].items():
    path='execution/receipts/'+ident+'.json';raw=blob(path);r=json.loads(raw)
    assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256']
    assert r['complete'] and r['source_sha256']==plan['source_sha256'] and r['recipe_sha256']==plan['recipe_sha256']
    assert r['code_sha']=='166fb8653b53439507183d76e3e36a79dd28d166'
    receipt_pins[ident]={'git_path':BASE+path,'bytes':len(raw),'sha256':sha(raw)}
    if r['kind']=='benchmark-classification':reports.setdefault(r['dataset'],{})[r['view']]=r
    else:pose=r
assert len(receipt_pins)==16 and len(reports)==5 and all(set(v)=={'pre','post','pool'} for v in reports.values())
profiles={};numeric=set()
for ds,views in reports.items():
    p=views['pool']['benchmark'];pool_work=sum(f['elapsed_s'] for f in p['fits'])+max(f['elapsed_s'] for f in p['fits'])
    for view,r in views.items():
        b=r['benchmark'];pol=r['policy'];numeric.add(canonical_sha(b['numeric_context']))
        assert b['timing_only'] and not b['validation_or_held_scored'] and b['full_fold0_train']
        assert pol==plan['classification_benchmark_policy']
        assert b['sample_rows']==b['full_train_rows'] and b['sample_classes']==b['full_train_classes']
        assert len(b['fits'])==6 and {f['C'] for f in b['fits']}=={.001,.01,.1,1.,10.,100.}
        assert all(f['converged_by_criterion'] and f['exception'] is None and f['convergence_warnings']==0 and f['n_iter']<12000 for f in b['fits'])
        evidence=r['historical_numeric_match'];assert evidence['compatible']
        old_numeric=evidence['historical_numeric_context'];new_numeric=b['numeric_context']
        assert canonical_sha(new_numeric)==evidence['numeric_sha256']
        assert canonical_sha(old_numeric)==evidence['historical_numeric_sha256']
        assert all(new_numeric[k]==old_numeric[k] for k in ('python','numpy','scipy','sklearn','features_dtype'))
        old_pools=Counter(json.dumps(x,sort_keys=True) for x in old_numeric['threadpools'])
        new_pools=Counter(json.dumps(x,sort_keys=True) for x in new_numeric['threadpools'])
        assert all(new_pools[k]>=v for k,v in old_pools.items())
        extras=list((new_pools-old_pools).elements())
        assert all(json.loads(x)['internal_api']=='openmp' and json.loads(x)['num_threads']==8 for x in extras)
        assert Counter(json.dumps(x,sort_keys=True) for x in evidence['extra_threadpools'])==Counter(extras)
        work=sum(f['elapsed_s'] for f in b['fits'])+max(f['elapsed_s'] for f in b['fits'])
        factors=[]
        for f,old in enumerate(r['historical_timing']['folds']):
            assert old['train_rows']==b['fold_train_rows'][f] and old['classes']==b['fold_train_classes'][f]
            measured=work*b['fold_train_rows'][f]/b['full_train_rows']*b['fold_train_classes'][f]/b['full_train_classes']
            floor=old['fit_seconds']*work/pool_work
            factors.append({'fold':f,'measured_scaled_seconds':measured,'historical_floor_seconds':floor,'selected':'measured' if measured>=floor else 'historical_floor','projected_work_seconds':max(measured,floor)*pol['safety_factor']*pol['data_and_prediction_multiplier']})
        profiles[ds+'/'+view]={'dataset':ds,'view':view,'work_seconds':work,'pool_work_seconds':pool_work,'folds':factors,'fixed_overhead_seconds':pol['fixed_seconds_per_job']+2*(b['load_s']+b['scale_s']),'projected_peak_rss_bytes':r['peak_rss_bytes']*pol['memory_multiplier'],'benchmark_train_rows':b['full_train_rows'],'benchmark_max_n_iter':max(f['n_iter'] for f in b['fits'])}
assert len(numeric)==1 and next(iter(numeric))==owner['classification']['numeric_sha256'][0]
# Independent recomputation of pose admission from its train-only timings.
t=pose['timings_seconds'];ratio=18548/1907
assert pose['held_payload_opened'] is False and pose['held_scored'] is False and pose['training_scores_persisted'] is False
pose_floor=4*(max(v['statistics_seconds'] for v in t['ridge_by_depth'].values())+27*max(v['three_eigensystems_seconds'] for v in t['ridge_by_depth'].values()))
other=t['read_validate']*(1+ratio)+t['fixed_alpha_fit']+t['predict']*ratio+t['summary']*ratio+t['E2_pairing']*ratio*ratio+t['bootstrap_diagnostic']*ratio+t['artifact_write']*ratio
pose_seconds=pose_floor+2*other+120
pose_ram=pose['peak_rss_bytes']*ratio
assert abs(pose_seconds-pose['projection']['seconds_per_dose'])<1e-9
assert pose_ram==pose['projection']['projected_peak_rss_bytes']
assert pose_seconds<=3300 and pose_ram<=.85*30*1024**3 and pose['projection']['admitted']
# Recompute allocation accounting from all30 completed sacct rows, not summaries.
accounting={};seen=set()
for phase in ('extract','benchmark'):
    rows=list(csv.DictReader(io.StringIO(blob('execution/accounting/sacct_'+phase+'.tsv').decode()),delimiter='\t'))
    gh=ch=0.
    for r in rows:
        assert r['State']=='COMPLETED' and r['ExitCode']=='0:0' and int(r['Restarts'])==0 and r['JobIDRaw'] not in seen
        seen.add(r['JobIDRaw']);tres=dict(v.split('=',1) for v in r['AllocTRES'].split(','));seconds=int(r['ElapsedRaw'])
        gh+=seconds*int(tres.get('gres/gpu','0'))/3600;ch+=seconds*int(r['AllocCPUS'])/3600
    accounting[phase]={'allocations':len(rows),'actual_gpu_hours':gh,'actual_allocated_core_hours':ch}
spent_core=sum(v['actual_allocated_core_hours'] for v in accounting.values());spent_gpu=sum(v['actual_gpu_hours'] for v in accounting.values())
cap=launch['caps']['allocated_core_hours']
remaining_other=[j for j in launch['jobs'] if j['stage'] in ('fit-pose','compare-classification','compare-pose','summary')]
other_cap=sum(j['allocation']['cpus']*j['allocation']['seconds']/3600 for j in remaining_other)+2*2*600/3600
assert abs(other_cap-41)<1e-9
completed_reserved_cap=sum(j['allocation']['cpus']*j['allocation']['seconds']/3600 for j in launch['jobs'] if j['phase'] in ('extract','benchmark'))+2*2*600/3600
# A1 classification IDs only. Unstarted pose/compare/summary and two checkout
# logical IDs stay unchanged. No identifier below is submitted or reserved.
def shapes(ds,view,option):
    if option=='original':return [[0,1],[2,3]] if ds=='ninapro_db2' else [[0,1,2,3]]
    if ds=='ninapro_db2':return [[0],[1],[2],[3]] if view in ('pre','pool') else [[0,1],[2,3]]
    if ds=='ninapro_db3' and view=='pre':return [[0,1],[2,3]] if option=='db3_pairs' else [[0],[1],[2],[3]]
    if ds=='ninapro_db6' and view=='pool':return [[0,1],[2,3]]
    return [[0,1,2,3]]
def matrix(option):
    jobs=[]
    for step in (19618,39237,49046):
        for view in ('pre','post','pool'):
            if step==19618 and view=='pool':continue
            for ds in reports:
                q=profiles[ds+'/'+view]
                for fs in shapes(ds,view,option):
                    projection=sum(q['folds'][f]['projected_work_seconds'] for f in fs)+q['fixed_overhead_seconds']
                    seconds=math.ceil((projection+600)/900)*900
                    stem=f'LONG-FIT-G{step}-{view}-{ds}-F'+''.join(map(str,fs))+'-20261105'
                    jobs.append({'id':stem+('' if option=='original' else '-A1'),'dataset':ds,'view':view,'step':step,'folds':fs,'projection_seconds':projection,'uniform4h_gate_passed':projection<=13800,'uniform4h_gate_margin_seconds':13800-projection,'measured_walltime_seconds':seconds,'measured_gate_limit_seconds':seconds-600,'measured_gate_margin_seconds':seconds-600-projection,'max4h_passed':seconds<=14400,'gpus':0,'cpus':8,'memory_gib':30,'out':plan['output_root']+('/fits/classification' if option=='original' else '/fits_a1/classification')+f'/{step}/{view}/{ds}/F'+''.join(map(str,fs))})
    assert len({j['id'] for j in jobs})==len(jobs) and len({j['out'] for j in jobs})==len(jobs)
    assert len({(j['step'],j['view'],j['dataset']) for j in jobs})==40
    for cell in {(j['step'],j['view'],j['dataset']) for j in jobs}:
        assert sorted(f for j in jobs if (j['step'],j['view'],j['dataset'])==cell for f in j['folds'])==[0,1,2,3]
    total=sum(j['projection_seconds'] for j in jobs);uniform=len(jobs)*8*4;measured=sum(j['measured_walltime_seconds']*8/3600 for j in jobs)
    budget={}
    for key,amount in (('uniform4h',uniform),('rounded15min',measured)):
        future=amount+other_cap;whole=spent_core+future
        budget[key]={'classification_reserved_core_hours':amount,'future_other_reserved_core_hours':other_cap,'future_total_reserved_core_hours':future,'past_actual_plus_future_cap_core_hours':whole,'headroom_to_original_total_cap_core_hours':cap-whole,'whole_campaign_using_past_original_reservations_core_hours':completed_reserved_cap+future,'within_original_total_cap':whole<=cap}
    return {'classification_fit_jobs':len(jobs),'new_logical_cells':40,'all_compute_jobs_including_completed':len(jobs)+34,'all_jobs_including_four_checkouts':len(jobs)+38,'future_jobs_including_two_checkouts':len(jobs)+8,'projected_classification_seconds':total,'projected_classification_core_hours':8*total/3600,'worst_projected_seconds':max(j['projection_seconds'] for j in jobs),'minimum_uniform4h_gate_margin_seconds':min(j['uniform4h_gate_margin_seconds'] for j in jobs),'minimum_rounded_gate_margin_seconds':min(j['measured_gate_margin_seconds'] for j in jobs),'all_uniform4h_admitted':all(j['uniform4h_gate_passed'] for j in jobs),'all_rounded_within4h':all(j['max4h_passed'] for j in jobs),'budget':budget,'jobs':jobs}
options={o:matrix(o) for o in ('original','db3_pairs','db3_singles')}
orig=options['original'];assert orig['classification_fit_jobs']==48
assert round(orig['projected_classification_seconds'],3)==owner['gate']['projected_seconds_total']
assert round(orig['projected_classification_core_hours'],6)==owner['gate']['projected_core_hours']
assert sum(not j['uniform4h_gate_passed'] for j in orig['jobs'])==owner['gate']['shards_over_limit']==15
blocked_cells={(j['step'],j['view'],j['dataset']) for j in orig['jobs'] if not j['uniform4h_gate_passed']};assert len(blocked_cells)==owner['gate']['cells_over_limit']==10
assert all(options[o]['all_rounded_within4h'] for o in ('db3_pairs','db3_singles'))
# List every original unsubmitted fit ID and the proposed successors; input
# receipts retain their old source hash and are never relabelled.
original_ids={j['id']:j for j in orig['jobs']}
for option in ('db3_pairs','db3_singles'):
    mapping=[]
    for ident,old in original_ids.items():
        successors=[j['id'] for j in options[option]['jobs'] if (j['step'],j['view'],j['dataset'])==(old['step'],old['view'],old['dataset']) and set(j['folds']).issubset(old['folds'])]
        assert successors
        mapping.append({'old_unsubmitted_id':ident,'proposed_successor_ids':successors,'disposition':'superseded only after explicit amendment approval; never submit original ID'})
    options[option]['old_to_new_fit_ids']=mapping
    options[option]['executable_with_current_frozen_runtime']=False
    options[option]['proposed_ids_reserved_or_submitted']=False
    options[option]['grouped_allocations']=[{'dataset':ds,'view':v,'fold_shards':shapes(ds,v,option),'per_dose_walltime_seconds':[j['measured_walltime_seconds'] for j in options[option]['jobs'] if j['step']==(39237 if v=='pool' else 19618) and j['view']==v and j['dataset']==ds],'per_dose_projected_seconds':[j['projection_seconds'] for j in options[option]['jobs'] if j['step']==(39237 if v=='pool' else 19618) and j['view']==v and j['dataset']==ds],'doses':2 if v=='pool' else 3} for ds in reports for v in ('pre','post','pool')]
assert all(j.get('slurm_job_id') is None for j in launch['jobs'] if j['stage']=='fit-classification')
all_input_receipts=dict(receipt_pins)
for ident,pin in read('execution/audit/receipts_extract.json')['receipts'].items():
    rel='execution/receipts/'+ident+'.json';raw=blob(rel);r=json.loads(raw)
    assert sha(raw)==pin['sha256'] and len(raw)==pin['bytes'] and r['complete']
    assert r['source_sha256']==plan['source_sha256'] and r['recipe_sha256']==plan['recipe_sha256']
    all_input_receipts[ident]={'git_path':BASE+rel,'bytes':len(raw),'sha256':sha(raw)}
assert len(all_input_receipts)==28
result={'status':'proposal_only_awaiting_Simone_decision_and_Claude_review_not_a_launch_plan','git_commit':COMMIT,'source_sha256':plan['source_sha256'],'recipe_sha256':plan['recipe_sha256'],'receipt_pins':receipt_pins,'proposed_original_source_bridge_28_input_pins':all_input_receipts,'recommended_option':'db3_singles/rounded15min','operational_order_proposed':'after review/authorization pose fits first on the same immutable fit-phase checkout; at most2 outstanding line2 CPU jobs including pose until scaling comparison has been enqueued; owner Claude coordinates','numeric_sha256':next(iter(numeric)),'verified_all16_receipt_hashes':True,'verified_30_completed_allocations':True,'pose_recalculation':{'seconds_per_dose':pose_seconds,'fit_floor_seconds':pose_floor,'projected_rss_bytes':pose_ram,'admitted':True,'score_policy':'train timings only, held not opened/scored'},'classification_profiles':profiles,'original_gate_independent_match':{'all_passed':False,'blocked_shards':15,'blocked_cells':10,'projected_seconds':orig['projected_classification_seconds'],'projected_core_hours':orig['projected_classification_core_hours'],'comparison':'owner_checks totals equal at its published decimal precision'},'accounting':{'phases':accounting,'spent_actual_gpu_hours':spent_gpu,'spent_actual_core_hours':spent_core,'original_gpu_cap':launch['caps']['gpu_hours'],'original_core_cap':cap,'remaining_core_cap_after_spend':cap-spent_core,'completed_phase_original_reservations_core_hours':completed_reserved_cap,'remaining_nonclassification_cap_core_hours':other_cap,'future_gpu_needed_if_cache_compatibility_amendment_approved':0},'allocation_proposal_rule':{'uniform_seconds':14400,'measured_seconds':'ceil((projection_seconds+600)/900)*900','maximum_seconds':14400,'gate':'projection_seconds <= selected_walltime_seconds - 600','scientific_terms_unchanged':'all six C candidates plus validation-selected refit per fold, all160 folds across40 logical cells; same recipe, data, dose, numerical context and statistical reading','note':'projection includes the already-frozen x3*x1.5 and fixed600s. Additional600s is allocation reserve, not a reduction of existing prudence. Rounded walltimes propose a policy amendment and are not accepted by current launcher.'},'options':options,'retained_unstarted_nonclassification_job_ids':[j['id'] for j in remaining_other]+[x['id'] for x in launch['phase_checkouts'] if x['code_sha'] is None],'requirements':['No source/pin/receipt changes implemented by this script.','Current strict-source policy rejects these changes and invalidates old extraction+benchmark receipts. Cache-preserving audited amendment and new phase source/pins/review required before any launch.','Do not lower safety factors or C grid. All projected timings remain empirical estimates; no automatic retry or dose extension.','Original unsubmitted classification fit IDs would be superseded by explicit A1 list only after approval. No duplicate original IDs may be submitted.','Owner Claude remains responsible for all operational checks and submission.']}
OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(OUT)
print('ACCOUNTING',json.dumps(result['accounting'],indent=2))
for option,v in options.items():
 print(option,json.dumps({k:x for k,x in v.items() if k!='jobs'},indent=2))
