import json,pathlib,hashlib,csv,re,math
R=pathlib.Path('/private/tmp/wearusfm-b19618-results-20261029')
P=pathlib.Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/results/infra/pilot_posa/b19618_20261028')
def read(p):return json.loads(p.read_text())
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def chk(name,v):checks.append({'name':name,'pass':bool(v)})
m=read(R/'manifest.json'); remote={f['remote_path']:R/f['local_path'] for f in m['files']}
for f in m['files']:chk('manifest '+f['local_path'],h(R/f['local_path'])==f['remote_sha256']==f['local_sha256'] and (R/f['local_path']).stat().st_size==f['bytes'])
plan=read(P/'evaluation_plan.json'); rp=read(P/'recovery/recovery_plan.json')
chk('original plan immutable',h(P/'evaluation_plan.json')==rp['original_plan_sha256'])
cont=read(R/'training/continuation_receipt.json'); prov=read(R/'training/continuation_provenance.json'); pre=read(R/'preflight/preflight_receipt.json')
chk('continuation complete/source unchanged/stop',cont['complete'] and cont['source_unchanged'] and cont['expected_step']==19618 and cont['summary']['steps']==19618 and cont['summary']['stopped']=='max_steps' and cont['trainer_exit_code']==0)
chk('continuation provenance hash',h(R/'training/continuation_provenance.json')==cont['provenance_sha256'])
for f in ['config.json','summary.json']:chk('continuation output '+f,h(R/'training'/f)==cont['outputs'][f]['sha256'])
chk('B5000 source checkpoint',prov['source_hashes']['checkpoint.pt']==pre['B5000']['sha256'])
chk('source step',prov['source_step']==pre['B5000']['step']==5000)
chk('source config hash',prov['source_hashes']['config.json']==pre['B5000']['config_file_sha256'])
chk('same fixed eval set',h(R/'training/fixed_eval_set.json')==prov['source_hashes']['fixed_eval_set.json'])
chk('same panel set',h(R/'training/pose_panel_set.json')==prov['source_hashes']['pose_panel_set.json'])
chk('same environment freeze',h(R/'training/pip_freeze_62582916.txt')==cont['environment']['pip_freeze_sha256']==prov['source_hashes']['pip_freeze_62523672.txt'])
metrics=[json.loads(l) for l in (R/'training/metrics.jsonl').read_text().splitlines()]
chk('continuous metrics steps 5001..19618',[x['step'] for x in metrics]==list(range(5001,19619)))
chk('LR constant plateau',all(x['lr']==0.0001 for x in metrics))
ck={'B':cont['outputs']['checkpoint_step0019618.pt']['sha256'],'R3c':pre['R3c19618']['sha256']}
for p in sorted((R/'extraction_metadata').glob('*/*/*.json')):
 j=read(p);model=j['model'];chk(str(p.relative_to(R))+' identity',j['checkpoint_sha256']==ck[model] and j['checkpoint_step']==19618 and j['code_sha']==plan['environment_code_sha'])
 chk(str(p.relative_to(R))+' upstream receipt',j['receipt_sha256']==h(remote[j['receipt']]))
 if p.name=='extraction_receipt.json':
  chk(str(p.relative_to(R))+' complete unchanged',j['complete'] and j['checkpoint_unchanged'])
  for v in j['outputs'].values():
   pp=remote[v['cache']+'.provenance.json'];pj=read(pp);chk(str(pp.relative_to(R))+' hash',h(pp)==v['provenance_sha256'])
   chk(str(pp.relative_to(R))+' cache pin',pj['cache_sha256']==v['sha256'])
for p in sorted((R/'reports').glob('*.provenance.json')):
 j=read(p);chk(p.name+' report hash',h(remote[j['report']])==j['report_sha256']);chk(p.name+' complete',j['complete'])
 if j['mode']=='probe':
  r=read(remote[j['report']]);chk(p.name+' checkpoint identity',j['checkpoint_sha256']==ck[j['model']]==r['checkpoint_sha256'])
  chk(p.name+' extraction receipt hash',h(remote[j['extraction_receipt']])==j['extraction_receipt_sha256'])
  chk(p.name+' OOF pin matches report',j['oof_sha256']==r['oof_sha256'])
 elif j['mode']=='compare':
  for model,v in j['inputs'].items():
   pp=remote[v['path']+'.provenance.json'];pj=read(pp)
   chk(p.name+' '+model+' input hash',h(remote[v['path']])==v['sha256'] and h(pp)==v['provenance_sha256'])
   chk(p.name+' '+model+' checkpoint/OOF pin',v['checkpoint_sha256']==ck[model] and v['oof_sha256']==pj['oof_sha256'])
rec=read(R/'reports/R3c_p2_pool.json.provenance.json')
chk('R3c recovered evidence unchanged',rec['recovery_evidence']==rp['original_recovery_evidence'])
chk('R3c original failed log unchanged',h(R/'logs/pose_b19618_eval_62583157.out')==rec['recovery_evidence']['failure_log_sha256'])
chk('R3c recovery status',rec['recovery'] and rec['validator_mode']=='validate-probe' and rec['validation_job_id']=='62618174')
pose=read(R/'reports/pose_comparison.json')
for model in ['B','R3c']:
 for split,fn in [('train','train_subset'),('val','val')]:
  pp=read(R/f'extraction_metadata/{model}/pose/{fn}.npz.provenance.json'); emb=dict(pose['provenance'][model+'_'+split]); ep=emb.pop('provenance_sha256'); er=emb.pop('extraction_receipt_sha256'); chk(f'pose embedded {model}/{split}',emb==pp and ep==h(R/f'extraction_metadata/{model}/pose/{fn}.npz.provenance.json') and er==h(R/f'extraction_metadata/{model}/pose/extraction_receipt.json'))
for task in ['p1','p2']:
 a=read(R/f'reports/B_{task}_pool.json');b=read(R/f'reports/R3c_{task}_pool.json');chk(task+' matching splits',a['splits_sha256']==b['splits_sha256'])
 for ds,v in a['datasets'].items():
  w=b['datasets'][ds];chk(task+' '+ds+' matching input/subjects',v['input_sha256']==w['input_sha256'] and v['subjects']==w['subjects'])
  chk(task+' '+ds+' matching fold roles',all(all(x[k]==y[k] for k in ['train','val','held']) for x,y in zip(v['folds'],w['folds'])))
rows=[x for x in csv.DictReader((R/'sacct_all.stdout.txt').read_text().splitlines(),delimiter='|') if x['JobIDRaw'].isdigit()]
jobs=[]
for x in rows:
 ng=int(dict(v.split('=',1) for v in x['AllocTRES'].split(',') if '=' in v).get('gres/gpu','0'));gh=ng*int(x['ElapsedRaw'])/3600;ch=int(x['CPUTimeRAW'])/3600
 jobs.append({'id':x['JobIDRaw'],'state':x['State'],'exit_code':x['ExitCode'],'gpu_hours':gh,'allocated_core_hours':ch,'cpu_only':not ng,'start':x['Start'],'end':x['End']})
gpu=sum(j['gpu_hours'] for j in jobs);cpu=sum(j['allocated_core_hours'] for j in jobs if j['cpu_only']);old_preflight=44/3600
warningcounts={}
for p in sorted((R/'logs').glob('*.out')):
 s=p.read_text();v={name:len(re.findall(pat,s)) for name,pat in {'convergence_warning_emissions':'ConvergenceWarning:','predicted_absent_class_warning_emissions':'UserWarning: y_pred contains classes not in y_true','malformed_row_warning_emissions':'righe malformate'}.items()}
 if sum(v.values()):warningcounts[p.name]=v
result={'collection':str(R),'independent_checks_count':len(checks),'checks_failed':[x for x in checks if not x['pass']],'manifest_file_count':len(m['files']),'checkpoint_sha256':ck,'continuation':{'source_step':5000,'stop_step':19618,'metric_rows':len(metrics),'source_unchanged_receipt':True,'all_steps_present':True,'lr':0.0001},'accounting':{'jobs':jobs,'gpu_hours_collected':gpu,'cpu_only_core_hours_collected':cpu,'all_allocated_core_hours':sum(j['allocated_core_hours'] for j in jobs),'preflight_cpu_hours_already_in_base':old_preflight,'incremental_cpu_only_core_hours':cpu-old_preflight,'base_spent_gpu_hours':52.92916666666667,'base_spent_cpu_only_core_hours':71.88722222222222,'spent_gpu_hours_total':52.92916666666667+gpu,'spent_cpu_only_core_hours_total':71.88722222222222+cpu-old_preflight,'residual_gpu_hours_of_132':132-52.92916666666667-gpu,'residual_cpu_only_core_hours_of_531':531-71.88722222222222-(cpu-old_preflight)},'warnings':warningcounts,'limits':['60 text files verified against remote collection hashes. Checkpoint/cache/OOF binaries were not independently rehashed or downloaded in this review; their cross-file hashes and successful guard receipts are coherent.','Convergence logs do not identify C, fold or final-vs-grid fit. There are 280 fit calls by recipe (240 grid fits +40 selected-C refits), not18 folds/models inferred from18 warning emissions. Selected-C convergence cannot be certified from these logs.','978 predicted-class warnings total are 489 per P1 run. Text occurs twice per message (header+warnings.warn source), not1956 distinct warnings or failed fits. Both arms have identical subject/fold support; warning count is not distinct fit count.','Malformed UCI warning reports the same one row in the same file for each arm:2 log events,1 source row, no evidence of asymmetric preprocessing.','Preflight and continuation metadata establish resume provenance but do not prove numerical equality to an uninterrupted run.'],'checks':checks}
out=pathlib.Path('/private/tmp/b19618_integrity_review_20261029.json');out.write_text(json.dumps(result,indent=2)+'\n');print('wrote',out,'checks',len(checks),'FAIL',result['checks_failed']);print(json.dumps(result['accounting'],indent=2)[-800:])
