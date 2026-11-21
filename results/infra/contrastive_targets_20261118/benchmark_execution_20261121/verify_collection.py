import pathlib,json,hashlib,statistics,sys
R=pathlib.Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-contrastive-targets')
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from contrastive_prepare import config_from_dict
from wearusfm.training import run as trainer
from contrastive_target_parity import validate_parity_report
p=R/'results/infra/contrastive_targets_20261118/benchmark_execution_20261121'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(f):return json.loads(f.read_text())
files=[]
for row in (p/'remote_sha256.txt').read_text().splitlines():
 h,n=row.split('  ',1); assert sha(p/n)==h,n; files.append(n)
for row in (p/'remote_bytes.txt').read_text().splitlines():
 size,n=row.split(' ',1); assert (p/n).stat().st_size==int(size),n
result={'execution_sha':'746112dd711903e5c92ade2efd90834264b2676a','verified_files':len(files),'arms':{},'training_approved':False}
origin_hashes=[]
for arm in ('C0','CN','CR'):
 d=p/arm;s=read(d/'summary.json');cfg=read(d/'config.json')
 job=read(R/f'results/infra/contrastive_targets_20261118/releases/benchmark_20261121/benchmark_{arm}.job.json')
 expected=read(R/job['config']['path'])
 assert cfg==trainer.config_to_dict(config_from_dict(expected)),arm+' recipe'
 assert s['steps']==576 and s['stopped']=='max_steps' and s['world_size']==4
 assert s['code']['sha']==result['execution_sha']
 assert s['contrastive']['provenance']['source_sha256']==job['source_sha256']
 assert s['contrastive']['provenance']['input_sha256']=={k:v['sha256'] for k,v in job['inputs'].items()}
 rows=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
 assert [r['step'] for r in rows]==list(range(1,577))
 for rank in range(4):
  profile=read(d/f'numeric_profile.rank{rank}.json')
  assert profile['rank']==rank and profile['flags']['profile']=='bf16_math_tf32_off'
 origin_hashes.append(sha(d/'origin_sessions.json'))
 out={'steps':576,'summary_sha256':sha(d/'summary.json'),'gpu_peak_gib':s['gpu_max_memory_gib'],'trainer_elapsed_s':s['elapsed_s'],
      'mean_t_step_after8_s':statistics.mean(r['t_step_s'] for r in rows[8:]),'origin_sessions_sha256':origin_hashes[-1]}
 if arm=='C0':
  parity=read(d/'zero_weight_parity.json');gate=validate_parity_report(parity)
  assert gate['status']=='PASSED';assert len(list(d.glob('zero_weight_parity.rank*.json')))==4
  for rank in range(4):
   individual=read(d/f'zero_weight_parity.rank{rank}.json');assert individual['rank']==rank
   assert [x['update'] for x in individual['points']]==[128,256]
   for point,aggregated in zip(individual['points'],parity['points']):
    assert point['update']==aggregated['update'] and point['record']==aggregated['ranks'][rank]
  out['parity']=gate;out['parity_sha256']=sha(d/'zero_weight_parity.json')
 else:
  for rank in range(4):
   trace=[json.loads(x) for x in (d/f'trace-{rank}.jsonl').read_text().splitlines()]
   assert [x['update'] for x in trace]==list(range(545,577))
   assert all(x['rank']==rank for x in trace)
  name='nce' if arm=='CN' else 'rnc'
  ratios=[r['contrast_'+name+'_weight']*r['contrast_global_grad_'+name]/r['contrast_global_grad_base'] for r in rows[512:576]]
  out['trace_range']=[545,576];out['active_gradient_ratio_median_descriptive']=statistics.median(ratios)
 result['arms'][arm]=out
assert len(set(origin_hashes))==1
result['same_origin_sessions_map']=True
trace_mismatches=[]
for rank in range(4):
 a,b=[[json.loads(x) for x in (p/arm/f'trace-{rank}.jsonl').read_text().splitlines()] for arm in ('CN','CR')]
 for left,right in zip(a,b):
  for field in ('batch_sha256','input_sha256'):
   if left[field]!=right[field]:trace_mismatches.append({'rank':rank,'update':left['update'],'field':field})
result['CN_CR_input_hash_differences_545_576']=trace_mismatches
(p/'collection_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
