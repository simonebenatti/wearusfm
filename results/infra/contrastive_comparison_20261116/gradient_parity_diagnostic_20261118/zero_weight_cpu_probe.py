from pathlib import Path
from dataclasses import replace
import sys,tempfile,json,hashlib,time
import torch
ROOT=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-logcontrast')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/cpu')]
from test_pose_training import _pose_setup
from test_contrastive_runtime import _config
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256

def equal(a,b,path=''):
 if isinstance(a,torch.Tensor):
  assert torch.equal(a,b),path
 elif isinstance(a,dict):
  assert set(a)==set(b),path
  for k in a:equal(a[k],b[k],path+'.'+str(k))
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
 else:assert a==b,(path,a,b)

if __name__=='__main__':
 torch.set_num_threads(1)
 out=Path(tempfile.mkdtemp(prefix='wearusfm-zero-parity-',dir='/private/tmp'))
 root,manifest,pose=_pose_setup(out)
 cfg=_config(pose,'C0'); cfg=replace(cfg,max_steps=12,eval_every=100,ckpt_every=100,fixed_eval_every=0,pose_grad_every=0,pose_panel_every=0,model=replace(cfg.model,muscle_dropout=.4,grad_checkpoint=True,window_ms=100.,tokenizer_activation='gelu'),schedule=replace(cfg.schedule,total_steps=20,keep_checkpoints=()),contrastive={**cfg.contrastive,'fill_steps':256,'ramp_steps':256,'grad_every':4,'grad_steps':(8,)})
 checkpoints={}; records={};start=time.monotonic()
 for arm in ('C0','CN','CR','C0_no_diagnostics'):
  conf={**cfg.contrastive,'arm':arm if arm!='C0_no_diagnostics' else 'C0'}
  if arm.endswith('diagnostics'):conf.update(grad_every=0,grad_steps=())
  current=replace(cfg,contrastive=conf)
  s=R.train(current,manifest,[root],out/arm,log=lambda _:None)
  assert s['steps']==12 and s['stopped']=='max_steps'
  ck=torch.load(out/arm/'checkpoint.pt',map_location='cpu',weights_only=False);checkpoints[arm]=ck
  records[arm]=[json.loads(x) for x in (out/arm/'metrics.jsonl').read_text().splitlines()]
  print(arm,state_sha256(ck['student']),flush=True)
 base=checkpoints['C0'];checks={}
 for arm,ck in checkpoints.items():
  for field in ('student','teacher','optimizer','torch_rng','stream','alarm_counts'):
   equal(base[field],ck[field],arm+'.'+field)
  for field in ('queue','coverage','coverage_reports'):
   equal(base['contrastive_state'][field],ck['contrastive_state'][field],arm+'.contrastive_state.'+field)
  for a,b in zip(records['C0'],records[arm]):
   for k in ('total','grad_norm','grad_clip_factor','contrast_projector_grad_norm','contrast_nce','contrast_rnc'):
    assert a[k]==b[k],(arm,a['step'],k,a[k],b[k])
  checks[arm]={'states_and_optimizer_rng_queue_exact':True,'loss_grad_clip_metrics_exact':True,'projector_gradients_zero':all(x['contrast_projector_grad_norm']==0 for x in records[arm])}
 result={'torch':torch.__version__,'platform':'CPU single-thread, synthetic real loader/trainer','steps':12,'global_diagnostics_updates':[4,8,12],'dropout':.4,'activation_checkpointing':True,'window_ms':100.,'tokenizer_activation':'gelu','results':checks,'duration_s':time.monotonic()-start,'limitations':['small 16-dim/4-slot model, CPU FP32; does not reproduce A100 bf16/NCCL nondeterminism','coverage warmup does not complete and production stream data absent']}
 (out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print('OUT',out);print(json.dumps(result,indent=2))
