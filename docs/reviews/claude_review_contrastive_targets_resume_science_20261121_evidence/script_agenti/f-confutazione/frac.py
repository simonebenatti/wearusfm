import json,sys
E=sys.argv[1]
for arm in ['CN','CR']:
  c=[r for r in map(json.loads,open(f'{E}/benchmark_execution_20261121/{arm}/metrics.jsonl'))]
  b2=0.999;b1=0.9
  def F(x,beta):
    num=sum(beta**(576-r['step'])*x(r) for r in c if r['step']>=545)
    den=sum(beta**(576-r['step'])*x(r) for r in c)
    return num/den
  post=lambda r:(r['grad_norm']*r['grad_clip_factor'])**2
  proj=lambda r:(r.get('contrast_projector_grad_norm',0)*r['grad_clip_factor'])**2
  print(arm,'post-clip total norm range',min(r['grad_norm']*r['grad_clip_factor'] for r in c),max(r['grad_norm']*r['grad_clip_factor'] for r in c), 'clip<1 in',sum(r['grad_clip_factor']<1 for r in c),'of',len(c))
  print(arm,'F_sq total %.4f'%F(post,b2), 'F_ea total %.4f'%F(post,b1))
  if arm=='CN': print(arm,'F_sq proj %.4f'%F(proj,b2),'F_ea proj %.4f'%F(proj,b1))
