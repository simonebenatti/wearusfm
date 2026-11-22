"""CPU diagnostic only: embed the known teacher in the existing head; no fit."""
import json
import numpy as np
import torch
from pose_temponet import controls as C, heads

def run(fold):
    rng=np.random.default_rng(20261122+fold)
    x=rng.standard_normal((4,160,384)).astype(np.float16)
    meta=[{'user':f'user{i}','window':i} for i in range(4)]
    support=np.ones((4,148,20),dtype=bool)
    a=C.generate_components(x,meta,support,np.array([0,1]),np.array([2,3]),[f'user{i}' for i in range(4)],outer_fold=fold)
    pupil=heads.make_head('TCN_post',0).eval()
    layers=[m for m in pupil.deep if isinstance(m,torch.nn.Conv1d)]
    teacher=a['generator_states']['T']; linear=a['generator_states']['L']
    weight_keys=[k for k in teacher if k.endswith('.weight')]
    coeff=.4
    with torch.no_grad():
        for layer,key in zip(layers,weight_keys):
            layer.weight.zero_();layer.bias.zero_()
            w=teacher[key];b=teacher[key[:-6]+'bias']
            layer.weight[:w.shape[0],:w.shape[1]].copy_(w)
            layer.bias[:len(b)].copy_(b)
        lstd=torch.tensor(a['normalization']['L']['std'],dtype=torch.float32)
        lmean=torch.tensor(a['normalization']['L']['mean'],dtype=torch.float32)
        tstd=torch.tensor(a['normalization']['T']['std'],dtype=torch.float32)
        tmean=torch.tensor(a['normalization']['T']['mean'],dtype=torch.float32)
        pupil.affine.weight.copy_(linear['weight']/lstd[:,None])
        pupil.affine.bias.copy_((linear['bias']-lmean)/lstd)
        layers[-1].weight.mul_((coeff/tstd)[:,None,None])
        layers[-1].bias.copy_((layers[-1].bias-tmean)*coeff/tstd)
        y=pupil(torch.from_numpy(x).float()).numpy()
    expected=a['L']+coeff*a['T']
    error=y[:,6:-6]-expected[:,6:-6]
    return {'fold':fold,'maximum_absolute_error':float(abs(error).max()),'rmse':float(np.sqrt(np.mean(error**2))),'relative_rmse':float(np.sqrt(np.mean(error**2)/np.mean(expected[:,6:-6]**2))), 'weights_fitted':False,'teacher_embedding_eval_only':True}
print(json.dumps({'scope':'construction proof on CPU synthetic features, not validation on actual cached inputs or A100; dropout off as in evaluation','rows':[run(0),run(1)]},indent=2))
