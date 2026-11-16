import itertools
import math
import pytest
import torch
from wearusfm.model.rank_contrast import pose_rank_contrast,rank_contrast_count,pose_distance

def oracle(z,target,mask,tv,temperature,stride,positives,per_window,resolution):
    points=[(s,p) for s in range(z.shape[0]) for p in range(stride//2,z.shape[1],stride) if bool(tv[s,p]) and bool(mask[s,p].any())]
    buckets={}
    for si,pi in points:
        hi=z[si,pi]-z[si,pi].mean()
        hi=hi/hi.norm()
        ds={};ls={}
        for sk,pk in points:
            if (si,pi)==(sk,pk):continue
            common=mask[si,pi]&mask[sk,pk]
            if not common.any():continue
            d=math.sqrt(sum(float(target[si,pi,j]-target[sk,pk,j])**2 for j in range(target.shape[-1]) if common[j])/int(common.sum()))
            ds[sk,pk]=math.floor(d/resolution)*resolution if resolution else d
            hk=z[sk,pk]-z[sk,pk].mean();hk=hk/hk.norm()
            ls[sk,pk]=(hi*hk).sum()/temperature
        js=[k for k in ds if positives=='all' or k[0]!=si]
        if len(js)<2:continue
        vals=[];chance=0.
        for j in js:
            eligible=[k for k in ds if ds[k]>=ds[j]]
            vals.append(torch.logsumexp(torch.stack([ls[k] for k in eligible]),0)-ls[j])
            chance+=math.log(len(eligible))
        buckets.setdefault(si,[]).append(torch.stack(vals).sum()/chance)
    if not buckets:return z.sum()*0.,0
    if per_window:return torch.stack([torch.stack(v).mean() for v in buckets.values()]).mean(),len(buckets)
    vals=[x for v in buckets.values() for x in v]
    return torch.stack(vals).mean(),len(vals)

@pytest.mark.parametrize('seed',range(16))
def test_independent_forward_and_backward_random_masks_ties(seed):
    g=torch.Generator().manual_seed(seed)
    s,p,j,d=3+seed%2,5+seed%3,3+seed%2,7
    z=torch.randn(s,p,d,generator=g,dtype=torch.float64,requires_grad=True)
    y=torch.randint(0,4,(s,p,j),generator=g).double()/4
    m=torch.rand(s,p,j,generator=g)>.35
    tv=torch.rand(s,p,generator=g)>.15
    if seed%4==0:y[~m]=float('nan')
    for positives,per_window,res in itertools.product(['all','cross'],[True,False],[0.,.5]):
        kw=dict(temperature=.07+seed*.013,stride=1+seed%3,positives=positives,per_window=per_window,resolution=res)
        actual,num,count,_=pose_rank_contrast(z,(y,m),tv,**kw)
        expected,ec=oracle(z,y,m,tv,**kw)
        assert int(count)==ec
        assert int(rank_contrast_count(m&tv[...,None],stride=kw['stride'],positives=positives,per_window=per_window))==ec
        assert torch.allclose(actual,expected,atol=1e-11,rtol=1e-10)
        ga,=torch.autograd.grad(actual,z)
        ge,=torch.autograd.grad(expected,z)
        assert torch.allclose(ga,ge,atol=1e-10,rtol=1e-9)

@pytest.mark.parametrize('positives',['all','cross'])
def test_gradcheck(positives):
    torch.manual_seed(104)
    z=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
    y=torch.randn(2,3,2,dtype=torch.float64)
    m=torch.ones_like(y,dtype=torch.bool);tv=torch.ones(2,3,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x:pose_rank_contrast(x,(y,m),tv,temperature=.1,stride=1,positives=positives,per_window=True)[0],(z,),atol=1e-5,rtol=1e-3)

@pytest.mark.parametrize('bf16_input',[False,True])
def test_autocast_cpu_fp32_computation_and_i64_count(bf16_input):
    torch.manual_seed(55)
    z=torch.randn(3,5,8)
    if bf16_input:z=z.bfloat16()
    z.requires_grad_(True)
    y=torch.randn(3,5,4);m=torch.ones_like(y,dtype=torch.bool);tv=torch.ones(3,5,dtype=torch.bool)
    kw=dict(temperature=.1,stride=1,positives='cross',per_window=True)
    plain=pose_rank_contrast(z,(y,m),tv,**kw)
    with torch.autocast('cpu',dtype=torch.bfloat16):amp=pose_rank_contrast(z,(y,m),tv,**kw)
    assert amp[0].dtype==torch.float32 and amp[2].dtype==torch.int64
    assert torch.equal(plain[0],amp[0])
    assert torch.equal(torch.autograd.grad(plain[0],z)[0],torch.autograd.grad(amp[0],z)[0])

def test_zero_support_and_padding_do_not_contaminate_distance():
    torch.manual_seed(90)
    z=torch.randn(2,6,8,requires_grad=True)
    y=torch.randn(2,6,3);m=torch.zeros_like(y,dtype=torch.bool);tv=torch.ones(2,6,dtype=torch.bool)
    y[:]=float('nan')
    val,num,count,_=pose_rank_contrast(z,(y,m),tv,temperature=.1,stride=1,positives='all',per_window=True)
    assert int(count)==0 and val==0
    assert torch.equal(torch.autograd.grad(val,z)[0],torch.zeros_like(z))
    m[:,0:3]=True;y[:,0:3]=0;tv[:,3:]=False
    val,num,count,_=pose_rank_contrast(z,(y,m),tv,temperature=.1,stride=1,positives='all',per_window=True)
    grad,=torch.autograd.grad(val,z)
    assert torch.isfinite(grad).all() and (grad[:,3:]==0).all()

@pytest.mark.xfail(reason='Fixed -1e4 masked sentinel contaminates accepted tiny-temperature logits',strict=True)
def test_accepted_temperature_preserves_mathematical_denominator():
    z=torch.tensor([[[1.,-1.,0.]],[[0.,1.,-1.]],[[-1.,0.,1.]]],dtype=torch.float64)
    y=torch.arange(3,dtype=torch.float64).reshape(3,1,1)
    m=torch.ones_like(y,dtype=torch.bool);tv=torch.ones(3,1,dtype=torch.bool)
    kw=dict(temperature=1e-5,stride=1,positives='all',per_window=True,resolution=0.)
    actual,*_=pose_rank_contrast(z,(y,m),tv,**kw)
    expected,_=oracle(z,y,m,tv,**kw)
    assert torch.allclose(actual,expected,atol=1e-8,rtol=1e-8),(actual.item(),expected.item())
