# Verifica indipendente dei conteggi (solo costruzione dei moduli e aritmetica; nessun training, nessun dato).
import torch, inspect
from torch import nn
import torch.nn.functional as F
D,J,T=384,20,160
def conv(ci,co,k): return ci*co*k, co
def mlp_parts():
    dims=[384,96,96,96,64,32,20]
    return [(dims[i]*dims[i+1], dims[i+1]) for i in range(6)]
def tcn_parts():
    return [conv(384,64,1),conv(64,64,3),conv(64,64,3),conv(64,64,3),conv(64,32,1),conv(32,20,1)]
LN=2*D; AFF_W=D*J; AFF_B=J
for name,parts in (('MLP',mlp_parts()),('TCN',tcn_parts())):
    w=sum(p[0] for p in parts); b=sum(p[1] for p in parts)
    print(name,'g pesi',w,'g bias',b,'tot',LN+w+b+AFF_W+AFF_B,'pesi trasf',w+AFF_W,'bias tot',b+AFF_B,'MAC160',(w+AFF_W)*T)
print('L0',LN+AFF_W+AFF_B,'MAC160',AFF_W*T,'L13',LN+13*D*J+J,'MAC160',13*D*J*T)
# costruzione torch
g=nn.Sequential(nn.Conv1d(D,64,1),nn.ReLU(),nn.Conv1d(64,64,3,dilation=1,padding=1),nn.ReLU(),nn.Conv1d(64,64,3,dilation=2,padding=2),nn.ReLU(),
  nn.Conv1d(64,64,3,dilation=3,padding=3),nn.ReLU(),nn.Dropout(0.1),nn.Conv1d(64,32,1),nn.ReLU(),nn.Dropout(0.1),nn.Conv1d(32,J,1))
print('torch g TCN',sum(p.numel() for p in g.parameters()),'+LN+aff',sum(p.numel() for p in g.parameters())+LN+AFF_W+AFF_B)
# campo ricettivo per supporto dei kernel (composizione delle dilatazioni) e padding
offs={0}
for d in (3,2,1):
    offs={o+k*d for o in offs for k in (-1,0,1)}
print('RF offsets',min(offs),max(offs),'n',len(offs))
# centri che non toccano padding: tutte le posizioni intermedie in [0,159]
def ok(t):
    s3={t+k*3 for k in (-1,0,1)}; s2={p+k*2 for p in s3 for k in (-1,0,1)}; s1={p+k for p in s2 for k in (-1,0,1)}
    return all(0<=p<T for s in (s3,s2,s1) for p in s)
good=[t for t in range(T) if ok(t)]
print('centri senza padding',good[0],'..',good[-1],len(good))
# API generatori
print('F.dropout sig:',inspect.signature(F.dropout))
print('nn.Dropout init sig:',inspect.signature(nn.Dropout.__init__))
print('kaiming_normal_ sig:',inspect.signature(nn.init.kaiming_normal_))
print('torch version',torch.__version__)
src=inspect.getsource(torch.random.manual_seed)
print('manual_seed chiama cuda.manual_seed_all:', 'manual_seed_all' in src or 'cuda' in src)
print('bernoulli_ doc has generator:', 'generator' in (torch.Tensor.bernoulli_.__doc__ or ''))
# fan_in Kaiming per conv k3 64->64
w=torch.empty(64,64,3); fi,fo=nn.init._calculate_fan_in_and_fan_out(w); print('fan_in conv k3',fi)
