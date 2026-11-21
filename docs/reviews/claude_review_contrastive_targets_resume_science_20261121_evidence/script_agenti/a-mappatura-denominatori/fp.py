import json, copy, torch
C='/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_resume/clone/'
from contrastive_prepare import config_from_dict
from wearusfm.training import run as trainer
from wearusfm.training.pairing import state_sha256
import inspect
print(inspect.getsource(state_sha256)[:1500])
for arm in ('CN','CR'):
    raw=json.load(open(C+f'results/infra/contrastive_targets_20261118/benchmark_execution_20261121/{arm}/config.json'))
    cfg=config_from_dict(raw)
    ps=[torch.nn.Parameter(torch.zeros(1)) for _ in range(270)]
    opt=torch.optim.AdamW(ps, lr=cfg.lr, weight_decay=cfg.weight_decay)
    g=copy.deepcopy(opt.state_dict()['param_groups'])
    for x in g: x['lr']=trainer._lr_at(544,cfg)
    print(arm, trainer._lr_at(544,cfg), state_sha256(g))
    for n in (269,271):
        ps2=[torch.nn.Parameter(torch.zeros(1)) for _ in range(n)]
        o2=torch.optim.AdamW(ps2, lr=cfg.lr, weight_decay=cfg.weight_decay); g2=copy.deepcopy(o2.state_dict()['param_groups'])
        for x in g2: x['lr']=trainer._lr_at(544,cfg)
        print('  n',n,state_sha256(g2))
    print(g[0].keys())
