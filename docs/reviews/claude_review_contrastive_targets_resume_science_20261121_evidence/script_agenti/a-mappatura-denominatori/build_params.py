import json, sys, hashlib
import torch
C='/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_resume/clone/'
from contrastive_prepare import config_from_dict
from wearusfm.model.fm import WearUsFM
from wearusfm.training.contrastive_objectives import attach_contrastive_head
for arm in ('CN','CR'):
    raw=json.load(open(C+f'results/infra/contrastive_targets_20261118/benchmark_execution_20261121/{arm}/config.json'))
    cfg=config_from_dict(raw)
    torch.manual_seed(cfg.seed)
    m=WearUsFM(cfg.model); attach_contrastive_head(m)
    names=[n for n,p in m.named_parameters() if p.requires_grad]
    allp=list(m.named_parameters())
    print(arm,'n trainable',len(names),'n all',len(allp),'total elems',sum(p.numel() for n,p in allp if p.requires_grad), 'freeze', cfg.freeze_encoder, 'lr',cfg.lr,'wd',cfg.weight_decay,'clip',cfg.grad_clip)
    for i,(n,p) in enumerate([(n,p) for n,p in m.named_parameters() if p.requires_grad]):
        if i>=258: print('  ',i,n,tuple(p.shape),p.numel(),p.dtype)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    print('  groups',len(opt.param_groups),[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups])
    print('  head elems', sum(p.numel() for p in m.contrastive_head.parameters()))
    json.dump(names, open(sys.argv[1]+f'/names_{arm}.json','w'))
