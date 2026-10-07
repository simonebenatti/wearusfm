#!/usr/bin/env python3
"""Small REAL WearUsFM CUDA/bf16/checkpointing invariants, within approved GPU job."""
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src"))
from wearusfm.harness.fm_features import myo8_montage
from wearusfm.model import channel_codes as CC
from wearusfm.model.anchor_targets import anchor_targets
from wearusfm.model.channel_identity import codes_to_tensors
from wearusfm.model.fm import ModelInputs, WearUsFM
from wearusfm.model.local_encoder import sets_to_tensors
from wearusfm.model.spectral_step1 import global_spectral_targets, keep_objective, pool_p1p2
from wearusfm.training import run as R
from wearusfm.training.jepa import ema_update, make_teacher


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    torch.manual_seed(0)
    cfg = replace(R.small_config().model, keep_readout=True, multiscale_anchor=True, grad_checkpoint=True)
    model = WearUsFM(cfg).cuda().train()
    rng = np.random.default_rng(0)
    signals = [rng.normal(size=(8, n)).astype(np.float32) for n in (2000,1500,1000)]
    m = myo8_montage("step1_synthetic",1000.)
    code = CC.anatomy_codes(m); sets = CC.attention_sets(CC.layout_from_montage(m),8)
    inp = ModelInputs([torch.from_numpy(x).cuda() for x in signals], [1000.]*3,
                      {k:v.cuda() for k,v in codes_to_tensors(CC.pack_codes([code]*3)).items()},
                      {k:v.cuda() for k,v in sets_to_tensors(CC.pack_attention_sets([sets]*3)).items()},[8]*3,
                      torch.ones(24,dtype=torch.bool,device="cuda"))
    targets = [anchor_targets(x,1000.,450.,multiscale=True) for x in signals]
    y, valid = global_spectral_targets(targets,inp.counts,inp.qc_valid)
    model.spectral_keep.fit_target_stats(y,valid)
    teacher = make_teacher(model)
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        loss, _ = keep_objective(model,inp,targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(norm)
    for name in ("tokenizer","local","pool","backbone","spectral_keep"):
        grads = [p.grad for p in getattr(model,name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    assert all(p.grad is None for p in teacher.parameters())
    opt.step(); ema_update(teacher,model,.996)
    assert torch.equal(model.spectral_keep.target_mean,teacher.spectral_keep.target_mean)
    print(json.dumps({"cuda_step1_checks":"passed", "torch":torch.__version__,"loss":float(loss.detach()),"grad_norm":float(norm),
                      "rng_paired":True,"bf16":True,"checkpointing":True}),flush=True)


if __name__ == "__main__":
    main()
