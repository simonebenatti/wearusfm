import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.model.anchor_targets import anchor_targets
from wearusfm.model.fm import WearUsFM
from wearusfm.model.spectral_step1 import (SpectralKeepHead, global_spectral_targets, keep_objective, pool_p1p2)
from wearusfm.training import run as R
from wearusfm.training.jepa import JEPAConfig, ema_update, make_teacher
from wearusfm.training.spectral_calibration import calibration_signature, read_calibration, sha256_file
from test_pretraining_loader import _tree


def test_pool_qc_padding_heterogeneous_and_gradient():
    local = torch.arange(5*4*2, dtype=torch.float32).reshape(5, 4, 2).requires_grad_()
    pv = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,0,0],[1,1,0,0],[1,1,0,0]], dtype=torch.bool)
    qc = torch.tensor([True,False,True,False,True])
    z = torch.randn(2, 4, 2, 2, requires_grad=True)
    tv = torch.tensor([[1,1,1,1],[1,1,0,0]], dtype=torch.bool)
    enc = SimpleNamespace(local=local, z=z, patch_valid=pv, visible=pv & qc[:,None], key_time_valid=tv)
    out = pool_p1p2(enc, [2,3], qc)
    torch.testing.assert_close(out[:,2:], torch.stack([local[0].mean(0), (local[2,:2].mean(0)+local[4,:2].mean(0))/2]))
    out.sum().backward()
    assert local.grad[1].abs().sum() == 0 and local.grad[3].abs().sum() == 0
    assert local.grad[2,2:].abs().sum() == 0 and z.grad[1,2:].abs().sum() == 0
    enc.visible[0,0] = False
    with pytest.raises(ValueError, match="UNMASKED"):
        pool_p1p2(enc, [2,3], qc)


def test_global_targets_gain_shape_linear_average_missing_and_silence():
    fs = 1000.
    t = np.arange(2000)/fs
    x = np.stack([np.sin(2*np.pi*100*t), 2*np.sin(2*np.pi*100*t)])
    a = anchor_targets(x, fs, 450., multiscale=True)
    b = anchor_targets(3*x, fs, 450., multiscale=True)
    y, m = global_spectral_targets([a], [2], np.ones(2,bool))
    y2, m2 = global_spectral_targets([b], [2], np.ones(2,bool))
    for offset, nb in [(0,6),(7,24)]:
        valid = m[0,offset:offset+nb]
        np.testing.assert_allclose(y[0,offset:offset+nb][valid], y2[0,offset:offset+nb][valid], atol=2e-5)
        assert y2[0,offset+nb]-y[0,offset+nb] == pytest.approx(2*np.log(3), abs=2e-6)
        assert np.exp(y[0,offset+nb]) == pytest.approx(1.25, rel=.002)
    assert np.array_equal(m,m2)
    # QC-excluded high-power channel must not affect either shape or energy.
    only_first, _ = global_spectral_targets([a], [2], np.array([True,False]))
    for idx in [6,31]:
        assert np.exp(only_first[0,idx]) == pytest.approx(.5,rel=.002)
    a.ms_fast_valid[1] = False
    _, masked = global_spectral_targets([a], [2], np.ones(2,bool))
    assert not masked[0,:7].any() and masked[0,7:].any()
    a.ms_slow_available[1,-1] = False
    _, masked = global_spectral_targets([a], [2], np.ones(2,bool))
    assert not masked[0,30] and masked[0,31]
    silent = anchor_targets(np.zeros_like(x), fs, 450., multiscale=True)
    ys, ms = global_spectral_targets([silent], [2], np.ones(2,bool))
    assert ms.sum() == 2 and ms[0,6] and ms[0,31]
    np.testing.assert_allclose(ys[0,[6,31]], np.log(1e-8), atol=1e-5)


def test_stats_mask_constant_buffers_and_equal_family_weight():
    h = SpectralKeepHead(4)
    y = np.tile(np.arange(4)[:,None], (1,32)).astype(float)
    m = np.ones_like(y,bool)
    y[:,3] = 7.; m[:,4] = False; y[:,4] = np.nan
    h.fit_target_stats(y,m)
    assert not h.target_active[3] and not h.target_active[4]
    h2 = SpectralKeepHead(4); h2.load_state_dict(h.state_dict())
    assert h2.stats_fitted and torch.equal(h2.target_mean,h.target_mean)
    pred = torch.ones(2,32, requires_grad=True)
    yt = h.target_mean.repeat(2,1)
    loss, metrics = h.loss(pred,yt, np.ones((2,32),bool))
    assert float(loss.detach()) == pytest.approx(.5)
    loss.backward(); assert torch.isfinite(pred.grad).all()
    assert set(metrics) == {"keep_fast_shape","keep_fast_energy","keep_slow_shape","keep_slow_energy","keep_valid_samples"}
    pred2 = torch.zeros(1,32); pred2[:,7:31] = 2
    l,_ = h.loss(pred2,h.target_mean[None,:],np.ones((1,32),bool))
    assert float(l) == pytest.approx(1.5/4)  # NOT 24 times the energy family
    missing = np.zeros((2,32),bool)
    missing[0,:6] = True; missing[1,31] = True
    varied = torch.ones(2,32); varied[1,31] = 2
    l, _ = h.loss(varied,h.target_mean.repeat(2,1),missing)
    assert float(l) == pytest.approx((.5+1.5)/2)  # equal samples, even with different families
    empty_prediction = torch.randn(2,32,requires_grad=True)
    l, _ = h.loss(empty_prediction,h.target_mean.repeat(2,1),np.zeros((2,32),bool))
    assert float(l.detach()) == 0.
    l.backward(); assert torch.equal(empty_prediction.grad,torch.zeros_like(empty_prediction))


def _setup(tmp_path):
    root, manifest = _tree(tmp_path)
    cfg = R.small_config(datasets=("ninapro_db2",), max_steps=1)
    cfg = replace(cfg, model=replace(cfg.model, multiscale_anchor=True, keep_readout=True),
                  loader=replace(cfg.loader, multiscale_anchor=True))
    idx = R.load_index(manifest,[root],cfg.datasets)
    batch = L.PretrainLoader(idx,cfg.loader).batch(3,np.random.default_rng(7))
    y,m = global_spectral_targets(batch.anchor_targets,batch.counts,batch.qc_valid)
    # Extra independent training windows ensure calibration variation in every observed family.
    more = L.PretrainLoader(idx,cfg.loader).batch(3,np.random.default_rng(11))
    y2,m2 = global_spectral_targets(more.anchor_targets,more.counts,more.qc_valid)
    return root, manifest, cfg, batch, np.concatenate([y,y2]), np.concatenate([m,m2])


@pytest.mark.parametrize("checkpointing", [False, True])
def test_real_small_encoder_keep_gradient_rng_optimizer_ema(tmp_path, checkpointing):
    _,_,cfg,batch,y,m = _setup(tmp_path)
    cfg = replace(cfg, model=replace(cfg.model, grad_checkpoint=checkpointing))
    torch.manual_seed(0)
    student = WearUsFM(cfg.model)
    student.spectral_keep.fit_target_stats(y,m)
    teacher = make_teacher(student)
    inp,visible,_ = L.to_model_inputs(batch)
    rng = torch.get_rng_state().clone()
    loss,metrics = keep_objective(student,inp,batch.anchor_targets)
    assert torch.equal(torch.get_rng_state(),rng)
    loss.backward()
    assert torch.equal(torch.get_rng_state(),rng)  # checkpoint recomputation also preserves RNG
    for name in ("tokenizer","local","pool","backbone","spectral_keep"):
        grad = [p.grad for p in getattr(student,name).parameters() if p.grad is not None]
        assert grad and all(torch.isfinite(g).all() for g in grad) and sum(float(g.abs().sum()) for g in grad) > 0
    assert all(p.grad is None for p in teacher.parameters())
    opt = torch.optim.AdamW(student.parameters(),lr=.001)
    before = student.spectral_keep.linear.weight.detach().clone()
    opt.step(); ema_update(teacher,student,.5)
    assert not torch.equal(before,student.spectral_keep.linear.weight)
    torch.testing.assert_close(teacher.spectral_keep.linear.weight, .5*(before+student.spectral_keep.linear.weight))
    assert torch.equal(teacher.spectral_keep.target_mean,student.spectral_keep.target_mean)
    with pytest.raises(ValueError,match="UNMASKED"):
        pool_p1p2(student.encode(inp,visible), inp.counts,inp.qc_valid)


def _calibration(path,cfg,manifest,y,m):
    meta = calibration_signature(cfg,manifest,{},None)
    meta.update(role="train",subjects=["ninapro_db2/s01"],split_sha256="a"*64,subset_sha256="b"*64)
    np.savez(path,y=y,valid=m,subject=np.repeat("ninapro_db2/s01",len(y)),split=np.repeat("train",len(y)),metadata_json=np.array(json.dumps(meta)))


def test_real_trainer_keep_checkpoint_resume_identity(tmp_path):
    root,manifest,cfg,batch,y,m = _setup(tmp_path)
    calibration = tmp_path/"calibration.npz"
    _calibration(calibration,cfg,manifest,y,m)
    cfg = R.with_spectral_keep(cfg,calibration,.05)
    cfg = replace(cfg, keep_grad_every=1)
    out = tmp_path/"run"
    res = R.train(cfg,manifest,[root],out,log=lambda _:None)
    assert res["steps"] == 1
    rec = json.loads((out/"metrics.jsonl").read_text().splitlines()[0])
    assert rec["keep"] > 0 and rec["keep_valid_samples"] > 0
    assert all(rec["shared_grad_"+k] > 0 for k in ("jepa", "anchors", "keep"))
    state = torch.load(out/"checkpoint.pt",weights_only=False)
    assert state["student"]["spectral_keep.stats_fitted"] and state["teacher"]["spectral_keep.stats_fitted"]
    assert state["config"]["keep_calibration_sha256"] == sha256_file(calibration)
    assert R.train(replace(cfg,max_steps=2),manifest,[root],out,log=lambda _:None)["steps"] == 2
    before = (out/"checkpoint.pt").read_bytes()
    with pytest.raises(ValueError,match="identity changed"):
        R.train(replace(cfg,jepa=replace(cfg.jepa,keep_weight=.1)),manifest,[root],out)
    assert (out/"checkpoint.pt").read_bytes() == before
    with pytest.raises(ValueError,match="NEW runs"):
        R.train(cfg,manifest,[root],tmp_path/"warm",init_from=out/"checkpoint.pt")


def test_paired_a_b_same_initialization(tmp_path):
    root,manifest,cfg,_,y,m = _setup(tmp_path)
    cal = tmp_path/"cal.npz"; _calibration(cal,cfg,manifest,y,m)
    weights = []
    for label,weight in [("a",0.),("b",.05)]:
        out = tmp_path/label
        R.train(R.with_spectral_keep(cfg,cal,weight),manifest,[root],out,time_limit_s=0.,log=lambda _:None)
        weights.append(torch.load(out/"checkpoint.pt",weights_only=False)["student"])
    assert all(torch.equal(weights[0][k],weights[1][k]) for k in weights[0])


@pytest.mark.parametrize("weight", [-1.,float("nan"),float("inf")])
def test_invalid_keep_weight(weight):
    with pytest.raises(ValueError,match="keep_weight"):
        JEPAConfig("b",.2,.99,keep_weight=weight)
