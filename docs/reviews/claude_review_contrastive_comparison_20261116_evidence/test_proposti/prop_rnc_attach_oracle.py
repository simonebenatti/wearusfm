"""Test proposto dalla review Claude (agente rnc): valore, gradiente sui parametri e scala del contratto di `contrast_rnc` prodotti
dal punto di chiamata VERO (ContrastiveRuntime.attach dentro R._accumulate_terms, un processo, float64, fixture sintetici) contro
una definizione coppia per coppia scritta dal protocollo par. 3 (stride 8, fase 4, temperatura 0,1, esclusione simmetrica dello
stesso istante fisico, media nella finestra e poi fra finestre)."""
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import torch

from rnc_oracle_common import oracle, instant
from test_pose_training import _pose_setup
from test_contrastive_runtime import _config
from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training import jepa as J
from wearusfm.training import contrastive_objectives as CO
from wearusfm.training.jepa import make_teacher


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return _pose_setup(tmp_path_factory.mktemp("attach_oracle"))


@pytest.mark.parametrize("seed,batch", [(0, 4), (1, 8), (3, 8), (10, 4)])
def test_attach_rnc_value_gradient_support_and_contract_match_protocol_oracle(setup, seed, batch):
    root, manifest, pose_root = setup
    cfg = _config(pose_root, "CR")
    cfg = replace(cfg, batch_size=batch, loader=replace(cfg.loader, min_window_s=1.0, max_window_s=4.0),
                  contrastive={**cfg.contrastive, "rnc_stride": 8, "rnc_temperature": 0.1})
    weight = cfg.contrastive["rnc_weight"]                       # fill_steps=0, ramp_steps=1: peso pieno al passo 0
    loader = L.PretrainLoader(R.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader, {}, None)
    data = loader.batch(batch, np.random.default_rng(100 + seed))
    if seed % 3 == 1:
        data.pose[0] = None
    torch.manual_seed(seed)
    student = WearUsFM(cfg.model).double()
    teacher = make_teacher(student)
    engine = CO.ContrastiveRuntime(cfg.contrastive)
    engine.prepare(data, 0)
    micro = R._prepare(data, "cpu", torch.float64)
    student.train()
    failure, metrics = R._accumulate_terms(None, student, teacher, [micro], cfg, R.DistContext(), "cpu", nullcontext(), contrastive=engine)
    assert failure is None, failure
    grads = {n: p.grad.clone() for n, p in student.named_parameters() if p.grad is not None}
    # riferimento: stesso student e stesso batch; termine RnC dall'oracolo, senza passare da attach ne' da rank_contrast.py
    student.zero_grad(set_to_none=True)
    seen = []
    terms = J.jepa_loss_terms(student, teacher, micro.inp, micro.visible, cfg.jepa, anchor_targets=micro.anchor_targets, kinds=micro.kinds,
                              regions=micro.regions, pose=micro.pose, capture_encoded=lambda a, b: seen.extend((a, b)))
    enc = seen[0]
    zm = enc.z.mean(2)
    tv = J.pose_time_valid(enc.patch_valid, micro.inp.counts)
    ids = engine.global_ids
    windows = [{"z": zm[s], "y": micro.pose[0][s], "m": micro.pose[1][s], "tv": tv[s], "ident": ids[s]} for s in range(zm.shape[0])]
    ref, count, detail = oracle(windows, temperature=0.1, stride=8, patch_ms=cfg.model.patch_ms)
    assert ref is not None and count >= 2
    assert metrics["contrast_rnc"] == pytest.approx(float(ref.detach()), rel=1e-10, abs=1e-12)   # il contratto riporta il valore globale
    assert metrics["contrast_rnc_windows"] == count and metrics["contrast_rnc_anchors"] == detail["anchors"]
    points = [instant(ids[s], p, cfg.model.patch_ms) for s in range(zm.shape[0]) for p in range(4, zm.shape[1], 8)
              if bool((micro.pose[1][s, p] & tv[s, p]).any())]
    assert metrics["contrast_rnc_duplicate_instants"] == len(points) - len(set(points)) > 0     # il fixture ha istanti duplicati
    base = sum(w * metrics[k] for k, w in (("log_rms", .2), ("band_shape", .2), ("log_env", .2), ("pose", 1.0)))
    assert metrics["total"] == pytest.approx(base + weight * float(ref.detach()), rel=1e-10)
    (terms.backward_loss() + weight * ref).backward()
    for name, p in student.named_parameters():
        expected = p.grad if p.grad is not None else torch.zeros_like(p)
        got = grads.get(name, torch.zeros_like(p))
        torch.testing.assert_close(got, expected, rtol=1e-8, atol=1e-11, msg=name)
