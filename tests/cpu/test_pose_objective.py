"""Obiettivo di posa R3 (`docs/pilot_posa_v1.md` §2 e §6, punto 2): testa a zero fuori dal percorso di encode; termine per finestra sugli elementi
validi e sugli istanti validi; contratto dei conteggi (`loss_counts` = `den` dei termini); micro-batch senza posa con la testa nel grafo e
gradiente nullo; totale = somma pesata; pannello fisso con l'errore per giunto; `_prepare` riempie i target alla larghezza del batch; il default
(peso 0, senza testa) non cambia."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace  # noqa: E402

from test_pose_targets import _pose_tree, _stats  # noqa: E402
from test_pretraining_loader import _cfg, _tree  # noqa: E402
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402
from wearusfm.training import jepa as J  # noqa: E402
from wearusfm.training import run as R  # noqa: E402

CFG = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)
JC = J.JEPAConfig(target="b", anchor_weight=0.2, ema_momentum=0.996, loss_per_window=True, jepa_weight=0.0, pose_weight=1.0)


def _models(pose_head=True, seed=0):
    torch.manual_seed(seed)
    student = WearUsFM(replace(CFG, pose_head=pose_head))
    return student, J.make_teacher(student)


def _batch_with_pose(tmp_path, n=6, seed=3, invalid=()):
    root, mpath = _tree(tmp_path)
    _pose_tree(tmp_path / "pose", 24000, invalid_blocks=invalid)
    _stats(tmp_path / "pose", mean=[0.5] * 20, std=[0.25] * 20)
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), _cfg(pose_root=str(tmp_path / "pose")))
    return loader.batch(n, np.random.default_rng(seed))


def test_head_is_zero_at_init_and_leaves_encode_and_the_other_weights_untouched():
    with_head, _ = _models(True)
    without, _ = _models(False)
    assert with_head.pose_head is not None and without.pose_head is None
    assert float(with_head.pose_head.out.weight.detach().abs().max()) == 0.0 and float(with_head.pose_head.out.bias.detach().abs().max()) == 0.0
    sd_h, sd = with_head.state_dict(), without.state_dict()
    assert set(sd_h) - set(sd) == {"pose_head.norm.weight", "pose_head.norm.bias", "pose_head.out.weight", "pose_head.out.bias"}
    assert all(torch.equal(sd_h[k], sd[k]) for k in sd)
    assert with_head.pose_head(torch.randn(2, 5, 16)).abs().max() == 0.0  # predice la media standardizzata


def test_pose_terms_per_window_mask_and_time_validity():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(3, 4, 20, generator=g, requires_grad=True)
    target = torch.randn(3, 4, 20, generator=g)
    mask = torch.ones(3, 4, 20, dtype=torch.bool)
    mask[1] = False  # finestra senza posa
    mask[2, :, 10:] = False  # meta' giunti
    tv = torch.ones(3, 4, dtype=torch.bool)
    tv[2, 3] = False  # istante senza canali visibili
    loss, num, den, valid = J.pose_terms(pred, (target, mask), tv, per_window=True)
    err = (pred - target).pow(2)
    w0 = err[0].mean()
    w2 = err[2, :3, :10].mean()
    assert int(den) == 2 and torch.allclose(loss, (w0 + w2) / 2) and torch.allclose(num, w0 + w2) and not valid[1].any() and not valid[2, 3].any()
    loss_t, num_t, den_t, _ = J.pose_terms(pred, (target, mask), tv, per_window=False)
    sel = valid
    assert int(den_t) == int(sel.sum()) and torch.allclose(loss_t, err[sel].mean()) and torch.allclose(num_t, err[sel].sum())
    # senza target: zero con gradiente (la testa resta nel grafo), conteggio zero
    z, zn, zd, zv = J.pose_terms(pred, None, tv, per_window=True)
    assert float(z.detach()) == 0.0 and z.requires_grad and int(zd) == 0 and not zv.any()
    with pytest.raises(ValueError, match="posa"):
        J.pose_terms(pred, (target[:, :2], mask[:, :2]), tv, True)


def test_loader_batch_counts_match_the_terms_and_the_total_is_the_weighted_sum(tmp_path):
    batch = _batch_with_pose(tmp_path)
    student, teacher = _models(True)
    m = R._prepare(batch, "cpu")
    assert m.pose is not None and m.pose[0].shape == (6, max(batch.n_patches), 20) and m.pose[1].dtype == torch.bool
    for s, item in enumerate(batch.pose):  # riempimento oltre le patch del campione: maschera falsa
        n_p = batch.n_patches[s]
        assert not m.pose[1][s, n_p:].any()
        if item is None:
            assert not m.pose[1][s].any()
    n_pose_windows = sum(1 for item in batch.pose if item is not None and item[1].any())
    assert n_pose_windows >= 1
    counts = J.loss_counts(m.inp, m.visible, JC, m.anchor_targets, model_cfg=student.cfg, pose_valid=m.pose[1])
    terms = J.jepa_loss_terms(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose)
    assert "pose" in counts and torch.equal(terms.den["pose"], counts["pose"]) and 1 <= int(counts["pose"]) <= n_pose_windows
    assert terms.weight["pose"] == 1.0 and int(terms.log_sum["pose_windows"]) == int(counts["pose"])
    assert all(torch.equal(terms.den[k], counts[k]) for k in counts)  # il contratto regge per tutte le componenti
    torch.manual_seed(1)
    losses = J.jepa_losses(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose)
    torch.manual_seed(1)
    terms2 = J.jepa_loss_terms(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose)
    assert torch.allclose(losses["pose"], terms2.num["pose"] / terms2.den["pose"]) and float(losses["pose_windows"]) == float(terms2.den["pose"])
    # testa a zero e target standardizzati: il termine iniziale e' la media dei quadrati dei target validi (~1 se standardizzati sul corpus)
    tgt, msk = m.pose
    expected = [float((tgt[s] ** 2)[msk[s]].mean()) for s in range(6) if msk[s].any()]
    # la media per finestra sugli istanti validi puo' differire dalla media su tutti gli elementi solo per gli istanti senza canali visibili
    assert abs(float(losses["pose"]) - float(np.mean(expected))) < 0.5 * float(np.mean(expected)) + 1e-6
    anchors = sum(float(losses[k]) for k in ("log_rms", "band_shape", "log_env"))
    total = 0.2 * anchors + 1.0 * float(losses["pose"])
    assert abs(float(losses["total"]) - total) < 1e-5 * max(1.0, abs(total))  # float32
    # il peso entra nel backward del contratto: componenti e pesi coerenti
    assert set(terms2.weight) == set(terms2.num) == set(terms2.den) and terms2.weight["pose"] == 1.0
    assert torch.isfinite(terms2.backward_loss(terms2.den))


def test_micro_batch_without_pose_keeps_the_head_in_the_graph_with_zero_gradient(tmp_path):
    batch = _batch_with_pose(tmp_path)
    student, teacher = _models(True)
    m = R._prepare(batch, "cpu")
    losses = J.jepa_losses(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=None)
    assert float(losses["pose"]) == 0.0 and float(losses["pose_windows"]) == 0.0
    losses["total"].backward()
    g = student.pose_head.out.weight.grad
    assert g is not None and float(g.abs().max()) == 0.0  # nel grafo, gradiente nullo: nessun parametro inutilizzato per DDP
    counts = J.loss_counts(m.inp, m.visible, JC, m.anchor_targets, model_cfg=student.cfg, pose_valid=None)
    assert int(counts["pose"]) == 0
    terms = J.jepa_loss_terms(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=None)
    assert int(terms.den["pose"]) == 0 and torch.isfinite(terms.backward_loss({k: v for k, v in terms.den.items()}))  # den 0 -> clamp(1)


def test_fixed_panel_logs_the_pose_error_per_joint(tmp_path):
    batch = _batch_with_pose(tmp_path)
    student, teacher = _models(True)
    m = R._prepare(batch, "cpu")
    std = torch.full((20,), 0.25)
    terms = J.jepa_loss_terms(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose, pose_std=std)
    keys = {"pose_mae", "pose_mae_std"} | {f"pose_mae_j{j:02d}" for j in range(20)}
    assert keys <= set(terms.log_num) and keys <= set(terms.log_den)
    pooled = R._pool_terms([{"num": {k: float(v) for k, v in terms.num.items()}, "den": {k: float(v) for k, v in terms.den.items()},
                             "log_num": {k: float(v) for k, v in terms.log_num.items()}, "log_den": {k: float(v) for k, v in terms.log_den.items()},
                             "log_sum": {k: float(v) for k, v in terms.log_sum.items()}}], dict(terms.weight))
    assert pooled["pose_mae"] == pytest.approx(0.25 * pooled["pose_mae_std"], rel=1e-6) and pooled["pose_count"] == float(terms.den["pose"])
    assert pooled["pose_mae_j00"] > 0 and "pose_windows" in pooled
    # senza pose_std (passo di training) le medie per giunto non si loggano
    plain = J.jepa_loss_terms(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose)
    assert "pose_mae" not in plain.log_num and "pose_windows" in plain.log_sum


def test_default_without_pose_is_unchanged_and_misconfigurations_fail(tmp_path):
    batch = _batch_with_pose(tmp_path)
    student, teacher = _models(False)
    m = R._prepare(batch, "cpu")
    base = replace(JC, pose_weight=0.0, jepa_weight=1.0)
    torch.manual_seed(0)
    a = J.jepa_losses(student, teacher, m.inp, m.visible, base, anchor_targets=m.anchor_targets, kinds=m.kinds)
    torch.manual_seed(0)
    b = J.jepa_losses(student, teacher, m.inp, m.visible, base, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose)
    assert "pose" not in a and set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)  # la posa nel batch non tocca il default
    assert "pose" not in J.loss_counts(m.inp, m.visible, base, m.anchor_targets, model_cfg=student.cfg, pose_valid=m.pose[1])
    with pytest.raises(ValueError, match="testa di posa"):
        J.jepa_losses(student, teacher, m.inp, m.visible, JC, anchor_targets=m.anchor_targets, kinds=m.kinds, pose=m.pose)
    with pytest.raises(ValueError, match="pose head"):
        J.loss_counts(m.inp, m.visible, JC, m.anchor_targets, model_cfg=student.cfg, pose_valid=m.pose[1])
    with pytest.raises(ValueError, match="pose_weight"):
        replace(JC, pose_weight=-1.0)
    # batch senza posa dal loader di default: `_prepare` non crea tensori di posa; con la posa i tensori del micro-batch sono due in piu'
    root, mpath = _tree(tmp_path / "plain")
    plain = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), _cfg()).batch(4, np.random.default_rng(0))
    assert R._prepare(plain, "cpu").pose is None
    m1 = R._prepare(batch, "cpu")
    assert len(R._micro_tensors([m1])) == len(R._micro_tensors([replace(m1, pose=None)])) + 2
