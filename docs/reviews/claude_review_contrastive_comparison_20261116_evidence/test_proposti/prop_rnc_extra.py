"""Test aggiuntivi proposti dalla review Claude (agente rnc) per i mutanti non equivalenti sopravvissuti ai 23 test.
Solo tensori sintetici. Si eseguono con PYTHONPATH=<src>:<agents/rnc>."""
import math

import pytest
import torch

import wearusfm.model.rank_contrast as RC
from wearusfm.model.rank_contrast import (anchor_support, pose_distance, pose_rank_contrast, rank_contrast_count,
                                        rank_contrast_pairs, readout_similarity)


def _brute_distance(y, valid):
    n = y.shape[0]
    out = torch.zeros(n, n, dtype=torch.float64)
    yd = y.double()
    for i in range(n):
        common = valid[i][None, :] & valid                      # (n, J)
        diff2 = torch.where(common, (yd[i][None, :] - yd).square(), torch.zeros((), dtype=torch.float64)).sum(-1)
        out[i] = (diff2 / common.sum(-1).clamp(min=1)).sqrt()
    return out


def test_pose_distance_beyond_one_chunk_matches_bruteforce_with_partial_masks():
    """Uccide m35/m36: oltre _CHUNK = 256 ancore (il batch globale arriva a 400) i blocchi successivi al primo non erano provati."""
    g = torch.Generator().manual_seed(0)
    n = 2 * RC._CHUNK + 37
    y = torch.randn(n, 5, generator=g)
    valid = torch.rand(n, 5, generator=g) > 0.3
    y = torch.where(valid, y, torch.full_like(y, float("nan")))
    got = pose_distance(y, valid)
    assert got.shape == (n, n) and got.dtype == torch.float64
    assert torch.allclose(got, _brute_distance(torch.nan_to_num(y), valid), atol=1e-12, rtol=1e-12)


def test_loss_beyond_one_chunk_equals_single_chunk_evaluation(monkeypatch):
    g = torch.Generator().manual_seed(1)
    n = RC._CHUNK + 44
    z = torch.randn(1, n, 6, dtype=torch.float64, generator=g, requires_grad=True)
    y = torch.randn(1, n, 4, generator=g)
    mask = torch.rand(1, n, 4, generator=g) > 0.2
    tv = torch.ones(1, n, dtype=torch.bool)
    loss, *_ = pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=1)
    grad, = torch.autograd.grad(loss, z)
    monkeypatch.setattr(RC, "_CHUNK", 10 ** 6)
    ref, *_ = pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=1)
    gref, = torch.autograd.grad(ref, z)
    assert torch.allclose(loss, ref, atol=1e-13, rtol=1e-12) and torch.allclose(grad, gref, atol=1e-13, rtol=1e-10)


def test_distances_keep_float64_order_of_float32_targets():
    """Uccide m34/m83/m86: due distanze diverse in float64 ma uguali in float32 NON sono un pareggio (risoluzione 0)."""
    eps = 2.0 ** -23
    y = torch.tensor([[[0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0, 1.0 + eps]]],
                     dtype=torch.float32)       # (3 finestre, 1 patch, 4 giunti): d(0,2) = 1 + 2^-25, sotto la risoluzione float32
    mask = torch.ones_like(y, dtype=torch.bool)
    tv = torch.ones(3, 1, dtype=torch.bool)
    d = pose_distance(y[:, 0], mask[:, 0])
    assert float(d[0, 2]) > float(d[0, 1]) and float(d[0, 2].float()) == float(d[0, 1].float())   # distinte solo in float64
    z = torch.tensor([[[1.0, 0.0, -1.0]], [[0.3, 1.0, -0.2]], [[-1.0, 0.5, 0.7]]], dtype=torch.float64, requires_grad=True)
    loss, _, count, _ = pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=1, per_window=False)
    u = z[:, 0] - z[:, 0].mean(-1, keepdim=True)
    u = u / u.norm(dim=-1, keepdim=True)
    s = (u @ u.T) / 0.1

    def anchor(i, order):                                   # order: candidati dal piu' vicino al piu' lontano, senza pareggi
        num = torch.logsumexp(torch.stack([s[i, k] for k in order]), 0) - s[i, order[0]]            # il piu' lontano contribuisce 0
        return num / math.log(2)
    expected = (anchor(0, [1, 2]) + anchor(1, [2, 0]) + anchor(2, [1, 0])) / 3
    assert int(count) == 3
    assert torch.allclose(loss, expected, atol=1e-12, rtol=1e-12)


def test_nonfinite_inputs_are_rejected_only_where_used():
    """Uccide m27/m37/m51 (guardie mai esercitate)."""
    g = torch.Generator().manual_seed(2)
    z = torch.randn(2, 4, 5, dtype=torch.float64, generator=g)
    y = torch.randn(2, 4, 3, generator=g)
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    tv = torch.ones(2, 4, dtype=torch.bool)
    kw = dict(temperature=0.1, stride=2)                     # ancore alle patch 1 e 3
    bad = y.clone(); bad[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite valid pose target"):
        pose_rank_contrast(z, (bad, mask), tv, **kw)
    bad = y.clone(); bad[0, 0, 0] = float("inf")              # patch 0: non e' un'ancora, non entra nel termine
    assert torch.isfinite(pose_rank_contrast(z, (bad, mask), tv, **kw)[0])
    zbad = z.clone(); zbad[1, 3, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pose_rank_contrast(zbad, (y, mask), tv, **kw)
    with pytest.raises(ValueError, match="finite"):
        readout_similarity(zbad[1])
    cand = ~torch.eye(3, dtype=torch.bool)
    logits = torch.zeros(3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="nonfinite"):
        rank_contrast_pairs(logits + float("nan"), torch.zeros(3, 3, dtype=torch.float64), cand)
    with pytest.raises(ValueError, match="nonfinite"):
        rank_contrast_pairs(logits, torch.full((3, 3), float("inf"), dtype=torch.float64), cand)
    with pytest.raises(ValueError, match="nonfinite valid pose target"):
        pose_distance(torch.tensor([[float("nan")]]), torch.ones(1, 1, dtype=torch.bool))


def test_argument_guards():
    """Uccide m16/m17/m18/m50/m59/m80/m81."""
    z = torch.randn(2, 4, 5, dtype=torch.float64)
    y = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    tv = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="boolean"):
        anchor_support(mask.float(), 1)
    with pytest.raises(ValueError, match="stride"):
        anchor_support(mask, 0)
    with pytest.raises(ValueError, match="pair_allowed"):
        anchor_support(mask, 1, pair_allowed=torch.ones(8, 8))
    with pytest.raises(ValueError, match="self pairs"):
        rank_contrast_pairs(torch.zeros(3, 3), torch.zeros(3, 3), torch.ones(3, 3, dtype=torch.bool))
    for resolution in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="resolution"):
            pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=1, resolution=resolution)
    with pytest.raises(ValueError, match="physical support"):
        pose_rank_contrast(z, (y, mask), tv[:, :3], temperature=0.1, stride=1)
    with pytest.raises(ValueError, match="physical support"):
        pose_rank_contrast(z, (y, mask), tv.float(), temperature=0.1, stride=1)
    with pytest.raises(ValueError, match="target and boolean mask"):
        pose_rank_contrast(z, (y[:, :3], mask[:, :3]), tv, temperature=0.1, stride=1)
    with pytest.raises(ValueError, match="target and boolean mask"):
        pose_rank_contrast(z, (y, mask.float()), tv, temperature=0.1, stride=1)


def test_logged_counts_and_empty_count_type():
    """Uccide m76/m77/m78/m79: i valori di log (ancore, coppie, raw, chance) non erano fissati da nessun test."""
    g = torch.Generator().manual_seed(3)
    z = torch.randn(3, 6, 5, dtype=torch.float64, generator=g)
    y = torch.randint(0, 3, (3, 6, 2), generator=g).float()
    mask = torch.rand(3, 6, 2, generator=g) > 0.35
    tv = torch.ones(3, 6, dtype=torch.bool)
    s_idx, p_idx, v, cand, _, ok = anchor_support(mask & tv[..., None], 2)
    assert bool(ok.any()) and not bool(ok.all()) or int(ok.sum()) >= 3
    loss, num, count, log = pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=2, per_window=False)
    assert int(log["anchors"]) == int(ok.sum()) == int(count)
    assert int(log["pairs"]) == int(cand[ok].sum())
    logits = readout_similarity(z[s_idx, p_idx]) / 0.1
    n, c, n_pos = rank_contrast_pairs(logits, pose_distance(y[s_idx, p_idx], v), cand)
    assert torch.allclose(log["raw"], (n[ok] / n_pos[ok]).sum(), atol=1e-12, rtol=1e-12)
    assert torch.allclose(log["chance"], (c[ok] / n_pos[ok]).sum(), atol=1e-12, rtol=1e-12)
    assert int(rank_contrast_count(mask & tv[..., None], stride=2, per_window=False)) == int(ok.sum())
    empty = pose_rank_contrast(z, None, tv, temperature=0.1, stride=2)
    assert empty[2].dtype == torch.int64 and empty[3]["anchors"].dtype == torch.int64 and empty[3]["pairs"].dtype == torch.int64


def test_proposed_stride_phase_and_temperature_on_full_length_windows():
    """I valori del protocollo (stride 8, fase 4, temperatura 0,1, finestre da 160 patch) non comparivano nei 23 test."""
    g = torch.Generator().manual_seed(4)
    z = torch.randn(3, 160, 6, dtype=torch.float64, generator=g, requires_grad=True)
    y = torch.randn(3, 160, 20, generator=g)
    mask = torch.ones(3, 160, 20, dtype=torch.bool)
    mask[1, 12] = False                                      # un'ancora senza posa valida
    tv = torch.ones(3, 160, dtype=torch.bool)
    tv[2, 120:] = False                                      # finestra piu' corta: riempimento
    s_idx, p_idx, *_ = anchor_support(mask & tv[..., None], 8)
    assert (p_idx % 8 == 4).all() and s_idx.numel() == 20 + 19 + 15
    loss, _, count, log = pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=8)
    grad, = torch.autograd.grad(loss, z)
    used = torch.zeros(3, 160, dtype=torch.bool); used[s_idx, p_idx] = True
    assert int(count) == 3 and int(log["anchors"]) == 54 and (grad[~used] == 0).all() and (grad[used].abs().sum(-1) > 0).all()


def test_window_count_skips_windows_without_support():
    """Uccide m21: nei 23 test ogni finestra ha sempre almeno un'ancora contata, quindi il conteggio per finestra di
    rank_contrast_count (usato dall'audit, dove la maggior parte delle finestre NON ha posa) non era fissato."""
    mask = torch.zeros(5, 6, 2, dtype=torch.bool)
    mask[0] = True                     # 3 ancore a stride 2
    mask[2, 1] = True                  # una sola ancora valida, ma con candidati nelle altre finestre
    mask[4, 0] = True                  # patch valida che non e' un'ancora
    assert int(rank_contrast_count(mask, stride=2, per_window=True)) == 2
    assert int(rank_contrast_count(mask, stride=2, per_window=False)) == 4
    z = torch.randn(5, 6, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    y = torch.randn(5, 6, 2, generator=torch.Generator().manual_seed(6))
    loss, num, count, log = pose_rank_contrast(z, (y, mask), torch.ones(5, 6, dtype=torch.bool), temperature=0.1, stride=2)
    assert int(count) == 2 and int(log["anchors"]) == 4 and torch.allclose(num, loss * 2)
    # nessuna finestra con almeno due candidati: zero finestre
    lonely = torch.zeros(5, 6, 2, dtype=torch.bool); lonely[1, 1] = True; lonely[3, 3] = True
    assert int(rank_contrast_count(lonely, stride=2, per_window=True)) == 0


def test_counted_and_uncounted_anchors_in_the_same_window():
    """Uccide m70. Nei 23 test nessuna chiamata ha insieme ancore contate e non contate (il commento del test dell'oracolo
    «some rows having fewer than two eligible candidates» non si avvera: 0 righe su tutte le combinazioni). Caso vero: una finestra
    con due ancore e una sua copia fisica con una sola ancora valida: l'ancora col gemello resta con un solo candidato."""
    g = torch.Generator().manual_seed(7)
    z = torch.randn(2, 4, 5, dtype=torch.float64, generator=g, requires_grad=True)
    y = torch.zeros(2, 4, 2)
    y[0, 1] = torch.tensor([0.5, -1.0]); y[0, 3] = torch.tensor([1.5, 0.25]); y[1, 1] = y[0, 1]       # la copia ha lo stesso target
    mask = torch.zeros(2, 4, 2, dtype=torch.bool)
    mask[0, 1] = mask[0, 3] = mask[1, 1] = True
    tv = torch.ones(2, 4, dtype=torch.bool)
    s_idx, p_idx, *_ = anchor_support(mask & tv[..., None], 2)
    assert s_idx.tolist() == [0, 0, 1] and p_idx.tolist() == [1, 3, 1]
    allowed = torch.ones(3, 3, dtype=torch.bool)
    allowed[0, 2] = allowed[2, 0] = False                      # stesso istante fisico: coppia esclusa nei due ruoli
    *_, ok = anchor_support(mask & tv[..., None], 2, pair_allowed=allowed)
    assert ok.tolist() == [False, True, False]
    for per_window in (True, False):
        loss, num, count, log = pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=2, per_window=per_window,
                                                   pair_allowed=allowed)
        u = torch.stack([z[0, 1], z[0, 3], z[1, 1]])
        u = u - u.mean(-1, keepdim=True)
        u = u / u.norm(dim=-1, keepdim=True)
        s = (u @ u.T) / 0.1
        expected = (2 * torch.logsumexp(torch.stack([s[1, 0], s[1, 2]]), 0) - s[1, 0] - s[1, 2]) / (2 * math.log(2))   # pareggio esatto
        assert int(count) == 1 and int(log["anchors"]) == 1 and int(log["pairs"]) == 2
        assert torch.isfinite(loss) and torch.allclose(loss, expected, atol=1e-12, rtol=1e-12)
        grad, = torch.autograd.grad(loss, z)
        assert torch.isfinite(grad).all() and (grad[0, 0] == 0).all() and grad[1, 1].abs().sum() > 0
