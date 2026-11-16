"""verifica-rnc: test "forti" indipendenti del modulo model/rank_contrast.py, con un oracolo numpy (float64, nessun autograd)
scritto dal paragrafo 3 del protocollo. Servono a due cose: (1) decidere se un mutante sopravvissuto ai 23 test di GPT e'
equivalente oppure no; (2) controllare con dati MIEI i comportamenti che il revisore dice scoperti. Solo tensori sintetici."""
import math

import numpy as np
import pytest
import torch

import wearusfm.model.rank_contrast as RC
from wearusfm.model.rank_contrast import (anchor_support, pose_distance, pose_rank_contrast, rank_contrast_count,
                                        rank_contrast_pairs, readout_similarity)


def np_oracle(z, y, m, tv, *, temperature, stride, per_window=True, resolution=0.0, allowed=None):
    """z (S,P,D) f64, y (S,P,J) qualunque float, m (S,P,J) bool, tv (S,P) bool. allowed: (N,N) bool sulle ancore in ordine
    finestra-patch. Ritorna dict con loss, count, anchors, pairs, raw, chance, values (per ancora, None se non contata)."""
    z, y, m, tv = (np.asarray(a) for a in (z, y, m, tv))
    S, P, _ = z.shape
    pts = [(s, p) for s in range(S) for p in range(stride // 2, P, stride) if tv[s, p] and m[s, p].any()]
    n = len(pts)
    U = np.zeros((n, z.shape[2]))
    for i, (s, p) in enumerate(pts):
        h = z[s, p].astype(np.float64)
        h = h - h.mean()
        U[i] = h / math.sqrt(float((h * h).sum()))
    sim = U @ U.T / temperature
    values, npairs, raws, chances = [None] * n, [0] * n, [0.0] * n, [0.0] * n
    for i, (si, pi) in enumerate(pts):
        ks, ds = [], []
        for k, (sk, pk) in enumerate(pts):
            if k == i:
                continue
            common = m[si, pi] & m[sk, pk]
            if not common.any() or (allowed is not None and not allowed[i][k]):
                continue
            diff = y[si, pi].astype(np.float64)[common] - y[sk, pk].astype(np.float64)[common]
            d = math.sqrt(float((diff * diff).sum()) / int(common.sum()))
            if resolution > 0:
                d = math.floor(d / resolution) * resolution
            ks.append(k)
            ds.append(d)
        if len(ks) < 2:
            continue
        d = np.array(ds)
        s = sim[i, ks]
        tail = d[None, :] >= d[:, None]                       # tail[j, k]: k sta nel denominatore di j
        x = np.where(tail, s[None, :], -np.inf)
        mx = x.max(axis=1)
        lse = mx + np.log(np.exp(x - mx[:, None]).sum(axis=1))
        num = float((lse - s).sum())
        ref = float(np.log(tail.sum(axis=1)).sum())
        values[i], npairs[i], raws[i], chances[i] = num / ref, len(ks), num / len(ks), ref / len(ks)
    counted = [i for i in range(n) if values[i] is not None]
    out = {"points": pts, "values": values, "anchors": len(counted), "pairs": sum(npairs[i] for i in counted),
           "raw": sum(raws[i] for i in counted), "chance": sum(chances[i] for i in counted)}
    if not counted:
        out.update(loss=0.0, count=0)
        return out
    if per_window:
        by = {}
        for i in counted:
            by.setdefault(pts[i][0], []).append(values[i])
        out.update(loss=float(np.mean([np.mean(v) for v in by.values()])), count=len(by))
    else:
        out.update(loss=float(np.mean([values[i] for i in counted])), count=len(counted))
    return out


def _population(seed, S=5, P=40, D=12, J=6, stride=8, sparse_allowed=False, joint_blocks=False, f32=True):
    g = np.random.default_rng(seed)
    z = g.standard_normal((S, P, D))
    y = g.standard_normal((S, P, J)).astype(np.float32 if f32 else np.float64)
    m = g.random((S, P, J)) > 0.25
    if joint_blocks:   # gruppi di giunti disgiunti: molte coppie senza giunti in comune -> ancore con 0 o 1 candidato
        grp = g.integers(0, 3, size=(S, P))
        m = np.zeros((S, P, J), bool)
        for a in range(3):
            m[grp == a, 2 * a:2 * a + 2] = True
    tv = g.random((S, P)) > 0.1
    m[int(g.integers(0, S))] = False                           # una finestra senza posa
    s_idx, p_idx, *_ = anchor_support(torch.from_numpy(m & tv[..., None]), stride)
    n = s_idx.numel()
    allowed = None
    if sparse_allowed:
        a = g.random((n, n)) > 0.8
        a = np.triu(a, 1)
        allowed = a | a.T
    return z, y, m, tv, allowed


def _call(z, y, m, tv, allowed, **kw):
    zt = torch.from_numpy(z).requires_grad_(True)
    out = pose_rank_contrast(zt, (torch.from_numpy(y), torch.from_numpy(m)), torch.from_numpy(tv),
                             pair_allowed=None if allowed is None else torch.from_numpy(allowed), **kw)
    return zt, out


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("variant", ["plain", "sparse", "blocks", "blocks_sparse"])
@pytest.mark.parametrize("per_window", [True, False])
def test_value_counts_and_logs_against_numpy_oracle_with_uncounted_anchors(seed, variant, per_window):
    z, y, m, tv, allowed = _population(seed, sparse_allowed="sparse" in variant, joint_blocks="blocks" in variant)
    kw = dict(temperature=0.1, stride=8, per_window=per_window)
    ref = np_oracle(z, y, m, tv, allowed=allowed, **kw)
    zt, (loss, num, count, log) = _call(z, y, m, tv, allowed, **kw)
    assert count.dtype == torch.int64 and int(count) == ref["count"]
    assert int(log["anchors"]) == ref["anchors"] and int(log["pairs"]) == ref["pairs"]
    assert float(loss) == pytest.approx(ref["loss"], rel=1e-11, abs=1e-12)
    assert float(num) == pytest.approx(ref["loss"] * ref["count"], rel=1e-11, abs=1e-12)
    assert float(log["raw"]) == pytest.approx(ref["raw"], rel=1e-10, abs=1e-12)
    assert float(log["chance"]) == pytest.approx(ref["chance"], rel=1e-10, abs=1e-12)
    assert not log["raw"].requires_grad and not log["chance"].requires_grad
    valid = torch.from_numpy(m & tv[..., None])
    pa = None if allowed is None else torch.from_numpy(allowed)
    assert int(rank_contrast_count(valid, stride=8, per_window=per_window, pair_allowed=pa)) == ref["count"]
    if ref["count"]:
        g, = torch.autograd.grad(loss, zt)
        assert torch.isfinite(g).all()
        # derivata direzionale per differenze centrali dell'oracolo (nessun autograd)
        d = np.random.default_rng(100 + seed).standard_normal(z.shape)
        eps = 1e-6
        fd = (np_oracle(z + eps * d, y, m, tv, allowed=allowed, **kw)["loss"] - np_oracle(z - eps * d, y, m, tv, allowed=allowed, **kw)["loss"]) / (2 * eps)
        ad = float((g.numpy() * d).sum())
        assert ad == pytest.approx(fd, rel=2e-5, abs=1e-8)


def test_the_populations_above_do_contain_uncounted_anchors_and_windows():
    """Controllo dei dati: senza ancore non contate i test sopra non proverebbero nulla sul ramo ok=False."""
    tot = mixed = 0
    for seed in range(6):
        for variant in ("sparse", "blocks", "blocks_sparse"):
            z, y, m, tv, allowed = _population(seed, sparse_allowed="sparse" in variant, joint_blocks="blocks" in variant)
            ref = np_oracle(z, y, m, tv, allowed=allowed, temperature=0.1, stride=8)
            n = len(ref["points"])
            tot += n - ref["anchors"]
            by = {}
            for (s, _), v in zip(ref["points"], ref["values"]):
                by.setdefault(s, set()).add(v is None)
            mixed += sum(len(v) == 2 for v in by.values())
    assert tot > 20 and mixed > 5


def test_single_counted_window_is_not_rescaled():
    """Un solo conteggio (una finestra di posa nel batch globale: 0,72% dei passi veri)."""
    g = np.random.default_rng(11)
    z = g.standard_normal((3, 160, 16)); y = g.standard_normal((3, 160, 20)).astype(np.float32)
    m = np.zeros((3, 160, 20), bool); m[1] = True
    tv = np.ones((3, 160), bool)
    ref = np_oracle(z, y, m, tv, temperature=0.1, stride=8)
    _, (loss, num, count, log) = _call(z, y, m, tv, None, temperature=0.1, stride=8)
    assert ref["count"] == 1 == int(count) and int(log["anchors"]) == 20 and int(log["pairs"]) == 380
    assert float(loss) == pytest.approx(ref["loss"], rel=1e-12) and float(num) == pytest.approx(ref["loss"], rel=1e-12)
    _, (loss1, num1, count1, _) = _call(z, y, m, tv, None, temperature=0.1, stride=8, per_window=False)
    assert int(count1) == 20 and float(loss1) == pytest.approx(ref["loss"], rel=1e-12)   # una sola finestra: le due medie coincidono


def test_more_than_256_anchors_with_ties_and_partial_masks():
    g = np.random.default_rng(12)
    S, P, D, J = 4, 80, 8, 3
    z = g.standard_normal((S, P, D))
    y = (g.integers(0, 3, size=(S, P, J)) / 2).astype(np.float32)          # molti pareggi esatti
    m = g.random((S, P, J)) > 0.2
    tv = np.ones((S, P), bool)
    kw = dict(temperature=0.1, stride=1)
    ref = np_oracle(z, y, m, tv, **kw)
    assert len(ref["points"]) > 300
    _, (loss, num, count, log) = _call(z, y, m, tv, None, **kw)
    assert int(count) == ref["count"] and int(log["anchors"]) == ref["anchors"] and int(log["pairs"]) == ref["pairs"]
    assert float(loss) == pytest.approx(ref["loss"], rel=1e-11)
    # pose_distance: stessa matrice a blocchi e in un blocco solo, bit per bit, e simmetrica
    s_idx, p_idx, v, *_ = anchor_support(torch.from_numpy(m & tv[..., None]), 1)
    yt = torch.from_numpy(y)[s_idx, p_idx]
    d = pose_distance(yt, v)
    old = RC._CHUNK
    try:
        RC._CHUNK = 10 ** 6
        d1 = pose_distance(yt, v)
    finally:
        RC._CHUNK = old
    assert d.dtype == torch.float64 and torch.equal(d, d1) and torch.equal(d, d.T)
    brute = np.zeros((len(yt), len(yt)))
    yn, vn = yt.numpy().astype(np.float64), v.numpy()
    for i in (0, 1, 255, 256, 257, len(yt) - 1):
        for k in range(len(yt)):
            c = vn[i] & vn[k]
            brute[i, k] = math.sqrt(((yn[i] - yn[k])[c] ** 2).sum() / max(int(c.sum()), 1))
        assert np.allclose(d[i].numpy(), brute[i], rtol=1e-13, atol=1e-15)


def test_float32_targets_are_ranked_in_float64():
    """d(0,1) e d(0,2) differiscono solo oltre la precisione float32: con risoluzione 0 NON sono un pareggio."""
    y = np.zeros((4, 1, 4), np.float32)
    y[1, 0] = [1.0, 1.0, 1.0, 1.0]
    y[2, 0] = [1.0, 1.0, 1.0, np.float32(1.0) + np.float32(2.0 ** -23)]
    y[3, 0] = [3.0, -2.0, 0.5, 1.0]
    m = np.ones((4, 1, 4), bool); tv = np.ones((4, 1), bool)
    z = np.random.default_rng(13).standard_normal((4, 1, 5))
    kw = dict(temperature=0.1, stride=1, per_window=False)
    ref = np_oracle(z, y, m, tv, **kw)
    y_tied = y.copy(); y_tied[2, 0] = y[1, 0]
    tied = np_oracle(z, y_tied, m, tv, **kw)
    assert abs(ref["loss"] - tied["loss"]) > 1e-3                         # il pareggio cambierebbe davvero il valore
    _, (loss, *_r) = _call(z, y, m, tv, None, **kw)
    assert float(loss) == pytest.approx(ref["loss"], rel=1e-12)
    d = pose_distance(torch.from_numpy(y[:, 0]), torch.from_numpy(m[:, 0]))
    assert float(d[0, 2]) > float(d[0, 1]) and np.float32(float(d[0, 2])) == np.float32(float(d[0, 1]))


def test_window_count_with_mixed_and_empty_windows_and_more_windows_than_patches():
    S, P, J = 9, 4, 2                                                      # piu' finestre che patch
    m = np.zeros((S, P, J), bool)
    m[0, 1] = m[0, 3] = True
    m[4, 1] = True
    m[7, 3] = True
    m[8, 0] = True                                                         # non e' un'ancora (stride 2: patch 1 e 3)
    valid = torch.from_numpy(m)
    n = 4
    allowed = np.ones((n, n), bool)
    allowed[0, 1] = allowed[1, 0] = False                                  # ancore: (0,1) (0,3) (4,1) (7,3)
    allowed[0, 2] = allowed[2, 0] = False
    # candidati: a0 {3}; a1 {2,3}; a2 {1,3}; a3 {0,1,2} -> contate a1, a2, a3; finestra 0 mista (a0 no, a1 si')
    pa = torch.from_numpy(allowed)
    assert int(rank_contrast_count(valid, stride=2, per_window=False, pair_allowed=pa)) == 3
    assert int(rank_contrast_count(valid, stride=2, per_window=True, pair_allowed=pa)) == 3
    assert int(rank_contrast_count(valid, stride=2, pair_allowed=pa)) == 3                         # default: per finestra
    allowed2 = allowed.copy(); allowed2[1, 3] = allowed2[3, 1] = False       # a1 {2}: finestra 0 senza ancore contate
    pa2 = torch.from_numpy(allowed2)
    assert int(rank_contrast_count(valid, stride=2, per_window=True, pair_allowed=pa2)) == 2
    assert int(rank_contrast_count(valid, stride=2, per_window=False, pair_allowed=pa2)) == 2
    g = np.random.default_rng(14)
    z = g.standard_normal((S, P, 6)); y = g.standard_normal((S, P, J)).astype(np.float32); tv = np.ones((S, P), bool)
    for al in (allowed, allowed2):
        for per_window in (True, False):
            ref = np_oracle(z, y, m, tv, temperature=0.2, stride=2, per_window=per_window, allowed=al)
            zt, (loss, num, count, log) = _call(z, y, m, tv, al, temperature=0.2, stride=2, per_window=per_window)
            assert int(count) == ref["count"] and int(log["anchors"]) == ref["anchors"] and int(log["pairs"]) == ref["pairs"]
            assert float(loss) == pytest.approx(ref["loss"], rel=1e-12)
            g_, = torch.autograd.grad(loss, zt)
            assert torch.isfinite(g_).all()


def test_defaults_are_per_window_and_resolution_zero():
    z, y, m, tv, _ = _population(3)
    y = np.round(y, 1)
    ref = np_oracle(z, y, m, tv, temperature=0.1, stride=8, per_window=True, resolution=0.0)
    other = np_oracle(z, y, m, tv, temperature=0.1, stride=8, per_window=False, resolution=1.0)
    assert abs(ref["loss"] - other["loss"]) > 1e-6
    _, (loss, _, count, _) = _call(z, y, m, tv, None, temperature=0.1, stride=8)
    assert float(loss) == pytest.approx(ref["loss"], rel=1e-12) and int(count) == ref["count"]


def test_resolution_floors_distances():
    z, y, m, tv, _ = _population(4)
    for res in (0.25, 0.5):
        ref = np_oracle(z, y, m, tv, temperature=0.1, stride=8, resolution=res)
        _, (loss, *_r) = _call(z, y, m, tv, None, temperature=0.1, stride=8, resolution=res)
        assert float(loss) == pytest.approx(ref["loss"], rel=1e-11)


def test_nonfinite_values_raise_only_where_they_are_used():
    g = np.random.default_rng(15)
    z = g.standard_normal((2, 24, 6)); y = g.standard_normal((2, 24, 3)).astype(np.float32)
    m = np.ones((2, 24, 3), bool); tv = np.ones((2, 24), bool)
    kw = dict(temperature=0.1, stride=8)                                   # ancore: 4, 12, 20
    for bad in (float("nan"), float("inf"), -float("inf")):
        yb = y.copy(); yb[1, 12, 2] = bad
        with pytest.raises(ValueError, match="nonfinite valid pose target"):
            _call(z, yb, m, tv, None, **kw)
        zb = z.copy(); zb[0, 20, 1] = bad
        with pytest.raises(ValueError, match="finite"):
            _call(zb, y, m, tv, None, **kw)
    yb = y.copy(); yb[1, 13, 2] = float("nan")                              # non e' un'ancora
    zb = z.copy(); zb[0, 21, 1] = float("nan")
    mb = m.copy(); mb[0, 4, 1] = False; yb[0, 4, 1] = float("inf")           # giunto mascherato su un'ancora
    ref = np_oracle(z, np.nan_to_num(yb, nan=0.0, posinf=0.0), mb, tv, **kw)
    _, (loss, *_r) = _call(zb, yb, mb, tv, None, **kw)
    assert float(loss) == pytest.approx(ref["loss"], rel=1e-12)
    cand = ~torch.eye(3, dtype=torch.bool)
    zero = torch.zeros(3, 3, dtype=torch.float64)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            rank_contrast_pairs(zero + bad, zero, cand)
        dist = zero.clone(); dist[0, 1] = bad
        with pytest.raises(ValueError, match="nonfinite"):
            rank_contrast_pairs(zero, dist, cand)
        with pytest.raises(ValueError, match="finite"):
            readout_similarity(torch.tensor([[1.0, bad], [0.0, 1.0]]))
        with pytest.raises(ValueError, match="nonfinite valid pose target"):
            pose_distance(torch.tensor([[bad, 0.0]]), torch.tensor([[True, False]]))
    dist = zero.clone(); dist[0, 0] = float("nan")                           # diagonale: non candidata, ignorata
    rank_contrast_pairs(zero, dist, cand)
    assert pose_distance(torch.tensor([[float("nan"), 0.0]]), torch.tensor([[False, True]])).shape == (1, 1)


def test_argument_guards_reject_bad_shapes_types_and_values():
    z = torch.randn(2, 4, 5, dtype=torch.float64); y = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, 3, dtype=torch.bool); tv = torch.ones(2, 4, dtype=torch.bool)
    ok = dict(temperature=0.1, stride=1)
    for stride in (0, -1, 1.0, True, "1"):
        with pytest.raises(ValueError, match="stride"):
            anchor_support(mask, stride)
        with pytest.raises(ValueError, match="stride"):
            pose_rank_contrast(z, (y, mask), tv, temperature=0.1, stride=stride)
    for bad in (mask.float(), mask[0], mask[..., 0]):
        with pytest.raises(ValueError, match="boolean"):
            anchor_support(bad, 1)
    n = 8
    for bad in (torch.ones(n, n), torch.ones(n, n - 1, dtype=torch.bool), torch.ones(n + 1, n + 1, dtype=torch.bool)):
        with pytest.raises(ValueError, match="pair_allowed"):
            anchor_support(mask, 1, pair_allowed=bad)
    for res in (-0.1, float("nan"), float("inf"), "0", None):
        with pytest.raises(ValueError, match="resolution"):
            pose_rank_contrast(z, (y, mask), tv, resolution=res, **ok)
    for temp in ("0.1", None, True):
        with pytest.raises(ValueError, match="temperature"):
            pose_rank_contrast(z, (y, mask), tv, temperature=temp, stride=1)
    for zz, tt in ((z[0], tv), (z, tv[:, :3]), (z, tv.float()), (z, tv[0])):
        with pytest.raises(ValueError, match="physical support"):
            pose_rank_contrast(zz, (y, mask), tt, **ok)
    for yy, mm in ((y[:, :3], mask[:, :3]), (y, mask.float()), (y[0], mask[0]), (y, mask[..., :2]), (y[:1], mask[:1])):
        with pytest.raises(ValueError, match="target and boolean mask"):
            pose_rank_contrast(z, (yy, mm), tv, **ok)
    for h in (torch.zeros(3), torch.zeros(2, 3, 4), torch.zeros(3, 0)):
        with pytest.raises(ValueError, match="readouts"):
            readout_similarity(h)
    assert readout_similarity(torch.tensor([[1.0], [2.0]])).shape == (2, 2)   # D = 1 e' ammesso dalla guardia
    for yy, vv in ((torch.zeros(3), torch.ones(3, dtype=torch.bool)), (torch.zeros(2, 3), torch.ones(2, 2, dtype=torch.bool)),
                   (torch.zeros(2, 3), torch.ones(2, 3))):
        with pytest.raises(ValueError, match="target and boolean validity"):
            pose_distance(yy, vv)
    sq = torch.zeros(3, 3, dtype=torch.float64); cand = ~torch.eye(3, dtype=torch.bool)
    for lg, ds, cd in ((sq[0], sq, cand), (torch.zeros(3, 4), torch.zeros(3, 4), torch.ones(3, 4, dtype=torch.bool)),
                       (sq, sq[:, :2], cand), (sq, sq, cand[:2]), (sq, sq, cand.float())):
        with pytest.raises(ValueError, match="square logits"):
            rank_contrast_pairs(lg, ds, cd)
    for cd in (torch.ones(3, 3, dtype=torch.bool), torch.eye(3, dtype=torch.bool)):
        with pytest.raises(ValueError, match="self pairs"):
            rank_contrast_pairs(sq, sq, cd)
    with pytest.raises(ValueError, match="identical"):
        rank_contrast_pairs(sq, sq, cand, cand.float())


def test_empty_results_keep_integer_types_and_detached_logs():
    z = torch.randn(2, 4, 5, dtype=torch.float32, requires_grad=True)
    tv = torch.ones(2, 4, dtype=torch.bool)
    for pose in (None, (torch.zeros(2, 4, 3), torch.zeros(2, 4, 3, dtype=torch.bool))):
        loss, num, count, log = pose_rank_contrast(z, pose, tv, temperature=0.1, stride=1)
        assert count.dtype == torch.int64 and log["anchors"].dtype == torch.int64 and log["pairs"].dtype == torch.int64
        assert loss.dtype == torch.float32 and loss.requires_grad and not log["raw"].requires_grad and not log["chance"].requires_grad
    assert pose_distance(torch.empty(0, 3), torch.empty(0, 3, dtype=torch.bool)).dtype == torch.float64
