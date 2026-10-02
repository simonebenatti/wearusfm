"""Generatore di maschere (schema della bozza D10) sui montaggi veri: quote, slab allineati, tetto spaziale, forme per topologia."""

import numpy as np
import pytest

from wearusfm.ingest import camargo, capgmyo, emg2pose, ninapro_std
from wearusfm.ingest.common import montage_to_dict
from wearusfm.model import anchor_targets as AT
from wearusfm.model import channel_codes as CC
from wearusfm.training import masking as M


def _setup(m, valid=None):
    d = montage_to_dict(m, valid if valid is not None else [True] * m.n_channels)
    return CC.layout_from_montage(d), CC.anatomy_codes(d).compartment_weights.argmax(axis=1)


def test_budget_shares_and_kinds_on_a_large_grid():
    lay, comp = _setup(capgmyo.build_montage_metadata(1))  # 128 canali
    vis, kind = M.generate_mask(lay, comp, 160, 160, True, M.MaskSpec.d10_proposal(0.5), np.random.default_rng(0))
    frac = kind_total = (kind != M.VISIBLE).sum() / (128 * 160)
    assert 0.45 <= frac <= 0.6, frac  # vicino al budget (le quote si riempiono a token nuovi, con un ultimo tentativo che puo' sforare)
    f = M.kind_fractions(kind, 128 * 160)
    assert f["spaziale"] > 0 and f["slab"] > 0 and f["lunga"] > 0 and f["media"] > 0 and f["corta"] > 0
    assert np.array_equal(vis, kind == M.VISIBLE) and kind_total == frac


def test_slabs_are_aligned_on_all_channels_and_give_rvq_targets():
    lay, comp = _setup(emg2pose.build_montage_metadata("u1", "s", "left"))
    vis, kind = M.generate_mask(lay, comp, 160, 160, True, M.MaskSpec.d10_proposal(0.5), np.random.default_rng(1))
    slab_t = np.flatnonzero((kind == M.SLAB).any(axis=0))
    assert len(slab_t) and ((kind[:, slab_t] != M.VISIBLE).all())  # uno slab copre tutti i canali
    runs = np.split(slab_t, np.flatnonzero(np.diff(slab_t) > 1) + 1)
    assert all(r[0] % 8 == 0 for r in runs)  # comincia sulla griglia del tokenizer
    win = AT.rvq_target_windows(vis, np.ones_like(vis), np.ones(16, dtype=bool))
    assert win.any()
    _, kind_off = M.generate_mask(lay, comp, 160, 160, False, M.MaskSpec.d10_proposal(0.5), np.random.default_rng(1))
    assert not (kind_off == M.SLAB).any()  # senza ancora RVQ niente slab: la quota torna al temporale normale


def test_spatial_cap_and_shapes_by_topology():
    spec = M.MaskSpec.d10_proposal(0.6)
    for m in (emg2pose.build_montage_metadata("u1", "s", "left"), camargo.build_montage_metadata(1, "d"),
              ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right")):
        lay, comp = _setup(m)
        for seed in range(5):
            _, kind = M.generate_mask(lay, comp, 80, 80, True, spec, np.random.default_rng(seed))
            spatial_ch = np.flatnonzero((kind == M.SPATIAL).any(axis=1))
            assert len(spatial_ch) <= lay.n_channels // 2  # al piu' meta' dei canali validi
    lay, comp = _setup(emg2pose.build_montage_metadata("u1", "s", "left"))
    rng = np.random.default_rng(3)
    for _ in range(20):  # sull'anello: archi contigui
        g = sorted(M._spatial_groups(lay, comp, np.ones(16, dtype=bool), rng))
        gaps = np.diff(g + [g[0] + 16])
        assert (gaps == 1).sum() >= len(g) - 1
    lay, comp = _setup(capgmyo.build_montage_metadata(1))
    for _ in range(20):  # sulla griglia: rettangoli
        g = M._spatial_groups(lay, comp, np.ones(128, dtype=bool), rng)
        rc = lay.grid_rc[g]
        assert len(g) == (rc[:, 0].max() - rc[:, 0].min() + 1) * (rc[:, 1].max() - rc[:, 1].min() + 1)


def test_short_never_alone_padding_and_invalid_channels_untouched():
    lay, comp = _setup(emg2pose.build_montage_metadata("u1", "s", "left"), [i != 5 for i in range(16)])
    vis, kind = M.generate_mask(lay, comp, 40, 64, True, M.MaskSpec.d10_proposal(0.4), np.random.default_rng(2))
    assert (kind[:, 40:] == M.VISIBLE).all() and vis[:, 40:].all()  # padding mai nascosto
    assert (kind[5] == M.VISIBLE).all()  # canale scartato: mai scelto
    only_short = M.MaskSpec(0.3, 1.0, 0.0, 0.0, 0.0)
    _, k2 = M.generate_mask(lay, comp, 40, 40, True, only_short, np.random.default_rng(0))
    assert not (k2 == M.SHORT).any()  # le corte non restano mai da sole
    long_cap = M.generate_mask(lay, comp, 40, 40, False, M.MaskSpec(0.5, 0.0, 0.0, 1.0, 0.0), np.random.default_rng(0))[1]
    runs = [np.diff(np.flatnonzero(np.r_[0, row == M.LONG, 0])) for row in long_cap]
    assert max((r[0::2].max() if len(r) else 0) for r in runs) <= 20  # al piu' meta' finestra


def test_spec_checks_and_determinism():
    with pytest.raises(ValueError):
        M.MaskSpec(1.0, 0.15, 0.25, 0.2, 0.4)
    with pytest.raises(ValueError):
        M.MaskSpec(0.5, 0.5, 0.5, 0.5, 0.5)
    lay, comp = _setup(emg2pose.build_montage_metadata("u1", "s", "left"))
    a = M.generate_mask(lay, comp, 80, 80, True, M.MaskSpec.d10_proposal(0.5), np.random.default_rng(7))[1]
    b = M.generate_mask(lay, comp, 80, 80, True, M.MaskSpec.d10_proposal(0.5), np.random.default_rng(7))[1]
    c = M.generate_mask(lay, comp, 80, 80, True, M.MaskSpec.d10_proposal(0.5), np.random.default_rng(8))[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)
