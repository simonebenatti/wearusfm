"""Montaggi virtuali dalle griglie HD (D6a, `data.virtual_montage`): sottogriglie monopolari e bipolari sui montaggi veri degli adattatori."""

import numpy as np
import pytest

from wearusfm.data import virtual_montage as VM
from wearusfm.ingest import capgmyo, csl_hdemg, emg2pose, hyser
from wearusfm.ingest.common import montage_to_dict
from wearusfm.model import channel_codes as CC


def _capgmyo(qc=None):
    return montage_to_dict(capgmyo.build_montage_metadata(1), qc if qc is not None else [True] * 128)


def _rc(montage, cols):
    chans = [c for g in montage["groups"] for c in g["channels"]]
    return [(chans[k]["sensor_coords"]["grid_row"], chans[k]["sensor_coords"]["grid_col"]) for k in cols]


def test_monopolar_subgrid_keeps_the_original_coordinates_and_positions():
    m = _capgmyo()
    rng = np.random.default_rng(0)
    spec = VM.VirtualSpec(p=1.0, p_bipolar=0.0)
    seen = set()
    for _ in range(200):
        vm = VM.draw(m, np.ones(128, bool), spec, rng)
        assert vm.kind == "mono" and vm.b is None and len(vm.a) <= 32
        rc = np.asarray(_rc(m, vm.a))
        presented = np.asarray([(c["sensor_coords"]["grid_row"], c["sensor_coords"]["grid_col"]) for c in vm.montage["groups"][0]["channels"]])
        assert np.array_equal(rc, presented)  # coordinate della griglia d'origine
        steps = np.unique(np.diff(np.unique(rc[:, 0])))
        assert steps.tolist() == [vm.stride] or len(np.unique(rc[:, 0])) == 1
        seen.add(vm.label)
        full = CC.anatomy_codes(m).sensor_pos[vm.a]
        assert np.allclose(CC.anatomy_codes(vm.montage).sensor_pos, full)  # stessa posizione nel sistema del sensore della griglia intera
        layout = CC.layout_from_montage(vm.montage)
        assert (layout.topology == CC.TOPOLOGIES.index("grid_2d")).all()
    assert {"mono 4x8 s1", "mono 8x4 s1", "mono 4x4 s2", "mono 4x8 s2"} <= seen


def test_bipolar_pairs_sign_midpoint_band_and_qc():
    qc = [True] * 128
    qc[17] = False
    m = _capgmyo(qc)
    rng = np.random.default_rng(1)
    x = np.random.default_rng(2).normal(size=(128, 50))
    for _ in range(100):
        vm = VM.draw(m, np.asarray(qc), VM.VirtualSpec(p=1.0, p_bipolar=1.0), rng)
        assert vm.kind == "bip" and vm.axis in VM.AXES
        assert np.array_equal(VM.apply(x, vm), x[vm.b] - x[vm.a])  # compagno meno elettrodo
        ra, rb = np.asarray(_rc(m, vm.a)), np.asarray(_rc(m, vm.b))
        step = np.array([vm.stride, 0] if vm.axis == "row" else [0, vm.stride])
        assert (rb - ra == step).all()
        ch = vm.montage["groups"][0]["channels"]
        mid = np.asarray([(c["sensor_coords"]["grid_row"], c["sensor_coords"]["grid_col"]) for c in ch])
        assert np.allclose(mid, (ra + rb) / 2.0)
        assert all(c["bipolar_orientation"] == f"{vm.axis}+{vm.stride}" for c in ch)
        assert np.array_equal(vm.qc_valid, (vm.a != 17) & (vm.b != 17))
        assert [c["qc_valid"] for c in ch] == vm.qc_valid.tolist()
    a, b = VM.all_pairs(m, "row", 1)
    assert len(a) == 7 * 16 and len(VM.all_pairs(m, "col", 2)[0]) == 8 * 14


def test_presented_whole_when_p_is_zero_or_without_grids():
    rng = np.random.default_rng(0)
    assert VM.draw(_capgmyo(), np.ones(128, bool), VM.VirtualSpec(p=0.0), rng) is None
    ring = montage_to_dict(emg2pose.build_montage_metadata("u1", "s", "left"), [True] * 16)
    assert VM.draw(ring, np.ones(16, bool), VM.VirtualSpec(p=1.0), rng) is None
    draws = [VM.draw(_capgmyo(), np.ones(128, bool), VM.VirtualSpec(p=0.5), rng) for _ in range(400)]
    assert 0.4 < np.mean([d is not None for d in draws]) < 0.6


def test_only_shapes_that_fit_and_one_grid_at_a_time():
    csl = montage_to_dict(csl_hdemg.build_montage_metadata(1, 1), [True] * 168)  # 7 righe x 24 colonne
    rng = np.random.default_rng(3)
    labels = {VM.draw(csl, np.ones(168, bool), VM.VirtualSpec(p=1.0, p_bipolar=0.0), rng).label for _ in range(300)}
    assert not any(lab.startswith("mono 8x4") for lab in labels)  # 8 righe non ci stanno
    layout = [(name, i, j) for name in hyser.ARRAY_NAMES for i in range(1, 9) for j in range(1, 9)]
    hy = montage_to_dict(hyser.build_montage_metadata(layout, 1, 1, "x"), [True] * 256)
    groups = set()
    for _ in range(200):
        vm = VM.draw(hy, np.ones(256, bool), VM.VirtualSpec(p=1.0), rng)
        g = set((vm.a // 64).tolist()) | (set((vm.b // 64).tolist()) if vm.b is not None else set())
        assert len(g) == 1  # una sola griglia
        groups |= g
        assert vm.montage["groups"][0]["group_id"] == hy["groups"][g.pop()]["group_id"]
    assert groups == {0, 1, 2, 3}


def test_band_limit_of_the_presented_montage():
    m = _capgmyo()
    vm = VM.draw(m, np.ones(128, bool), VM.VirtualSpec(p=1.0, p_bipolar=1.0), np.random.default_rng(0))
    assert VM.band_limit_hz(vm.montage, 1000.0) == pytest.approx(np.full(len(vm.a), 500.0))


def test_virtual_montage_before_the_filter_equals_after(tmp_path):
    """Il montaggio virtuale si applica prima del filtro (si filtrano solo i canali che servono): stesso risultato che dopo, filtro lineare."""
    from wearusfm.data import pretraining_loader as L

    cfg = L.signed_config((20.0, 450.0))
    x = np.random.default_rng(0).normal(size=(128, 3000)).astype(np.float32)
    vm = VM.draw(_capgmyo(), np.ones(128, bool), VM.VirtualSpec(p=1.0, p_bipolar=1.0), np.random.default_rng(4))
    before = L._filter(VM.apply(x, vm), 1000.0, cfg.filter_band_hz, cfg.notch_hz)
    after = VM.apply(L._filter(x, 1000.0, cfg.filter_band_hz, cfg.notch_hz), vm)
    assert np.allclose(before, after, atol=1e-4 * np.abs(after).max())
