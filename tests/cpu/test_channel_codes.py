"""Codici di canale (passo 6) sui montaggi veri prodotti dagli adattatori di ingest: anello (emg2pose), sparso (Camargo), griglia (CapgMyo),
anello piu' mirati (NinaPro DB2)."""

import copy
import math

import numpy as np
import pytest

from wearusfm.ingest import camargo, capgmyo, emg2pose, ninapro_std
from wearusfm.ingest.common import montage_to_dict
from wearusfm.metadata import taxonomy as T
from wearusfm.model import channel_codes as CC


def _dict(m, valid=None):
    return montage_to_dict(m, valid if valid is not None else [True] * m.n_channels)


def test_layout_and_relative_geometry_stay_inside_metric_groups():
    m = _dict(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"))  # anello da 8 + 4 mirati
    lay = CC.layout_from_montage(m)
    assert lay.n_channels == 12 and lay.topology.tolist() == [0] * 8 + [2] * 4
    rel = CC.relative_geometry(lay)
    assert rel["metric"][:8, :8].all() and not rel["metric"][8:].any() and not rel["metric"][:, 8:].any()  # niente geometria verso i mirati
    assert rel["d_angle"][0, 1] == pytest.approx(math.radians(45)) and rel["d_angle"][1, 0] == pytest.approx(-math.radians(45))
    assert rel["d_angle"][0, 7] == pytest.approx(-math.radians(45))  # 315 gradi = -45: l'anello si chiude
    assert np.isnan(rel["distance"][0, 8])


def test_neighbors_ring_grid_sparse():
    ring = CC.layout_from_montage(_dict(emg2pose.build_montage_metadata("u1", "s", "left")))  # 16 canali, passo 22,5 gradi
    nb = CC.neighbors(ring, 2)
    for i in range(16):
        assert set(nb[i]) == {(i + 1) % 16, (i - 1) % 16}
    assert nb[0, 0] == 1  # a parita' di distanza, prima lo spostamento angolare positivo
    grid = CC.layout_from_montage(_dict(capgmyo.build_montage_metadata(1)))  # 8 x 16
    nb = CC.neighbors(grid, 4)
    i = 2 * 16 + 5  # riga 2, colonna 5: interno
    assert sorted(nb[i]) == sorted([1 * 16 + 5, 3 * 16 + 5, 2 * 16 + 4, 2 * 16 + 6])
    assert set(nb[0]) == {1, 16, 17, 2}  # all'angolo: i due lati, la diagonale, poi (0,2) prima di (2,0) a parita' di distanza
    sparse = CC.layout_from_montage(_dict(camargo.build_montage_metadata(1, "d")))
    assert (CC.neighbors(sparse, 3) == -1).all()  # v10 §5.4: su Camargo il vicinato metrico e' vuoto


def test_neighbors_depend_only_on_relative_geometry():
    m = _dict(emg2pose.build_montage_metadata("u1", "s", "left"))
    base = CC.neighbors(CC.layout_from_montage(m), 3)
    rotated = copy.deepcopy(m)
    for c in rotated["groups"][0]["channels"]:
        c["sensor_coords"]["ring_angle_deg"] = (c["sensor_coords"]["ring_angle_deg"] + 3 * 22.5) % 360.0  # rotazione dell'anello
    assert np.array_equal(CC.neighbors(CC.layout_from_montage(rotated), 3), base)
    perm = np.random.default_rng(0).permutation(16)  # permutare le colonne permuta l'uscita allo stesso modo
    shuffled = copy.deepcopy(m)
    shuffled["groups"][0]["channels"] = [m["groups"][0]["channels"][p] for p in perm]
    inv = np.argsort(perm)
    assert np.array_equal(CC.neighbors(CC.layout_from_montage(shuffled), 3), inv[base[perm]])


def test_reflection_flips_angular_displacements():
    m = _dict(emg2pose.build_montage_metadata("u1", "s", "left"))
    mirrored = copy.deepcopy(m)
    for c in mirrored["groups"][0]["channels"]:
        c["sensor_coords"]["ring_angle_deg"] = (-c["sensor_coords"]["ring_angle_deg"]) % 360.0
    a = CC.relative_geometry(CC.layout_from_montage(m))
    b = CC.relative_geometry(CC.layout_from_montage(mirrored))
    away = ~np.isclose(np.abs(a["d_angle"]), math.pi)  # sulle coppie opposte il segno e' convenzione
    assert np.allclose(b["d_angle"][away], -a["d_angle"][away]) and np.allclose(b["distance"], a["distance"])


def test_qc_invalid_channels_are_nobodys_neighbors():
    m = _dict(emg2pose.build_montage_metadata("u1", "s", "left"), [i != 1 for i in range(16)])
    nb = CC.neighbors(CC.layout_from_montage(m), 2)
    assert 1 not in nb and (nb[1] == -1).all() and set(nb[0]) == {2, 15}
    assert set(CC.neighbors(CC.layout_from_montage(m), 2, only_valid=False)[0]) == {1, 15}


def test_missing_geometry_is_an_error():
    m = _dict(emg2pose.build_montage_metadata("u1", "s", "left"))
    m["groups"][0]["channels"][3]["sensor_coords"]["ring_angle_deg"] = None
    with pytest.raises(ValueError, match="mai geometria inventata"):
        CC.layout_from_montage(m)


def test_anatomy_codes_follow_precision():
    db2 = CC.anatomy_codes(_dict(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right")))
    assert (db2.region[:8] == 0).all() and (db2.compartment_weights[:8, 0] == 1).all() and not db2.muscle_known[:8].any()  # anello: ignoto
    assert db2.muscle_known[8:].all() and CC.MUSCLE_KEYS[db2.muscle[8]] == "FDS"
    assert CC.COMPARTMENT_KEYS[int(db2.compartment_weights[8].argmax())] == "flexor_pronator_ulnar"
    assert CC.REGION_KEYS[db2.region[11]] == "upper_arm" and db2.topology.tolist() == [0] * 8 + [2] * 4
    cam = CC.anatomy_codes(_dict(camargo.build_montage_metadata(1, "d")))
    assert cam.muscle_known.all() and np.allclose(cam.compartment_weights.sum(axis=1), 1.0)
    assert CC.REGION_KEYS[cam.region[list(np.array(CC.MUSCLE_KEYS)[cam.muscle]).index("EO")]] == "trunk"  # l'obliquo esterno e' tronco (D7b)


def test_anatomy_codes_soft_weights_sector_and_errors():
    m = _dict(emg2pose.build_montage_metadata("u1", "s", "left"))
    ch = m["groups"][0]["channels"]
    soft = T.atlas_identity(30.0, 0.3)  # funzione atlante: regione nota, compartimento stimato
    ch[0]["anatomical_identity"].update(region=soft.region, precision="region", soft_compartment_weights=soft.soft_compartment_weights)
    ch[1]["anatomical_identity"].update(region="wrist", sector="wrist_volar_radial", precision="sector")
    codes = CC.anatomy_codes(m)
    w = codes.compartment_weights[0]
    assert w.sum() == pytest.approx(1.0) and w[0] == 0.0 and CC.REGION_KEYS[codes.region[0]] == "forearm_proximal"
    for comp, v in soft.soft_compartment_weights.items():
        assert w[CC.COMPARTMENT_KEYS.index(comp)] == pytest.approx(v)
    assert CC.COMPARTMENT_KEYS[int(codes.compartment_weights[1].argmax())] == "wrist_volar_radial" and CC.REGION_KEYS[codes.region[1]] == "wrist"
    ch[2]["anatomical_identity"].update(muscle="FLEXOR_INVENTATO", precision="muscle")
    with pytest.raises(KeyError, match="sconosciuta"):
        CC.anatomy_codes(m)
    ch[2]["anatomical_identity"].update(muscle="FCU", region="upper_arm")
    with pytest.raises(ValueError, match="incoerente"):
        CC.anatomy_codes(m)


def test_attention_sets_distances_in_electrode_steps_sparse_sets_and_packing():
    db2 = CC.attention_sets(CC.layout_from_montage(_dict(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"))), 2)
    assert db2.index[0].tolist()[:3] == [0, 1, 7] and db2.dist[0, :3].tolist() == pytest.approx([0.0, 1.0, 1.0])  # 45 gradi = 1 passo
    assert (db2.pair_type[0, 1:3] == CC.PAIR_RING).all() and (db2.index[0, 3:] == -1).all()
    assert db2.index[8].tolist() == [8, *[j for j in range(12) if j != 8]] and (db2.pair_type[8, 1:] == CC.PAIR_SET).all()
    ring16 = CC.attention_sets(CC.layout_from_montage(_dict(emg2pose.build_montage_metadata("u1", "s", "left"))), 2)
    assert ring16.dist[0, 1:3].tolist() == pytest.approx([1.0, 1.0])  # 22,5 gradi = 1 passo anche qui: stessa scala su anelli diversi
    grid = CC.attention_sets(CC.layout_from_montage(_dict(capgmyo.build_montage_metadata(1))), 4)
    assert grid.d_row[0].tolist()[1:] == [0.0, 1.0, 1.0, 0.0] and grid.d_col[0].tolist()[1:] == [1.0, 0.0, 1.0, 2.0]
    bad = CC.attention_sets(CC.layout_from_montage(_dict(emg2pose.build_montage_metadata("u1", "s", "left"), [i != 3 for i in range(16)])), 2)
    assert bad.index[3].tolist() == [3, -1, -1] and 3 not in bad.index[[i for i in range(16) if i != 3]]
    packed = CC.pack_attention_sets([db2, ring16])
    assert packed.index.shape == (28, 12) and packed.index[12, :3].tolist() == [12, 13, 27]  # spostati di 12, larghezza massima
    assert ((packed.index[:12] < 12) | (packed.index[:12] == -1)).all() and (packed.index[12:][packed.index[12:] >= 0] >= 12).all()
