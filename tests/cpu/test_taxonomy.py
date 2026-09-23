import dataclasses
import json
import math
import re

import pytest

from wearusfm.metadata.schema import (
    AnatomicalIdentity,
    AnatomicalPrecision,
    ChannelMetadata,
    RawOrEnvelope,
    SensorCoordinates,
)
from wearusfm.metadata.taxonomy import (
    CAMARGO_EMG_COLUMNS,
    COMPARTMENTS,
    MUSCLES,
    PROXIMAL_DISTAL_BOUNDARY,
    REGIONS,
    WRIST_BOUNDARY,
    ancestors,
    atlas_identity,
    atlas_weights,
    identity,
    region_at_level,
)

UPPER_LIMB = ("FCU", "PL", "FDS", "FCR", "PT", "BRD", "ECRL", "ECRB", "EDC", "EDM", "ECU",
              "APL", "EPB", "BB", "BRA", "TB")


def _muscles_of(compartment: str) -> set[str]:
    return {m.key for m in MUSCLES.values() if m.compartment == compartment}


def _compartments_of(region: str) -> set[str]:
    return {c.key for c in COMPARTMENTS.values() if c.region == region}


# --- albero -----------------------------------------------------------------

def test_keys_unique_across_levels():
    keys = [*REGIONS, *COMPARTMENTS, *MUSCLES]
    assert len(keys) == len(set(keys))


def test_every_parent_exists():
    for m in MUSCLES.values():
        assert m.compartment in COMPARTMENTS
    for c in COMPARTMENTS.values():
        assert c.region in REGIONS


def test_draft_table_of_v10_forearm_and_arm():
    # v10 §4.5, tabella "Bozza dei compartimenti", riga per riga
    assert _compartments_of("forearm_proximal") == {
        "flexor_pronator_ulnar", "flexor_pronator_radial", "radial_group", "dorsal_extensors",
    }
    assert _muscles_of("flexor_pronator_ulnar") == {"FCU", "PL", "FDS"}
    assert _muscles_of("flexor_pronator_radial") == {"FCR", "PT"}
    assert _muscles_of("radial_group") == {"BRD", "ECRL", "ECRB"}
    assert _muscles_of("dorsal_extensors") == {"EDC", "EDM", "ECU"}
    assert _compartments_of("forearm_distal") == {"dorsoradial_deep_outcropping"}
    assert _muscles_of("dorsoradial_deep_outcropping") == {"APL", "EPB"}
    assert _compartments_of("upper_arm") == {"arm_anterior", "arm_posterior"}
    assert _muscles_of("arm_anterior") == {"BB", "BRA"}
    assert _muscles_of("arm_posterior") == {"TB"}


def test_wrist_has_four_sectors_and_no_muscles():
    sectors = _compartments_of("wrist")
    assert sectors == {
        "wrist_volar_ulnar", "wrist_volar_radial", "wrist_dorsal_radial", "wrist_dorsal_ulnar",
    }
    for s in sectors:
        assert COMPARTMENTS[s].is_sector
        assert _muscles_of(s) == set()  # tendini e ventri distali, v10 §4.5


def test_sectors_only_at_wrist():
    for c in COMPARTMENTS.values():
        assert c.is_sector == (c.region == "wrist")


def test_camargo_has_eleven_muscles_and_keeps_them_distinguishable():
    assert len(CAMARGO_EMG_COLUMNS) == 11
    assert len(set(CAMARGO_EMG_COLUMNS.values())) == 11
    assert all(k in MUSCLES for k in CAMARGO_EMG_COLUMNS.values())
    assert list(CAMARGO_EMG_COLUMNS)[0] == "gastrocmed"  # ordine delle colonne del file
    # v10 §4.5, "Perche' non solo compartimento": questi collassano a livello di compartimento
    assert len({MUSCLES[k].compartment for k in ("VM", "VL", "RF")}) == 1
    assert len({MUSCLES[k].compartment for k in ("BF", "ST")}) == 1
    assert len({MUSCLES[k].compartment for k in ("GASMED", "SOL")}) == 1


def test_ontology_ids_well_formed_and_unique():
    for key in (*UPPER_LIMB, *CAMARGO_EMG_COLUMNS.values()):
        m = MUSCLES[key]
        assert m.uberon_id is not None and re.fullmatch(r"UBERON:\d{7}", m.uberon_id), key
        assert m.fma_id is not None and re.fullmatch(r"FMA:\d+", m.fma_id), key
    uberon = [m.uberon_id for m in MUSCLES.values() if m.uberon_id]
    fma = [m.fma_id for m in MUSCLES.values() if m.fma_id]
    assert len(uberon) == len(set(uberon))
    assert len(fma) == len(set(fma))


# --- annotazione a mano -------------------------------------------------------

def test_ancestors_derived_by_lookup():
    assert ancestors("FCU") == ("FCU", "flexor_pronator_ulnar", "forearm_proximal")
    assert ancestors("wrist_dorsal_ulnar") == ("wrist_dorsal_ulnar", "wrist")
    assert ancestors("upper_arm") == ("upper_arm",)


def test_identity_at_muscle_level():
    ident = identity("FCU")
    assert ident == AnatomicalIdentity(
        region="forearm_proximal", compartment="flexor_pronator_ulnar", muscle="FCU",
        precision=AnatomicalPrecision.MUSCLE, muscle_ontology_id="UBERON:0001522",
    )


def test_identity_at_sector_compartment_region_level():
    sec = identity("wrist_volar_radial")
    assert (sec.precision, sec.region, sec.sector, sec.compartment) == (
        AnatomicalPrecision.SECTOR, "wrist", "wrist_volar_radial", None,
    )
    comp = identity("radial_group")
    assert (comp.precision, comp.region, comp.compartment, comp.muscle) == (
        AnatomicalPrecision.COMPARTMENT, "forearm_proximal", "radial_group", None,
    )
    reg = identity("upper_arm")
    assert (reg.precision, reg.region, reg.compartment) == (
        AnatomicalPrecision.REGION, "upper_arm", None,
    )


def test_identity_nominal_flag_for_amputees():
    assert identity("ECU", nominal=True).nominal is True


def test_unknown_label_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        identity("flexor_digitorum_profundus")


# --- funzione atlante ---------------------------------------------------------

LEVELS = (0.0, 0.2, 0.49, 0.5, 0.7, 0.89, 0.9, 1.0)
ANGLES = (-725.0, -1.0, 0.0, 17.0, 90.0, 179.9, 180.0, 271.3, 359.999, 360.0, 1000.0)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("angle", ANGLES)
def test_atlas_weights_sum_to_one(angle, level):
    w = atlas_weights(angle, level)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)
    assert all(v >= 0.0 for v in w.values())


@pytest.mark.parametrize("sigma", (1.0, 20.0, 90.0, 400.0))
def test_atlas_weights_sum_to_one_for_any_sigma(sigma):
    assert math.isclose(sum(atlas_weights(123.0, 0.3, sigma).values()), 1.0, abs_tol=1e-12)


def test_atlas_keys_are_the_compartments_of_the_region():
    assert set(atlas_weights(0.0, 0.2)) == _compartments_of("forearm_proximal")
    assert set(atlas_weights(0.0, 0.7)) == (
        _compartments_of("forearm_proximal") | _compartments_of("forearm_distal")
    )
    assert set(atlas_weights(0.0, 0.95)) == _compartments_of("wrist")


def test_region_boundaries():
    assert region_at_level(0.0) == "forearm_proximal"
    assert region_at_level(PROXIMAL_DISTAL_BOUNDARY) == "forearm_distal"
    assert region_at_level(WRIST_BOUNDARY) == "wrist"
    assert region_at_level(1.0) == "wrist"


def test_atlas_follows_circumferential_order_of_v10():
    # v10 §4.5: dall'ulna in direzione volare FCU, PL/FDS, FCR, PT, BRD, ECRL, ECRB, EDC, EDM, ECU
    expected = ["flexor_pronator_ulnar"] * 2 + ["flexor_pronator_radial"] * 2 + \
        ["radial_group"] * 3 + ["dorsal_extensors"] * 3
    for i, comp in enumerate(expected):
        w = atlas_weights(36.0 * i + 18.0, 0.2)
        assert max(w, key=w.get) == comp


def test_distal_forearm_inserts_apl_epb_between_ecrb_and_edc():
    slot = 360.0 / 11
    for i, comp in ((6, "radial_group"), (7, "dorsoradial_deep_outcropping"), (8, "dorsal_extensors")):
        w = atlas_weights((i + 0.5) * slot, 0.7)
        assert max(w, key=w.get) == comp


def test_wrist_sectors_by_quadrant():
    for angle, sector in ((45.0, "wrist_volar_ulnar"), (135.0, "wrist_volar_radial"),
                          (225.0, "wrist_dorsal_radial"), (315.0, "wrist_dorsal_ulnar")):
        w = atlas_weights(angle, 0.95)
        assert max(w, key=w.get) == sector


def test_atlas_wraps_around_the_ulna():
    assert atlas_weights(-1.0, 0.2) == pytest.approx(atlas_weights(359.0, 0.2))
    assert atlas_weights(720.0, 0.2) == pytest.approx(atlas_weights(0.0, 0.2))
    # sull'ulna l'elettrodo sta a cavallo fra FCU ed ECU, quasi in parti uguali (gli archi dei
    # due compartimenti hanno larghezze diverse, quindi le code no)
    w = atlas_weights(0.0, 0.2)
    assert w["flexor_pronator_ulnar"] == pytest.approx(0.5, abs=1e-3)
    assert w["dorsal_extensors"] == pytest.approx(0.5, abs=1e-3)


def test_smaller_sigma_is_sharper():
    wide = atlas_weights(54.0, 0.2, sigma_deg=40.0)["flexor_pronator_ulnar"]
    narrow = atlas_weights(54.0, 0.2, sigma_deg=5.0)["flexor_pronator_ulnar"]
    assert narrow > wide


def test_unknown_angle_gives_no_weights():
    assert atlas_weights(None, 0.3) is None
    ident = atlas_identity(None, 0.3)
    assert ident.precision == AnatomicalPrecision.REGION
    assert ident.region == "forearm_proximal"
    assert ident.soft_compartment_weights is None
    assert ident.compartment is None and ident.sector is None and ident.muscle is None


def test_unknown_level_gives_unknown_even_with_known_angle():
    assert atlas_weights(90.0, None) is None
    for angle in (None, 90.0):
        assert atlas_identity(angle, None) == AnatomicalIdentity()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        atlas_weights(10.0, 1.2)
    with pytest.raises(ValueError):
        atlas_weights(10.0, -0.1)
    with pytest.raises(ValueError):
        atlas_weights(float("nan"), 0.3)
    with pytest.raises(ValueError):
        atlas_weights(10.0, 0.3, sigma_deg=0.0)


def test_atlas_identity_is_a_schema_identity():
    ident = atlas_identity(200.0, 0.3, nominal=True)
    assert isinstance(ident, AnatomicalIdentity)
    assert ident.precision == AnatomicalPrecision.REGION
    assert ident.nominal is True
    assert ident.compartment is None  # il compartimento e' solo stimato: niente etichetta dura
    ch = ChannelMetadata(
        sensor_coords=SensorCoordinates(channel_index=0, ring_angle_deg=200.0),
        electrode_type="Ag/AgCl", native_fs_hz=200.0, effective_band_hz=(20.0, 95.0),
        mains_frequency_hz=50, raw_or_envelope=RawOrEnvelope.RAW, anatomical_identity=ident,
    )
    json.dumps(dataclasses.asdict(ch))
