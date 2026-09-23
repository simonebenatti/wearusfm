import json

import pytest

from wearusfm.metadata.schema import (
    AnatomicalIdentity,
    AnatomicalPrecision,
    CalibrationStatus,
    ChannelGroup,
    ChannelMetadata,
    Chirality,
    MontageMetadata,
    RawOrEnvelope,
    SensorCoordinates,
    Topology,
    export_json_schema,
)


def _make_channel(idx: int, precision=AnatomicalPrecision.MUSCLE, muscle="FCU") -> ChannelMetadata:
    return ChannelMetadata(
        sensor_coords=SensorCoordinates(channel_index=idx),
        electrode_type="Ag/AgCl",
        native_fs_hz=2000.0,
        effective_band_hz=(20.0, 450.0),
        mains_frequency_hz=50,
        raw_or_envelope=RawOrEnvelope.RAW,
        anatomical_identity=AnatomicalIdentity(precision=precision, muscle=muscle if precision == AnatomicalPrecision.MUSCLE else None),
    )


def test_minimal_channel_constructs():
    ch = _make_channel(0)
    assert ch.qc_valid is True
    assert ch.calibration_status == CalibrationStatus.ABSENT


def test_mains_frequency_must_be_50_or_60():
    with pytest.raises(ValueError):
        ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=0),
            electrode_type="Ag/AgCl",
            native_fs_hz=2000.0,
            effective_band_hz=(20.0, 450.0),
            mains_frequency_hz=45,
            raw_or_envelope=RawOrEnvelope.RAW,
            anatomical_identity=AnatomicalIdentity(),
        )


def test_mvc_reference_requires_calibration_available():
    with pytest.raises(ValueError):
        ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=0),
            electrode_type="Ag/AgCl",
            native_fs_hz=2000.0,
            effective_band_hz=(20.0, 450.0),
            mains_frequency_hz=50,
            raw_or_envelope=RawOrEnvelope.RAW,
            anatomical_identity=AnatomicalIdentity(),
            calibration_status=CalibrationStatus.ABSENT,
            mvc_reference=0.8,
        )


def test_channel_group_rejects_empty_channel_list():
    with pytest.raises(ValueError):
        ChannelGroup(group_id="ring", topology=Topology.RING, symmetry="D_8", channels=[])


def test_sparse_group_must_have_no_symmetry():
    with pytest.raises(ValueError):
        ChannelGroup(
            group_id="targeted", topology=Topology.SPARSE, symmetry="D_8",
            channels=[_make_channel(0)],
        )
    # "none" e' invece valido
    ChannelGroup(
        group_id="targeted", topology=Topology.SPARSE, symmetry="none",
        channels=[_make_channel(0)],
    )


def test_ninapro_standard_montage_mixed_groups():
    # v10 §2.1/§3.4: montaggio a 12 elettrodi = anello da 8 (D_8) + mirati da 4 (nessuna simmetria)
    # SECTOR e' riservato al polso (v10 §4.5, "Due cautele"); questo anello e' all'altezza
    # radio-omerale, quindi REGION (stima via funzione atlante, taxonomy.py) e' la precisione
    # corretta - non SECTOR.
    ring_group = ChannelGroup(
        group_id="ring", topology=Topology.RING, symmetry="D_8",
        channels=[_make_channel(i, precision=AnatomicalPrecision.REGION, muscle=None) for i in range(8)],
    )
    targeted_group = ChannelGroup(
        group_id="targeted", topology=Topology.SPARSE, symmetry="none",
        channels=[_make_channel(8 + i) for i in range(4)],
    )
    montage = MontageMetadata(
        dataset_name="ninapro_standard", subject_id="ninapro_s01", session_id="s1",
        groups=[ring_group, targeted_group],
    )
    assert montage.n_channels == 12


def test_montage_rejects_duplicate_group_ids():
    g1 = ChannelGroup(group_id="a", topology=Topology.RING, symmetry="D_8", channels=[_make_channel(0)])
    g2 = ChannelGroup(group_id="a", topology=Topology.SPARSE, symmetry="none", channels=[_make_channel(1)])
    with pytest.raises(ValueError):
        MontageMetadata(dataset_name="x", subject_id="s1", session_id="sess1", groups=[g1, g2])


def test_montage_rejects_no_groups():
    with pytest.raises(ValueError):
        MontageMetadata(dataset_name="x", subject_id="s1", session_id="sess1", groups=[])


def test_nominal_flag_for_amputee_channels():
    identity = AnatomicalIdentity(precision=AnatomicalPrecision.MUSCLE, muscle="FCU", nominal=True)
    ch = ChannelMetadata(
        sensor_coords=SensorCoordinates(channel_index=0), electrode_type="Ag/AgCl",
        native_fs_hz=2000.0, effective_band_hz=(20.0, 450.0), mains_frequency_hz=60,
        raw_or_envelope=RawOrEnvelope.RAW, anatomical_identity=identity,
        chirality=Chirality.LEFT,
    )
    assert ch.anatomical_identity.nominal is True


def test_json_schema_exports_and_is_serializable():
    schema = export_json_schema()
    text = json.dumps(schema)  # deve essere serializzabile senza errori
    assert '"title"' in text
    assert schema["required"] == ["dataset_name", "subject_id", "session_id", "groups"]


def test_json_schema_topology_enum_matches_python_enum():
    schema = export_json_schema()
    topology_enum = schema["properties"]["groups"]["items"]["properties"]["topology"]["enum"]
    assert set(topology_enum) == {t.value for t in Topology}
