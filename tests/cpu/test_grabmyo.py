from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wearusfm.ingest.grabmyo import (
    EMG_CHANNEL_NAMES,
    RINGS,
    build_montage_metadata,
    load_dat,
    parse_hea,
    qc_channel_validity,
    scan_grabmyo,
    select_emg_channels,
)

ALL_32_CHANNEL_NAMES = list(EMG_CHANNEL_NAMES[:8]) + ["U1"] + list(EMG_CHANNEL_NAMES[8:16]) + \
    ["U2", "U3"] + list(EMG_CHANNEL_NAMES[16:22]) + list(EMG_CHANNEL_NAMES[22:28]) + ["U4"]
# ordine arbitrario ma coerente col fatto che U1-U4 sono intercalati nel file reale (colonne 17,24,25,32)


def _write_record(dir_: Path, session: int, participant: int, gesture: int, trial: int,
                   n_samples: int = 20, channel_names: list[str] | None = None) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    names = channel_names or ALL_32_CHANNEL_NAMES
    name = f"session{session}_participant{participant}_gesture{gesture}_trial{trial}"
    hea_lines = [f"{name} {len(names)} 2048 {n_samples}"]
    for cname in names:
        hea_lines.append(f"{name}.dat 16 207197.368(4547)/mV 16 0 0 0 0 {cname}")
    (dir_ / f"{name}.hea").write_text("\n".join(hea_lines) + "\n")

    rng = np.random.default_rng(hash((session, participant, gesture, trial)) % (2**31))
    data = rng.integers(-1000, 1000, size=(n_samples, len(names)), dtype=np.int16)
    data.tofile(dir_ / f"{name}.dat")
    return dir_ / f"{name}.hea"


def test_scan_grabmyo_parses_filenames(tmp_path: Path) -> None:
    _write_record(tmp_path / "Session1" / "session1_participant1", 1, 1, 3, 2)
    _write_record(tmp_path / "Session1" / "session1_participant1", 1, 1, 3, 3)

    files = scan_grabmyo(tmp_path)

    assert len(files) == 2
    assert files[0].session == 1 and files[0].participant == 1
    assert files[0].gesture == 3 and files[0].trial == 2


def test_parse_hea_reads_header_fields(tmp_path: Path) -> None:
    hea_path = _write_record(tmp_path, 1, 26, 3, 2, n_samples=10240)

    header = parse_hea(hea_path)

    assert header.n_channels == 32
    assert header.fs_hz == 2048.0
    assert header.n_samples == 10240
    assert header.channels[0].signal_name == "F1"
    assert header.channels[0].gain == pytest.approx(207197.368)
    assert header.channels[0].baseline == 4547
    assert header.channels[0].units == "mV"


def test_load_dat_shape_matches_header(tmp_path: Path) -> None:
    hea_path = _write_record(tmp_path, 1, 1, 1, 1, n_samples=20)
    header = parse_hea(hea_path)

    data = load_dat(header, hea_path.with_suffix(".dat"))

    assert data.shape == (20, 32)
    assert data.dtype == np.int16


def test_load_dat_raises_on_size_mismatch(tmp_path: Path) -> None:
    hea_path = _write_record(tmp_path, 1, 1, 1, 1, n_samples=20)
    header = parse_hea(hea_path)
    # tronca il .dat per simulare un file corrotto
    dat_path = hea_path.with_suffix(".dat")
    dat_path.write_bytes(dat_path.read_bytes()[:-10])

    with pytest.raises(ValueError, match="campioni letti"):
        load_dat(header, dat_path)


def test_select_emg_channels_drops_unused(tmp_path: Path) -> None:
    hea_path = _write_record(tmp_path, 1, 1, 1, 1, n_samples=20)
    header = parse_hea(hea_path)
    data = load_dat(header, hea_path.with_suffix(".dat"))

    emg_data, names = select_emg_channels(header, data)

    assert emg_data.shape == (20, 28)
    assert set(names) == set(EMG_CHANNEL_NAMES)
    assert "U1" not in names and "U4" not in names


def test_select_emg_channels_raises_if_missing() -> None:
    from wearusfm.ingest.grabmyo import RecordHeader, ChannelHeader

    header = RecordHeader(
        record_name="x", n_channels=1, fs_hz=2048.0, n_samples=10,
        channels=(ChannelHeader(signal_name="F1", gain=1.0, baseline=0, units="mV"),),
    )
    data = np.zeros((10, 1), dtype=np.int16)

    with pytest.raises(ValueError, match="mancanti"):
        select_emg_channels(header, data)


def test_qc_channel_validity_flags_flat_channel() -> None:
    rng = np.random.default_rng(0)
    data = rng.integers(-500, 500, size=(1000, 4)).astype(np.int16)
    data[:, 1] = 0

    valid = qc_channel_validity(data)

    assert valid.tolist() == [True, False, True, True]


def test_build_montage_metadata_has_four_rings() -> None:
    montage = build_montage_metadata(participant=26, session=1)

    assert montage.dataset_name == "grabmyo"
    assert montage.subject_id == "grabmyo_p26"
    assert montage.session_id == "session1"
    assert montage.day_index == 1
    assert montage.n_channels == 28
    assert [g.group_id for g in montage.groups] == list(RINGS.keys())
    assert [g.symmetry for g in montage.groups] == ["D_8", "D_8", "D_6", "D_6"]
    assert all(g.topology.value == "ring" for g in montage.groups)
