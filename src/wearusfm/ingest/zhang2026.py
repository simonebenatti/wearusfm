"""Ingest di Zhang et al. 2026 (Groningen, «anatomical» contro «random»; passo 2). Opzione B (Simone, 30/09/2026) e modo `random` sparso, etichette non
allineate (Simone, 01/10/2026): `docs/decisioni.md`, «Zhang 2026».

Fonti (README del dataset, abstract del paper; citazioni verificate, `docs/fatti_da_verificare.md`, fatto 10b): 64 partecipanti, gesti fatti con la
mano NON dominante; un sensore Delsys Trigno Quattro (4 elettrodi, EMG a 2000 Hz, una IMU a 148 Hz) e quattro Trigno Avanti (un elettrodo ciascuno,
EMG a 4000 Hz, IMU a 74 Hz); due parti dell'esperimento: `anatomical` (sensori sugli 8 muscoli) e `random` (attorno all'avambraccio; ordine ignoto per
32 partecipanti); `label.csv` con tempi contati dall'inizio della registrazione VIDEO.

Formato verificato su file reali il 01/10/2026 (`scripts/inspect_zhang_csv.py`): `<raw>/HG_<id>/<anatomical|random>/sequence_NN/sensor_data.csv`
(NN da 01 a 10, non consecutivi), separatore `,`, intestazione di **7** righe (il README dice 8: si riconosce, non si conta) = sensore, tipo, muscolo
(`NaN` in `random`), frequenza («2000 Hz», «148,1481 Hz»), colore/ordine, numero del dispositivo, unita' (`mV`). Ogni colonna e' una serie a se',
**contigua dall'inizio** del file: i canali a 4000 Hz occupano tutte le righe, quelli a 2000 Hz la prima meta'. Punto decimale nei valori.

Cosa fa:
- tiene solo le 8 colonne EMG; i 4 canali a 4000 Hz si portano a 2000 Hz con `resample_poly` (polifase, filtro anti-alias) e si allineano campione per
  campione dall'inizio ai 4 a 2000 Hz (stessa durata nei file letti); il sidecar registra `resampled_from_hz: 4000` per quei canali;
- un soggetto x un modo = UNA sessione (`<out>/<HG_id>/<modo>/`), le sequenze concatenate come prove (`trials` con `sequence`, offset, lunghezza);
- canali in ordine di sensore S1...S8; un solo gruppo sparso. `anatomical`: muscolo dall'intestazione (FDS, PT, EDC, ECRL, ECU, FCR, FCU; il
  supinatore, che non e' nella tassonomia firmata D7b, al livello di regione «avambraccio prossimale»); `random`: identita' ignota, nessun angolo;
- lateralita' = lato opposto alla mano dominante di `participants.csv` (deduzione dalla frase «with their non-dominant hand»);
- `label.csv` copiato tale e quale in `labels_video_time/`, segnato `labels_aligned_to_emg: false`;
- rete 50 Hz: deduzione (Groningen), non dichiarata.
Nessun dato anagrafico dei partecipanti entra nel processato (solo la mano dominante, per la lateralita').
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from wearusfm.ingest.common import relative_min_std
from wearusfm.metadata.schema import (
    AnatomicalIdentity,
    AnatomicalPrecision,
    ChannelGroup,
    ChannelMetadata,
    Chirality,
    MontageMetadata,
    RawOrEnvelope,
    SensorCoordinates,
    Topology,
)
from wearusfm.metadata.taxonomy import identity

GRID_FS_HZ = 2000.0
RAW_FS = (2000.0, 4000.0)
N_EMG = 8
N_HEADER = 7  # visto sui file; il README dice 8
HEADER_SCAN_ROWS = 50
MODES = ("anatomical", "random")
# abbreviazione dell'intestazione -> chiave della tassonomia; None = muscolo non in tassonomia (supinatore): regione
MUSCLE_KEYS = {"FDS": "FDS", "PTE": "PT", "SUP": None, "EDC": "EDC", "ECR": "ECRL", "ECU": "ECU", "FCR": "FCR", "FCU": "FCU"}
SUP_REGION = "forearm_proximal"
MAX_LEN_MISMATCH = 2  # campioni a 2 kHz fra canali a 2000 Hz e canali a 4000 Hz ricampionati, oltre i quali si solleva
_SEQ_RE = re.compile(r"^sequence_(\d+)$")


def _finite(s: str) -> bool:
    try:
        return math.isfinite(float(s.strip().replace(",", ".")))
    except ValueError:
        return False


def read_header(path: Path) -> list[list[str]]:
    """Le righe d'intestazione: fino all'ultima riga, fra le prime HEADER_SCAN_ROWS, con una cella non vuota che non e' un numero finito."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        head = [row for _, row in zip(range(HEADER_SCAN_ROWS), csv.reader(f))]
    last_text = max((i for i, row in enumerate(head) if any(c.strip() and not _finite(c) for c in row)), default=-1)
    return head[: last_text + 1]


def emg_columns(header: list[list[str]]) -> list[dict]:
    """Le colonne EMG: indice, sensore (S1-S8), muscolo (None se `NaN`), frequenza nativa, numero del dispositivo."""
    if len(header) != N_HEADER:
        raise ValueError(f"intestazione di {len(header)} righe, attese {N_HEADER}")
    sensor, kind, muscle, freq, _color, device, unit = header
    out = []
    for j, k in enumerate(kind):
        if k.strip() != "EMG":
            continue
        fs = float(freq[j].replace("Hz", "").strip().replace(",", "."))
        if fs not in RAW_FS:
            raise ValueError(f"colonna {j}: frequenza {freq[j]!r}, attese {RAW_FS}")
        if unit[j].strip() != "mV":
            raise ValueError(f"colonna {j}: unita' {unit[j]!r}, attesa mV")
        m = re.fullmatch(r"S([1-8])", sensor[j].strip())
        if not m:
            raise ValueError(f"colonna {j}: sensore {sensor[j]!r}")
        mus = muscle[j].strip()
        out.append({"col": j, "sensor": int(m.group(1)), "muscle": None if mus in ("", "NaN") else mus, "raw_fs_hz": fs,
                    "device": device[j].strip()})
    if len(out) != N_EMG or sorted(c["sensor"] for c in out) != list(range(1, N_EMG + 1)):
        raise ValueError(f"colonne EMG {[(c['col'], c['sensor']) for c in out]}, attesi i sensori S1-S8")
    return sorted(out, key=lambda c: c["sensor"])


def _contiguous(x: np.ndarray, name: str) -> np.ndarray:
    ok = ~np.isnan(x)
    n = int(ok.sum())
    if not ok[:n].all():
        raise ValueError(f"{name}: valori non contigui dall'inizio (buchi o righe sparse)")
    return x[:n]


def read_sequence(path: Path) -> dict:
    """Una sequenza -> {emg (T, 8) float64 a 2 kHz in ordine S1..S8, columns, n_raw per frequenza, trimmed}."""
    import pandas as pd
    from scipy.signal import resample_poly

    header = read_header(path)
    cols = emg_columns(header)
    df = pd.read_csv(path, header=None, skiprows=len(header), usecols=[c["col"] for c in cols], dtype=np.float64, engine="c")
    series = {c["sensor"]: _contiguous(df[c["col"]].to_numpy(), f"{path} S{c['sensor']}") for c in cols}
    n_raw = {}
    for fs in RAW_FS:
        lens = {len(series[c["sensor"]]) for c in cols if c["raw_fs_hz"] == fs}
        if len(lens) != 1:
            raise ValueError(f"{path}: canali a {fs:g} Hz di lunghezze diverse {sorted(lens)}")
        n_raw[fs] = lens.pop()
    out = {}
    for c in cols:
        x = series[c["sensor"]]
        out[c["sensor"]] = resample_poly(x, 1, 2) if c["raw_fs_hz"] == 4000.0 else x
    n = min(len(v) for v in out.values())
    trimmed = max(len(v) for v in out.values()) - n
    if trimmed > MAX_LEN_MISMATCH:
        raise ValueError(f"{path}: 2000 Hz {n_raw[2000.0]} campioni contro 4000 Hz {n_raw[4000.0]} (dopo il ricampionamento {trimmed} di differenza)")
    emg = np.stack([out[s][:n] for s in range(1, N_EMG + 1)], axis=1)
    return {"emg": emg, "columns": cols, "n_raw": {f"{k:g}": v for k, v in n_raw.items()}, "trimmed": trimmed}


def scan_raw(raw_root: Path) -> dict[str, dict[str, list[tuple[int, Path]]]]:
    """{soggetto: {modo: [(numero di sequenza, cartella), ...]}} per le cartelle che hanno `sensor_data.csv`."""
    out: dict[str, dict[str, list]] = {}
    for subj in sorted(p for p in Path(raw_root).glob("HG_*") if p.is_dir()):
        for mode in MODES:
            seqs = []
            for d in sorted((subj / mode).glob("sequence_*")):
                m = _SEQ_RE.match(d.name)
                if m and (d / "sensor_data.csv").is_file():
                    seqs.append((int(m.group(1)), d))
            if seqs:
                out.setdefault(subj.name, {})[mode] = seqs
    return out


def load_dominant_hands(participants_csv: Path) -> dict[str, str]:
    """{ID: 'right'|'left'|''} da `participants.csv` (separatore `;`); nessun altro dato anagrafico."""
    with open(participants_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))
    head = [h.strip().lower() for h in rows[0]]
    i_id, i_hand = head.index("id"), head.index("dominant hand")
    return {r[i_id].strip(): r[i_hand].strip().lower() for r in rows[1:] if len(r) > max(i_id, i_hand) and r[i_id].strip()}


def lookup_participant(subject: str, hands: dict[str, str]) -> tuple[str | None, str]:
    """(ID trovato nel CSV, mano dominante). Prima uguale, poi l'unico ID del CSV che e' prefisso della cartella (HG_M7873Q per HG_M7873Q0)."""
    if subject in hands:
        return subject, hands[subject]
    cands = [k for k in hands if subject.startswith(k)]
    return (cands[0], hands[cands[0]]) if len(cands) == 1 else (None, "")


def chirality_from_dominant(hand: str) -> Chirality:
    return {"right": Chirality.LEFT, "left": Chirality.RIGHT}.get(hand, Chirality.UNKNOWN)


def build_montage_metadata(subject: str, mode: str, columns: list[dict], chirality: Chirality) -> MontageMetadata:
    channels = []
    for c in columns:
        if mode == "anatomical":
            if c["muscle"] not in MUSCLE_KEYS:
                raise ValueError(f"muscolo {c['muscle']!r} non previsto (S{c['sensor']})")
            key = MUSCLE_KEYS[c["muscle"]]
            ident = identity(key) if key else identity(SUP_REGION)
        else:
            if c["muscle"] is not None:
                raise ValueError(f"modo random con muscolo {c['muscle']!r} (S{c['sensor']})")
            ident = AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN)
        channels.append(ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=c["sensor"] - 1),
            electrode_type="Delsys_Trigno_Avanti" if c["raw_fs_hz"] == 4000.0 else "Delsys_Trigno_Quattro",
            native_fs_hz=GRID_FS_HZ, effective_band_hz=(0.0, GRID_FS_HZ / 2),
            mains_frequency_hz=50,  # deduzione (Groningen), non dichiarata
            raw_or_envelope=RawOrEnvelope.RAW, anatomical_identity=ident, chirality=chirality, body_region="forearm",
        ))
    group = ChannelGroup(group_id=f"{mode}8", topology=Topology.SPARSE, symmetry="none", channels=channels)
    return MontageMetadata(dataset_name="zhang2026", subject_id=f"zhang2026_{subject}", session_id=mode, groups=[group])


def ingest_session(subject: str, mode: str, seqs: list[tuple[int, Path]], out_root: Path, dominant_hand: str, participant_id: str | None) -> dict:
    """Un soggetto x un modo -> `<out_root>/<soggetto>/<modo>/{data_int16.npy, labels_video_time/, metadata.json}`."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    parts, trials, offset, columns = [], [], 0, None
    for num, d in seqs:
        r = read_sequence(d / "sensor_data.csv")
        sig = [(c["sensor"], c["muscle"], c["raw_fs_hz"], c["device"]) for c in r["columns"]]
        if columns is None:
            columns, first_sig = r["columns"], sig
        elif sig != first_sig:
            raise ValueError(f"{d}: canali diversi dalle altre sequenze dello stesso modo")
        parts.append(r["emg"])
        trials.append({"sequence": num, "offset": offset, "n_samples": int(r["emg"].shape[0]), "n_raw": r["n_raw"], "trimmed": r["trimmed"]})
        offset += r["emg"].shape[0]
    emg = np.concatenate(parts, axis=0)
    valid = qc_channel_validity(emg, min_std=relative_min_std(emg))
    quantized, scale = to_int16(emg)
    out_dir = Path(out_root) / subject / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    lab_dir = out_dir / "labels_video_time"
    lab_dir.mkdir(exist_ok=True)
    labels = []
    for num, d in seqs:
        src = d / "label.csv"
        if src.is_file():
            shutil.copyfile(src, lab_dir / f"sequence_{num:02d}.csv")
            labels.append({"sequence": num, "bytes": src.stat().st_size})
    chir = chirality_from_dominant(dominant_hand)
    montage_dict = montage_to_dict(build_montage_metadata(subject, mode, columns, chir), valid)
    validate_montage_dict(montage_dict)
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape), "native_fs_hz": GRID_FS_HZ, "mode": mode, "trials": trials,
        "channels": [{"index": c["sensor"] - 1, "sensor": f"S{c['sensor']}", "muscle_header": c["muscle"], "raw_fs_hz": c["raw_fs_hz"],
                      "resampled_from_hz": c["raw_fs_hz"] if c["raw_fs_hz"] != GRID_FS_HZ else None, "device": c["device"]} for c in columns],
        "unit": "mV", "laterality": {Chirality.LEFT: "l", Chirality.RIGHT: "r"}.get(chir, ""),
        "laterality_source": "lato opposto alla mano dominante di participants.csv (gesti con la mano non dominante)",
        "participant_id_in_csv": participant_id, "n_channels_discarded_by_qc": int((~valid).sum()),
        "labels_dir": "labels_video_time", "labels_time_base": "video", "labels_aligned_to_emg": False,
        "random_sensor_order_note": ("ordine dei sensori nel modo random ignoto per 32 partecipanti (README); nessun angolo dichiarato"
                                     if mode == "random" else None),
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
    return {"subject": subject, "mode": mode, "n_sequences": len(seqs), "sequences": [n for n, _ in seqs], "n_samples": int(quantized.shape[0]),
            "hours": quantized.shape[0] / GRID_FS_HZ / 3600, "discarded_channels": [int(i) for i in np.flatnonzero(~valid)],
            "trimmed_by_sequence": {t["sequence"]: t["trimmed"] for t in trials if t["trimmed"]}, "laterality": sidecar["laterality"],
            "participant_id_in_csv": participant_id, "labels_copied": len(labels), "empty_label_files": [x["sequence"] for x in labels if x["bytes"] == 0],
            "int16_scale": scale, "emg_max_abs": float(np.abs(emg).max())}
