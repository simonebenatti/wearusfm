"""Ingest di CapgMyo-DBa (passo 2, piano_operativo_v10.md): il piu' piccolo fra i
dataset che entrano nel pretraining (v10 §2.3 esclude solo EPN-612 e UCI-EMG).

18 soggetti (corretto il 23/09/2026, non 23 - v10 §2.1), 8 gesti, 10 trial/gesto,
128 canali @ 1000 Hz, griglia 2D (v10 §3.4). File .mat per trial: chiavi `data`
(1000, 128) float64, `subject`/`gesture`/`trial` scalari.

Nessun README nel dataset scaricato (verificato in sessione, 23/09/2026): la
disposizione fisica 8x16 della griglia viene dal paper originale (Geng et al. 2016),
NON da una pagina ufficiale letta in questa sessione - resta "da verificare"
(docs/fatti_da_verificare.md). Il codice sotto la usa solo per popolare
`sensor_coords`, che non influenza QC ne' la conversione int16.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wearusfm.metadata.schema import (
    AnatomicalIdentity,
    AnatomicalPrecision,
    ChannelGroup,
    ChannelMetadata,
    MontageMetadata,
    RawOrEnvelope,
    SensorCoordinates,
    Topology,
)

N_CHANNELS = 128
GRID_ROWS = 8  # da verificare (fatti_da_verificare.md): non confermato da fonte ufficiale
GRID_COLS = 16  # idem
NATIVE_FS_HZ = 1000.0
WINDOW_SAMPLES = 1000  # un trial = 1s a 1kHz
N_GESTURES = 8
N_TRIALS_PER_GESTURE = 10
N_SUBJECTS = 18

_FILENAME_RE = re.compile(r"^(\d{3})-(\d{3})-(\d{3})\.mat$")


@dataclass(frozen=True)
class TrialFile:
    path: Path
    subject: int
    gesture: int
    trial: int


def scan_capgmyo(root: Path) -> list[TrialFile]:
    """Elenca i file .mat sotto `root`, parsando soggetto/gesto/trial dal nome.
    Non apre i file: solo il nome, per essere veloce su migliaia di file."""
    files = []
    for path in sorted(root.rglob("*.mat")):
        m = _FILENAME_RE.match(path.name)
        if m is None:
            continue
        subject, gesture, trial = (int(g) for g in m.groups())
        files.append(TrialFile(path=path, subject=subject, gesture=gesture, trial=trial))
    return files


def check_completeness(files: list[TrialFile]) -> list[str]:
    """Ritorna una lista di avvisi per combinazioni soggetto/gesto/trial mancanti
    rispetto a N_SUBJECTS x N_GESTURES x N_TRIALS_PER_GESTURE - non solleva
    eccezioni, un dataset reale puo' avere buchi legittimi da dichiarare."""
    seen = {(f.subject, f.gesture, f.trial) for f in files}
    missing = [
        f"{s:03d}-{g:03d}-{t:03d}"
        for s in range(1, N_SUBJECTS + 1)
        for g in range(1, N_GESTURES + 1)
        for t in range(1, N_TRIALS_PER_GESTURE + 1)
        if (s, g, t) not in seen
    ]
    if not missing:
        return []
    return [f"{len(missing)} combinazioni soggetto-gesto-trial mancanti (prime 10): {missing[:10]}"]


def verify_mat_consistency(mat_dict: dict, trial_file: TrialFile) -> None:
    """`mat_dict`: il risultato di scipy.io.loadmat su trial_file.path. Controlla che i
    campi scalari dentro il file combacino col nome file (visto in sessione: combaciano
    sempre nei 3 file campionati, ma un dataset reale puo' avere errori di copia)."""
    for key, expected in (("subject", trial_file.subject), ("gesture", trial_file.gesture),
                           ("trial", trial_file.trial)):
        actual = int(np.asarray(mat_dict[key]).item())
        if actual != expected:
            raise ValueError(
                f"{trial_file.path}: campo {key!r}={actual} non combacia col nome file "
                f"({expected}) - possibile file rinominato o copiato male"
            )


def qc_channel_validity(data: np.ndarray, *, min_std: float = 1e-6, max_abs_frac_clipped: float = 0.01) -> np.ndarray:
    """QC per canale (v10 §4.4): un canale e' valido se non e' piatto (std troppo bassa,
    elettrodo scollegato) e non e' saturato per piu' di `max_abs_frac_clipped` dei
    campioni (soglia al 99.5 percentile assoluto del canale stesso, dato che non
    conosciamo il fondo scala del sensore qui). `data`: (T, C) -> (C,) bool."""
    std = data.std(axis=0)
    not_flat = std >= min_std
    abs_data = np.abs(data)
    per_channel_max = abs_data.max(axis=0, keepdims=True)
    near_max = abs_data >= 0.999 * np.where(per_channel_max > 0, per_channel_max, 1.0)
    clipped_frac = near_max.mean(axis=0)
    not_clipped = clipped_frac <= max_abs_frac_clipped
    return not_flat & not_clipped


def to_int16(data: np.ndarray, *, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Quantizza EMG float64 a int16 (v10 §11, passo 2: "int16 memory-mapped").

    `scale` e' il fattore per cui MOLTIPLICARE i dati prima di arrotondare a intero -
    se non dato, si sceglie il piu' grande che tiene il MASSIMO assoluto dentro il range
    int16: i picchi di contrazione (ampiezza alta) sono spesso il segnale piu'
    informativo in EMG, non vanno tagliati per guadagnare risoluzione sul resto. Ritorna
    (dati_int16, scale): la ricostruzione approssimata e' `dati_int16.astype(float64) /
    scale`.
    """
    if scale is None:
        peak = np.max(np.abs(data))
        scale = (32000.0 / peak) if peak > 0 else 1.0
    quantized = np.clip(np.round(data * scale), -32768, 32767).astype(np.int16)
    return quantized, scale


def build_montage_metadata(subject: int, session_id: str = "s1") -> MontageMetadata:
    """v10 §3.4: griglia 2D, traslazione lungo gli assi, asse fibre privilegiato.
    Nessuna identita' anatomica per canale: HD senza mappatura muscolo-per-muscolo
    (v10 §4.5, "restano a mano poche decine di canali" - CapgMyo non e' fra questi)."""
    channels = []
    for idx in range(N_CHANNELS):
        row, col = divmod(idx, GRID_COLS)
        channels.append(ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=idx, grid_row=row, grid_col=col),
            electrode_type="dry_electrode_array",
            native_fs_hz=NATIVE_FS_HZ,
            effective_band_hz=(0.0, NATIVE_FS_HZ / 2),  # da verificare: banda reale non documentata
            mains_frequency_hz=50,  # da verificare: paese/luogo di acquisizione non documentato qui
            raw_or_envelope=RawOrEnvelope.RAW,
            anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
            body_region="forearm",
        ))
    group = ChannelGroup(
        group_id="grid", topology=Topology.GRID_2D, symmetry="translational_2d",
        channels=channels,
    )
    return MontageMetadata(
        dataset_name="capgmyo_dba", subject_id=f"capgmyo_s{subject:02d}",
        session_id=session_id, groups=[group],
    )
