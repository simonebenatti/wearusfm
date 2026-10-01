"""scripts/inspect_mat.py su un .mat sintetico: variabili, scalari, etichette, quota di potenza sopra una frequenza (sovracampionamento)."""

import importlib.util
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal

_spec = importlib.util.spec_from_file_location("inspect_mat", Path(__file__).resolve().parents[2] / "scripts" / "inspect_mat.py")
IM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IM)


def _file(tmp_path):
    rng = np.random.default_rng(0)
    sos = signal.butter(10, 500, fs=2000, output="sos")
    low = signal.sosfiltfilt(sos, rng.normal(size=(20000, 2)), axis=0)  # come un segnale a 1111 Hz portato a 2 kHz
    white = rng.normal(size=(20000, 1))
    emg = np.concatenate([low, white, np.zeros((20000, 1))], axis=1).astype(np.float32)
    p = tmp_path / "S1_E1_A1.mat"
    sio.savemat(p, {"emg": emg, "stimulus": np.repeat(np.arange(4), 5000)[:, None], "subject": 1, "exercise": np.array([[2]])})
    return p


def test_inspect_reports_variables_scalars_labels_and_band(tmp_path):
    r = IM.inspect(_file(tmp_path), "emg", 2000.0, 555.0, 10.0)
    names = {v["name"]: v for v in r["variables"]}
    assert names["emg"]["shape"] == [20000, 4] and r["format"] == "classico"
    assert r["scalars"] == {"subject": [1], "exercise": [2]}
    assert r["labels"]["stimulus"] == {"length": 20000, "n_distinct": 4, "min": 0, "max": 3}
    assert r["emg"]["n_constant_columns"] == 1 and r["emg"]["duration_s_at_fs"] == 10.0
    frac = r["emg"]["power_fraction_above_555_hz_first_10_s"]
    assert frac[0] < 1e-3 and frac[1] < 1e-3  # filtrato: quasi nulla sopra 555 Hz
    assert 0.35 < frac[2] < 0.55  # bianco: (1000 - 555) / 1000 della potenza
    assert np.isnan(frac[3])  # colonna a zero
