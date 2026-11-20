"""Baseline classiche di [Z21] su DB8 (passo 1 della diagnosi di TEMPONet): WL, cancello, catena su dati finti."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import db8_classical_baselines as cb  # noqa: E402
import temponet_db8_cache  # noqa: E402
from test_temponet_db8_scripts import _raw_root  # noqa: E402


def test_wl_features_match_a_naive_loop():
    rng = np.random.default_rng(0)
    emg = rng.normal(size=(1000, 3))
    ends = np.array([119, 500, 999])
    got = cb.wl_features(emg, ends)
    for i, t in enumerate(ends):
        w = emg[t - 119 : t + 1]
        np.testing.assert_allclose(got[i], np.abs(np.diff(w, axis=0)).sum(axis=0))
    with pytest.raises(ValueError):
        cb.wl_features(emg, np.array([118]))
    with pytest.raises(ValueError):
        cb.wl_features(emg, np.array([1000]))


@pytest.mark.parametrize("fold,full,expected", [
    (7.28, 9.0, "fold coerente"), (7.58, 9.0, "fold coerente"), (6.98, 6.0, "fold coerente"),
    (8.5, 7.4, "indizio H1"), (8.5, 7.6, "indizio H2: stop alla TCN"), (6.5, 7.0, "non conclusivo"), (8.5, 6.5, "non conclusivo"),
])
def test_gate(fold, full, expected):
    assert cb.gate(fold, full) == expected


def test_end_to_end_on_fake_cache(tmp_path):
    cache = tmp_path / "cache"
    assert temponet_db8_cache.main(["--raw-root", str(_raw_root(tmp_path)), "--out", str(cache), "--workers", "1", "--subjects", "1"]) == 0
    out = tmp_path / "run"
    assert cb.main(["--cache", str(cache), "--out", str(out), "--subjects", "1", "--workers", "1"]) == 0
    u = json.loads((out / "units" / "S1.json").read_text())
    for model in ("svr", "mlp"):
        assert set(u[model]) == {"fold1", "fold2", "full"}
        assert np.isfinite(u[model]["full"]["test_ema"]["mae_deg"]) and len(u[model]["fold1"]["alpha"]) == 5
    assert u["svr"]["fold1"]["n_train"] == len(range(119, 3000, 200)) and u["svr"]["full"]["n_train"] == 2 * u["svr"]["fold1"]["n_train"]
    assert u["mlp"]["fold2"]["n_train"] == len(range(119, 3000, 50))
    assert all(c in cb.C_GRID for c in u["svr"]["full"]["C"]) and np.asarray(u["svr"]["fold1"]["val_mae_by_C"]).shape == (4, 5)
    s = json.loads((out / "summary.json").read_text())
    assert s["gate"] == "incompleto" and not s["complete"]
    assert s["models"]["svr"]["fold_mae_deg"] == pytest.approx(np.mean([u["svr"][f]["test_ema"]["mae_deg"] for f in ("fold1", "fold2")]))
    assert "| 1 |" in (out / "summary.md").read_text()
    assert cb.main(["--summarize", str(out)]) == 0
    assert cb.main(["--cache", str(cache), "--out", str(out)]) == 2  # cartella non vuota


def test_deadline_stops_before_fitting(tmp_path):
    d = cb.Deadline(0.0)
    with pytest.raises(TimeoutError):
        d.check("qualcosa")


def _unit(s, svr_fold, svr_full, mlp=7.0):
    def proto(mae, c=1.0):
        return {"test_ema": {"mae_deg": mae, "mae_per_doa_deg": [mae] * 5, "acc_10deg": 0.8, "acc_15deg": 0.9},
                "test_raw": {"mae_deg": mae + 0.5}, "alpha": [0.9] * 5, "C": [c] * 5, "converged": True}
    return {"subject": s, "svr": {"fold1": proto(svr_fold), "fold2": proto(svr_fold), "full": proto(svr_full)},
            "mlp": {"fold1": proto(mlp), "fold2": proto(mlp), "full": proto(mlp)}}


def test_summary_complete_gate_and_groups():
    units = [_unit(s, 8.6 if s < 11 else 9.6, 7.3) for s in range(1, 13)]
    s = cb.summarize(units)
    assert s["complete"] and s["gate"] == "indizio H1"
    svr = s["models"]["svr"]
    assert svr["groups"]["amputati"]["fold_mae_deg"] == pytest.approx(9.6) and svr["groups"]["normodotati"]["fold_mae_deg"] == pytest.approx(8.6)
    assert svr["C_counts_fold"]["1.0"] == 12 * 2 * 5
    assert "indizio H1" in cb.to_markdown(s) and "| 12 (amputato) |" in cb.to_markdown(s)
