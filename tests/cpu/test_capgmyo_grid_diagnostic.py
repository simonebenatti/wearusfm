"""Diagnostic invariants; synthetic data only, no external files or downloads."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.ndimage import gaussian_filter

PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_capgmyo_grid.py"
spec = importlib.util.spec_from_file_location("capgmyo_grid_diagnostic", PATH)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def test_maps_are_bijections_and_have_expected_equivalences():
    maps = D.candidate_maps()
    for grid in maps.values():
        assert sorted(grid.ravel()) == list(range(128))
    assert maps["row16x8_C"][1, 0] == 8
    assert maps["column16x8_F"][1, 0] == 1
    assert D.graph_key(maps["row16x8_C"]) == D.graph_key(maps["row16x8_C_mirrored"])
    assert D.graph_key(maps["row16x8_C"]) == D.graph_key(maps["modules8x2_F_transposed"])
    assert D.graph_key(maps["legacy8x16_C"]) == D.graph_key(maps["column16x8_F"])
    assert D.graph_key(maps["row16x8_C"]) != D.graph_key(maps["legacy8x16_C"])


def test_rectangle_edge_counts_and_no_wrap():
    edges = D.edge_axes(np.arange(128).reshape(16, 8))
    assert len(edges["row_axis"]) == 120
    assert len(edges["column_axis"]) == 112
    assert (7, 8) not in map(tuple, edges["column_axis"])


def test_correlation_matches_numpy_and_masks_flat_channels():
    x = np.random.default_rng(2).normal(size=(100, 128))
    x[:, 0] = 0
    valid = np.ones(128, dtype=bool)
    valid[0] = False
    r = D.correlation(x, valid)
    assert np.isnan(r[0]).all()
    np.testing.assert_allclose(r[1:, 1:], np.corrcoef(x[:, 1:].T), atol=1e-12)


def test_lag_sign_and_scale_invariance():
    rng = np.random.default_rng(7)
    a = rng.normal(size=1000)
    b = np.r_[rng.normal(size=2), a[:-2]]
    x = np.column_stack([a, b])
    peak, lag = D.lagged_correlations(x, np.array([[0, 1]]), 3)
    assert peak[0] == pytest.approx(1)
    assert lag[0] == 2
    p2, l2 = D.lagged_correlations(x * [100, 0.1], np.array([[0, 1]]), 3)
    np.testing.assert_allclose(p2, peak)
    np.testing.assert_allclose(l2, lag)


def test_synthetic_smooth_grid_prefers_true_adjacencies_and_preserves_reflection():
    rng = np.random.default_rng(12)
    grid = gaussian_filter(rng.normal(size=(1000, 16, 8)), sigma=(0, 1.2, 1.2))
    result, _ = D.analyze_trial(grid.reshape(1000, 128), D.candidate_maps())
    scores = result["candidates"]
    assert scores["row16x8_C"]["all"]["raw_abs"] > scores["legacy8x16_C"]["all"]["raw_abs"]
    for key in ("raw_abs", "cmr_abs", "log_rms_tv", "lag_peak_abs"):
        assert scores["row16x8_C"]["all"][key] == pytest.approx(scores["row16x8_C_mirrored"]["all"][key])


@pytest.mark.parametrize("bad", [np.zeros((1000, 128)), np.full((1000, 128), np.nan), np.zeros((128, 1000))])
def test_invalid_signal_refused(bad):
    with pytest.raises(ValueError):
        D.analyze_trial(bad, D.candidate_maps())


def test_flat_channels_do_not_enter_edge_metrics():
    x = np.random.default_rng(3).normal(size=(1000, 128))
    x[:, 0] = 1.0
    result, _ = D.analyze_trial(x, D.candidate_maps())
    assert result["invalid_channels"] == [0]
    assert result["candidates"]["row16x8_C"]["all"]["edges"] == 230


def test_mat_metadata_consistency_and_original_column_order(tmp_path):
    p = tmp_path / "001-002-003.mat"
    x = np.tile(np.arange(128), (1000, 1))
    savemat(p, {"data": x, "subject": 1, "gesture": 2, "trial": 3})
    data, meta = D.load_trial(p)
    np.testing.assert_array_equal(data, x)
    assert meta["subject"] == 1 and len(meta["sha256"]) == 64
    savemat(p, {"data": x, "subject": 1, "gesture": 4, "trial": 3})
    with pytest.raises(ValueError, match="gesture"):
        D.load_trial(p)


def test_cli_writes_finite_json_without_touching_input(tmp_path):
    import hashlib
    import json
    p = tmp_path / "001-001-001.mat"
    savemat(p, {"data": np.random.default_rng(4).normal(size=(1000, 128)), "subject": 1, "gesture": 1, "trial": 1})
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    out = tmp_path / "output"
    assert D.main(["--raw-root", str(tmp_path), "--out-dir", str(out), "--subjects", "1", "--gestures", "1", "--trials", "1", "--no-plots"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["n_files"] == 1
    assert report["source_audit"]["status"].startswith("not supplied")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before
    with pytest.raises(SystemExit):
        D.main(["--raw-root", str(tmp_path), "--out-dir", str(out)])


def test_cli_refuses_output_inside_repo(tmp_path):
    with pytest.raises(SystemExit):
        D.main(["--raw-root", str(tmp_path), "--out-dir", str(D.REPO / "results" / "raw_maps")])


def test_plot_selection_fails_before_creating_partial_output(tmp_path):
    out = tmp_path / "output"
    with pytest.raises(SystemExit):
        D.main(["--raw-root", str(tmp_path), "--out-dir", str(out), "--gestures", "2"])
    assert not out.exists()
