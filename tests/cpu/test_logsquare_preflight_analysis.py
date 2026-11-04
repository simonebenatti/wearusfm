"""Check the preregistered R1 boundary rules independently of the CUDA run."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "logsquare_preflight_analysis", REPO / "scripts/logsquare_preflight_analysis.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def feature(*, p=.1, d=.1, mean=0., std=1., n=17):
    return {"n_tokens": n, "logsquare": {
        "below_floor_fraction_per_kernel": [p] * 64,
        "derivative_rms_per_kernel": [d] * 64,
        "mean_per_kernel": [mean] * 64,
        "std_per_kernel": [std] * 64}}


def pair(masked=None, unmasked=None):
    prefix = "initial/train200/200/"
    return {prefix + "masked_visible": masked or feature(),
            prefix + "unmasked_same_visible": unmasked or feature()}


def test_r1_threshold_equality_is_inclusive_for_all_three_rules():
    # sqrt(.25) is exact: d=1 gives the declared normalized threshold .5.
    rows, _ = A.r1_rows(pair(feature(p=.5, d=1., mean=1.),
                            feature(p=.5, d=1., std=1.)), .25)
    assert len(rows) == 64
    for row in rows:
        assert row["floor_trigger"]
        assert row["derivative_trigger_masked"]
        assert row["derivative_trigger_unmasked"]
        assert row["mask_log_mean_trigger"]
        assert not row["mask_floor_crossing_trigger"]


@pytest.mark.parametrize("masked_p, unmasked_p, crossing", [
    (.5, .499, True), (.499, .5, True), (.5, .5, False), (.499, .499, False)])
def test_mask_floor_crossing_is_symmetric(masked_p, unmasked_p, crossing):
    rows, _ = A.r1_rows(pair(feature(p=masked_p), feature(p=unmasked_p)), .001)
    assert all(row["mask_floor_crossing_trigger"] is crossing for row in rows)
    assert all(row["floor_trigger"] is (unmasked_p >= .5) for row in rows)


@pytest.mark.parametrize("masked_mean, expected", [(0., False), (.125, True), (-.125, True)])
def test_zero_unmasked_std_flags_only_nonzero_mean_shift(masked_mean, expected):
    rows, _ = A.r1_rows(pair(feature(mean=masked_mean), feature(std=0.)), .001)
    assert all(row["mask_log_mean_trigger"] is expected for row in rows)
    assert all(row["mask_mean_std_ratio"] is None for row in rows)
    json.dumps(rows, allow_nan=False)


def test_r1_rejects_different_token_support_counts():
    with pytest.raises(ValueError, match="unpaired feature support"):
        A.r1_rows(pair(feature(n=17), feature(n=18)), .001)


def test_s1_keeps_every_frequency_pair_and_never_exempts_floor_flags():
    features = {}
    for phase in ("initial", "trajectory_preupdate"):
        for mode in ("masked_visible", "unmasked_same_visible"):
            for ds, fs, p in (("low_a", 200, .8), ("low_b", 200, .2),
                              ("high_a", 2000, .1), ("high_b", 2000, .9),
                              ("other", 1000, .1)):
                features[f"{phase}/{ds}/{fs}/{mode}"] = feature(p=p)
    rows, pairs = A.r1_rows(features, .001)
    assert len(pairs) == 16  # 2 phases x 2 modes x 2 low-fs x 2 high-fs.
    identities = {(p["group_200"], p["group_2000"]) for p in pairs}
    assert len(identities) == 16
    for p in pairs:
        expected = list(range(64)) if "/low_a/" in p["group_200"] and "/high_a/" in p["group_2000"] else []
        assert p["compatible_fs_attenuation_kernels"] == expected
        assert p["exempted_kernels"] == []
        assert p["group_200"].split("/")[0] == p["group_2000"].split("/")[0]
        assert p["group_200"].split("/")[-1] == p["group_2000"].split("/")[-1]
    assert all(row["floor_trigger"] for row in rows if row["dataset"] == "low_a")


def test_audit_rejects_modified_original_report_before_reading_artifacts(tmp_path):
    stored = REPO / "results/infra/logsquare_execution_20261104/report.json.gz"
    with gzip.open(stored, "rt") as stream:
        report = json.load(stream)
    report["epsilon"] = .002
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report differs from remote receipt"):
        A.audit(tmp_path, REPO)
