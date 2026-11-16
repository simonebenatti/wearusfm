"""Review Claude, dimensione `calibrazione`: casi aggiuntivi sul lettore della calibrazione (regola della mediana, limiti dell'intervallo,
soglie del cancello di copertura sul pannello). Stessa fixture dei test consegnati."""
import pytest
from test_contrastive_prepare import P, fixture_run, write_run


def _report(tmp_path, mutate):
    folder = tmp_path / "run"
    cfg, summary, records = fixture_run(folder)
    mutate(summary, records)
    write_run(folder, cfg, summary, records)
    return P.calibration_report(folder)


def test_lambda_is_the_median_not_mean_or_extreme(tmp_path):
    def m(summary, records):  # 30 batch a rapporto 2, il primo a rapporto 1000 e uno a 0,02
        records[256]["contrast_global_grad_nce"] = .001  # il PRIMO scelto: la regola non e' «il primo batch»
        records[261]["contrast_global_grad_nce"] = 50.
    r = _report(tmp_path, m)
    assert r["status"] == "PASSED_FOR_REVIEW" and r["weights"]["nce"] == pytest.approx(.2) and r["weights"]["rnc"] == pytest.approx(.4)


@pytest.mark.parametrize("nce,ok", [(1000., True), (1001., False), (.1, True), (.0999, False)])
def test_lambda_range_is_inclusive_on_both_sides(tmp_path, nce, ok):
    def m(summary, records):
        for r in records[256:]:
            r["contrast_global_grad_nce"] = nce
    r = _report(tmp_path, m)
    assert (r["status"] == "PASSED_FOR_REVIEW") is ok
    assert ("weight_outside_frozen_range_nce" in r["failure_reasons"]) is (not ok)


@pytest.mark.parametrize("queries,eligible,ok", [(32, 32, True), (31, 31, False), (40, 32, True), (40, 31, False)])
def test_per_dataset_gate_32_queries_and_80_percent(tmp_path, queries, eligible, ok):
    def m(summary, records):
        b = summary["contrastive"]["coverage_partial_block"]
        b["datasets"] = {"A": {"queries": 8192 - queries, "eligible": 8192 - queries, "fraction": 1.},
                         "B": {"queries": queries, "eligible": eligible, "fraction": eligible / queries}}
        b["nce_fraction"] = (8192 - queries + eligible) / 8192
        b["passed"] = ok
    r = _report(tmp_path, m)
    assert (r["status"] == "PASSED_FOR_REVIEW") is ok and r["coverage"]["datasets"]["B"]["passed"] is ok


@pytest.mark.parametrize("eligible_a,ok", [(3480, True), (3400, False)])  # con B a 3900: 7380/8192 = 0,9009; 7300/8192 = 0,8911
def test_overall_gate_90_percent_with_every_dataset_above_80(tmp_path, eligible_a, ok):
    def m(summary, records):
        b = summary["contrastive"]["coverage_partial_block"]
        b["datasets"] = {"A": {"queries": 4096, "eligible": eligible_a, "fraction": eligible_a / 4096},
                         "B": {"queries": 4096, "eligible": 3900, "fraction": 3900 / 4096}}
        b["nce_fraction"] = (eligible_a + 3900) / 8192
        b["passed"] = ok
    r = _report(tmp_path, m)
    assert (r["status"] == "PASSED_FOR_REVIEW") is ok and all(d["passed"] for d in r["coverage"]["datasets"].values())


@pytest.mark.parametrize("active,ok", [(231, True), (230, False)])  # 231/256 = 0,9023; 230/256 = 0,8984
def test_rnc_gate_90_percent_of_batches(tmp_path, active, ok):
    def m(summary, records):
        b = summary["contrastive"]["coverage_partial_block"]
        b["rnc_active_fraction"] = active / 256
        b["passed"] = ok
    r = _report(tmp_path, m)
    assert (r["status"] == "PASSED_FOR_REVIEW") is ok


def test_declared_coverage_verdict_must_match_the_recomputed_one(tmp_path):
    def m(summary, records):  # 19 query in un dataset: il cancello ricalcolato fallisce, il summary dichiara «superato»
        b = summary["contrastive"]["coverage_partial_block"]
        b["datasets"] = {"A": {"queries": 8173, "eligible": 8173, "fraction": 1.}, "B": {"queries": 19, "eligible": 19, "fraction": 1.}}
        b["passed"] = True
    with pytest.raises(ValueError, match="coverage verdict"):
        _report(tmp_path, m)
