"""R1-N2: readout output on two different parity failure modes (read-only use)."""
import json

import pytest

import contrastive_benchmark_readout as B
import contrastive_target_parity as P
from test_contrastive_targets_readout import readout_fixture, write


def _mutate(parity, mode):
    for row in parity["points"][0]["ranks"]:
        if mode == "within_C0b":
            row["cases"]["C0b"]["gradient_sha256"] = "e" * 64
        else:
            row["cases"]["CNa"]["gradient_sha256"] = "e" * 64
            row["cases"]["CNb"]["gradient_sha256"] = "e" * 64


@pytest.mark.parametrize("mode", ["within_C0b", "between_CN"])
def test_readout_failure_text_and_embedded_report(tmp_path, monkeypatch, mode, capsys):
    kwargs, path, parity, _, _ = readout_fixture(tmp_path, monkeypatch)
    _mutate(parity, mode)
    write(path, parity)
    result = B.read_preflight(**kwargs)
    zw = result["zero_weight_parity"]
    pt = zw["assessment"]["points"][0]
    # Per-case/per-rank evidence is embedded verbatim in the readout output.
    embedded = zw["report"]["points"][0]["ranks"]
    breakdown = {a + "/" + b: all(r["cases"][a]["gradient_sha256"] == r["cases"][b]["gradient_sha256"]
                                  for r in embedded) for a, b in P.PAIRS}
    keys = sorted(pt)
    print("MODE", mode, json.dumps({"status": result["status"],
                                    "failure_reasons": result["failure_reasons"],
                                    "point_errors": pt["errors"], "point_keys": keys,
                                    "n_unique": len(pt["unique_gradient_hashes"]),
                                    "embedded_has_norms": all("gradient_norm" in c for r in embedded
                                                              for c in r["cases"].values()),
                                    "manual_breakdown": breakdown}))
    assert result["status"] == "FAILED_REQUIRES_REVIEW"
    assert result["failure_reasons"] == ["production zero-weight gradients/invariants not byte-identical"]
    assert pt["errors"] == ["reduced gradients are not byte-identical within/between arms/ranks"]
    assert keys == ["errors", "inconclusive", "status", "unique_gradient_hashes",
                    "within_and_between_byte_identical"]
