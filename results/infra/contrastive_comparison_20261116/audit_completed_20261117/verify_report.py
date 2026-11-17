"""Recheck archived audit content; does NOT waive its SLURM TIMEOUT."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from contrastive_prepare import _coverage, source_digest

EXPECTED_REPORT_SHA256 = "5c720fadbcb12fd5458bc0e19cd03f28fec30ad5973ca72739f01bf1b3d7547b"


def main():
    archive = HERE / "audit.json.gz"
    raw = gzip.decompress(archive.read_bytes()) if archive.exists() else (HERE / "audit.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_REPORT_SHA256
    report = json.loads(raw)
    base = HERE.parent
    job = json.loads((base / "releases/audit.job.json").read_text())
    config = json.loads((ROOT / job["config"]["path"]).read_text())
    assert report["source_sha256"] == job["source_sha256"] == source_digest()
    assert report["inputs"] == {k: v["sha256"] for k, v in job["inputs"].items()}
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    assert report["config_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert (report["steps"], report["world_size"], report["workers_per_rank"]) == (1024, 4, 6)
    records = report["records"]
    assert len(records) == 1024
    assert [r["step"] for r in records] == list(range(1024))
    assert all(r["queries"] == 32 for r in records)
    datasets = report["expected_datasets"]
    assert set(datasets) == set(config["contrastive"]["coverage_quotas"])
    assert report["coverage_completed_blocks"] == []
    recounted = {}
    for key, stop in (("calibration_panel", 512), ("coverage_partial_block", 1024)):
        panel = records[256:stop]
        counts = {d: [0, 0] for d in datasets}
        for r in panel:
            assert sum(v[0] for v in r["datasets"].values()) == r["queries"]
            assert sum(v[1] for v in r["datasets"].values()) == r["eligible_nce"]
            for d, pair in r["datasets"].items():
                assert all(type(v) is int for v in pair) and 0 <= pair[1] <= pair[0]
                counts[d] = [a + b for a, b in zip(counts[d], pair)]
        n = sum(v[0] for v in counts.values())
        eligible = sum(v[1] for v in counts.values())
        active = sum(r["rnc_windows"] > 0 for r in panel)
        stored = report[key]
        assert stored["updates"] == len(panel) and stored["through_step"] == stop - 1
        assert stored["nce_fraction"] == eligible / n
        assert stored["rnc_active_fraction"] == active / len(panel)
        cfg = config["contrastive"]
        assert stored["exempt_datasets"] == cfg["coverage_exempt_datasets"] == ["capgmyo"]
        passed = eligible / n >= cfg["coverage_nce_fraction"] and active / len(panel) >= cfg["coverage_rnc_fraction"]
        for d, (total, good) in counts.items():
            exempt = d in cfg["coverage_exempt_datasets"]
            assert stored["datasets"][d] == {"queries": total, "eligible": good, "fraction": good / max(total, 1), "exempt": exempt}
            passed &= exempt or (total >= cfg["coverage_min_queries"] and good / total >= cfg["coverage_dataset_fraction"])
        assert stored["passed"] == passed is True
        recounted[key] = {"queries": n, "eligible": eligible, "nce_fraction": eligible / n,
                          "rnc_active_batches": active, "batches": len(panel), "datasets": counts, "passed": passed}
    runtime_panel = _coverage({"config": config["contrastive"], "expected_datasets": datasets,
                              "coverage_completed_blocks": [], "coverage_partial_block": report["calibration_panel"]})
    assert runtime_panel["passed"] and report["passed"] is True
    assert report["descriptor_budget"]["estimated_open_descriptor_bound"] <= report["descriptor_budget"]["soft_limit"]
    remote = (HERE / "frozen_inputs_rechecked.txt").read_text().splitlines()
    assert remote[0] == "f09bf857eee24f02aff25fa743853d60fcab7f03"
    assert len(remote) == 5  # SHA, no dirty files, four input SHA lines.
    actual = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in remote[1:]}
    assert actual == {v["path"]: v["sha256"] for v in job["inputs"].values()}
    result = {"content_gate": "PASSED", "execution_gate": "FAILED_TIME_LIMIT",
              "calibration_released": False, "report_sha256": EXPECTED_REPORT_SHA256,
              "report_bytes": len(raw), "source_sha256": report["source_sha256"],
              "steps": len(records), "total_windows": sum(r["queries"] for r in records),
              "recomputed": recounted, "runtime_panel_gate": runtime_panel["passed"],
              "elapsed_seconds": 3787, "allocated_core_hours": 3787 * 8 / 3600,
              "audit_core_hours_cap": 8, "allocated_gpu_hours": 0,
              "max_rss_gib": 22555604 / 1024**2,
              "cpu_seconds": 35 * 60 + 38.449,
              "scope": "Recounts reported candidate/support records; does not rerun raw signal processing or waive the timeout."}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "recomputed"}, indent=2))


if __name__ == "__main__":
    main()
