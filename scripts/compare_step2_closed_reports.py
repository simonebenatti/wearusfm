#!/usr/bin/env python3
"""Replay the pre-specified 2C secondary endpoint RPW-MS, without new probes.

Uses closed, hash-verified reports, not the NPZ caches. A/B in compare_reports
mean reference/variant here; all source teachers belong to Step 1 family B.
Never rewrites either campaign's comparison.json or changes primary verdicts.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearusfm.harness.downstream_ab import compare_reports

CAMPAIGNS = {
    "step2a": ("step2a_stats_v1", "3dba14c1176f9ac525aee0e851590ca7db4b4c39", "MS", "model+stats"),
    "step2c": ("step2c_frozen_branches_v1", "1c9485260aaf32d5936346b50225f5cff1d1b8ea", "RPW", "model"),
}
THRESHOLDS = {"useful_delta": .01, "harm_delta": -.01}
DATASETS = {"p1": {"ninapro_db2", "ninapro_db3", "ninapro_db6"}, "p2": {"epn612", "uci_emg"}}
CHECKPOINTS = {
    "B_seed0": "b264df4b75f7047980212ab068fe524148e71dbdef442d4e75e95648f841d3cd",
    "B_seed1": "27b3a819c35f798992b2bcddf7f48998be03d9dc4bc113b79e5170c199577a29",
}


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def pairing_inventory(report):
    """Physical feature hashes/dimensions differ; actual inputs and rows must not."""
    cache = report["provenance"]["cache"]
    return {dataset: {key: row[key] for key in ("input_sha256", "rows")}
            for dataset, row in cache.items()}


def decision(result):
    tasks = {}
    for task, row in result["tasks"].items():
        tasks[task] = {
            "useful": row["delta"] >= THRESHOLDS["useful_delta"] and row["ci95"][0] > 0
                      and all(d > 0 for d in row["delta_by_model_seed"]),
            "not_harmful": row["delta"] >= THRESHOLDS["harm_delta"],
        }
    return {"tasks": tasks, "positive": any(r["useful"] for r in tasks.values())
            and all(r["not_harmful"] for r in tasks.values())}


def replay(roots, n_boot=1000, seed=20261009):
    if set(roots) != set(CAMPAIGNS):
        raise ValueError("both closed campaigns are required")
    reports = {"p1": {}, "p2": {}}
    hashes, manifest_hashes, pairing = {}, {}, {}
    for campaign, root in roots.items():
        protocol, source, variant, features = CAMPAIGNS[campaign]
        manifest, manifest_hashes[campaign] = read_json(root / "comparison.json")
        if manifest.get("protocol") != protocol or manifest.get("source_commit") != source:
            raise ValueError(f"{campaign}: changed campaign identity")
        # Both closed campaigns recorded exactly the same signed thresholds.
        if manifest.get("thresholds", manifest.get("decision", {}).get("thresholds")) != THRESHOLDS:
            raise ValueError(f"{campaign}: changed thresholds")
        if any(manifest.get("checkpoint_sha256", {}).get(k) != v for k, v in CHECKPOINTS.items()):
            raise ValueError(f"{campaign}: changed source teachers")
        for task in reports:
            for model_seed in (0, 1):
                arm = f"B_seed{model_seed}"
                key = f"{task}/{arm}/{variant}"
                report, digest = read_json(root / task / arm / f"{variant}.json")
                if manifest["report_sha256"].get(key) != digest:
                    raise ValueError(f"{campaign}/{key}: report hash changed")
                identity = report["provenance"]["identity"]
                expected = {"protocol": protocol, "source_commit": source, "task": task,
                            "arm": arm, "variant": variant, "which": "teacher", "probe_seed": 0}
                if any(identity.get(k) != v for k, v in expected.items()) or report.get("feature_set") != features:
                    raise ValueError(f"{campaign}/{key}: report identity changed")
                source_identity = (report["provenance"]["downstream"]["identity"]
                                   if campaign == "step2a" else identity)
                checkpoint_key = "checkpoint_sha256" if campaign == "step2a" else "source_checkpoint_sha256"
                if (source_identity.get(checkpoint_key) != CHECKPOINTS[arm] or source_identity.get("arm") != arm
                        or source_identity.get("which") != "teacher"):
                    raise ValueError(f"{campaign}/{key}: different source teacher")
                inventory = pairing_inventory(report)
                if set(inventory) != DATASETS[task] or set(report["datasets"]) != DATASETS[task]:
                    raise ValueError(f"{campaign}/{key}: incomplete dataset inventory")
                signature = (inventory, identity["split_sha256"])
                if task in pairing and pairing[task] != signature:
                    raise ValueError(f"{campaign}/{key}: unpaired inputs, rows or split")
                pairing[task] = signature
                hashes[f"{campaign}/{key}"] = digest
                label = "A" if campaign == "step2a" else "B"
                reports[task][f"{label}_seed{model_seed}"] = report
    result = compare_reports(reports, n_boot=n_boot, seed=seed)
    result.pop("both_practical_guards_pass")
    result.pop("guard_note")
    for row in result["tasks"].values():
        row.pop("practical_guard_pass")
    return {
        "analysis": "step2c_secondary_RPW_vs_fixed_stats_B_v1",
        "endpoint": "RPW - MS; two paired Step 1 family-B teachers",
        "thresholds": THRESHOLDS, "decision": decision(result), "paired": result,
        "source_campaigns": {k: {"protocol": v[0], "source_commit": v[1]} for k, v in CAMPAIGNS.items()},
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_sha256": manifest_hashes, "report_sha256": hashes,
        "scope": "Closed report replay: input/row/split identities and per-subject counts checked; NPZ caches not reopened.",
        "note": "Secondary endpoint pre-specified by 2C; no primary verdict changed, no automatic adoption. "
                "Two teachers only; P2 interval descriptive; multiple comparisons without correction.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2a-root", type=Path, required=True)
    parser.add_argument("--step2c-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay({"step2a": args.step2a_root, "step2c": args.step2c_root})
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"decision": result["decision"], "tasks": {
        t: {k: row[k] for k in ("delta", "ci95", "delta_by_model_seed")}
        for t, row in result["paired"]["tasks"].items()}}, allow_nan=False))


if __name__ == "__main__":
    main()
