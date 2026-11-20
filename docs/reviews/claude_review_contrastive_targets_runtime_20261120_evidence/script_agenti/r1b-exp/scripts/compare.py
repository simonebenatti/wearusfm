"""Confronto fra run (checkpoint e metriche) e lettura del rapporto di parita'."""
import hashlib
import json
import sys
from pathlib import Path

import torch

from wearusfm.training.pairing import state_sha256
import contrastive_target_parity as P
P.UPDATES = (2, 4)  # come nell esperimento


def load(run):
    ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    metrics = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return ck, metrics


def strip(rec):
    return {k: v for k, v in rec.items() if not k.startswith("t_") and k not in ("elapsed_s",)}


def main():
    base, ref, others = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    ck_ref, m_ref = load(base / ("run_" + ref))
    for name in others:
        ck, m = load(base / ("run_" + name))
        out = {key: state_sha256(ck_ref[key]) == state_sha256(ck[key])
               for key in ("student", "teacher", "optimizer", "contrastive_state", "rank_rng", "step")}
        out["n_metrics"] = (len(m_ref), len(m))
        out["metrics_equal_without_timing"] = [strip(a) for a in m_ref] == [strip(b) for b in m]
        out["grad_norm"] = [(a.get("grad_norm"), b.get("grad_norm")) for a, b in zip(m_ref, m)]
        print(ref, "vs", name, json.dumps(out, default=str))
    for name in others:
        run = base / ("run_" + name)
        parity = run / "zero_weight_parity.json"
        if not parity.exists():
            continue
        rep = json.loads(parity.read_text())
        cfg = json.loads((run / "config.json").read_text())
        canonical = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        summary = json.loads((run / "summary.json").read_text())
        print(name, "binding config_sha256 == canonical(config.json):", rep["config_sha256"] == canonical,
              "| summary_sha256 == sha(summary.json):",
              rep["summary_sha256"] == hashlib.sha256((run / "summary.json").read_bytes()).hexdigest(),
              "| initial_state_sha256 == summary:", rep["initial_state_sha256"] == summary["contrastive"]["initial_sha256"],
              "| provenance:", rep["provenance"])
        gate = P.validate_parity_report(rep)
        print(name, "status serializzato", rep["status"], "| ricalcolato", gate["status"],
              "| inconclusive", sorted({i for p in gate["points"] for i in p["inconclusive"]})[:4],
              "| errors", sorted({e for p in gate["points"] for e in p["errors"]}))
        for p in rep["points"]:
            hashes = {c["gradient_sha256"] for r in p["ranks"] for c in r["cases"].values()}
            befores = {json.dumps(c["before"], sort_keys=True) for r in p["ranks"] for c in r["cases"].values()}
            print(" update", p["update"], "state_completed_steps", p["state_completed_steps"],
                  "| hash unici", len(hashes), "| live_restored", [r["live_state_restored"] for r in p["ranks"]],
                  "| before distinti fra casi e rank", len(befores),
                  "| supporto rank0", {n: (c["nce_eligible"], c["rnc_windows"], c["rnc_pairs"], c["global_windows"],
                                          c["nce_weight"], c["rnc_weight"]) for n, c in p["ranks"][0]["cases"].items()},
                  "| observer_s", [round(r["observer_seconds_including_setup"], 2) for r in p["ranks"]])
            print("   cpu_rng diversi fra rank:", len({r["live_before"]["cpu_rng_sha256"] for r in p["ranks"]}),
                  "| student uguale fra rank:", len({r["live_before"]["student_sha256"] for r in p["ranks"]}),
                  "| batch diversi fra rank:", len({r["live_before"]["batch_sha256"] for r in p["ranks"]}))
        exp = run / "exp.rank0.json"
        if exp.exists():
            data = json.loads(exp.read_text())
            if "live_grads" in data:
                for lg in data["live_grads"]:
                    probe = [p for p in rep["points"] if p["update"] == lg["update"]][0]
                    h = probe["ranks"][0]["cases"]["C0a"]["gradient_sha256"]
                    print("   live gradient after live backward, update", lg["update"], "== probe C0a:", lg["live_gradient_sha256"] == h)


if __name__ == "__main__":
    main()
