"""Compare final checkpoints/metrics between runs; summarize a parity report."""
import json
import sys
from pathlib import Path

import torch

from wearusfm.training.pairing import state_sha256


def load(run):
    ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    metrics = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return ck, metrics


def strip(rec):
    return {k: v for k, v in rec.items() if not k.startswith("t_")}


def main():
    base = Path(sys.argv[1])
    ref = sys.argv[2]
    others = sys.argv[3:]
    ck_ref, m_ref = load(base / ("run_" + ref))
    for name in others:
        ck, m = load(base / ("run_" + name))
        out = {}
        for key in ("student", "teacher", "optimizer", "contrastive_state", "rank_rng", "step"):
            out[key] = state_sha256(ck_ref[key]) == state_sha256(ck[key])
        out["metrics_equal_without_timing"] = [strip(a) for a in m_ref] == [strip(b) for b in m]
        diffs = []
        for a, b in zip(m_ref, m):
            for k in a:
                if k.startswith("t_"):
                    continue
                if a.get(k) != b.get(k):
                    diffs.append((a["step"], k, a.get(k), b.get(k)))
        out["first_metric_diffs"] = diffs[:6]
        print(ref, "vs", name, json.dumps(out, default=str))
    parity = base / ("run_" + others[-1]) / "zero_weight_parity.json"
    if parity.exists():
        rep = json.loads(parity.read_text())
        print("parity status", rep["status"], json.dumps(rep["gate"], default=str)[:1500])
        for p in rep["points"]:
            hashes = {c["gradient_sha256"] for r in p["ranks"] for c in r["cases"].values()}
            print("update", p["update"], "unique gradient hashes", len(hashes),
                  "live_restored", [r["live_state_restored"] for r in p["ranks"]],
                  "support", {n: (c["nce_eligible"], c["rnc_windows"], c["rnc_pairs"], c["global_windows"], c["nce_weight"], c["rnc_weight"])
                              for n, c in p["ranks"][0]["cases"].items()},
                  "observer_s", [round(r["observer_seconds_including_setup"], 2) for r in p["ranks"]],
                  "backward_s", [round(c["backward_seconds"], 3) for c in p["ranks"][0]["cases"].values()])


if __name__ == "__main__":
    main()
