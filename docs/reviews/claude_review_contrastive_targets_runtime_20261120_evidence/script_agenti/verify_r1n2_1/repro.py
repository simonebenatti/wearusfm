"""R1-N2 reproduction: distinct parity failure modes give the same gate output.

Read-only use of the reviewed module; synthetic evidence built like the
repository tests (tests/cpu/test_contrastive_target_parity.py case()/point()).
"""
import copy
import json

import contrastive_target_parity as P
from contrastive_numeric_profile import numeric_profile

with numeric_profile(require_cuda=False) as flags:
    FLAGS = {**flags, "autocast_cuda_enabled": True, "autocast_cuda_dtype": "torch.bfloat16"}

H = "c" * 64


def case(grad=H, norm=1.0):
    return {"before": {key: "a" * 64 for key in P.MATCH_FIELDS},
            "after_rng": {key: "b" * 64 for key in ("cpu_rng_sha256", "cuda_rng_sha256",
                                                     "python_rng_sha256", "numpy_rng_sha256")},
            "gradient_sha256": grad, "gradient_norm": norm, "loss": 1.,
            "nce_weight": 0., "rnc_weight": 0., "nce_eligible": 4., "rnc_windows": 3., "rnc_pairs": 20.,
            "global_windows": 32., "backward_seconds": .1, "flags": copy.deepcopy(FLAGS),
            "memory": {"cuda_measured": True, "peak_allocated_bytes": 100}}


def point(update=128):
    return {"update": update, "state_completed_steps": update - 1, "ranks": [
        {"rank": rank, "profile_status": "AVAILABLE", "global_batch_size": 32,
         "global_ids_sha256": "d" * 64, "live_before": case()["before"], "live_state_restored": True,
         "cases": {name: case() for name in P.CASES}}
        for rank in range(4)]}


def scenario(name):
    p = point()
    if name == "baseline":
        pass
    elif name == "within_only_C0b_repeatability":
        # C0 not repeatable: C0b differs from C0a; CN/CR identical to C0a.
        for row in p["ranks"]:
            row["cases"]["C0b"]["gradient_sha256"] = "e" * 64
    elif name == "between_only_CN_neutrality":
        # CN deterministic but different from C0 (repeatable, not neutral).
        for row in p["ranks"]:
            for c in ("CNa", "CNb"):
                row["cases"][c]["gradient_sha256"] = "e" * 64
    elif name == "ranks_only_rank2":
        # All cases on rank 2 differ: reduced gradients differ across ranks.
        for c in P.CASES:
            p["ranks"][2]["cases"][c]["gradient_sha256"] = "e" * 64
    elif name == "within_and_between_mixed":
        for row in p["ranks"]:
            row["cases"]["CRb"]["gradient_sha256"] = "e" * 64
            row["cases"]["CNa"]["gradient_sha256"] = "f" * 64
    return p


def pair_breakdown(p):
    """What a reader must reconstruct by hand from the embedded report."""
    out = {}
    for a, b in P.PAIRS:
        out[a + "/" + b] = all(r["cases"][a]["gradient_sha256"] == r["cases"][b]["gradient_sha256"]
                               for r in p["ranks"])
    out["ranks_agree_per_case"] = {c: len({r["cases"][c]["gradient_sha256"] for r in p["ranks"]}) == 1
                                   for c in P.CASES}
    return out


results = {}
for name in ("baseline", "within_only_C0b_repeatability", "between_only_CN_neutrality",
             "ranks_only_rank2", "within_and_between_mixed"):
    p = scenario(name)
    g = P.assess_point(p)
    results[name] = {"gate": {k: g[k] for k in ("status", "errors", "inconclusive",
                                                "within_and_between_byte_identical")},
                     "n_unique": len(g["unique_gradient_hashes"]),
                     "manual_pair_breakdown": pair_breakdown(p)}

print(json.dumps(results, indent=1))
failed = [r["gate"] for k, r in results.items() if k != "baseline"]
print("IDENTICAL_GATE_OUTPUT_ACROSS_FAILURE_MODES:",
      all(f == failed[0] for f in failed))
print("PAIRS_USED_IN_ASSESS_POINT:", "PAIRS" in P.assess_point.__code__.co_names)
print("NAMES_USED_IN_ASSESS_POINT:", sorted(P.assess_point.__code__.co_names))
