"""Recompute aggregate claims from the delivered diagnostic reports; no model or training imports."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def analyze(repo):
    root = repo / "results/infra/pilot_posa"
    paths = [root / "pose_audit_62473340.json", root / "pose_eval_R3c-e2_62443474.json"]
    audit, e3 = [json.loads(p.read_text()) for p in paths]
    users = list(audit["ridge"].values())
    paired = [u["disallineato"] for u in users]
    hist = audit["sintesi"]["blocchi_validi_per_patch_istogramma_0_5"]
    out = {
        "sources_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "audit": {
            "recordings": audit["n_recordings"], "users": audit["n_users"], "windows": audit["n_windows_total"],
            "verified_checks": sum(v["verificato"] for v in audit["copertura"].values()),
            "failed_checks": sum(v["fallito"] for v in audit["copertura"].values()),
            "unverified_checks": sum(v["non_verificato"] for v in audit["copertura"].values()),
            "users_ratio_below_1": sum(u["ratio_all"] < 1 for u in users),
            "users_ratio_below_09": sum(u["ratio_all"] < .9 for u in users),
            "median_ratio": statistics.median(u["ratio_all"] for u in users),
            "ratio_range": [min(u["ratio_all"] for u in users), max(u["ratio_all"] for u in users)],
            "users_positive_misaligned_gap": sum(p["gap"] > 0 for p in paired),
            "users_negative_misaligned_gap": sum(p["gap"] < 0 for p in paired),
            "median_relative_misaligned_gap_pct": 100 * statistics.median(p["gap"] / p["mae_allineata"] for p in paired),
            "users_misaligned_below_constant": sum(p["mae_disallineata"] < p["mae_const_sui_bersagli_disallineati"] for p in paired),
            "fit_gap_test_windows": [sum(u[k] for u in users) for k in ("n_fit", "n_gap", "n_test")],
            "test_patches": sum(u["n_test_patches"] for u in users),
            "paired_windows": sum(p["n_coppie"] for p in paired),
            "paired_patches": sum(p["n_patch"] for p in paired),
            "patch_support_pct": {"full": 100 * hist[5] / sum(hist), "empty": 100 * hist[0] / sum(hist),
                                  "partial": 100 * sum(hist[1:5]) / sum(hist)},
            "recordings_with_raw_zero_rows": sum(r["posa_ricostruzione"]["frac_zero_rows"] > 0 for r in audit["per_recording"]),
            "max_raw_zero_fraction": max(r["posa_ricostruzione"]["frac_zero_rows"] for r in audit["per_recording"]),
            "max_invalid_fraction": max(r["posa_ricostruzione"]["frac_invalid_rows"] for r in audit["per_recording"]),
        },
        "e3": {}, "probes": [],
    }
    for support in ("dynamic", "all"):
        d = e3["E3"][support]
        out["e3"][support] = {
            "ratio": d["ratio"], "ci95_users": d["ratio_ci95_users"],
            "mae_reduction_pct": 100 * (1 - d["ratio"]),
            "mae_reduction_ci95_pct": [100 * (1 - r) for r in reversed(d["ratio_ci95_users"])],
            "users_ratio_below_1": sum(v["ratio"] < 1 for v in d["per_user"].values()),
            "users_ratio_below_09": sum(v["ratio"] < .9 for v in d["per_user"].values()),
            "n_windows": d["n_windows"], "n_users": d["n_users"],
        }
    for p in sorted((root / "confronti").glob("descrittivo_R3c_vs_R0_*.json")):
        d = json.loads(p.read_text())
        c = d["comparison"]
        out["sources_sha256"][str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
        out["probes"].append({"name": p.stem, "delta_pp": 100 * c["delta"],
                              "ci95_pp": [100 * v for v in c["ci95"]], "beats": c["beats"], "model_seeds": d["seeds"]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = json.dumps(analyze(args.repo), indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
    else:
        print(payload, end="")
