"""Independent local summary audit; no cache loading, refitting or remote calls."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REL = "results/infra/scaling_300m_20261104/cpu_fit_release_20261105/submissions/collected_jobs/SCALING-pose-ridge-20261105/report.json"
EXPECTED = "8f70ff3b58c8f54413f6c8a89a861a4fdb7e6c9d934a65e73a11d4172a47e4ff"


def audit():
    raw = (ROOT / REL).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    report = json.loads(raw)
    cells = report["cells"]
    assert report["complete"] and report["step"] == 5000
    result = {"source": REL, "source_sha256": EXPECTED,
              "scope": "Published pose report only; no final classification or S8 verdict. Raw predictions not revalidated here.",
              "independent_paired_recomputations": 0, "groups": {}}
    for group, paired in report["paired"].items():
        reference = next(iter(cells.values()))["analysis"]["groups"][group]["centered"]
        users = sorted(reference["per_user"])
        den = np.array([reference["per_user"][u]["sums"]["vary"] for u in users])
        draws = np.random.default_rng(20261102).integers(0, len(users), (2000, len(users)))
        weights = np.stack([np.bincount(row, minlength=len(users)) for row in draws])
        errors, metrics = {}, {}
        for name, cell in cells.items():
            summary = cell["analysis"]["groups"][group]["centered"]
            rows, sums = summary["per_user"], summary["sums"]
            assert sorted(rows) == users
            assert np.array_equal(den, [rows[u]["sums"]["vary"] for u in users])
            errors[name] = np.array([rows[u]["sums"]["centered"] for u in users])
            ratio = float(errors[name].sum() / den.sum())
            assert math.isclose(ratio, paired["cell_ratio"][name], abs_tol=1e-12)
            variance_ratio = sums["varp"] / sums["vary"]
            metrics[name] = {
                "E_centered": ratio, "centered_correlation": sums["cov"] / math.sqrt(sums["varp"] * sums["vary"]),
                "predicted_target_variance_ratio": variance_ratio,
                "predicted_target_std_ratio": math.sqrt(variance_ratio),
                "total_standardized_mse_on_centered_support": sums["total"] / summary["n_valid_elements"],
                "mean_level_standardized_mse": sums["mean_error"] / summary["n_valid_elements"],
                "alpha_train_selected": cell["fit"]["selection"]["alpha"],
                "alpha_boundary_unresolved": cell["fit"]["selection"]["boundary_unresolved"],
                "n_users": summary["n_users"], "n_windows": summary["n_windows"],
            }
        contrasts = {}
        for name, expected in paired["contrasts"].items():
            delta_errors = errors[expected["candidate"]] - errors[expected["control"]]
            point = float(delta_errors.sum() / den.sum())
            ci = np.quantile((weights @ delta_errors) / (weights @ den), [.025, .975])
            relative = float(-delta_errors.sum() / errors[expected["control"]].sum())
            assert math.isclose(point, expected["delta_ratio"], abs_tol=1e-12)
            assert np.allclose(ci, expected["ci95"], rtol=0, atol=1e-12)
            assert math.isclose(relative, expected["relative_error_reduction"], abs_tol=1e-12)
            contrasts[name] = {"delta_E": point, "ci95": ci.tolist(), "relative_error_reduction": relative,
                               "users_improved": int((delta_errors < 0).sum()), "users_total": len(users)}
            result["independent_paired_recomputations"] += 1
        result["groups"][group] = {"metrics": metrics, "contrasts": contrasts}
    assert result["independent_paired_recomputations"] == 9
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
