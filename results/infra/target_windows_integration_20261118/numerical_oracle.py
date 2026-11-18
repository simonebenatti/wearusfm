"""Independent float64 FFT/index oracle for the 125 ms production targets.

Run from any directory with the project's CPU venv. Writes only the requested
JSON evidence file; imports the source tree containing this evidence script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy
from wearusfm.model.anchor_targets import anchor_targets

REVIEWED_COMMIT = "0d91297776a5ee63ccd3f3d10b48cf63163172c9"
RATES = (200, 1000, 1024, 1200, 1926, 2000, 2048, 2148, 4000, 4096, 5120, 10240)
TOLERANCE = 3e-5


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_files = {}
    for name in ("src/wearusfm/model/anchor_targets.py", "src/wearusfm/model/spectral_targets.py"):
        current = (ROOT / name).read_bytes()
        reviewed = subprocess.check_output(["git", "show", f"{REVIEWED_COMMIT}:{name}"], cwd=ROOT)
        assert current == reviewed, f"production source differs from reviewed commit: {name}"
        source_files[name] = {"sha256": sha(current), "identical_to_reviewed_commit": True}

    edges = np.geomspace(20, 450, 7)
    rows = []
    for fs in RATES:
        x = np.random.default_rng(18).normal(size=(1, 2 * fs)).astype(np.float32)
        limit = min(fs / 2, 450)
        got = anchor_targets(x, fs, limit, multiscale=True)
        assert np.isfinite(got.ms_fast).all(), f"nonfinite production target: fs={fs}"

        # Rational half-even support fs/8, not production's floating point round.
        q, r = divmod(fs, 8)
        n = q + (r > 4 or (r == 4 and q % 2 == 1))
        # Center (2*p+1)/80 seconds, subtract half the support, round half up.
        starts = np.array([(fs * (2 * p + 1) - 40 * n + 40) // 80 for p in range(80)])
        valid = (starts >= 0) & (starts + n <= 2 * fs)
        freq = np.fft.rfftfreq(n, 1 / fs)
        bands = [(freq >= lo) & (freq < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        counts = [int(b.sum()) for b in bands]
        available = np.array([hi <= limit and b.any() for hi, b in zip(edges[1:], bands)])
        assert np.array_equal(got.ms_fast_available[0], available), f"availability: fs={fs}"
        assert np.array_equal(got.ms_fast_valid[0], valid), f"validity mask: fs={fs}"
        assert counts[:3] == [2, 3, 4]
        if fs != 200:
            assert counts == [2, 3, 4, 8, 14, 23], (fs, counts)
        else:
            assert available.tolist() == [True, True, True, False, False, False]

        errors = []
        for patch in np.flatnonzero(valid):
            h = np.hanning(n)
            segment = x[:, starts[patch]:starts[patch] + n].astype(np.float64)
            # NumPy float64 FFT, independent of production scipy float32 FFT.
            density = np.abs(np.fft.rfft(segment * h, axis=-1)) ** 2 / (fs * (h * h).sum())
            density[:, 1:(-1 if n % 2 == 0 else None)] *= 2
            expected = np.log(np.stack([density[:, b].sum(-1) * fs / n for b in bands], axis=-1) + 1e-8)
            errors.append(float(np.max(np.abs(expected[0, available] - got.ms_fast[0, patch, available]))))
        error = max(errors)
        assert error < TOLERANCE, (fs, error)
        rows.append({"fs_hz": fs, "support_samples": n, "bins_per_band": counts,
                     "available_bands": available.tolist(), "valid_patches": int(valid.sum()),
                     "mask_equal": True, "all_targets_finite": True,
                     "max_absolute_log_error": error})

    withdrawn_counts = {}
    for fs in (1000, 1926, 2000, 2048):
        n = round(fs / 10)
        freq = np.fft.rfftfreq(n, 1 / fs)
        withdrawn_counts[str(fs)] = [int(((freq >= lo) & (freq < hi)).sum())
                                     for lo, hi in zip(edges[:-1], edges[1:])]
    assert withdrawn_counts["1000"][0] == withdrawn_counts["2000"][0] == 2
    assert withdrawn_counts["1926"][0] == withdrawn_counts["2048"][0] == 1
    report = {"passed": True, "reviewed_commit": REVIEWED_COMMIT,
              "checkout_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": sha(Path(__file__).read_bytes()), "source_files": source_files,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "input": {"seed": 18, "seconds": 2, "channels": 1, "distribution": "standard_normal",
                        "dtype": "float32", "band_limit": "min(fs/2,450)"},
              "oracle": {"support_rounding": "rational_half_even_fs_over_8", "start_rounding": "rational_half_up",
                         "fft": "numpy_float64", "window": "symmetric_hann", "one_sided_density": True,
                         "absolute_log_tolerance": TOLERANCE},
              "rates": rows, "max_absolute_log_error": max(row["max_absolute_log_error"] for row in rows),
              "withdrawn_100ms_bin_counts": withdrawn_counts,
              "scope": "Finite synthetic oracle; not evidence of scientific superiority or exact sampling-rate invariance."}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "rates": len(rows), "max_absolute_log_error": report["max_absolute_log_error"]}))


if __name__ == "__main__":
    main()
