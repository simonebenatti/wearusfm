"""Skeptic probe R6-4: does read_preflight/main turn a missing trace/start.json into a receipt?

Uses the repository readout fixture (all heavy readers stubbed) and replaces the
compare_three_paths stub with one that calls the REAL readers used at
contrastive_target_resume.py:324 (read start.json) and :359 (rows trace-rank.jsonl)
on directories that lack those files. Then contrasts with the two designed
INCONCLUSIVE paths (missing parity report, missing numeric receipt).
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

import contrastive_benchmark_readout as B
import contrastive_target_resume as C
from test_contrastive_targets_readout import readout_fixture


def fixture(base):
    mp = pytest.MonkeyPatch()
    tmp = Path(tempfile.mkdtemp(dir=base))
    return mp, tmp, readout_fixture(tmp, mp)


def run_main(kwargs, out):
    argv = ["readout", "--cohort", str(kwargs["cohort_path"]), "--cuda-report", str(kwargs["cuda_path"]),
            "--cpu-audit", str(kwargs["cpu_audit_path"]), "--cpu-audit-sacct", str(kwargs["cpu_audit_sacct_path"]),
            "--calibration-report", str(kwargs["calibration_path"]),
            "--calibration-run-dir", str(kwargs["calibration_run_dir"]), "--out", str(out)]
    for arm, path in kwargs["full_dirs"].items():
        argv += ["--full", f"{arm}={path}"]
    for arm, path in kwargs["resumed_dirs"].items():
        argv += ["--resumed", f"{arm}={path}"]
    old = sys.argv
    sys.argv = argv
    try:
        B.main()
        return "returned"
    except SystemExit as exc:
        return f"SystemExit({exc.code})"
    except Exception as exc:
        return type(exc).__name__ + " | " + str(exc).split("/")[-1]
    finally:
        sys.argv = old


def main(base):
    # 1. Real readers on a missing start.json and on a missing trace.
    for what in ("start.json", "trace-0.jsonl"):
        mp, tmp, (kwargs, *_rest) = fixture(base)
        import contrastive_target_resume as module

        def compare(paths, *args, **kw):
            if what == "start.json":
                C.read(Path(paths["reload_a"]) / "start.json")           # as at :324
            else:
                C.rows(Path(paths["continuous"]) / "trace-0.jsonl", 544, 576)  # as at :359
            return {"status": "PASSED", "passed": True}

        mp.setattr(module, "compare_three_paths", compare)
        try:
            B.read_preflight(**kwargs)
            print(what, "read_preflight -> returned")
        except Exception as exc:
            print(what, "read_preflight ->", type(exc).__name__, "|", str(exc).split("/")[-1])
        out = tmp / "receipt" / "preflight.json"
        print(what, "main ->", run_main(kwargs, out), "| receipt written:", out.exists())
        mp.undo()

    # 2. Designed INCONCLUSIVE paths, for contrast.
    mp, tmp, (kwargs, parity_path, *_rest) = fixture(base)
    parity_path.unlink()
    out = tmp / "receipt" / "preflight.json"
    print("missing parity main ->", run_main(kwargs, out), "| receipt written:", out.exists())
    if out.exists():
        r = json.loads(out.read_text())
        print("  status", r["status"], "| inconclusive", r["inconclusive_reasons"],
              "| ratio present", "ratio_CN_over_CR" in r, "| resume read", sorted(k for k in r["arms"] if "resume" in r["arms"][k]))
    mp.undo()

    mp, tmp, (kwargs, *_rest) = fixture(base)
    (kwargs["resumed_dirs"]["CN_A"] / "numeric_profile.rank1.json").unlink()
    out = tmp / "receipt" / "preflight.json"
    print("missing numeric receipt main ->", run_main(kwargs, out), "| receipt written:", out.exists())
    if out.exists():
        r = json.loads(out.read_text())
        print("  status", r["status"], "| arms read", sorted(r["arms"]), "| ratio present", "ratio_CN_over_CR" in r,
              "| parity present", "zero_weight_parity" in r)
    mp.undo()


if __name__ == "__main__":
    main(sys.argv[1])
