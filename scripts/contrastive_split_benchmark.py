"""Run a reviewed 544+32-update resume check inside ONE bounded allocation.

The two torchrun processes are distinct; no implicit retries or recipe changes.
A signed split release must explicitly authorize the full config and total cap.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from contrastive_prepare import _read, source_digest
from contrastive_run import sha, validate_resume

RUNNER = Path(__file__).with_name("contrastive_run.py")


def checked_command(command, deadline):
    """Bound the launcher group; torchrun closes workers and SLURM bounds other sessions."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("paired benchmark allocation exhausted before phase")
    child = subprocess.Popen(command, start_new_session=True)
    try:
        code = child.wait(timeout=remaining)
        if code:
            raise RuntimeError(f"paired benchmark phase failed ({code}); no retry")
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise


def validate_pair_policy(initial, split_config, full_config):
    policy = initial.get("paired_resume")
    if not isinstance(policy, dict) or policy.get("config_sha256") != sha(full_config):
        raise ValueError("split release does not authorize this continuation config")
    if initial.get("stage") != "benchmark" or initial.get("phase") != "split" or initial.get("arm") not in ("CN", "CR"):
        raise ValueError("paired resume check requires a reviewed CN/CR split release")
    before, after = _read(split_config), _read(full_config)
    expected = copy.deepcopy(before)
    expected["max_steps"] = 576
    if before.get("max_steps") != 544 or after != expected:
        raise ValueError("resume changes more than the frozen 544-to-576 endpoint")
    if initial.get("config_sha256") != sha(split_config) or initial.get("source_sha256") != source_digest():
        raise ValueError("split config/source changed after review")
    maximum = policy.get("maximum_resume_time_limit_s")
    if type(maximum) not in (int, float) or not 0 < maximum <= 1800:
        raise ValueError("invalid reviewed continuation time bound")
    return maximum


def continuation_release(initial, split_config, full_config, out_dir, remaining):
    """Derive only the continuation explicitly authorized in the reviewed input."""
    maximum = validate_pair_policy(initial, split_config, full_config)
    if type(remaining) not in (int, float) or not 0 < remaining <= maximum:
        raise ValueError("resume time is outside the reviewed bound")
    release = copy.deepcopy(initial)
    release.update(phase="resume", config_sha256=sha(full_config), time_limit_s=remaining,
                   resume_checkpoint_sha256=sha(Path(out_dir) / "checkpoint.pt"),
                   resume_summary_sha256=sha(Path(out_dir) / "summary.json"))
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--resume-config", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--total-time-limit-s", type=int, default=1800)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    shared = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    forbidden = {"--config", "--release", "--out-dir", "--stage", "--time-limit-s", "--validate-only", "--recipe-only"}
    if any(value.split("=", 1)[0] in forbidden for value in shared):
        parser.error("shared arguments cannot override phase/config/release/time or validation mode")
    initial = _read(args.release)
    policy = initial.get("paired_resume", {})
    if (type(args.total_time_limit_s) is not int or args.total_time_limit_s != 1800 or
            policy.get("total_time_limit_s") != args.total_time_limit_s or
            initial.get("time_limit_s") != args.total_time_limit_s):
        raise ValueError("paired benchmark requires one explicitly reviewed 1800-second allocation")
    if args.out_dir.exists():
        raise ValueError("paired benchmark requires a new output directory")
    validate_pair_policy(initial, args.split_config, args.resume_config)
    if "--cohort" not in shared:
        parser.error("shared --cohort is required")
    cohort = _read(Path(shared[shared.index("--cohort") + 1]))
    deadline = time.monotonic() + args.total_time_limit_s - 30  # reserve process cleanup outside model timing
    base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", str(RUNNER),
            "--stage", "benchmark", "--out-dir", str(args.out_dir), *shared]
    checked_command([*base, "--config", str(args.split_config), "--release", str(args.release),
                     "--time-limit-s", str(initial["time_limit_s"])], deadline)
    remaining = deadline - time.monotonic()
    release = continuation_release(initial, args.split_config, args.resume_config, args.out_dir, remaining)
    validate_resume(release, args.out_dir, cohort)
    # Preserve the split receipt before the resumed trainer overwrites summary.json.
    (args.out_dir / "split_summary.json").write_bytes((args.out_dir / "summary.json").read_bytes())
    (args.out_dir / "split_checkpoint_sha256.txt").write_text(release["resume_checkpoint_sha256"] + "\n")
    resume_path = args.out_dir / "resume_release.json"
    with resume_path.open("x") as out:
        json.dump(release, out, indent=2, allow_nan=False)
        out.write("\n")
    checked_command([*base, "--config", str(args.resume_config), "--release", str(resume_path),
                     "--time-limit-s", str(remaining)], deadline)
    summary = _read(args.out_dir / "summary.json")
    if summary.get("steps") != 576 or summary.get("stopped") != "max_steps":
        raise RuntimeError("paired benchmark did not complete all 576 updates")
    print(json.dumps({"status": "PAIRED_BENCHMARK_COMPLETE", "arm": initial["arm"], "steps": 576}))


if __name__ == "__main__":
    main()
