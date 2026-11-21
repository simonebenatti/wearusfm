"""Verify physical receipts and exercise the frozen CLI parser without executing read_preflight."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

import contrastive_benchmark_readout as reader
from contrastive_job import build_command
from contrastive_prepare import _read
from contrastive_release import checked_pin, make_job

PACKAGE = Path("results/infra/contrastive_targets_20261118")
RELEASE = PACKAGE / "releases/readout_20261121"
BINDING = PACKAGE / "releases/audit_20261120/review_binding.json"
CALIBRATION = PACKAGE / "calibration_execution_20261121"
BENCHMARK = PACKAGE / "benchmark_execution_20261121"
RELOAD = PACKAGE / "reloads_execution_20261121"
FULL_ARMS = ("C0", "CN", "CR")
CELLS = ("CN_A", "CN_B", "CR_A", "CR_B")
JOB_ID = "CONTRAST-READOUT-targets-v3-20261118"


def verify_file(path, expected_hash, expected_size):
    if path.is_symlink() or path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected type or size: {path}")
    # Stream checkpoint bytes once, without torch.load or allocating tensor storage.
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
            raise ValueError(f"Physical file SHA differs: {path}")


def check_checkpoints(root):
    lines = (CALIBRATION / "checkpoint_remote_receipt.txt").read_text().splitlines()
    digest, path = lines[0].split("  ")
    calibration = root / "calibration/checkpoint.pt"
    if Path(path) != calibration:
        raise ValueError("Unexpected calibration checkpoint path")
    verify_file(calibration, digest, int(lines[1]))

    lines = (BENCHMARK / "checkpoints_remote.txt").read_text().splitlines()
    hashes = dict(reversed(line.split("  ")) for line in lines if "  " in line)
    sizes = dict(reversed(line.split(" bytes ")) for line in lines if " bytes " in line)
    required = [f"{arm}/checkpoint.pt" for arm in FULL_ARMS]
    required.extend(f"{arm}/checkpoint_step0000544.pt" for arm in ("CN", "CR"))
    for name in required:
        verify_file(root / "benchmark" / name, hashes[name], int(sizes[name]))


def check_reloads(root):
    collection = _read(RELOAD / "collection_readout.json")
    if (set(collection["cells"]) != set(CELLS) or collection.get("status") != "COLLECTED_AND_BYTE_VERIFIED"
            or collection.get("source_sha") != "12e215a978b9d30ea6d670f58af1a0f97387d477"):
        raise ValueError("Four verified reload receipts required")
    for cell in CELLS:
        folder = root / "reloads" / cell
        lines = (RELOAD / f"{cell}_remote_integrity_and_log.txt").read_text().splitlines()
        sha = {}
        sizes = {}
        for line in lines:
            match = re.fullmatch(r"([a-f0-9]{64})  (/.+)", line)
            size = re.fullmatch(r"(\d+) (/.+)", line)
            if match and Path(match[2]).parent == folder:
                sha[Path(match[2]).name] = match[1]
            if size and Path(size[2]).parent == folder:
                sizes[Path(size[2]).name] = int(size[1])
        mandatory = {"summary.json", "config.json", "start.json", "metrics.jsonl",
                     "origin_sessions.json", "checkpoint.pt"}
        for rank in range(4):
            mandatory.update((f"numeric_profile.rank{rank}.json", f"trace-{rank}.jsonl", f"trace-{rank}.ddp.jsonl"))
        receipt = collection["cells"][cell]
        if (not mandatory <= sha.keys() or sha.keys() != sizes.keys()
                or receipt["checkpoint"] != {"path": str(folder / "checkpoint.pt"), "bytes": sizes["checkpoint.pt"],
                                              "sha256": sha["checkpoint.pt"]}
                or receipt["summary_sha256"] != sha["summary.json"]):
            raise ValueError(f"Inconsistent reload inventory: {cell}")
        for name, digest in sha.items():
            verify_file(folder / name, digest, sizes[name])


def main():
    root = Path(os.environ["WORK"]).resolve() / "wearusfm_runs/results/contrastive_targets_20261118"
    output = root / "preflight_readout.json"
    job_path = RELEASE / "readout.job.json"
    job = _read(job_path)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Readout output already present: {output}")
    expected_job = make_job(_read(BINDING), JOB_ID, _read(RELEASE / "readout.evidence.json"), output)
    if (json.dumps(expected_job, indent=2, allow_nan=False) + "\n").encode() != job_path.read_bytes():
        raise ValueError("Regenerated readout job differs from release")

    command = build_command(job, "cpu", 3600)
    if command[:2] != [sys.executable, str(Path(reader.__file__).resolve())]:
        raise ValueError("Unexpected readout worker")
    full = {arm: root / "benchmark" / arm for arm in FULL_ARMS}
    reloads = {cell: root / "reloads" / cell for cell in CELLS}
    check_checkpoints(root)
    check_reloads(root)
    for key, dirs in (("full", full), ("resumed", reloads)):
        for label, folder in dirs.items():
            summary = checked_pin(job[key][label])
            values = _read(summary)
            if (summary != folder / "summary.json" or values["steps"] != 576 or values["stopped"] != "max_steps"
                    or values["contrastive"]["config"]["arm"] != label.split("_")[0] or (folder / "STOP").exists()):
                raise ValueError(f"Mismatch between summary, arm and physical endpoint: {label}")
    validate_reader_command(command, root, full, reloads)
    print(json.dumps({"id": JOB_ID, "execution_started": False, "reader_argv_valid": True,
                      "verified_full_arms": list(full), "verified_reloads": list(reloads),
                      "physical_checkpoints_verified": 10, "output": str(output),
                      "execution_argv": command}, indent=2))


def validate_reader_command(command, root, full, reloads):
    # argparse parses individual receipt paths as Path; ARM=directory mappings keep strings.
    cohort = (PACKAGE / "releases/benchmark_20261121/cohort/cohort.json").resolve()
    expected_args = (cohort, {arm: str(folder) for arm, folder in full.items()},
                     {cell: str(folder) for cell, folder in reloads.items()}, root / "cuda.json")
    expected_kwargs = {"cpu_audit_path": root / "audit.json",
                       "calibration_path": root / "calibration_report.json",
                       "calibration_run_dir": root / "calibration",
                       "cpu_audit_sacct_path": (PACKAGE / "audit_execution_20261120/audit_accounting.json").resolve()}
    if command.count("--out") != 1 or command[command.index("--out") + 1] != str(root / "preflight_readout.json"):
        raise ValueError("Wrong readout output")

    class Parsed(Exception):
        pass

    def stop_after_parse(*args, **kwargs):
        if args != expected_args or kwargs != expected_kwargs:
            raise ValueError("Frozen reader parsed different arm/receipt paths")
        raise Parsed()

    # Use the actual reader CLI up to the dispatch boundary. The scientific function
    # is replaced by this sentinel only on login; no evidence is evaluated or written.
    with patch.object(sys, "argv", command[1:]), patch.object(reader, "read_preflight", stop_after_parse):
        try:
            reader.main()
        except Parsed:
            pass
        else:
            raise ValueError("Frozen reader did not reach expected dispatch")


if __name__ == "__main__":
    main()
