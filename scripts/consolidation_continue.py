"""Review-gated continuation of T100-G, preserving the original WSD recipe.

No submission is performed here. Run only in a frozen compute-node checkout.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import tokenizer_train as T
import train_jepa as J
from wearusfm.training import run as R

PROTOCOL = "consolidation-t100g-20261104-v1"
PHASES = (30000, 39237, 49046)
BASE = ROOT / "results/infra/tokenizer_execution_20261103"
SOURCE = BASE / "19618/T100-G/receipt.json"
WORK_ROOT = "/leonardo_work/IscrB_WearUsFM"


def quota_values(project_text, quota_text, work_root=WORK_ROOT):
    """Read numeric Lustre project quota, never filesystem-wide df availability."""
    rows = [line.split() for line in project_text.splitlines() if line.strip()]
    T.require(len(rows) == 1 and len(rows[0]) == 3 and rows[0][2] == work_root and
              rows[0][0].isdigit(), "unrecognised lfs project response")
    project_id = int(rows[0][0])
    quota_project = re.search(r"\(pid\s+(\d+)\)", quota_text)
    T.require(quota_project is not None and int(quota_project.group(1)) == project_id,
              "project quota header identity mismatch or absent")
    tokens = quota_text.split()
    filesystem_rows = [i for i, token in enumerate(tokens) if token in (work_root, str(Path(work_root).parent))]
    T.require("kbytes" in tokens and "files" in tokens and len(filesystem_rows) == 1,
              "unrecognised numeric lfs quota response")
    fields = tokens[filesystem_rows[0] + 1:][:8]
    T.require(len(fields) == 8, "incomplete project quota row")
    values = []
    for index in (0, 1, 2, 4, 5, 6):
        value = fields[index].rstrip("*")
        T.require(value.isdigit(), "non-numeric project quota field")
        values.append(int(value))
    used_kib, soft_kib, hard_kib, used_files, soft_files, hard_files = values
    limits = [v for v in (soft_kib, hard_kib) if v > 0]
    T.require(limits, "project block quota missing; cannot certify headroom")
    file_limits = [v for v in (soft_files, hard_files) if v > 0]
    return {"project_id": project_id, "available_bytes": (min(limits) - used_kib) * 1024,
            "available_files": min(file_limits) - used_files if file_limits else None}


def check_quota_evidence(evidence, policy, *, submission_time=None):
    T.require(isinstance(evidence, dict), "WORK quota preflight is still pending")
    T.require(evidence.get("work_root") == WORK_ROOT, "wrong quota filesystem")
    when = datetime.fromisoformat(evidence["checked_at_utc"])
    T.require(when.utcoffset() == timedelta(0), "quota timestamp must be UTC")
    if submission_time is not None:
        T.require(submission_time.tzinfo is not None and timedelta(0) <= submission_time - when <= timedelta(hours=24),
                  "quota evidence must be within 24 hours before submission")
    values = quota_values(evidence["project_output"], evidence["quota_output"])
    T.require(values["available_bytes"] >= policy["min_headroom_bytes"], "insufficient WORK project byte quota")
    T.require(values["available_files"] is None or values["available_files"] >= policy["min_headroom_files"],
              "insufficient WORK project inode quota")
    return {**evidence, **values}


def live_quota(policy):
    project = subprocess.check_output(["lfs", "project", "-d", WORK_ROOT], text=True, timeout=20)
    rows = project.split()
    T.require(rows and rows[0].isdigit(), "cannot read WORK project ID")
    quota = subprocess.check_output(["lfs", "quota", "-p", rows[0], WORK_ROOT], text=True, timeout=20)
    return check_quota_evidence({"work_root": WORK_ROOT,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_output": project, "quota_output": quota}, policy)


def archive_checkpoint(source, destination, expected_sha, *, allow_existing=False):
    """Persist verified bytes without overwriting any earlier endpoint or partial copy."""
    source, destination = Path(source), Path(destination)
    before = source.stat()
    metadata = {"mtime_ns": before.st_mtime_ns, "atime_ns": before.st_atime_ns,
                "ctime_ns": before.st_ctime_ns}
    source_identity = T.P.file_identity(source)
    T.require(source_identity["sha256"] == expected_sha, "archive source differs from validated checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    T.require(not destination.is_symlink(), "archive destination cannot be a symlink")
    if destination.exists():
        T.require(allow_existing, "archive checkpoint already exists; no blind retry")
    else:
        partial = destination.with_name(destination.name + ".partial")
        with source.open("rb") as src, partial.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        T.require(T.P.file_identity(partial)["sha256"] == expected_sha, "archive copy hash mismatch")
        T.require(T.P.file_identity(source) == source_identity, "archive source changed while copying")
        # link is atomic and fails if a destination appeared concurrently; never replace it.
        os.link(partial, destination)
        partial.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    saved = T.P.file_identity(destination)
    T.require(saved["sha256"] == expected_sha and saved["bytes"] == source_identity["bytes"],
              "persistent checkpoint hash mismatch")
    return {"path": str(destination), "sha256": saved["sha256"], "bytes": saved["bytes"],
            "source": str(source), "source_metadata_before_archive": metadata,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(), "persistent_storage": "WORK project quota",
            "scratch_purge_estimate_utc": (datetime.fromtimestamp(before.st_mtime, timezone.utc) +
                                           timedelta(days=40)).isoformat(),
            "purge_estimate_basis": "source mtime + 40 days; conservative estimate, not scheduler guarantee"}


def command_for(receipt, step, run_dir, seconds=10800):
    T.require(step in PHASES and seconds == 10800, "unregistered continuation phase/budget")
    T.require(receipt["arm"] == "T100-G" and receipt["checkpoint_step"] == 19618 and receipt["complete"], "wrong source")
    argv = list(receipt["command"])
    for name, value in (("--max-steps", step), ("--out-dir", run_dir), ("--time-limit-s", seconds - 300)):
        T.require(argv.count(name) == 1, "ambiguous argument " + name)
        argv[argv.index(name) + 1] = str(value)
    if step == PHASES[0]:
        argv += ["--branch-from", receipt["checkpoint"]]
    args = J.build_parser().parse_args(argv[4:])
    cfg = J.build_config(args)
    old = json.loads((BASE / "19618/T100-G/config.json").read_text())
    new = R.config_to_dict(cfg)
    T.require({**new, "max_steps": old["max_steps"]} == old, "continuation changed scientific recipe")
    T.require(cfg.schedule.decay_start == 39237 and cfg.schedule.total_steps == 49046, "wrong WSD endpoints")
    return argv, cfg


def execute(plan_path, step, expected_code):
    code = T.P.code_identity(expected_code)
    identity = T.P.file_identity(plan_path)
    plan = json.loads(Path(plan_path).read_text())
    T.require(plan["protocol"] == PROTOCOL and plan["status"] == "claude_review_converged", "Claude review is required before execution")
    T.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_NODELIST"),
              "continuation and persistent archiving require the compute-node SLURM launcher")
    T.require(plan["phases"] == list(PHASES), "phase plan changed")
    policy = plan["retention"]
    T.require(policy["scratch_retention_days"] == 40 and policy["protected_steps"] == [19618, *PHASES],
              "retention policy changed")
    check_quota_evidence(policy["quota_preflight"], policy)
    quota_at_start = live_quota(policy)
    receipt = json.loads(SOURCE.read_text())
    T.require(T.P.file_identity(SOURCE)["sha256"] == plan["source_receipt_sha256"], "source receipt changed")
    # Training code remains byte-identical to the original tokenizer campaign.
    for rel, sha in json.loads((BASE / "training_plan.json").read_text())["scientific_source_sha256"].items():
        T.require(R.sha256_file(ROOT / rel) == sha, "scientific source changed: " + rel)
    run_dir, report_root = Path(plan["run_dir"]), Path(plan["report_root"])
    for path in (run_dir, report_root):
        T.require(path.is_absolute() and not path.resolve().is_relative_to(ROOT) and
                  "wearusfm_campaigns" not in path.parts and "wearusfm_runs" in path.parts, "unsafe output path")
    archive_root = Path(policy["persistent_root"])
    T.require(archive_root.resolve().is_relative_to(Path(WORK_ROOT) / "wearusfm_runs/results") and
              archive_root.resolve() == (report_root / "checkpoints").resolve(), "unsafe persistent archive path")
    out = report_root / str(step)
    T.require(not out.exists() and not out.is_symlink(), "phase output already exists; no blind retry")
    command, cfg = command_for(receipt, step, run_dir)
    _, configs, inputs = T.P.load_plan(BASE / "preflight_plan.json")
    scales = json.loads(Path(inputs["scales"]["path"]).read_text())["scales"]
    windows = json.loads(Path(inputs["windows"]["path"]).read_text())["window_s"]
    signature = json.loads(json.dumps(R.calibration_signature(cfg, inputs["manifest"]["path"], scales, windows)))
    source = Path(receipt["checkpoint"])
    source_id = T.checkpoint_identity(source, replace(configs["T100-G"], max_steps=19618), 19618, signature)
    T.require(source_id["sha256"] == receipt["checkpoint_sha256"], "source checkpoint changed")
    if step == PHASES[0]:
        T.require(not run_dir.exists() and not run_dir.is_symlink(), "new branch directory required")
    else:
        previous = PHASES[PHASES.index(step) - 1]
        prior = json.loads((report_root / str(previous) / "receipt.json").read_text())
        T.require(prior["complete"] is True and prior["step"] == previous and prior["code_sha"] == code and
                  prior["plan_sha256"] == identity["sha256"], "previous phase not verified")
        resume = T.checkpoint_identity(run_dir / "checkpoint.pt", replace(cfg, max_steps=previous), previous, signature)
        T.require(resume["sha256"] == prior["resume_sha256"], "resume checkpoint differs from receipt")
        T.require(not (run_dir / "STOP").exists(), "STOP present")
    out.mkdir(parents=True)
    T.P.write_json(out / "launch.json", {"command": command, "code_sha": code, "source": source_id,
                                        "quota_at_start": quota_at_start})
    source_archive = archive_checkpoint(source, archive_root / "checkpoint_step0019618.pt",
                                        source_id["sha256"], allow_existing=step != PHASES[0])
    T.P.write_json(out / "source_archive.json", source_archive)
    subprocess.run(command, cwd=ROOT, check=True)
    summary = json.loads((run_dir / "summary.json").read_text())
    T.require(summary.get("steps") == step and summary.get("stopped") == "max_steps" and
              not (run_dir / "STOP").exists(), "phase incomplete; stop dependent jobs")
    final = T.checkpoint_identity(run_dir / "checkpoint.pt", cfg, step, signature)
    names = {30000: "checkpoint_step0030000.pt", 39237: "checkpoint_predecay.pt", 49046: "checkpoint_postdecay.pt"}
    named = run_dir / names[step]
    if step == 30000:  # operational boundary only; exact copy, no optimizer/RNG edits
        T.require(not named.exists(), "named checkpoint already exists")
        shutil.copyfile(run_dir / "checkpoint.pt", named)
    saved = T.checkpoint_identity(named, cfg, step, signature)
    T.require(saved["trained_state_sha256"] == final["trained_state_sha256"], "named/final state mismatch")
    # A training endpoint is complete only after durable WORK retention succeeds.
    quota_before_archive = live_quota(policy)
    archived = archive_checkpoint(named, archive_root / names[step], saved["sha256"])
    T.require(T.P.file_identity(source)["sha256"] == source_id["sha256"], "source was modified")
    T.P.check_unchanged(inputs)
    T.require(T.P.file_identity(plan_path) == identity and T.P.code_identity(code) == code, "code or plan changed")
    T.P.write_json(out / "receipt.json", {"protocol": PROTOCOL, "complete": True, "step": step,
        "code_sha": code, "plan_sha256": identity["sha256"], "checkpoint": str(named),
        "checkpoint_sha256": saved["sha256"], "resume_sha256": final["sha256"], "summary": summary,
        "persistent_checkpoint": archived, "persistent_source": source_archive,
        "quota_at_start": quota_at_start, "quota_before_archive": quota_before_archive})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--step", type=int, choices=PHASES, required=True)
    p.add_argument("--expected-code", required=True)
    a = p.parse_args()
    execute(a.plan, a.step, a.expected_code)
