"""Review-gated continuation of T100-G, preserving the original WSD recipe.

No submission is performed here. Run only in a frozen compute-node checkout.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
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
    T.require(plan["phases"] == list(PHASES), "phase plan changed")
    receipt = json.loads(SOURCE.read_text())
    T.require(T.P.file_identity(SOURCE)["sha256"] == plan["source_receipt_sha256"], "source receipt changed")
    # Training code remains byte-identical to the original tokenizer campaign.
    for rel, sha in json.loads((BASE / "training_plan.json").read_text())["scientific_source_sha256"].items():
        T.require(R.sha256_file(ROOT / rel) == sha, "scientific source changed: " + rel)
    run_dir, report_root = Path(plan["run_dir"]), Path(plan["report_root"])
    for path in (run_dir, report_root):
        T.require(path.is_absolute() and not path.resolve().is_relative_to(ROOT) and
                  "wearusfm_campaigns" not in path.parts and "wearusfm_runs" in path.parts, "unsafe output path")
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
    T.P.write_json(out / "launch.json", {"command": command, "code_sha": code, "source": source_id})
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
    T.require(T.P.file_identity(source)["sha256"] == source_id["sha256"], "source was modified")
    T.P.check_unchanged(inputs)
    T.require(T.P.file_identity(plan_path) == identity and T.P.code_identity(code) == code, "code or plan changed")
    T.P.write_json(out / "receipt.json", {"protocol": PROTOCOL, "complete": True, "step": step,
        "code_sha": code, "plan_sha256": identity["sha256"], "checkpoint": str(named),
        "checkpoint_sha256": saved["sha256"], "resume_sha256": final["sha256"], "summary": summary})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--step", type=int, choices=PHASES, required=True)
    p.add_argument("--expected-code", required=True)
    a = p.parse_args()
    execute(a.plan, a.step, a.expected_code)
