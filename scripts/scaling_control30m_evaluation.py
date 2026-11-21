"""Extract the existing 30M terminal control independently of the 300M training.

Only the plan-shape guard changes. The numeric extraction, input/checkpoint
validation, output finalization and receipt writer are the frozen 49k functions.
No placeholder 300M endpoint is accepted or introduced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import scaling_evaluation49046 as E


FROZEN_EVALUATION_SHA256 = "3720836c6a01d3239ba0af2d513eb2f3a1830f6307cb026ec5ca1010fe476890"
_ORIGINAL_LOAD_CONTEXT = E.load_context
_ORIGINAL_GUARD = '''    require(plan.get("protocol") == PROTOCOL and plan.get("arms") == list(ARMS) and
            type(step) is int and step == plan.get("step") == STEP, "wrong scaling extraction protocol/dose")
    require(arm in ARMS and set(plan["checkpoints"]) == set(ARMS) and
            plan.get("control_checkpoint_sha256") == CONTROL_CHECKPOINT_SHA256,
            "exactly the actual 300M endpoint pin and the registered control are required; both checkpoint pins are registered")'''
_CONTROL_GUARD = '''    require(plan.get("protocol") == PROTOCOL and plan.get("control_only") is True and
            plan.get("arms") == [ARMS[0]] and arm == ARMS[0] and
            type(step) is int and step == plan.get("step") == STEP,
            "explicit control-only 30M terminal plan required")
    require(set(plan["checkpoints"]) == {ARMS[0]} and
            plan.get("control_checkpoint_sha256") == CONTROL_CHECKPOINT_SHA256,
            "exactly the registered 30M control checkpoint pin is required")'''


def derived_loader_source():
    """Declare and verify the single non-numeric substitution against 87da975."""
    data = Path(E.__file__).read_bytes()
    E.require(hashlib.sha256(data).hexdigest() == FROZEN_EVALUATION_SHA256,
              "frozen 49k extraction source differs")
    source = inspect.getsource(_ORIGINAL_LOAD_CONTEXT)
    E.require(source.count(_ORIGINAL_GUARD) == 1, "control guard derivation is ambiguous")
    return source.replace(_ORIGINAL_GUARD, _CONTROL_GUARD, 1)


def _make_control_loader():
    namespace = dict(vars(E))
    exec(compile(derived_loader_source(), "<scaling-control30m-load-context>", "exec"), namespace)
    return namespace["load_context"]


load_context = _make_control_loader()


@contextmanager
def _control_context():
    """Temporarily select the derived loader in this single-process adapter."""
    E.require(E.load_context is _ORIGINAL_LOAD_CONTEXT, "extraction loader already rebound")
    E.load_context = load_context
    try:
        yield
    finally:
        E.load_context = _ORIGINAL_LOAD_CONTEXT


def execute(stage, plan_path, arm, step, expected_code_sha, out, task=None, device="cuda"):
    E.require(arm == E.ARMS[0], "control-only extraction cannot select the 300M")
    with _control_context():
        return E.execute(stage, plan_path, arm, step, expected_code_sha, out, task, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "extract-pose", "extract-classification"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=(E.ARMS[0],), required=True)
    parser.add_argument("--step", type=int, choices=(E.STEP,), required=True)
    parser.add_argument("--task", choices=tuple(E.DATASETS))
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    execute(args.stage, args.plan, args.arm, args.step, args.expected_code_sha,
            args.out, args.task, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
