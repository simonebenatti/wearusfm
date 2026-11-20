"""pytest plugin (skeptic 1, R5-2): record in-process line hits in four runtime modules.
Env SK1_CLONE = clone root, SK1_OUT = output JSON. Subprocesses are not traced."""
import json
import os
import sys
import threading
from pathlib import Path

_CLONE = Path(os.environ["SK1_CLONE"]).resolve()
_TARGETS = {str(_CLONE / "scripts" / name): name for name in
            ("contrastive_job.py", "contrastive_run.py", "contrastive_release.py", "contrastive_cuda_check.py")}
_HITS = {name: set() for name in _TARGETS.values()}
_CURRENT = {"test": None}
_BY_TEST = {}  # (file, line) -> set of test files


def _local(frame, event, arg):
    if event == "line":
        name = _TARGETS[frame.f_code.co_filename]
        _HITS[name].add(frame.f_lineno)
        if _CURRENT["test"]:
            _BY_TEST.setdefault(f"{name}:{frame.f_lineno}", set()).add(_CURRENT["test"])
    return _local


def _global(frame, event, arg):
    if event == "call" and frame.f_code.co_filename in _TARGETS:
        return _local
    return None


def pytest_configure(config):
    sys.settrace(_global)
    threading.settrace(_global)


def pytest_runtest_setup(item):
    _CURRENT["test"] = item.nodeid.split("::")[0]
    if sys.gettrace() is not _global:  # a test may have replaced it
        sys.settrace(_global)


def pytest_unconfigure(config):
    sys.settrace(None)
    out = {"hits": {k: sorted(v) for k, v in _HITS.items()},
           "by_test": {k: sorted(v) for k, v in _BY_TEST.items()}}
    Path(os.environ["SK1_OUT"]).write_text(json.dumps(out))
