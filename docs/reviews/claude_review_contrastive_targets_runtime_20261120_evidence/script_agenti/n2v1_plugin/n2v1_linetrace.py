"""Minimal line tracer (no coverage package): records executed lines of selected files."""
import json, os, sys, threading
TARGETS = ("contrastive_target_resume.py", "contrastive_run.py", "contrastive_objectives.py")
HITS = {}
def _local(frame, event, arg):
    if event == "line":
        HITS.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
    return _local
def _global(frame, event, arg):
    if frame.f_code.co_filename.endswith(TARGETS):
        HITS.setdefault(frame.f_code.co_filename, set()).add(frame.f_lineno)
        return _local
    return None
sys.settrace(_global); threading.settrace(_global)
def pytest_unconfigure(config):
    sys.settrace(None)
    out = os.environ["N2V1_TRACE_OUT"]
    with open(out, "w") as f:
        json.dump({k: sorted(v) for k, v in HITS.items()}, f)
