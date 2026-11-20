"""pytest plugin (-p r53_lines): record executed lines of contrastive_target_resume.py in this process."""
import json
import os
import sys
import threading

TARGET = "/scripts/contrastive_target_resume.py"   # not tests/cpu/test_contrastive_target_resume.py
HITS = set()


def _local(frame, event, arg):
    if event == "line":
        HITS.add(frame.f_lineno)
    return _local


def _global(frame, event, arg):
    if frame.f_code.co_filename.endswith(TARGET):
        HITS.add(frame.f_lineno)
        return _local
    return None


def pytest_configure(config):
    sys.settrace(_global)
    threading.settrace(_global)


def pytest_unconfigure(config):
    sys.settrace(None)
    out = os.environ.get("R53_LINES_OUT")
    if out:
        with open(out, "w") as stream:
            json.dump(sorted(HITS), stream)
