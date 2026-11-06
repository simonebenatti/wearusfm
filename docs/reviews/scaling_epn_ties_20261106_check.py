"""Reproduce floating-point ties in Claude's descriptive EPN subject counts.

Only reads the published final report. With 25 windows per class and six
classes for every subject, equal integer totals imply exactly equal BAcc.
No subject IDs are printed. This does not alter any prespecified score or CI.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
raw = args.report.read_bytes()
report = json.loads(raw)
epn = report["secondary"]["classification"]["datasets"]["epn612"]
counts = epn["paired_counts"]
control = "T100-G-30M-natural"
result = {"source_sha256": hashlib.sha256(raw).hexdigest(), "cells": {}}

def categories(values):
    return {"improved": sum(x > 0 for x in values),
            "worse": sum(x < 0 for x in values), "ties": sum(x == 0 for x in values)}

for cell in ("T100-G-300M-natural", "T100-G-300M-matched"):
    exact, floating, false_nonzero = [], [], []
    for subject in epn["subjects"]:
        a, b = (counts[c]["subjects"][subject] for c in (control, cell))
        assert a["n"] == b["n"] == [25] * 6
        integer_delta = sum(b["correct"]) - sum(a["correct"])
        float_delta = (sum(h / n for h, n in zip(b["correct"], b["n"])) / 6
                       - sum(h / n for h, n in zip(a["correct"], a["n"])) / 6)
        exact.append(integer_delta)
        floating.append(float_delta)
        if integer_delta == 0 and float_delta != 0:
            false_nonzero.append(float_delta)
    result["cells"][cell] = {"integer_counts": categories(exact),
                             "float_counts": categories(floating),
                             "false_nonzero_ties": false_nonzero}

args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2, allow_nan=False))
