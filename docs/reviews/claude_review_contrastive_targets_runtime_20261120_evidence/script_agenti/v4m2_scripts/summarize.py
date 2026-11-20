"""Summarize mut_full.tsv: verdict counts and the mapping to the reviewer's MU ids."""
import collections
import re
import sys
from pathlib import Path

rows = [line.rstrip("\n").split("\t") for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
counts = collections.Counter(row[1].split()[0] for row in rows)
print("verdicts:", dict(counts))
for row in rows:
    mid, verdict, elapsed, detail = (row + ["", "", ""])[:4]
    mu = re.search(r"\((MU\d+)\)", mid)
    print(f"{mid:40s} {verdict:28s} {elapsed:>6s}  {detail[:160]}")
survivors = sorted({re.search(r'\((MU\d+)\)', r[0]).group(1) for r in rows if r[1].startswith("SURVIVED") and '(' in r[0]},
                   key=lambda s: int(s[2:]))
print("surviving reviewer ids (full 465-test set):", len(survivors), survivors)
