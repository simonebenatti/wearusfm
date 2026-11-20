"""Skeptic R5-1: instrument scripts/contrastive_release.py with zero-cost sentinels.

Each `raise ValueError` gets a preceding `_sk("L<orig line>")`; each top-level def gets
`_sk("enter:<name>")` as first body statement; a few value probes are added.
Sentinels append `PYTEST_CURRENT_TEST|tag` to $SK1_TRACE (no-op when unset).
Usage: instrument.py <clone>
"""
import re
import sys
from pathlib import Path

clone = Path(sys.argv[1])
path = clone / "scripts/contrastive_release.py"
lines = path.read_text().splitlines()
out = []
HELPER = ("import os as _sk_os\n"
          "def _sk(tag):\n"
          "    p = _sk_os.environ.get('SK1_TRACE')\n"
          "    if p:\n"
          "        with open(p, 'a') as s:\n"
          "            s.write(_sk_os.environ.get('PYTEST_CURRENT_TEST', '?').replace('|', '/') + '|' + tag + '\\n')\n")
probes = {
    # after original line N (1-based) insert code with given indentation
    91: '    _sk("L91:nblocks=%d" % len(blocks))',
    94: '    _sk("L94:status=%r keys_ok=%r schema_ok=%r" % (payload.get("status"), set(payload) == {"schema", "status", "reviewed_commit", "source_sha256", "science_plan_sha256", "preflight_plan_sha256"}, payload.get("schema") == REVIEW_SCHEMA))',
    116: '    _sk("L116:src_ok=%r sci_ok=%r pre_ok=%r" % (declared["source_sha256"] == source_digest(), declared["science_plan_sha256"] == file_sha(science), declared["preflight_plan_sha256"] == file_sha(preflight)))',
    125: '        _sk("L125:verify_reviewed_sources_done")',
    128: '        _sk("L128:plan_bytes_done")',
    133: '    _sk("L134:binding_ok")',
}
for number, line in enumerate(lines, start=1):
    if number == 17:  # after `import sys`
        out.append(line)
        out.extend(HELPER.rstrip("\n").split("\n"))
        continue
    m = re.match(r"^(\s*)raise ValueError", line)
    if m:
        out.append(f'{m.group(1)}_sk("L{number}")')
    out.append(line)
    d = re.match(r"^def (\w+)\(.*\):\s*$", line)
    if d and d.group(1) != "_sk":
        out.append(f'    _sk("enter:{d.group(1)}")')
    if number in probes:
        out.append(probes[number])
path.write_text("\n".join(out) + "\n")
print("instrumented", path)
