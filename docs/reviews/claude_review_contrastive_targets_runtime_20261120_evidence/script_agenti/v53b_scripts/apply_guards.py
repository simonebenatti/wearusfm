"""Insert env-switchable mutation guards into the temp clone's resume module.

With V53B_MUT unset the module behaves exactly as at fbfde797 (each guard is
`if <env> != id:` in front of the original statement). Line numbers are those of
fbfde797; each target line's expected text is asserted before editing.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text().split("\n")
SPECS = [  # (1-based line at fbfde797, id, expected stripped prefix)
    (82, "fork_rankrng", 'raise ValueError("common fork lacks separate CPU/CUDA RNG'),
    (105, "copy_bytes", 'raise ValueError("reload copy is not byte-identical'),
    (329, "start_json", 'raise ValueError("reload did not start from the same immutable'),
    (340, "endpoint", 'raise ValueError("incomplete or mismatched common-fork endpoint")'),
    (344, "counters", 'raise ValueError("common-fork summary counters or provenance differ")'),
    (354, "origin_map", 'raise ValueError("restart origin session mapping differs")'),
    (356, "origin_pin", 'raise ValueError("restart origin session mapping pin differs")'),
    (366, "boundary", 'boundary_errors.append({"phase": phase, "rank": rank, "field": field})'),
    (371, "rng_boundary", 'boundary_errors.append({"phase": phase, "rank": rank, "field": f"rng_{domain}_before_sha256"})'),
    (380, "row_inv", 'raise ValueError("invalid runtime restart trace invariants")'),
    (385, "weights", 'raise ValueError("restart contrastive support has inactive or changed weights")'),
    (388, "origin_trace", 'raise ValueError("restart origin trace is incomplete")'),
    (391, "trace_metrics", 'raise ValueError("restart support trace disagrees with production metrics")'),
    (394, "support_missing", 'support_missing.append({"phase": phase, "rank": rank, "update": row["update"], "arm": arm})'),
    (402, "pair_diff", 'differences.append({"rank": rank, "update": a["update"], "field": field})'),
]
for number, ident, prefix in SPECS:
    text = lines[number - 1]
    stripped = text.lstrip()
    assert stripped.startswith(prefix), (number, text)
    indent = text[: len(text) - len(stripped)]
    lines[number - 1] = f'{indent}if __import__("os").environ.get("V53B_MUT") != "{ident}": {stripped}'
path.write_text("\n".join(lines))
print("guards:", " ".join(ident for _, ident, _ in SPECS))
