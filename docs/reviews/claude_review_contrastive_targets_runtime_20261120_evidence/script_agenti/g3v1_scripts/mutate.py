"""Replace given 1-based line ranges with an indented 'pass' and print what was removed."""
import sys
from pathlib import Path
path = Path(sys.argv[1])
ranges = [tuple(map(int, r.split('-'))) for r in sys.argv[2:]]
lines = path.read_text().splitlines(keepends=True)
for start, end in sorted(ranges, reverse=True):
    removed = lines[start-1:end]
    indent = removed[0][:len(removed[0]) - len(removed[0].lstrip())]
    sys.stdout.write(f"--- {path.name}:{start}-{end} removed:\n" + "".join(removed))
    lines[start-1:end] = [indent + "pass  # MUTATION\n"]
path.write_text("".join(lines))
