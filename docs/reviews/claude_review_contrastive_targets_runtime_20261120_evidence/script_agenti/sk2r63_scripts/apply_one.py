import json, sys
from pathlib import Path
clone, mutfile, mid = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
m = {x["id"]: x for x in json.loads(mutfile.read_text())}[mid]
p = clone / m["file"]; t = p.read_text()
assert t.count(m["old"]) == 1
p.write_text(t.replace(m["old"], m["new"])); print("applied", mid, m["file"])
