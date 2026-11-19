"""Skeptic's own whitelist-widening mutation (W): compat manifest/whitelist accept the CURRENT sources.

Usage: python -B my_widen.py <temporary clone>
Only for the temporary clone below scratchpad/rev_targets/tmp/p2m1v1/.
"""
import hashlib
import json
import sys
from pathlib import Path

M = Path(sys.argv[1]).resolve()
assert "/scratchpad/rev_targets/tmp/p2m1v1/" in str(M), "refusing to touch anything else"
compat = M / "scripts/contrastive_source_compat.py"
manifest_path = M / "results/infra/contrastive_comparison_20261116/compatibility_20261118/manifest.json"
manifest = json.loads(manifest_path.read_text())
old = {r["path"]: r["sha256"] for r in manifest["historical_sources"]}


def inventory():
    paths = sorted((M / "src/wearusfm").rglob("*.py"))
    paths += sorted((M / "scripts").glob("contrastive_*.py"))
    paths += sorted((M / "scripts/slurm").glob("contrastive*.sbatch"))
    paths += [M / "scripts/train_jepa.py", M / "scripts/slurm/lib_code_dir.sh"]
    return [{"path": str(p.relative_to(M)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def digest(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(r["path"].encode() + b"\0" + bytes.fromhex(r["sha256"]))
    return h.hexdigest()


text = compat.read_text()
added = "    _require(set(current) - set(old) == {HELPER_PATH} and not set(old) - set(current),\n"
assert text.count(added) == 1
text = text.replace(added, "    _require(set(current) - set(old) <= ALLOWED_CHANGED_SOURCES and not set(old) - set(current),\n")
compat.write_text(text)

rows = inventory()
current = {r["path"]: r["sha256"] for r in rows}
changed = sorted({p for p in set(old) | set(current) if old.get(p) != current.get(p)})
assert "scripts/contrastive_source_compat.py" in changed
text = compat.read_text()
start = text.index("ALLOWED_CHANGED_SOURCES = frozenset({")
end = text.index("})\n", start) + 3
text = text[:start] + "ALLOWED_CHANGED_SOURCES = frozenset({\n" + "".join("    %r,\n" % p for p in changed) + "})\n" + text[end:]
compat.write_text(text)

rows = inventory()  # helper bytes changed again; the changed SET is unchanged
current = {r["path"]: r["sha256"] for r in rows}
assert sorted({p for p in set(old) | set(current) if old.get(p) != current.get(p)}) == changed
manifest["current_sources"] = rows
manifest["current_source_sha256"] = digest(rows)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"widened_changed_paths": len(changed), "added_paths": sorted(set(current) - set(old)),
                  "current_source_sha256": manifest["current_source_sha256"]}))
