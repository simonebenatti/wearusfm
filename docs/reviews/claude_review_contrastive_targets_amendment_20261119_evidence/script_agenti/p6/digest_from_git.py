"""Recompute contrastive_prepare.source_digest() from git blobs of a commit (read-only)."""
import fnmatch, hashlib, subprocess, sys
from pathlib import PurePosixPath

repo, commits = sys.argv[1], sys.argv[2:]

def tree(commit):
    out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--full-tree", commit],
                         capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        rows[path] = (mode, kind, sha)
    return rows

def blob(commit, path):
    return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, check=True).stdout

def digest(commit):
    rows = tree(commit)
    names = list(rows)
    g1 = sorted(PurePosixPath(p) for p in names if p.startswith("src/wearusfm/") and fnmatch.fnmatchcase(PurePosixPath(p).name, "*.py"))
    g2 = sorted(PurePosixPath(p) for p in names if PurePosixPath(p).parent == PurePosixPath("scripts") and fnmatch.fnmatchcase(PurePosixPath(p).name, "contrastive_*.py"))
    g3 = sorted(PurePosixPath(p) for p in names if PurePosixPath(p).parent == PurePosixPath("scripts/slurm") and fnmatch.fnmatchcase(PurePosixPath(p).name, "contrastive*.sbatch"))
    g4 = [PurePosixPath("scripts/train_jepa.py"), PurePosixPath("scripts/slurm/lib_code_dir.sh")]
    paths = g1 + g2 + g3 + g4
    symlinks = [str(p) for p in paths if rows.get(str(p), ("",))[0] == "120000"]
    h = hashlib.sha256()
    for p in paths:
        data = blob(commit, str(p))
        h.update(str(p).encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest(), len(paths), symlinks

for c in commits:
    d, n, links = digest(c)
    print(c, d, n, "symlinks=" + ",".join(links) if links else "")
