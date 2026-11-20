"""Recompute contrastive_prepare.source_digest() from git objects of a commit (no working tree)."""
import fnmatch, hashlib, subprocess, sys
from pathlib import PurePosixPath
repo, commit = sys.argv[1], sys.argv[2]
names = subprocess.check_output(['git', '-C', repo, 'ls-tree', '-r', '--name-only', commit], text=True).splitlines()
def blob(p): return subprocess.check_output(['git', '-C', repo, 'show', f'{commit}:{p}'])
src = sorted([n for n in names if n.startswith('src/wearusfm/') and n.endswith('.py')], key=lambda s: PurePosixPath(s).parts)
scr = sorted([n for n in names if PurePosixPath(n).parent == PurePosixPath('scripts') and fnmatch.fnmatch(PurePosixPath(n).name, 'contrastive_*.py')])
sb = sorted([n for n in names if PurePosixPath(n).parent == PurePosixPath('scripts/slurm') and fnmatch.fnmatch(PurePosixPath(n).name, 'contrastive*.sbatch')])
paths = src + scr + sb + ['scripts/train_jepa.py', 'scripts/slurm/lib_code_dir.sh']
d = hashlib.sha256()
for p in paths: d.update(p.encode() + b'\0' + hashlib.sha256(blob(p)).digest())
print(commit, d.hexdigest(), 'files', len(paths), 'scripts', len(scr), 'sbatch', sb)
