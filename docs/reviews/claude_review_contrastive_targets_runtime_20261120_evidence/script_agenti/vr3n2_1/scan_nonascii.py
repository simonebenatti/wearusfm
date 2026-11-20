import subprocess, sys, pathlib
root = pathlib.Path(sys.argv[1])
files = subprocess.check_output(["git", "ls-files", "results"], cwd=root, text=True).split("\n")
hits = []
for rel in files:
    if not rel or not (rel.endswith(".out") or rel.endswith(".log") or rel.endswith(".err") or rel.endswith("stdout.log") or rel.endswith(".txt")):
        continue
    p = root / rel
    try:
        b = p.read_bytes()
    except Exception:
        continue
    n = sum(1 for x in b if x > 127)
    if n:
        hits.append((rel, n, len(b)))
for h in hits[:80]:
    print(*h)
print("total files with non-ascii:", len(hits))
