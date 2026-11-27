"""Full positive flow: Claude review commit on REF -> bind at 8c72655c -> child commit -> clean checkout -> require_binding."""
import json, subprocess, sys
from bind_sim import *

body_bytes = body(json_text=json.dumps(block(), indent=2), post="```text\nesempio con cn-restart-review in prosa\n```\n")
c = review_commit(body_bytes)
git("checkout", "-q", "--detach", TARGET)
r = subprocess.run([PY, "-B", "scripts/diagnostics/contrastive_cn_restart.py", "--bind-review", c],
                   cwd=COPY, env=ENV, capture_output=True, text=True)
print("bind rc", r.returncode, r.stderr[-300:])
git("add", BINDING)
git("commit", "-q", "-m", "binding")
release = git("rev-parse", "HEAD")
print("release", release, "parent", git("rev-parse", "HEAD^"))
# second bind attempt in same checkout must refuse (write_new uses 'x')
r2 = subprocess.run([PY, "-B", "scripts/diagnostics/contrastive_cn_restart.py", "--bind-review", c],
                    cwd=COPY, env=ENV, capture_output=True, text=True)
print("rebind rc", r2.returncode, r2.stderr.strip().splitlines()[-1])
wt = W / "release_wt"
git("worktree", "add", "-q", "--detach", str(wt), release)
print("status clean:", repr(git("status", "--porcelain", cwd=wt)))
code = ("import sys; sys.path.insert(0,'scripts/diagnostics'); import contrastive_cn_restart as g; "
        "p=g.require_binding(); print('require_binding OK', p['status'], p['training_approved']); print(g.package()[1])")
r3 = subprocess.run([PY, "-B", "-c", code], cwd=wt, env=ENV, capture_output=True, text=True)
print(r3.stdout, r3.stderr[-400:])
# tamper: modify review_text in binding (and keep sha) -> provenance error
p = wt / BINDING
subprocess.run(["chmod", "u+w", str(p)])
d = json.loads(p.read_text()); d["review_text"] += "x"; p.write_text(json.dumps(d))
r4 = subprocess.run([PY, "-B", "-c", code], cwd=wt, env=ENV, capture_output=True, text=True)
print("tampered:", r4.stderr.strip().splitlines()[-1])
git("checkout", "--", BINDING, cwd=wt)
# release where a FILES member changed after binding -> must refuse
(wt / "docs/contrastive_cn_restart_probe_20261121.md").write_text("altered\n")
r5 = subprocess.run([PY, "-B", "-c", code], cwd=wt, env=ENV, capture_output=True, text=True)
print("protocol altered:", r5.stderr.strip().splitlines()[-1][:200])
git("checkout", "--", ".", cwd=wt)
git("worktree", "remove", "--force", str(wt))
